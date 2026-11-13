"""c15 risk & fraud investigator: rules in plain code over the customer's history, the claim's size against the order
and accounts that share a device, an address or a upi id with this one (blueprint 7.4; design: no trained model,
no model at all yet). it reads the history and the claimed value from the case board, and the shared-account counts
from the business.

what it may say (blueprint c15): a score from 0 to 1, a band (low, elevated, high), the signals that counted, and a
handling: proceed, proceed_with_cap, require_evidence or human_review. "deny" is never an output. it only decides
whether hivemind may act alone; proof of a store or delivery fault outranks it (the eligibility engine, phase 3),
and the customer is never told of it (the progress reporter shows it as "reviewing your request").

finding: risk.assessment.
"""

from hivemind_agents.capabilities.checking import finding, rupees
from hivemind_agents.capabilities.history_analyst import WINDOW_DAYS, ratio
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Finding, Task

RULES_VERSION = "risk_rules@1"
# signal -> weight. the score is their sum, at most 1
WEIGHTS = {
    "claim_velocity": 0.35,               # this claim makes 3 or more in 30 days
    "high_claim_ratio": 0.3,              # 3 or more claims, on 30% or more of delivered orders
    "high_claim_value_share": 0.2,        # the claim is over 60% of the order's item total
    "new_account_high_value": 0.3,        # account under 7 days old, claim of ₹500 or more
    "linked_accounts_with_claims": 0.4,   # another account sharing a device, address or upi id has claims
    "linked_accounts": 0.1,               # shared with another account that has no claims
    "earlier_claim_rejected_by_person": 0.15,
}
BANDS = ((0.6, "high"), (0.3, "elevated"), (0.0, "low"))
HANDLING = {"low": "proceed", "elevated": "proceed_with_cap", "high": "human_review"}


def claimed_value(check: CheckInput) -> tuple[int | None, int | None, list[str]]:
    """(the claimed value, the order's item total, the evidence they come from), from the order check's findings on
    the board; the customer's own amount when no item was matched."""
    refs: list[str] = []
    total = value = None
    for f in check.on_board("order.status"):
        total, refs = f.facts.get("item_total_paise"), refs + f.evidence_refs
    for f in check.on_board("order.claimed_lines"):
        value, refs = f.facts.get("claim_value_paise"), refs + f.evidence_refs
    if value is None and check.issue is not None:
        value = check.issue.claimed_amount_paise
    return value, total, refs


def assess(history: dict, links: list[dict], value: int | None, total: int | None) -> tuple[float, str, list[str]]:
    """(score, band, signals): the rules on their own, for tests and for the record."""
    signals = []
    claims, recent = history["claims"], history["claims_last_30_days"]
    if recent + 1 >= 3:
        signals.append("claim_velocity")
    if claims >= 3 and ratio(claims, history["delivered_orders"]) >= 0.3:
        signals.append("high_claim_ratio")
    if value and total and value > 0.6 * total:
        signals.append("high_claim_value_share")
    if history["account_age_days"] < 7 and value and value >= 50_000:
        signals.append("new_account_high_value")
    if any(link["accounts_with_claims"] for link in links):
        signals.append("linked_accounts_with_claims")
    elif any(link["accounts"] for link in links):
        signals.append("linked_accounts")
    if history["escalations_rejected"]:
        signals.append("earlier_claim_rejected_by_person")
    score = round(min(1.0, sum(WEIGHTS[s] for s in signals)), 2)
    band = next(name for floor, name in BANDS if score >= floor)
    return score, band, signals


def _history(check: CheckInput) -> tuple[dict, list[str]]:
    on_board: list[Finding] = check.on_board("history.profile")
    if on_board:
        return on_board[-1].facts, on_board[-1].evidence_refs
    profile = check.backend.support_profile(WINDOW_DAYS)  # the history check did not run: read the profile here
    return profile, [profile["evidence_id"]]


def run(task: Task, check: CheckInput) -> CheckResult:
    history, history_refs = _history(check)
    signals_read = check.backend.risk_signals(WINDOW_DAYS)
    value, total, order_refs = claimed_value(check)
    score, band, signals = assess(history, signals_read["links"], value, total)
    links = {row["link_type"]: row["accounts"] for row in signals_read["links"] if row["accounts"]}
    claim = (f"Risk band {band} (score {score:.2f}) from {len(signals)} signal(s); handling "
             f"{HANDLING[band]}." + (f" Claimed value {rupees(value)}." if value else ""))
    return CheckResult(findings=[finding(
        task, check, "risk.assessment", claim, [signals_read["evidence_id"], *history_refs, *order_refs],
        confidence="medium", basis=f"{RULES_VERSION}: rules over history, claim size and shared accounts",
        facts={"score": score, "band": band, "signals": signals, "handling": HANDLING[band],
               "rules_version": RULES_VERSION, "claim_value_paise": value, "order_item_total_paise": total,
               "linked_accounts": links})])


CAPABILITY = Capability(
    name="risk",
    description="rules over the customer's claims, the claim's size and accounts sharing a device, address or upi "
                "id; decides only whether hivemind may act alone",
    params=(), scopes=frozenset({"risk:read", "history:read:self"}), run=run, cost="low")
