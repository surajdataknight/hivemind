"""c14 customer history analyst: the customer's support profile over 180 days, as the business counts it from its
own records (blueprint 6.2, l3): orders, earlier claims by type and how they ended, money refunded, account age.
plain code. counts only: no transcript or free text of earlier cases reaches any model (data minimisation).

finding: history.profile, with the counts as facts and the claim-to-order ratio worked out here.
"""

from hivemind_agents.capabilities.checking import finding, rupees
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task

WINDOW_DAYS = 180


def ratio(claims: int, delivered: int) -> float:
    if delivered == 0:
        return 1.0 if claims else 0.0
    return round(claims / delivered, 3)


def run(task: Task, check: CheckInput) -> CheckResult:
    profile = check.backend.support_profile(WINDOW_DAYS)
    facts = {k: v for k, v in profile.items() if k not in ("evidence_id", "customer_id")}
    facts["claim_ratio"] = ratio(profile["claims"], profile["delivered_orders"])
    claim = (f"In {WINDOW_DAYS} days: {profile['delivered_orders']} orders delivered, {profile['claims']} earlier "
             f"claims ({profile['claims_last_30_days']} in the last 30 days), {rupees(profile['refunded_paise'])} "
             f"refunded for them; the account is {profile['account_age_days']} days old.")
    return CheckResult(findings=[finding(task, check, "history.profile", claim, [profile["evidence_id"]],
                                         facts=facts)])


CAPABILITY = Capability(
    name="customer_history",
    description="the customer's orders and earlier claims in 180 days, counted, and the account's age",
    params=(), scopes=frozenset({"history:read:self"}), run=run, cost="low")
