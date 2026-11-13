"""the business's rules for what support may do on a case (blueprint c18): given what hivemind found, which parts
of the claim may be refunded or cancelled, up to how much, and whether support may do it alone or a person decides.

plain code, no model. amounts never come from the caller: a line's value, a capture, a failed debit and what was
paid for an order are read here from the business's own records. what the caller says is only what its
investigation concluded: the status of each claimed part, the customer's risk band, and whether the records conflict.
the action gateway runs this same engine again before it moves any money (defence in depth).

the rules are one versioned table; each names the policy sections that explain it (blueprint a.5), and a test fails
when a rule names no active section or an active section has no rule.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import psycopg

IST = timezone(timedelta(hours=5, minutes=30))
WINDOW_DAYS = 180
CLAIM_TYPES = ("item_missing_or_wrong", "quality", "refund")
DUPLICATE_WITHIN = timedelta(minutes=10)
NOT_PACKED = ("PLACED", "CONFIRMED")

# the example business's amounts (paise) and limits; tunable, and any change is a new rules version
LIMITS = {
    "auto_limit_per_case_paise": 50_000,          # claim refunds support may make alone on one case
    "bod_cap_per_claim_paise": 30_000,            # benefit of doubt on one case
    "bod_cap_per_day_paise": 50_000,              # benefit of doubt for one customer in one day
    "bod_max_earlier_claims": 2,                  # earlier claims in 180 days
    "bod_max_claim_ratio": 0.25,                  # earlier claims over delivered orders, 180 days (under this)
    "auto_budget_per_customer_day_paise": 100_000,  # claim refunds made alone for one customer in one day
    "auto_budget_global_day_paise": 5_000_000,    # every refund made alone, all customers, in one day
}

# each rule: the case types it decides, and the policy sections that explain it
RULES = {
    # money proven owed goes back where it came from, or to the wallet if asked; benefit of doubt to the
    # wallet only (blueprint c18 rule). the ways are in the table, so they change its version
    "item.corroborated": {"for": ["item_missing_or_wrong"], "policy": ["refund_policy", "missing_item_sop"],
                          "ways": ["refund_to_source", "wallet_credit"]},
    "item.benefit_of_doubt": {"for": ["item_missing_or_wrong"], "policy": ["bod_policy", "refund_policy"],
                              "ways": ["wallet_credit"]},
    "item.to_person": {"for": ["item_missing_or_wrong"], "policy": ["refund_policy", "bod_policy"]},
    "item.records_unclear": {"for": ["item_missing_or_wrong", "payment_failed", "refund"],
                             "policy": ["refund_policy"]},
    "quality.to_person": {"for": ["quality"], "policy": ["damage_sop"]},
    "payment.duplicate_capture": {"for": ["payment_failed", "refund"],
                                  "policy": ["payment_failed_sop", "refund_policy"]},
    "payment.failed_debit": {"for": ["payment_failed", "refund"], "policy": ["payment_failed_sop", "refund_policy"]},
    "payment.not_shown": {"for": ["payment_failed", "refund"], "policy": ["payment_failed_sop", "refund_policy"]},
    "already_refunded": {"for": ["item_missing_or_wrong", "payment_failed", "refund", "cancellation"],
                         "policy": ["refund_policy"]},
    "cancel.before_packing": {"for": ["cancellation"], "policy": ["cancellation_policy", "refund_policy"]},
    "cancel.packing_started": {"for": ["cancellation"], "policy": ["cancellation_policy"]},
    "cancel.nothing_to_cancel": {"for": ["cancellation"], "policy": ["cancellation_policy"]},
    "partner.to_person": {"for": ["delivery_partner_behaviour"], "policy": ["delivery_partner_sop"]},
    "order_status.answer_only": {"for": ["where_is_my_order"], "policy": ["order_status_sop"]},
}

def rules_version(limits: dict, rules: dict) -> str:
    """changes with any amount or rule, so a decision record says which rules it was made under."""
    return "rules@" + hashlib.sha256(json.dumps({"limits": limits, "rules": rules}, sort_keys=True)
                                     .encode()).hexdigest()[:8]


RULES_VERSION = rules_version(LIMITS, RULES)


@dataclass(frozen=True)
class Claim:
    part: str
    status: str  # corroborated, uncorroborated, contradicted


def ist_day(now: datetime) -> tuple[datetime, datetime]:
    start = now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


# what the records hold


def claims_profile(conn: psycopg.Connection, customer_id: str, now: datetime) -> dict:
    """earlier claims and delivered orders in 180 days, counted as the support profile counts them."""
    since = now - timedelta(days=WINDOW_DAYS)
    delivered = conn.execute(
        "select count(*) filter (where status = 'DELIVERED') as n from orders.orders where customer_id = %s "
        "and placed_at >= %s", (customer_id, since)).fetchone()["n"]
    claims = conn.execute(
        "select count(*) as n from support.cases where customer_id = %s and issue_type = any(%s) and created_at >= %s "
        "and state in ('RESOLVED', 'CLOSED')", (customer_id, list(CLAIM_TYPES), since)).fetchone()["n"]
    ratio = (claims / delivered) if delivered else (1.0 if claims else 0.0)
    return {"earlier_claims": claims, "delivered_orders": delivered, "claim_ratio": round(ratio, 3)}


def usage_today(conn: psycopg.Connection, customer_id: str, now: datetime) -> dict:
    """money refunded today (the business's day, ist): benefit of doubt and claim refunds made alone for this
    customer, and every refund made alone for anyone."""
    start, end = ist_day(now)
    row = conn.execute(
        "select coalesce(sum(p.amount_paise) filter (where r.customer_id = %(c)s and p.basis = 'benefit_of_doubt'), 0) "
        "as bod, coalesce(sum(p.amount_paise) filter (where r.customer_id = %(c)s and r.route = 'auto' and p.basis in "
        "('proven', 'benefit_of_doubt')), 0) as mine, coalesce(sum(p.amount_paise) filter (where r.route = 'auto'), 0) "
        "as everyone from refunds_wallet.refunds r join refunds_wallet.refund_parts p using (refund_id) "
        "where r.created_at >= %(s)s and r.created_at < %(e)s", {"c": customer_id, "s": start, "e": end}).fetchone()
    return {"bod_today_paise": int(row["bod"]), "auto_today_paise": int(row["mine"]),
            "global_auto_today_paise": int(row["everyone"])}


def refunded_parts(conn: psycopg.Connection, order_id: str) -> dict[str, dict]:
    rows = conn.execute(
        "select p.part, p.amount_paise, r.refund_id, r.case_id, r.destination, r.created_at "
        "from refunds_wallet.refund_parts p join refunds_wallet.refunds r using (refund_id) where p.order_id = %s",
        (order_id,)).fetchall()
    return {r["part"]: r for r in rows}


def paid_for_order(conn: psycopg.Connection, order_id: str) -> int:
    """what the customer paid for the order and has not had back: captures less refunds already made."""
    captured = conn.execute(
        "select coalesce(sum(c.amount_paise), 0) as n from payments.captures c join payments.payment_intents i "
        "using (intent_id) where i.order_id = %s", (order_id,)).fetchone()["n"]
    refunded = conn.execute("select coalesce(sum(amount_paise), 0) as n from refunds_wallet.refunds "
                            "where order_id = %s", (order_id,)).fetchone()["n"]
    return int(captured) - int(refunded)


def extra_captures(conn: psycopg.Connection, order_id: str) -> dict[str, int]:
    """capture id -> amount, for each capture that repeats an earlier one: the same amount within minutes, under
    another gateway reference (the business's own rule for a duplicate)."""
    captures = conn.execute(
        "select c.capture_id, c.amount_paise, c.gateway_ref, c.captured_at from payments.captures c "
        "join payments.payment_intents i using (intent_id) where i.order_id = %s order by c.captured_at, c.capture_id",
        (order_id,)).fetchall()
    extra: dict[str, int] = {}
    for i, first in enumerate(captures):
        if first["capture_id"] in extra:
            continue
        refs = {first["gateway_ref"]}
        for later in captures[i + 1:]:
            if (later["capture_id"] not in extra and later["amount_paise"] == first["amount_paise"]
                    and later["captured_at"] - first["captured_at"] <= DUPLICATE_WITHIN
                    and later["gateway_ref"] not in refs):
                extra[later["capture_id"]] = int(later["amount_paise"])
                refs.add(later["gateway_ref"])
    return extra


def failed_debits(conn: psycopg.Connection, order_id: str) -> dict[str, int]:
    """intent id -> amount, for each failed attempt the bank debited and never gave back."""
    rows = conn.execute(
        "select i.intent_id, "
        "(select e.amount_paise from payments.gateway_events e where e.intent_id = i.intent_id "
        " and e.type = 'debit_success' order by e.at limit 1) as debited, "
        "exists (select 1 from payments.gateway_events e where e.intent_id = i.intent_id and e.type = 'reversal') "
        " as reversed from payments.payment_intents i where i.order_id = %s and i.status = 'failed'",
        (order_id,)).fetchall()
    return {r["intent_id"]: int(r["debited"]) for r in rows if r["debited"] and not r["reversed"]}


def policy_versions(conn: psycopg.Connection) -> dict[str, str]:
    return {r["policy_id"]: f"{r['policy_id']}@{r['version']}" for r in conn.execute(
        "select policy_id, version from policy.sections where active").fetchall()}


# the rules


class Decider:
    def __init__(self, conn: psycopg.Connection, case: dict, now: datetime):
        self.conn, self.case, self.now = conn, case, now
        self.versions = policy_versions(conn)
        self.parts: list[dict] = []

    def add(self, part: str, outcome: str, rule: str, why: str, *, amount: int = 0, basis: str | None = None,
            kind: str | None = None, allowed: list[str] | None = None, name: str | None = None) -> dict:
        row = {"part": part, "name": name, "outcome": outcome, "kind": kind, "allowed": allowed or [],
               "max_amount_paise": amount, "basis": basis, "requires": "auto" if outcome == "remedy" else None,
               "rule": rule, "why": why, "policy_refs": [self.versions[p] for p in RULES[rule]["policy"]
                                                         if p in self.versions]}
        self.parts.append(row)
        return row


def evaluate(conn: psycopg.Connection, case: dict, claims: list[Claim], risk_band: str | None,
             records_conflict: bool, verdict: str | None, now: datetime) -> dict:
    """the eligibility of one case: a row per part with its outcome (remedy, person or nothing_owed), and for a
    remedy its kind, allowed types, the most it may be, its basis and whether support may make it alone."""
    d = Decider(conn, case, now)
    issue, order_id = case["issue_type"], case["order_id"]
    profile = claims_profile(conn, case["customer_id"], now)
    usage = usage_today(conn, case["customer_id"], now)
    order = conn.execute("select status from orders.orders where order_id = %s", (order_id,)).fetchone() \
        if order_id else None
    paid = refunded_parts(conn, order_id) if order_id else {}
    unclear = records_conflict or verdict == "insufficient_escalate"

    if order is None and issue != "where_is_my_order":
        d.add("order", "person", "item.records_unclear", "no_order_on_the_case")
    elif issue == "cancellation":
        _cancellation(d, order["status"], paid)
    elif issue == "quality":
        for c in claims or [Claim("order", "uncorroborated")]:
            d.add(c.part, "person", "quality.to_person", "photo_check_not_built")
    elif issue == "delivery_partner_behaviour":
        d.add("order", "person", "partner.to_person", "a_person_reviews_partner_reports")
    elif issue == "where_is_my_order":
        d.add("order", "nothing_owed", "order_status.answer_only", "status_answer_only")
    elif issue == "item_missing_or_wrong":
        _items(d, claims, risk_band, unclear, profile, usage, paid)
    elif issue in ("payment_failed", "refund"):
        _payments(d, claims, unclear, paid)
    else:
        d.add("order", "person", "item.records_unclear", "no_rule_for_this_case_type")
    _budgets(d, usage)
    return {"case_id": case["case_id"], "order_id": order_id, "issue_type": issue, "rules_version": RULES_VERSION,
            "risk_band": risk_band, "parts": d.parts, "profile": profile, "usage": usage, "limits": LIMITS}


def _already(d: Decider, part: str, paid: dict[str, dict], name: str | None = None) -> bool:
    if part not in paid:
        return False
    d.add(part, "nothing_owed", "already_refunded", f"already_refunded:{paid[part]['refund_id']}",
          amount=int(paid[part]["amount_paise"]), name=name)
    return True


def _items(d: Decider, claims: list[Claim], risk_band: str | None, unclear: bool, profile: dict, usage: dict,
           paid: dict[str, dict]) -> None:
    lines = {f"line:{r['line_no']}": r for r in d.conn.execute(
        "select line_no, name, qty, unit_paise, status from orders.order_lines where order_id = %s",
        (d.case["order_id"],)).fetchall()}
    if not claims:
        d.add("order", "person", "item.records_unclear", "no_item_named")
        return
    bod_used = 0
    limits = LIMITS
    bod_open = (risk_band == "low" and profile["earlier_claims"] <= limits["bod_max_earlier_claims"]
                and profile["claim_ratio"] < limits["bod_max_claim_ratio"])
    for c in claims:
        line = lines.get(c.part)
        if line is None:
            d.add(c.part, "person", "item.records_unclear", "not_a_line_of_this_order")
            continue
        value, name = int(line["qty"] * line["unit_paise"]), line["name"]
        if _already(d, c.part, paid, name):
            continue
        if unclear:
            d.add(c.part, "person", "item.records_unclear", "records_in_conflict_or_unclear", name=name)
        elif c.status == "corroborated":
            d.add(c.part, "remedy", "item.corroborated", "records_show_it_missing", amount=value, basis="proven",
                  kind="refund", allowed=RULES["item.corroborated"]["ways"], name=name)
        elif not bod_open:
            why = ("risk_not_low" if risk_band != "low" else "too_many_earlier_claims")
            d.add(c.part, "person", "item.to_person", why, name=name)
        elif bod_used + value > limits["bod_cap_per_claim_paise"]:
            d.add(c.part, "person", "item.to_person", "over_the_benefit_of_doubt_cap", name=name)
        elif usage["bod_today_paise"] + bod_used + value > limits["bod_cap_per_day_paise"]:
            d.add(c.part, "person", "item.to_person", "over_the_daily_benefit_of_doubt_cap", name=name)
        else:
            bod_used += value
            d.add(c.part, "remedy", "item.benefit_of_doubt", "not_proven_small_claim_low_risk", amount=value,
                  basis="benefit_of_doubt", kind="refund", allowed=RULES["item.benefit_of_doubt"]["ways"], name=name)


def _payments(d: Decider, claims: list[Claim], unclear: bool, paid: dict[str, dict]) -> None:
    extra = extra_captures(d.conn, d.case["order_id"])
    debits = failed_debits(d.conn, d.case["order_id"])
    if not claims:
        d.add("order", "person" if d.case["issue_type"] == "refund" else "nothing_owed", "payment.not_shown",
              "records_show_no_money_owed")
        return
    for c in claims:
        if _already(d, c.part, paid):
            continue
        if unclear:
            d.add(c.part, "person", "item.records_unclear", "records_in_conflict_or_unclear")
        elif c.part.startswith("capture:") and c.part[8:] in extra:
            d.add(c.part, "remedy", "payment.duplicate_capture", "charged_twice", amount=extra[c.part[8:]],
                  basis="payment_error", kind="refund", allowed=["refund_to_source"])
        elif c.part.startswith("intent:") and c.part[7:] in debits:
            d.add(c.part, "remedy", "payment.failed_debit", "failed_payment_took_money", amount=debits[c.part[7:]],
                  basis="payment_error", kind="refund", allowed=["refund_to_source"])
        else:
            d.add(c.part, "person", "payment.not_shown", "records_do_not_show_it")


def _cancellation(d: Decider, status: str, paid: dict[str, dict]) -> None:
    if status in NOT_PACKED:
        if _already(d, "order", paid):
            return
        d.add("order", "remedy", "cancel.before_packing", "not_packed_yet",
              amount=max(0, paid_for_order(d.conn, d.case["order_id"])), basis="order_cancelled", kind="cancel",
              allowed=["refund_to_source"])
    elif status in ("PACKING", "PACKED", "OUT_FOR_DELIVERY"):
        d.add("order", "person", "cancel.packing_started", "packing_started")
    else:
        d.add("order", "nothing_owed", "cancel.nothing_to_cancel", f"order_{status.lower()}")


def _budgets(d: Decider, usage: dict) -> None:
    """the caps on what support may do alone: the case's claim refunds, the customer's day and everyone's day. a
    remedy over a cap stays a remedy, but a person must approve it."""
    claim = [p for p in d.parts if p["outcome"] == "remedy" and p["basis"] in ("proven", "benefit_of_doubt")]
    claim_total = sum(p["max_amount_paise"] for p in claim)
    every = sum(p["max_amount_paise"] for p in d.parts if p["outcome"] == "remedy" and p["kind"] == "refund")
    over = None
    if claim_total > LIMITS["auto_limit_per_case_paise"]:
        over = "over_the_case_limit"
    elif usage["auto_today_paise"] + claim_total > LIMITS["auto_budget_per_customer_day_paise"]:
        over = "over_the_customer_daily_budget"
    if over:
        for p in claim:
            p["requires"], p["why"] = "human", f"{p['why']};{over}"
    if usage["global_auto_today_paise"] + every > LIMITS["auto_budget_global_day_paise"]:
        for p in d.parts:
            if p["outcome"] == "remedy" and p["kind"] == "refund":
                p["requires"], p["why"] = "human", f"{p['why']};over_the_global_daily_budget"
