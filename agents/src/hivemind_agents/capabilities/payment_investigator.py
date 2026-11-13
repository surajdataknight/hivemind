"""c10 payment investigator: the money side of an order, by arithmetic only (blueprint c10). what was captured, a
second capture of the same amount (a duplicate), a failed attempt that took money and gave none back by the time the
rbi allows (t+1 day for upi and wallets, t+5 days for cards), and refunds already made.

findings: payment.captured, payment.duplicate_capture, payment.failed_debit_not_reversed,
payment.failed_debit_reversed, payment.failed_without_debit, payment.refunds_made.
"""

from datetime import timedelta

from hivemind_agents.capabilities.checking import at, finding, iso, rupees
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task

DUPLICATE_WITHIN = timedelta(minutes=10)
REVERSAL_DUE = {"upi": timedelta(days=1), "wallet": timedelta(days=1), "card": timedelta(days=5)}  # rbi tat


def duplicates(captures: list[dict]) -> list[list[dict]]:
    """groups of captures of one amount, each within DUPLICATE_WITHIN of the first, with distinct gateway refs."""
    groups, used = [], set()
    ordered = sorted(captures, key=lambda c: (c["captured_at"], c["capture_id"]))
    for i, first in enumerate(ordered):
        if first["capture_id"] in used:
            continue
        group = [first] + [c for c in ordered[i + 1:] if c["capture_id"] not in used
                           and c["amount_paise"] == first["amount_paise"]
                           and at(c["captured_at"]) - at(first["captured_at"]) <= DUPLICATE_WITHIN]
        if len({c["gateway_ref"] for c in group}) > 1:
            groups.append(group)
            used.update(c["capture_id"] for c in group)
    return groups


def run(task: Task, check: CheckInput) -> CheckResult:
    order_id = task.params["order_id"]
    backend = check.backend
    payments = backend.payments(order_id)
    refunds = backend.refunds(order_id)
    now = at(backend.clock()["now"])
    ref = payments["evidence_id"]
    intents = payments["intents"]
    captures = [c for i in intents for c in i["captures"]]
    captured = sum(c["amount_paise"] for c in captures)
    found = [finding(task, check, "payment.captured",
                     f"{rupees(captured)} was captured in {len(captures)} capture(s) over {len(intents)} attempt(s).",
                     [ref], facts={"captured_paise": captured, "captures": len(captures), "attempts": len(intents),
                                   "methods": sorted({i["method"] for i in intents})})]
    for group in duplicates(captures):
        extra = sum(c["amount_paise"] for c in group[1:])
        found.append(finding(task, check, "payment.duplicate_capture",
                             f"The same amount, {rupees(group[0]['amount_paise'])}, was captured {len(group)} times "
                             f"within minutes; {rupees(extra)} was taken twice.", [ref],
                             facts={"capture_ids": [c["capture_id"] for c in group],
                                    "amount_paise": group[0]["amount_paise"], "extra_paise": extra}))
    for intent in intents:
        if intent["status"] != "failed":
            continue
        events = intent["gateway_events"]
        debits = [e for e in events if e["type"] == "debit_success"]
        reversals = [e for e in events if e["type"] == "reversal"]
        if not debits:
            found.append(finding(task, check, "payment.failed_without_debit",
                                 f"A {intent['method']} attempt of {rupees(intent['amount_paise'])} failed and took "
                                 "no money.", [ref], facts={"intent_id": intent["intent_id"],
                                                            "amount_paise": intent["amount_paise"]}))
            continue
        debited = at(debits[0]["at"])
        due = debited + REVERSAL_DUE.get(intent["method"], timedelta(days=5))
        if reversals:
            found.append(finding(task, check, "payment.failed_debit_reversed",
                                 f"A failed {intent['method']} attempt took {rupees(debits[0]['amount_paise'])}, and "
                                 "the bank gave it back.", [ref],
                                 facts={"intent_id": intent["intent_id"], "amount_paise": debits[0]["amount_paise"],
                                        "debited_at": debits[0]["at"], "reversed_at": reversals[0]["at"]}))
        else:
            found.append(finding(task, check, "payment.failed_debit_not_reversed",
                                 f"A failed {intent['method']} attempt took {rupees(debits[0]['amount_paise'])} and "
                                 f"it has not come back; it was due back by {iso(due)}.", [ref],
                                 facts={"intent_id": intent["intent_id"], "method": intent["method"],
                                        "amount_paise": debits[0]["amount_paise"], "debited_at": debits[0]["at"],
                                        "reversal_due_by": iso(due), "overdue": now > due}))
    if refunds["refunds"]:
        total = sum(r["amount_paise"] for r in refunds["refunds"])
        found.append(finding(task, check, "payment.refunds_made",
                             f"{len(refunds['refunds'])} refund(s) of {rupees(total)} in all were already made for "
                             "this order.", [refunds["evidence_id"]],
                             facts={"total_paise": total, "refunds": [
                                 {k: r[k] for k in ("refund_id", "amount_paise", "status", "reason", "case_id")}
                                 for r in refunds["refunds"]]}))
    return CheckResult(findings=found)


CAPABILITY = Capability(
    name="payment",
    description="what was paid and captured for the order, a double capture, a failed payment that took money, and "
                "refunds already made",
    params=("order_id",), scopes=frozenset({"payments:read:self"}), run=run, cost="low")
