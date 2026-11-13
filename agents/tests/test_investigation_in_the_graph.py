"""the real investigators inside the case graph, on the test database: a customer's message opens a case, the
minimum checks run side by side (risk after the history and the order), and every finding posted to the board is
proven by a read made in this case. then the business's rules decide (phase 3): here the scripted reconciler
settles nothing, so a small claim of a customer of low risk gets the benefit of the doubt."""

import pytest

from hivemind_agents import settings
from hivemind_agents.case_store import CaseStore
from scripted import NO_EXTRA_CHECKS, draft, issue

pytestmark = [pytest.mark.db]


def words_for(data, order_id, sku) -> str:
    line = next(x for x in data.t["orders.order_lines"] if x["order_id"] == order_id and x["sku"] == sku)
    return line["name"].split("(")[0].split(" ", 1)[1].lower()


def board_of(case_id):
    store = CaseStore(settings.agent_conninfo())
    try:
        return store.findings(case_id), store.evidence_ids(case_id)
    finally:
        store.close()


def claim_missing(chat_as, data, model, fault, sku):
    words = words_for(data, fault["order_id"], sku)
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=[words])],
                               sentiment="negative"))
    model.intakes.append(NO_EXTRA_CHECKS)
    return chat_as(fault["customer_id"]).say(f"The {words} was missing from my order {fault['order_id']}.")


def check_spans(spans):
    checks = [s for s in spans.get_finished_spans() if s.name.startswith("check ")]
    assert checks and all(s.attributes["hivemind.check.status"] == "done" for s in checks)
    dropped = [s.attributes.get("hivemind.findings.dropped", 0) for s in spans.get_finished_spans()
               if s.name == "scheduler"]
    assert sum(dropped) == 0  # evidence integrity: no finding without a read in this case
    return {s.name.removeprefix("check ") for s in checks}


def test_a_store_miss_is_found_by_the_real_checks(chat_as, data, model, cases, spans):
    fault = data.fault("missing_item", "store_miss")
    sku = fault["facts"]["missing"][0]["sku"]
    done = claim_missing(chat_as, data, model, fault, sku)
    assert done["message"]["kind"] == "wallet_credited"  # nothing settled: benefit of doubt, wallet
    ran = check_spans(spans)
    assert ran == {"order", "inventory_store", "delivery", "customer_history", "risk"}
    (case,) = cases()
    findings, ledger = board_of(case["case_id"])
    assert all(set(f.evidence_refs) <= ledger for f in findings)
    types = {f.type: f for f in findings}
    assert types["store.item_not_scanned"].facts["sku"] == sku
    assert types["store.stock_out_before_pack"].facts["sku"] == sku
    assert types["order.claimed_lines"].facts["claim_value_paise"] == fault["expected"]["owed_paise"]
    assert types["risk.assessment"].facts["claim_value_paise"] == fault["expected"]["owed_paise"]
    # risk ran after the history and the order, in a later wave (findings keep the order they were posted in)
    seq = [f.type for f in findings]
    assert seq.index("risk.assessment") > max(seq.index("history.profile"), seq.index("order.status"))


def test_a_ring_members_claim_is_high_risk_on_the_board(chat_as, data, model, cases, spans):
    fault = data.fault("fraud_ring", "false_claim")
    done = claim_missing(chat_as, data, model, fault, fault["claim"]["skus"][0])
    assert done["message"]["kind"] == "waiting_for_person"
    check_spans(spans)
    (case,) = cases()
    findings, _ = board_of(case["case_id"])
    risk = next(f for f in findings if f.type == "risk.assessment")
    assert risk.facts["band"] == "high" and risk.facts["handling"] == "human_review"
    assert not [f for f in findings if f.type == "store.item_not_scanned"]
    # the customer is never told of the risk check
    assert "risk" not in done["message"]["text"].lower() and "fraud" not in done["message"]["text"].lower()


def test_the_items_named_before_a_question_survive_the_answer():
    """a missing item with no order said: the question keeps the items and the amount, and the answer, typed or
    pressed, brings them back (they were lost before phase 2.4)."""
    from hivemind_agents import graph
    from hivemind_agents.capabilities import intake
    from hivemind_agents.contracts import Issue, IntentHint

    asked = Issue(case_type="item_missing_or_wrong", items_mentioned=["paneer", "butter"], claimed_amount_paise=15700)
    pending = {"asks": "order", "case_type": asked.case_type, "coupon_code": None, "queue": [],
               **graph._claim_words(asked)}
    typed = intake.from_pending(pending, "it was ORD-100123")
    pressed = intake.from_button(IntentHint(case_type=asked.case_type, order_id="ORD-100123"), pending)
    for answer in (typed, pressed):
        (got,) = answer.issues
        assert (got.order_id, got.items_mentioned, got.claimed_amount_paise) == ("ORD-100123", ["paneer", "butter"],
                                                                                  15700)
    other = intake.from_button(IntentHint(case_type="where_is_my_order", order_id="ORD-100123"), pending)
    assert other.issues[0].items_mentioned == []  # a button for another problem takes nothing from the question


def stored(case_id):
    store = CaseStore(settings.agent_conninfo())
    try:
        return store.fact_sheets(case_id), store.plans(case_id)
    finally:
        store.close()


def test_the_fact_sheet_is_kept_with_the_plan_and_cites_only_this_cases_reads(chat_as, data, model, cases, spans):
    fault = data.fault("missing_item", "store_miss")
    sku = fault["facts"]["missing"][0]["sku"]
    done = claim_missing(chat_as, data, model, fault, sku)
    assert done["message"]["kind"] == "wallet_credited"
    (case,) = cases()
    findings, ledger = board_of(case["case_id"])
    (sheet,), (plan,) = stored(case["case_id"])
    assert (sheet.round, sheet.source, sheet.follow_ups) == (1, "reconciler", [])
    assert len(sheet.facts) == len(findings)  # the scripted reconciler restates each finding, citing it
    on_board = {f.finding_id for f in findings}
    assert all(set(x["finding_ids"]) <= on_board and x["evidence_refs"] and set(x["evidence_refs"]) <= ledger
               for x in sheet.facts)
    signals = {(r["about"], r["signal"]) for r in sheet.cross_checks}
    assert {(sku, "not_scanned_at_packing"), (sku, "stock_out_before_packing")} <= signals
    assert {o["status"] for o in plan["outcomes"].values()} == {"done"}
    assert len(plan["outcomes"]) == len(plan["plan"]["tasks"])
    (span,) = [s for s in spans.get_finished_spans() if s.name == "evidence_reconciler"]
    assert span.attributes["hivemind.fact_sheet.verdict"] == "sufficient_with_uncertainty"


def test_the_customer_sees_each_check_start_and_finish_in_fixed_words(chat_as, data, model, cases):
    from hivemind_agents.capabilities import progress_reporter

    fault = data.fault("fraud_ring", "false_claim")
    chat = chat_as(fault["customer_id"])
    words = words_for(data, fault["order_id"], fault["claim"]["skus"][0])
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=[words])]))
    model.intakes.append(NO_EXTRA_CHECKS)
    chat.say(f"The {words} was missing from my order {fault['order_id']}.")
    kinds = [e["type"] for e in chat.events]
    shown = [e for e in chat.events if e["type"] == "progress"]
    allowed = set(progress_reporter.LABELS.values()) | {progress_reporter.STARTING}
    assert shown and {e["text"] for e in shown} <= allowed
    assert shown[0]["text"] == progress_reporter.STARTING
    # the five checks, then the step that works out what to do
    assert sum(e["state"] == "started" for e in shown) - 1 == sum(e["state"] == "finished" for e in shown) == 6
    assert max(i for i, k in enumerate(kinds) if k == "progress") < kinds.index("done")
    assert not any(w in e["text"].lower() for e in shown for w in ("risk", "fraud"))
