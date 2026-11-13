"""the deciding steps on their own, with no database: what is sent to the rules (c18's side), the approval router
(c21) and the resolution planner (c19)."""

import pytest

from hivemind_agents.capabilities import approval_router, eligibility, resolver
from hivemind_agents.contracts import FactSheet, Finding, Issue

LIMITS = {"auto_limit_per_case_paise": 50_000}


def part(p="line:1", outcome="remedy", basis="proven", amount=6200, requires="auto", allowed=None, rule=None,
         kind="refund", why="x", name="Butter"):
    return {"part": p, "name": name, "outcome": outcome, "kind": kind if outcome == "remedy" else None,
            "allowed": allowed if allowed is not None else ["refund_to_source", "wallet_credit"],
            "max_amount_paise": amount, "basis": basis if outcome == "remedy" else None,
            "requires": requires if outcome == "remedy" else None, "rule": rule or f"rule.{outcome}", "why": why,
            "policy_refs": []}


def answer(*parts, band="low"):
    return {"parts": list(parts), "risk_band": band, "limits": LIMITS, "rules_version": "rules@test"}


def sheet(contradictions=(), claims=(), verdict="sufficient"):
    return FactSheet(case_id="CASE-0123456789AB", round=1, verdict=verdict, source="reconciler",
                     claims=list(claims), contradictions=list(contradictions))


# c21


def test_a_small_proven_claim_of_a_low_risk_customer_is_done_alone():
    assert approval_router.route(answer(part()), sheet(), False)[0] == "auto"


@pytest.mark.parametrize("change, why", [
    ({"band": "elevated"}, "risk band elevated"),
    ({"band": None}, "risk band unknown"),
    ({"amount": 50_100}, "over the case limit"),
    ({"several": True}, "several issues"),
    ({"contradictions": [{"finding_ids": ["a", "b"], "why": "x"}]}, "contradict"),
])
def test_a_claim_the_reviewer_would_check_waits_for_a_person(change, why):
    a = answer(part(amount=change.get("amount", 6200)), band=change.get("band", "low"))
    route, reasons = approval_router.route(a, sheet(change.get("contradictions", ())), change.get("several", False))
    assert route == "person" and any(why in r for r in reasons), reasons


def test_a_payment_error_is_put_right_whatever_its_size_or_the_risk():
    a = answer(part("capture:CAP-000366", basis="payment_error", amount=57_200, allowed=["refund_to_source"]),
               band="high")
    assert approval_router.route(a, sheet(), True)[0] == "auto"


@pytest.mark.parametrize("money_to, route", [(None, "person"), ("card_or_bank", "person"), ("wallet", "auto")])
def test_owed_money_and_benefit_of_doubt_in_one_case_are_paid_alone_only_to_a_wallet_asked_for(money_to, route):
    a = answer(part(), part("line:2", basis="benefit_of_doubt", allowed=["wallet_credit"], name="Bread"))
    got, why = approval_router.route(a, sheet(), False, money_to)
    assert got == route
    if route == "person":
        assert why == ["owed money and benefit of doubt in one case, and the customer did not ask for the wallet"]


def test_benefit_of_doubt_alone_goes_to_the_wallet_whatever_was_asked():
    a = answer(part(basis="benefit_of_doubt", allowed=["wallet_credit"]))
    assert approval_router.route(a, sheet(), False, "card_or_bank")[0] == "auto"
    p = resolver.propose(a, "card_or_bank")
    assert (p.destination, p.source) == ("wallet_credit", "no_choice")
    assert p.rationale_summary == ("only the HiveMart wallet is allowed; the customer asked for the original payment "
                                   "method")


def test_any_part_for_a_person_sends_the_whole_case():
    a = answer(part(), part("line:2", outcome="person", why="risk_not_low"))
    route, why = approval_router.route(a, sheet(), False)
    assert route == "person" and why == ["line:2: risk_not_low"]


def test_a_remedy_that_needs_approval_is_not_done_alone():
    assert approval_router.route(answer(part(requires="human")), sheet(), False)[0] == "person"


def test_nothing_owed_and_already_refunded_are_told_apart():
    assert approval_router.route(answer(part(outcome="nothing_owed", rule="payment.not_shown")), None,
                                 False)[0] == "nothing_owed"
    assert approval_router.route(answer(part(outcome="nothing_owed", rule="already_refunded")), None,
                                 False)[0] == "already_refunded"


def test_parts_with_no_common_way_to_pay_go_to_a_person():
    a = answer(part(allowed=["wallet_credit"]), part("capture:CAP-000366", basis="payment_error",
                                                     allowed=["refund_to_source"]))
    assert approval_router.route(a, sheet(), False) == ("person", ["no one way to pay every part"])


# c18, hivemind's side


def finding(type_, facts, fid):
    return Finding(finding_id=fid, case_id="CASE-0123456789AB", task_id="t1", capability="x", type=type_,
                   claim="c", evidence_refs=["ev_x"], confidence="high", facts=facts)


CLAIMED = finding("order.claimed_lines", {"lines": [{"line_no": 2, "sku": "SKU-DAI-003", "name": "Butter",
                                                     "qty": 1, "value_paise": 6200}]}, "f1")


def test_a_corroborated_line_with_no_record_behind_it_is_sent_as_unproven():
    """the model said corroborated, but no record points that way: the rules see uncorroborated."""
    said = sheet(claims=[{"about": "Butter", "sku": "SKU-DAI-003", "status": "corroborated", "finding_ids": ["f1"]}])
    issue = Issue(case_type="item_missing_or_wrong", order_id="ORD-100311")
    sent = eligibility.inputs_for(issue, said, [CLAIMED])
    assert sent["claims"] == [{"part": "line:2", "status": "uncorroborated"}]
    proof = finding("store.item_not_scanned", {"sku": "SKU-DAI-003"}, "f2")
    sent = eligibility.inputs_for(issue, said, [CLAIMED, proof])
    assert sent["claims"] == [{"part": "line:2", "status": "corroborated"}]


def test_the_payment_parts_come_from_the_payment_checks_findings():
    board = [finding("payment.duplicate_capture", {"capture_ids": ["CAP-000001", "CAP-000366"]}, "f1"),
             finding("payment.failed_debit_not_reversed", {"intent_id": "PI-000386"}, "f2"),
             finding("risk.assessment", {"band": "low"}, "f3")]
    sent = eligibility.inputs_for(Issue(case_type="payment_failed", order_id="ORD-100217"),
                                  sheet(contradictions=[{"finding_ids": ["f1", "f2"], "why": "x"}]), board)
    assert sent == {"claims": [{"part": "capture:CAP-000366", "status": "corroborated"},
                               {"part": "intent:PI-000386", "status": "corroborated"}],
                    "risk_band": "low", "records_conflict": True, "verdict": "sufficient"}


def test_a_cancellation_sends_no_claims():
    assert eligibility.inputs_for(Issue(case_type="cancellation", order_id="ORD-100364"), None, []) == {"claims": []}


# c19


def test_with_one_way_allowed_that_way_is_used_and_a_different_ask_is_recorded():
    a = answer(part("capture:CAP-000366", basis="payment_error", allowed=["refund_to_source"]))
    p = resolver.propose(a, "wallet")
    assert (p.destination, p.source) == ("refund_to_source", "no_choice")
    assert p.rationale_summary == ("only the original payment method is allowed; the customer asked for the HiveMart "
                                   "wallet")


@pytest.mark.parametrize("money_to, way, source", [
    ("wallet", "wallet_credit", "asked"),
    ("card_or_bank", "refund_to_source", "asked"),
    (None, "refund_to_source", "default"),  # a wallet credit they did not want keeps their money in the app
])
def test_the_way_is_what_the_customer_asked_for_and_the_card_when_they_did_not_say(money_to, way, source):
    p = resolver.propose(answer(part()), money_to)
    assert (p.destination, p.source, p.total_paise, [x.amount_paise for x in p.parts]) == (way, source, 6200, [6200])


def test_the_reason_says_only_what_the_customer_did():
    """seen 2026-11-13: a model recorded "the customer asked for their money back" for "please check"."""
    assert resolver.propose(answer(part()), None).rationale_summary == (
        "the customer did not say where; the original payment method")
    assert resolver.propose(answer(part()), "wallet").rationale_summary == "the customer asked for the HiveMart wallet"


def test_the_points_come_from_the_rules_basis():
    p = resolver.propose(answer(part(), part("line:3", basis="benefit_of_doubt", name="Bread")), "wallet")
    assert p.explanation_points == ["Butter: the records show it was not delivered.",
                                    "Bread: the records could not confirm it either way; it is credited as a goodwill "
                                    "gesture."]


def test_when_a_part_allows_only_the_wallet_the_wallet_is_used():
    a = answer(part(allowed=["wallet_credit"]), part("line:3", name="Bread"))
    p = resolver.propose(a, "card_or_bank")
    assert (p.destination, p.source) == ("wallet_credit", "no_choice")


def test_a_cancellation_goes_back_where_it_came_from():
    a = answer(part("order", basis="order_cancelled", kind="cancel", allowed=["refund_to_source"]))
    p = resolver.propose(a, "wallet")
    assert (p.kind, p.destination, p.source) == ("cancel", "refund_to_source", "no_choice")
