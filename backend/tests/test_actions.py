"""the business's rules (c18) and the action gateway (c22), on the seeded test database.

rules: what each kind of claim may get, from what the investigation concluded and the business's own records; every
rule names active policy text and every active section has a rule (blueprint a.5).
gateway: once per key (a repeat returns the first answer; a key reused for another request is refused); the rules
are run again and anything over them is refused; a part is paid once whichever case asks; the money, the outbox
event and the decision record are one database step; the decision records of a case form a hash chain.
"""

import threading
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from hivemind_backend import db, eligibility, gateway
from hivemind_backend.app import app
from test_contract import (bearer, check_body, check_problem, deciding_case, extra_capture, fault, not_packed, owner,
                           store_miss)

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def client(test_db):
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean(clean_chat_state, clean_actions):
    """each test opens its own cases: one case per issue would otherwise hand it a case an earlier test moved on."""
    yield


def rows(sql: str, *args) -> list[dict]:
    with db.connect("admin", row_factory=dict_row) as conn:
        return conn.execute(sql, args).fetchall()


def case_for(client, t, issue_type: str, order_id: str) -> tuple[str, dict]:
    headers = bearer(owner(t, order_id))
    return deciding_case(client, headers, issue_type, order_id), headers


def eligible(client, t, issue_type: str, order_id: str, claims=(), **inputs) -> dict:
    case_id, headers = case_for(client, t, issue_type, order_id)
    body = {"claims": [{"part": p, "status": s} for p, s in claims], **inputs}
    response = client.post(f"/v1/cases/{case_id}/eligibility", json=body, headers=headers)
    return check_body("/v1/cases/{case_id}/eligibility", response, "post")


def one(answer: dict) -> dict:
    (part,) = answer["parts"]
    return part


def line_part(t, order_id: str, value_at_most: int | None = None, value_over: int | None = None) -> tuple[str, int]:
    for line in sorted((x for x in t["orders.order_lines"] if x["order_id"] == order_id), key=lambda x: x["line_no"]):
        value = line["qty"] * line["unit_paise"]
        if (value_at_most is None or value <= value_at_most) and (value_over is None or value > value_over):
            return f"line:{line['line_no']}", value
    raise LookupError(order_id)


def delivered_order_with_line(t, customer_ok, value_at_most=None, value_over=None) -> tuple[str, str, int]:
    """a delivered order, untouched by a fault, of a customer `customer_ok` accepts, with a line in range."""
    faulted = {f["order_id"] for f in t["truth.order_faults"]}
    claimed = {r["order_id"] for r in t["refunds_wallet.refunds"]}
    for order in t["orders.orders"]:
        if order["status"] != "DELIVERED" or order["order_id"] in faulted | claimed:
            continue
        if not customer_ok(order["customer_id"]):
            continue
        try:
            part, value = line_part(t, order["order_id"], value_at_most, value_over)
        except LookupError:
            continue
        return order["order_id"], part, value
    raise LookupError("no such order")


def no_claims(t):
    claimants = {c["customer_id"] for c in t["support.cases"]}
    return lambda customer_id: customer_id not in claimants


# the rules


def test_a_line_the_records_show_missing_may_be_refunded_alone_up_to_its_value(client, test_db):
    order_id, part, value = store_miss(test_db)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "corroborated")], risk_band="low"))
    assert (p["outcome"], p["basis"], p["requires"], p["max_amount_paise"]) == ("remedy", "proven", "auto", value)
    assert p["allowed"] == ["refund_to_source", "wallet_credit"]
    assert {r.split("@")[0] for r in p["policy_refs"]} == {"refund_policy", "missing_item_sop"}


def test_proof_of_a_store_fault_outranks_a_high_risk_band(client, test_db):
    order_id, part, _ = store_miss(test_db)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "corroborated")], risk_band="high"))
    assert (p["outcome"], p["requires"]) == ("remedy", "auto")
    assert p["allowed"] == ["refund_to_source", "wallet_credit"]  # money proven owed: either way


@pytest.mark.parametrize("status", ["uncorroborated", "contradicted"])
def test_a_small_unproven_claim_of_a_low_risk_customer_gets_the_benefit_of_doubt(client, test_db, status):
    order_id, part, value = delivered_order_with_line(test_db, no_claims(test_db), value_at_most=30_000)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, status)], risk_band="low"))
    assert (p["outcome"], p["basis"], p["requires"], p["max_amount_paise"]) == (
        "remedy", "benefit_of_doubt", "auto", value)
    assert any(r.startswith("bod_policy@") for r in p["policy_refs"])
    assert p["allowed"] == ["wallet_credit"]  # benefit of doubt: the wallet only


@pytest.mark.parametrize("band", ["elevated", "high"])
def test_an_unproven_claim_of_a_customer_not_low_risk_goes_to_a_person_never_refused(client, test_db, band):
    order_id, part, _ = delivered_order_with_line(test_db, no_claims(test_db), value_at_most=30_000)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "uncorroborated")], risk_band=band))
    assert (p["outcome"], p["why"]) == ("person", "risk_not_low")


def test_an_unproven_claim_over_the_cap_goes_to_a_person(client, test_db):
    order_id, part, _ = delivered_order_with_line(test_db, no_claims(test_db), value_over=30_000)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "uncorroborated")], risk_band="low"))
    assert (p["outcome"], p["why"]) == ("person", "over_the_benefit_of_doubt_cap")


def test_a_customer_with_many_earlier_claims_gets_no_benefit_of_doubt(client, test_db):
    order_id = fault(test_db, "fraud_ring")["order_id"]
    part, _ = line_part(test_db, order_id)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "uncorroborated")], risk_band="low"))
    assert (p["outcome"], p["why"]) == ("person", "too_many_earlier_claims")


@pytest.mark.parametrize("inputs", [{"records_conflict": True}, {"verdict": "insufficient_escalate"}])
def test_records_in_conflict_or_too_thin_go_to_a_person_even_when_corroborated(client, test_db, inputs):
    order_id, part, _ = store_miss(test_db)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "corroborated")], risk_band="low",
                     **inputs))
    assert (p["outcome"], p["why"]) == ("person", "records_in_conflict_or_unclear")


def test_a_refund_over_the_case_limit_needs_a_persons_approval(client, test_db):
    order_id, part, value = delivered_order_with_line(test_db, lambda c: True, value_over=50_000)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [(part, "corroborated")], risk_band="low"))
    assert (p["outcome"], p["requires"], p["max_amount_paise"]) == ("remedy", "human", value)
    assert "over_the_case_limit" in p["why"]


def test_a_part_not_on_the_order_goes_to_a_person(client, test_db):
    order_id, _, _ = store_miss(test_db)
    p = one(eligible(client, test_db, "item_missing_or_wrong", order_id, [("line:99", "corroborated")],
                     risk_band="low"))
    assert (p["outcome"], p["why"]) == ("person", "not_a_line_of_this_order")


def test_a_double_capture_is_refunded_alone_whatever_its_size(client, test_db):
    order_id = fault(test_db, "duplicate_capture")["order_id"]
    part, amount = extra_capture(test_db)
    p = one(eligible(client, test_db, "payment_failed", order_id, [(part, "corroborated")]))
    assert (p["outcome"], p["basis"], p["requires"], p["max_amount_paise"], p["allowed"]) == (
        "remedy", "payment_error", "auto", amount, ["refund_to_source"])
    assert amount > eligibility.LIMITS["auto_limit_per_case_paise"]  # the generated duplicates are all over ₹500


def test_the_first_capture_is_not_a_duplicate(client, test_db):
    f = fault(test_db, "duplicate_capture")
    first = next(c for c in test_db["payments.captures"] if c["capture_id"] != f["facts"]["capture_id"]
                 and c["intent_id"] == next(x["intent_id"] for x in test_db["payments.captures"]
                                            if x["capture_id"] == f["facts"]["capture_id"]))
    p = one(eligible(client, test_db, "payment_failed", f["order_id"], [(f"capture:{first['capture_id']}",
                                                                          "corroborated")]))
    assert (p["outcome"], p["why"]) == ("person", "records_do_not_show_it")


def test_a_failed_payment_that_took_money_is_refunded_alone(client, test_db):
    f = fault(test_db, "failed_debit_not_reversed")
    p = one(eligible(client, test_db, "payment_failed", f["order_id"], [(f"intent:{f['facts']['intent_id']}",
                                                                          "corroborated")]))
    assert (p["outcome"], p["basis"], p["requires"], p["max_amount_paise"]) == (
        "remedy", "payment_error", "auto", f["expected"]["owed_paise"])


def test_a_failed_payment_that_took_nothing_owes_nothing(client, test_db):
    order_id = fault(test_db, "missing_item", "store_miss")["order_id"]
    p = one(eligible(client, test_db, "payment_failed", order_id, []))
    assert (p["outcome"], p["rule"]) == ("nothing_owed", "payment.not_shown")


def test_quality_waits_for_the_photo_check(client, test_db):
    order_id, part, _ = store_miss(test_db)
    p = one(eligible(client, test_db, "quality", order_id, [(part, "corroborated")], risk_band="low"))
    assert (p["outcome"], p["rule"]) == ("person", "quality.to_person")


@pytest.mark.parametrize("status, outcome, rule", [
    ("not_packed", "remedy", "cancel.before_packing"),
    ("PACKING", "person", "cancel.packing_started"),
    ("DELIVERED", "nothing_owed", "cancel.nothing_to_cancel"),
])
def test_an_order_can_be_cancelled_alone_only_before_packing(client, test_db, status, outcome, rule):
    order = not_packed(test_db) if status == "not_packed" else next(
        o for o in test_db["orders.orders"] if o["status"] == status)
    p = one(eligible(client, test_db, "cancellation", order["order_id"]))
    assert (p["outcome"], p["rule"]) == (outcome, rule)
    if outcome == "remedy":
        assert p["max_amount_paise"] == order["total_paise"] and p["requires"] == "auto"


def test_every_rule_names_active_policy_text_and_every_section_has_a_rule(test_db):
    active = {r["policy_id"] for r in rows("select policy_id from policy.sections where active")}
    named = {p for rule in eligibility.RULES.values() for p in rule["policy"]}
    assert named <= active, named - active
    assert active <= named, active - named


def test_the_rules_version_changes_with_any_amount_or_rule():
    assert eligibility.RULES_VERSION == eligibility.rules_version(eligibility.LIMITS, eligibility.RULES)
    more = {**eligibility.LIMITS, "bod_cap_per_claim_paise": 30_001}
    assert eligibility.rules_version(more, eligibility.RULES) != eligibility.RULES_VERSION
    fewer = {k: v for k, v in eligibility.RULES.items() if k != "quality.to_person"}
    assert eligibility.rules_version(eligibility.LIMITS, fewer) != eligibility.RULES_VERSION


# the gateway


def refund_request(case_id: str, part: str, amount: int, status="corroborated", **inputs) -> dict:
    return {"case_id": case_id, "parts": [{"part": part, "amount_paise": amount}],
            "eligibility": {"claims": [{"part": part, "status": status}], **inputs},
            "decision": {"issue": {"type": "test"}}}


def double_capture_case(client, t) -> tuple[str, dict, str, int]:
    order_id = fault(t, "duplicate_capture")["order_id"]
    case_id, headers = case_for(client, t, "payment_failed", order_id)
    part, amount = extra_capture(t)
    return case_id, headers, part, amount


def post_refund(client, case_id, headers, body, key=None, path="/v1/actions/refunds"):
    return client.post(path, json=body, headers={**headers, "Idempotency-Key": key or f"{case_id}:remedy:1"})


def test_the_same_key_twice_gives_the_same_refund_once(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    body = refund_request(case_id, part, amount)
    first = check_body("/v1/actions/refunds", post_refund(client, case_id, headers, body), "post", 201)
    again = check_body("/v1/actions/refunds", post_refund(client, case_id, headers, body), "post", 201)
    assert (first["replayed"], again["replayed"]) == (False, True)
    assert again["refund_id"] == first["refund_id"] and again["decision_record_id"] == first["decision_record_id"]
    assert len(rows("select 1 from refunds_wallet.refunds where case_id = %s", case_id)) == 1
    assert first["amount_paise"] == amount and first["status"] == "INITIATED"
    assert first["expected_settlement"] == "3-5 business days"


def test_a_key_reused_for_another_request_is_refused(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    check_body("/v1/actions/refunds", post_refund(client, case_id, headers, refund_request(case_id, part, amount)),
               "post", 201)
    other = post_refund(client, case_id, headers, refund_request(case_id, part, amount - 100))
    check_problem("/v1/actions/refunds", other, 422, "post")


def test_a_part_is_paid_once_whatever_the_key(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    check_body("/v1/actions/refunds", post_refund(client, case_id, headers, refund_request(case_id, part, amount)),
               "post", 201)
    # the rules now say it was already refunded, so a new key is refused before any money moves
    second = post_refund(client, case_id, headers, refund_request(case_id, part, amount), key=f"{case_id}:remedy:2")
    assert check_problem("/v1/actions/refunds", second, 403, "post")["detail"].endswith("already_refunded:" + rows(
        "select refund_id from refunds_wallet.refunds where case_id = %s", case_id)[0]["refund_id"])
    assert len(rows("select 1 from refunds_wallet.refund_parts where part = %s", part)) == 1


def test_many_callers_with_one_key_at_once_make_one_refund(test_db):
    client = TestClient(app)
    case_id, headers, part, amount = double_capture_case(client, test_db)
    body = refund_request(case_id, part, amount)
    answers, errors = [], []

    def send():
        try:
            answers.append(post_refund(TestClient(app), case_id, headers, body).json())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=send) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len({a["refund_id"] for a in answers}) == 1, answers
    assert sum(not a["replayed"] for a in answers) == 1
    assert len(rows("select 1 from refunds_wallet.refunds where case_id = %s", case_id)) == 1


def test_the_budgets_are_read_and_spent_under_one_lock(client, test_db, monkeypatch):
    """two refunds at once must not both use the same room under a daily budget: the rules run with the lock held."""
    from hivemind_backend import gateway

    held, real = [], gateway.eligible

    def watching(conn, *args):
        held.append(conn.execute("select count(*) as n from pg_locks where locktype = 'advisory' and granted and "
                                 "pid = pg_backend_pid() and objid = %s", (gateway.BUDGET_LOCK,)).fetchone()["n"])
        return real(conn, *args)

    monkeypatch.setattr(gateway, "eligible", watching)
    case_id, headers, part, amount = double_capture_case(client, test_db)
    check_body("/v1/actions/refunds", post_refund(client, case_id, headers, refund_request(case_id, part, amount)),
               "post", 201)
    assert held == [1]


def test_an_amount_over_the_rules_is_refused(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    response = post_refund(client, case_id, headers, refund_request(case_id, part, amount + 1))
    assert "over the" in check_problem("/v1/actions/refunds", response, 403, "post")["detail"]
    assert not rows("select 1 from refunds_wallet.refunds where case_id = %s", case_id)


def test_a_refund_a_person_must_approve_is_refused_without_an_approval(client, test_db):
    order_id, part, value = delivered_order_with_line(test_db, lambda c: True, value_over=50_000)
    case_id, headers = case_for(client, test_db, "item_missing_or_wrong", order_id)
    response = post_refund(client, case_id, headers, refund_request(case_id, part, value, risk_band="low"))
    assert "approval" in check_problem("/v1/actions/refunds", response, 403, "post")["detail"]


def test_a_wallet_credit_may_not_pay_a_double_capture(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    response = post_refund(client, case_id, headers, refund_request(case_id, part, amount),
                           path="/v1/actions/wallet-credits")
    check_problem("/v1/actions/wallet-credits", response, 403, "post")


def test_support_acts_only_on_a_case_it_is_deciding(client, test_db):
    order_id = fault(test_db, "duplicate_capture")["order_id"]
    headers = bearer(owner(test_db, order_id))
    case_id = client.post("/v1/cases", headers=headers, json={"issue_type": "payment_failed",
                                                              "order_id": order_id}).json()["case_id"]
    part, amount = extra_capture(test_db)
    check_problem("/v1/actions/refunds", post_refund(client, case_id, headers, refund_request(case_id, part, amount)),
                  409, "post")


def test_another_customers_case_looks_like_no_case(client, test_db):
    case_id, _, part, amount = double_capture_case(client, test_db)
    other = next(o["customer_id"] for o in test_db["orders.orders"] if o["customer_id"] != owner(
        test_db, fault(test_db, "duplicate_capture")["order_id"]))
    check_problem("/v1/actions/refunds", post_refund(client, case_id, bearer(other),
                                                     refund_request(case_id, part, amount)), 404, "post")


def test_a_key_must_belong_to_the_case(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    response = post_refund(client, case_id, headers, refund_request(case_id, part, amount),
                           key="CASE-000000000000:remedy:1")
    check_problem("/v1/actions/refunds", response, 422, "post")


def test_actions_need_their_scope(client, test_db):
    case_id, _, part, amount = double_capture_case(client, test_db)
    customer = owner(test_db, fault(test_db, "duplicate_capture")["order_id"])
    headers = bearer(customer, scopes=["cases:write:self", "eligibility:read"])
    check_problem("/v1/actions/refunds", post_refund(client, case_id, headers, refund_request(case_id, part, amount)),
                  403, "post")


def test_money_outbox_and_decision_record_are_written_together(client, test_db):
    case_id, headers, part, amount = double_capture_case(client, test_db)
    made = post_refund(client, case_id, headers, refund_request(case_id, part, amount)).json()
    (refund,) = rows("select * from refunds_wallet.refunds where refund_id = %s", made["refund_id"])
    assert (refund["route"], refund["idempotency_key"], refund["decision_record_id"]) == (
        "auto", f"{case_id}:remedy:1", made["decision_record_id"])
    (parts,) = rows("select * from refunds_wallet.refund_parts where refund_id = %s", made["refund_id"])
    assert (parts["part"], parts["basis"], parts["amount_paise"]) == (part, "payment_error", amount)
    (event,) = rows("select * from notifications.notifications_outbox where type = 'refund.initiated'")
    assert event["data"]["refund_id"] == made["refund_id"] and event["source"] == "commerce-sim/refunds_wallet"
    (record,) = rows("select * from support.decision_records where case_id = %s", case_id)
    assert record["record"]["actions"][0]["refund_id"] == made["refund_id"]
    assert record["record"]["rules_version"] == eligibility.RULES_VERSION


def test_a_failure_after_the_money_leaves_nothing_behind(client, test_db, monkeypatch):
    case_id, headers, part, amount = double_capture_case(client, test_db)

    def broken(*args, **kwargs):
        raise RuntimeError("the record could not be written")

    monkeypatch.setattr(gateway, "keep_record", broken)
    with pytest.raises(RuntimeError):
        post_refund(client, case_id, headers, refund_request(case_id, part, amount))
    assert not rows("select 1 from refunds_wallet.refunds where case_id = %s", case_id)
    assert not rows("select 1 from actions.idempotency_keys where key like %s", case_id + "%")
    assert not rows("select 1 from notifications.notifications_outbox")


def test_a_wallet_credit_balances_its_ledger(client, test_db):
    order_id, part, value = store_miss(test_db)
    case_id, headers = case_for(client, test_db, "item_missing_or_wrong", order_id)
    made = check_body("/v1/actions/wallet-credits", post_refund(
        client, case_id, headers, refund_request(case_id, part, value, risk_band="low"),
        path="/v1/actions/wallet-credits"), "post", 201)
    assert (made["destination"], made["status"], made["expected_settlement"]) == ("wallet", "COMPLETED", "right away")
    entries = rows("select account, amount_paise from refunds_wallet.wallet_ledger where ref = %s", made["refund_id"])
    assert sorted(e["amount_paise"] for e in entries) == [-value, value]
    assert sum(e["amount_paise"] for e in entries) == 0


def test_the_decision_records_of_a_case_are_a_hash_chain(client, test_db):
    order_id = fault(test_db, "missing_item", "false_claim")["order_id"]
    case_id, headers = case_for(client, test_db, "item_missing_or_wrong", order_id)
    for n in (1, 2, 3):
        response = client.post("/v1/actions/outcomes", json={"case_id": case_id, "outcome": "to_a_person"},
                               headers={**headers, "Idempotency-Key": f"{case_id}:outcome:{n}"})
        check_body("/v1/actions/outcomes", response, "post", 201)
    records = rows("select record, record_hash from support.decision_records where case_id = %s order by seq",
                   case_id)
    assert len(records) == 3 and gateway.record_chain_ok(records)
    records[1]["record"]["outcome"] = "nothing_owed"  # an edited record breaks the chain
    assert not gateway.record_chain_ok(records)


def test_cancelling_before_packing_cancels_and_refunds_in_one_step(client, test_db):
    order = not_packed(test_db)
    case_id, headers = case_for(client, test_db, "cancellation", order["order_id"])
    key = {**headers, "Idempotency-Key": f"{case_id}:cancel"}
    made = check_body("/v1/actions/cancellations", client.post("/v1/actions/cancellations", json={"case_id": case_id},
                                                               headers=key), "post", 201)
    again = client.post("/v1/actions/cancellations", json={"case_id": case_id}, headers=key).json()
    assert again["replayed"] and again["refund"]["refund_id"] == made["refund"]["refund_id"]
    (now,) = rows("select status, cancelled_at from orders.orders where order_id = %s", order["order_id"])
    assert now["status"] == "CANCELLED"
    history = rows("select status from orders.order_status_history where order_id = %s order by seq",
                   order["order_id"])
    assert history[-1]["status"] == "CANCELLED"
    assert made["refund"]["amount_paise"] == order["total_paise"]
    types = sorted(r["type"] for r in rows("select type from notifications.notifications_outbox"))
    assert types == ["order.cancelled", "refund.initiated"]


def test_an_order_being_packed_cannot_be_cancelled_by_support(client, test_db):
    order = next(o for o in test_db["orders.orders"] if o["status"] == "PACKING")
    case_id, headers = case_for(client, test_db, "cancellation", order["order_id"])
    response = client.post("/v1/actions/cancellations", json={"case_id": case_id},
                           headers={**headers, "Idempotency-Key": f"{case_id}:cancel"})
    check_problem("/v1/actions/cancellations", response, 409, "post")
    assert rows("select status from orders.orders where order_id = %s", order["order_id"])[0]["status"] == "PACKING"


def test_today_counts_in_the_business_day():
    start, end = eligibility.ist_day(datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc))
    assert start == datetime(2026, 10, 4, 18, 30, tzinfo=timezone.utc) and end - start == timedelta(days=1)


@pytest.mark.parametrize("role, table, allowed", [
    ("agent", "refunds_wallet.refunds", False),
    ("agent", "actions.idempotency_keys", False),
    ("eval", "refunds_wallet.refund_parts", True),
    ("eval", "support.decision_records", True),
])
def test_only_the_business_writes_money(role, table, allowed, test_db):
    with db.connect(role) as conn:
        try:
            conn.execute(f"select count(*) from {table}").fetchone()
            ok = True
        except psycopg.errors.InsufficientPrivilege:
            ok = False
    assert ok is allowed
