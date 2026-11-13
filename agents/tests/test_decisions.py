"""phase 3 in whole chats: the business's rules (c18), the router (c21), the resolution planner (c19) and the
action gateway (c22), through the chat service, the real backend and the test database, on the scripted model.

blueprint scenarios: s07 (a store miss refunded exactly), s08 (benefit of the doubt, no fraud words), s09 (a ring
member goes to a person, neutral words), s12 (a double charge refunded), s27 (the gateway's answer lost after the
refund committed: one refund). and: records in conflict, a failed payment that kept money, cancellations, a second
claim on what was already refunded, a business that does not answer, and retries that never pay twice.
"""

import random

import httpx
import pytest
from psycopg.rows import dict_row

from hivemind_agents import replies
from hivemind_agents.backend_client import ActionRefused, BackendClient, BackendError
from hivemind_agents.capabilities import action_gateway
from hivemind_agents.capabilities.communicator import RISK_WORDS
from hivemind_agents.contracts import Decision, Proposal, ProposedPart
from hivemind_agents.tracing import Tracer
from hivemind_backend import db
from scripted import NO_EXTRA_CHECKS, careful_sheet, draft, issue

pytestmark = [pytest.mark.db]


@pytest.fixture
def rows(test_db_name):
    def query(sql: str, *args) -> list[dict]:
        with db.connect("admin", dbname=test_db_name, row_factory=dict_row) as conn:
            return conn.execute(sql, args).fetchall()
    return query


def refunds_of(rows, case_id):
    return rows("select r.*, coalesce(json_agg(p order by p.part) filter (where p.part is not null), '[]') parts "
                "from refunds_wallet.refunds r left join refunds_wallet.refund_parts p using (refund_id) "
                "where r.case_id = %s group by r.refund_id", case_id)


def records_of(rows, case_id):
    return rows("select record, record_hash from support.decision_records where case_id = %s order by seq", case_id)


def claim_item(chat_as, model, fault, words, text, careful=True, case_type="item_missing_or_wrong"):
    model.intakes.append(draft(issues=[issue(case_type, order_id=fault["order_id"], items=[words] if words else [])],
                               sentiment="negative"))
    model.intakes.append(NO_EXTRA_CHECKS)
    if careful:
        model.reconciles.append(careful_sheet)
    chat = chat_as(fault["customer_id"])
    return chat, chat.say(text)


def fault_of(data, fault_type, variant=None, n=0):
    return [f for f in data.t["truth.order_faults"]
            if f["fault_type"] == fault_type and (variant is None or f["variant"] == variant)][n]


def no_risk_words(text):
    assert not RISK_WORDS.search(text), text


# s07, s08, s09


def test_s07_a_store_miss_is_refunded_alone_to_the_rupee(chat_as, data, model, cases, rows):
    fault = data.fault("missing_item", "store_miss")
    chat, done = claim_item(chat_as, model, fault, "coriander",
                            f"The coriander was missing from my order {fault['order_id']}. Please refund it.")
    message = done["message"]
    assert (message["kind"], done["case_state"]) == ("refunded", "RESOLVED"), message
    owed = fault["expected"]["owed_paise"]
    assert "₹12" in message["text"] and "3-5 business days" in message["text"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert (refund["amount_paise"], refund["destination"], refund["status"], refund["route"]) == (
        owed, "source", "INITIATED", "auto")
    assert [(p["part"], p["basis"]) for p in refund["parts"]] == [("line:1", "proven")]
    (record,) = records_of(rows, case["case_id"])
    body = record["record"]
    assert body["route"] == "auto" and body["proposal"]["destination"] == "refund_to_source"
    assert body["actions"][0]["refund_id"] == refund["refund_id"] and body["fact_sheet"]["verdict"] == "sufficient"
    # "please refund it" names no way: back where the money came from, and the record says so
    assert (body["proposal"]["source"], body["issue"]["money_to"]) == ("default", None)
    assert body["proposal"]["rationale_summary"] == "the customer did not say where; the original payment method"
    events = rows("select type from notifications.notifications_outbox where data->>'refund_id' = %s",
                  refund["refund_id"])
    assert [e["type"] for e in events] == ["refund.initiated"]
    meta = chat.stored()[-1]["meta"]
    assert meta["decision"]["route"] == "auto" and meta["action"]["refund_id"] == refund["refund_id"]


def test_a_refund_asked_for_beside_the_missing_item_is_one_issue_and_paid_alone(chat_as, data, model, cases, rows):
    """seen 2026-11-13 on flash-lite: the money back ask read as a second issue sent the case to a person."""
    fault = data.fault("missing_item", "store_miss")
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=["coriander"]),
                                       issue("refund", order_id=fault["order_id"])], sentiment="negative"))
    model.intakes.append(NO_EXTRA_CHECKS)
    model.reconciles.append(careful_sheet)
    done = chat_as(fault["customer_id"]).say(
        f"The coriander did not come in my order {fault['order_id']}. Please give my money back.")
    assert (done["message"]["kind"], done["case_state"]) == ("refunded", "RESOLVED"), done["message"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert (refund["amount_paise"], refund["destination"]) == (fault["expected"]["owed_paise"], "source")


def test_a_customer_who_asks_for_the_wallet_gets_a_wallet_credit(chat_as, data, model, cases, rows):
    fault = data.fault("missing_item", "store_miss")
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=["coriander"],
                                             money_to="wallet")], sentiment="negative"))
    model.intakes.append(NO_EXTRA_CHECKS)
    model.reconciles.append(careful_sheet)
    done = chat_as(fault["customer_id"]).say(
        f"No coriander in my order {fault['order_id']}. Please add the money to my wallet.")
    assert (done["message"]["kind"], done["case_state"]) == ("wallet_credited", "RESOLVED"), done["message"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert (refund["amount_paise"], refund["destination"]) == (fault["expected"]["owed_paise"], "wallet")
    (record,) = records_of(rows, case["case_id"])
    assert record["record"]["proposal"]["rationale_summary"] == "the customer asked for the HiveMart wallet"


def test_s08_a_packed_item_of_a_low_risk_customer_gets_the_benefit_of_the_doubt(chat_as, data, model, cases, rows):
    fault = data.fault("missing_item", "store_miss")  # the butter was scanned at packing and the bags all handed over
    _, done = claim_item(chat_as, model, fault, "butter", f"No butter in my order {fault['order_id']} :(")
    message = done["message"]
    assert (message["kind"], done["case_state"]) == ("wallet_credited", "RESOLVED"), message
    no_risk_words(message["text"])
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert (refund["amount_paise"], refund["destination"], refund["status"]) == (6200, "wallet", "COMPLETED")
    assert refund["parts"][0]["basis"] == "benefit_of_doubt"
    ledger = rows("select amount_paise from refunds_wallet.wallet_ledger where ref = %s", refund["refund_id"])
    assert sum(e["amount_paise"] for e in ledger) == 0 and len(ledger) == 2


def test_s09_a_ring_members_claim_goes_to_a_person_in_neutral_words(chat_as, data, model, cases, rows):
    fault = data.fault("fraud_ring", "false_claim")
    sku = fault["claim"]["skus"][0]
    line = next(x for x in data.t["orders.order_lines"] if x["order_id"] == fault["order_id"] and x["sku"] == sku)
    words = line["name"].split("(")[0].split(" ", 1)[1].lower()
    _, done = claim_item(chat_as, model, fault, words, f"The {words} never came in {fault['order_id']}. Refund me.")
    message = done["message"]
    assert (message["kind"], done["case_state"]) == ("waiting_for_person", "AWAITING_HUMAN")
    assert message["text"] == replies.with_a_person("item_missing_or_wrong")
    no_risk_words(message["text"])
    (case,) = cases()
    assert refunds_of(rows, case["case_id"]) == []
    (record,) = records_of(rows, case["case_id"])
    assert (record["record"]["outcome"], record["record"]["route"]) == ("to_a_person", "person")
    assert "proposal" not in record["record"]  # nothing to pay: nothing proposed


def test_records_that_contradict_each_other_go_to_a_person(chat_as, data, model, cases, rows):
    fault = fault_of(data, "conflicting_records", "item_missing")
    sku = fault["facts"].get("sku") or fault["claim"]["skus"][0]
    line = next(x for x in data.t["orders.order_lines"] if x["order_id"] == fault["order_id"] and x["sku"] == sku)
    words = line["name"].split("(")[0].split(" ", 1)[1].lower()
    _, done = claim_item(chat_as, model, fault, words, f"My {words} is missing from {fault['order_id']}")
    assert (done["message"]["kind"], done["case_state"]) == ("waiting_for_person", "AWAITING_HUMAN")
    (case,) = cases()
    assert refunds_of(rows, case["case_id"]) == []
    (record,) = records_of(rows, case["case_id"])
    assert record["record"]["inputs"]["records_conflict"] is True


# s12 and the payments


def test_s12_a_double_charge_is_refunded_to_the_source_whatever_its_size(chat_as, data, model, cases, rows):
    fault = fault_of(data, "duplicate_capture")
    _, done = claim_item(chat_as, model, fault, None, f"I was charged twice for {fault['order_id']}!",
                         case_type="payment_failed")
    message = done["message"]
    assert (message["kind"], done["case_state"]) == ("refunded", "RESOLVED"), message
    owed = fault["expected"]["owed_paise"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert (refund["amount_paise"], refund["destination"]) == (owed, "source")
    assert refund["parts"] == [{"refund_id": refund["refund_id"], "order_id": fault["order_id"],
                                "part": f"capture:{fault['facts']['capture_id']}", "amount_paise": owed,
                                "basis": "payment_error"}]


def test_a_failed_payment_that_kept_money_is_refunded(chat_as, data, model, cases, rows):
    fault = fault_of(data, "failed_debit_not_reversed")
    _, done = claim_item(chat_as, model, fault, None, f"My payment failed on {fault['order_id']} but money was cut",
                         case_type="payment_failed")
    assert (done["message"]["kind"], done["case_state"]) == ("refunded", "RESOLVED"), done["message"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert refund["amount_paise"] == fault["expected"]["owed_paise"]
    assert refund["parts"][0]["part"] == f"intent:{fault['facts']['intent_id']}"


def test_a_second_claim_on_what_was_refunded_gets_nothing_more(chat_as, data, model, cases, rows):
    fault = fault_of(data, "duplicate_capture")
    chat, first = claim_item(chat_as, model, fault, None, f"Charged twice on {fault['order_id']}",
                             case_type="payment_failed")
    assert first["message"]["kind"] == "refunded"
    model.intakes.append(draft(issues=[issue("payment_failed", order_id=fault["order_id"])]))
    model.intakes.append(NO_EXTRA_CHECKS)
    model.reconciles.append(careful_sheet)
    again = chat.say(f"I was charged twice on {fault['order_id']}, refund it")
    assert (again["message"]["kind"], again["case_state"]) == ("already_refunded", "RESOLVED"), again["message"]
    assert len(rows("select 1 from refunds_wallet.refunds where order_id = %s and idempotency_key is not null",
                    fault["order_id"])) == 1


# s27 and a business that does not answer


class LosesAnswers:
    """the backend, but the answer to the first `lose` action calls never reaches hivemind: the request went
    through (or, with before=True, never left)."""

    def __init__(self, inner, lose: int, before: bool = False, path: str = "/v1/actions/"):
        self.inner, self.lose, self.before, self.path = inner, lose, before, path
        self.sent = 0

    def request(self, method, url, **kwargs):
        if url.startswith(self.path) and self.lose > 0:
            self.lose -= 1
            self.sent += 1
            if not self.before:
                self.inner.request(method, url, **kwargs)  # committed at the business
            raise httpx.ReadTimeout("the answer was lost")
        return self.inner.request(method, url, **kwargs)


def test_s27_a_lost_answer_after_the_refund_committed_makes_one_refund(chat_as, data, model, cases, rows, service,
                                                                       backend_http):
    service.deps.http = LosesAnswers(backend_http, lose=1)
    fault = fault_of(data, "duplicate_capture")
    _, done = claim_item(chat_as, model, fault, None, f"Charged twice for {fault['order_id']}",
                         case_type="payment_failed")
    assert (done["message"]["kind"], done["case_state"]) == ("refunded", "RESOLVED"), done["message"]
    (case,) = cases()
    (refund,) = refunds_of(rows, case["case_id"])
    assert service.deps.http.sent == 1
    meta_action = [r for r in rows("select response from actions.idempotency_keys where key = %s",
                                   f"{case['case_id']}:remedy:1")]
    assert meta_action[0]["response"]["refund_id"] == refund["refund_id"]
    assert len(records_of(rows, case["case_id"])) == 1


def test_a_business_that_does_not_answer_leaves_the_case_pending_and_the_next_message_finishes_it(
        chat_as, data, model, cases, rows, service, backend_http):
    service.deps.http = LosesAnswers(backend_http, lose=action_gateway.TRIES, before=True)
    fault = fault_of(data, "duplicate_capture")
    chat, done = claim_item(chat_as, model, fault, None, f"Charged twice for {fault['order_id']}",
                            case_type="payment_failed")
    assert (done["message"]["kind"], done["case_state"]) == ("pending_system", "PENDING_SYSTEM")
    assert done["message"]["text"] == replies.PENDING_SYSTEM
    (case,) = cases()
    assert refunds_of(rows, case["case_id"]) == []
    model.intakes.append(draft(issues=[issue("payment_failed", order_id=fault["order_id"])]))
    model.intakes.append(NO_EXTRA_CHECKS)
    model.reconciles.append(careful_sheet)
    again = chat.say("any update on my double charge?")
    assert (again["message"]["kind"], again["case_state"]) == ("refunded", "RESOLVED"), again["message"]
    (same,) = cases()
    assert same["case_id"] == case["case_id"] and len(refunds_of(rows, case["case_id"])) == 1


def test_a_refusal_at_the_gateway_goes_to_a_person(chat_as, data, model, cases, rows, monkeypatch):
    """the business's own re-check says no (here: someone else paid the part in between)."""
    def refused(*args, **kwargs):
        raise ActionRefused("refundToSource", 403, "capture:CAP-000000: already_refunded:RF-000000000000")

    monkeypatch.setattr(BackendClient, "refund", refused)
    fault = fault_of(data, "duplicate_capture")
    _, done = claim_item(chat_as, model, fault, None, f"Charged twice for {fault['order_id']}",
                         case_type="payment_failed")
    assert (done["message"]["kind"], done["case_state"]) == ("waiting_for_person", "AWAITING_HUMAN")
    (case,) = cases()
    (record,) = records_of(rows, case["case_id"])
    assert record["record"]["outcome"] == "to_a_person" and "already_refunded" in record["record"]["refused"]


def test_retries_never_pay_twice(data, rows, backend_http, chat_as, test_db):
    """0 double actions under retry fuzz: every way a call can be lost (before it left, after it committed), many
    times over, on one key and on new keys: one refund, one record."""
    from hivemind_agents.capabilities.identity import IdentityVerifier
    from hivemind_agents import settings

    fault = fault_of(data, "duplicate_capture", n=1)
    chat = chat_as(fault["customer_id"])
    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"), settings.signing_key("HM_CASE_TOKEN_SECRET"))
    _, token = verifier.mint_case_token(fault["customer_id"], "fuzz", chat.conversation_id)
    from opentelemetry.sdk.trace import TracerProvider

    tracer = Tracer(TracerProvider())
    plain = BackendClient(backend_http, token, tracer)
    case = plain.open_case("payment_failed", fault["order_id"])
    for state in ("INVESTIGATING", "DECIDING"):
        plain.move_case(case["case_id"], state)
    part = f"capture:{fault['facts']['capture_id']}"
    owed = fault["expected"]["owed_paise"]
    decision = Decision(route="auto", inputs={"claims": [{"part": part, "status": "corroborated"}]},
                        proposal=Proposal(kind="refund", destination="refund_to_source", source="no_choice",
                                          parts=[ProposedPart(part=part, amount_paise=owed, basis="payment_error")]))
    rng = random.Random(7)
    made = set()
    for _ in range(25):
        http = LosesAnswers(backend_http, lose=rng.randint(0, 4), before=rng.random() < 0.5)
        client = BackendClient(http, token, tracer)
        try:
            made.add(action_gateway.act(case["case_id"], decision, {"fuzz": True}, client, lambda s: None)["refund_id"])
        except BackendError:
            pass  # every try lost: the case would wait on the system; the next round tries again
    assert len(made) == 1
    assert len(refunds_of(rows, case["case_id"])) == 1 and len(records_of(rows, case["case_id"])) == 1
    with pytest.raises(ActionRefused):  # a new key for the same part: the rules say it is paid already
        plain.refund({"case_id": case["case_id"], "parts": [{"part": part, "amount_paise": owed}],
                      "eligibility": decision.inputs, "decision": {}}, f"{case['case_id']}:remedy:2")


# cancellation: the short path, no check and no model


def cancel(chat_as, model, order):
    model.intakes.append(draft(issues=[issue("cancellation", order_id=order["order_id"])]))
    return chat_as(order["customer_id"]).say(f"Please cancel my order {order['order_id']}")


def test_an_order_not_packed_yet_is_cancelled_and_refunded(chat_as, data, model, cases, rows):
    order = next(o for o in data.orders if o["status"] in ("PLACED", "CONFIRMED"))
    done = cancel(chat_as, model, order)
    assert (done["message"]["kind"], done["case_state"]) == ("cancelled", "RESOLVED"), done["message"]
    assert done["llm_calls"] == 1  # the intake only
    assert rows("select status from orders.orders where order_id = %s", order["order_id"])[0]["status"] == "CANCELLED"
    (case,) = cases()
    refunds = refunds_of(rows, case["case_id"])
    intents = {i["intent_id"] for i in data.t["payments.payment_intents"] if i["order_id"] == order["order_id"]}
    paid = sum(c["amount_paise"] for c in data.t["payments.captures"] if c["intent_id"] in intents)
    assert paid and sum(r["amount_paise"] for r in refunds) == paid
    assert done["message"]["text"] == replies.cancelled(order["order_id"], {
        "amount_paise": paid, "expected_settlement": "3-5 business days"})
    types = sorted(r["type"] for r in rows("select type from notifications.notifications_outbox where subject = %s",
                                           order["order_id"]))
    assert "order.cancelled" in types


def test_an_order_being_packed_goes_to_a_person(chat_as, data, model, cases, rows):
    order = data.order_with_status("PACKING")
    done = cancel(chat_as, model, order)
    assert (done["message"]["kind"], done["case_state"]) == ("waiting_for_person", "AWAITING_HUMAN")
    assert done["message"]["text"] == replies.cannot_cancel(order["order_id"])
    assert rows("select status from orders.orders where order_id = %s", order["order_id"])[0]["status"] == "PACKING"


def test_a_delivered_order_has_nothing_to_cancel(chat_as, data, model, cases, rows):
    order = data.order_with_status("DELIVERED")
    done = cancel(chat_as, model, order)
    assert (done["message"]["kind"], done["case_state"]) == ("nothing_owed", "RESOLVED")
    assert done["message"]["text"] == replies.nothing_to_cancel(order["order_id"], "order_delivered")


@pytest.mark.parametrize("status, kind, state", [("PACKING", "waiting_for_person", "AWAITING_HUMAN"),
                                                 ("DELIVERED", "nothing_owed", "RESOLVED")])
def test_a_decision_record_the_business_does_not_answer_still_ends_the_chat(chat_as, data, model, cases, rows,
                                                                            service, backend_http, status, kind,
                                                                            state):
    """the reply goes out and the case moves on; the decision stays in the chat's own record, marked pending."""
    service.deps.http = LosesAnswers(backend_http, lose=action_gateway.TRIES, before=True,
                                     path="/v1/actions/outcomes")
    order = data.order_with_status(status)
    model.intakes.append(draft(issues=[issue("cancellation", order_id=order["order_id"])]))
    chat = chat_as(order["customer_id"])
    done = chat.say(f"Please cancel my order {order['order_id']}")
    assert (done["message"]["kind"], done["case_state"]) == (kind, state), done["message"]
    (case,) = cases()
    assert records_of(rows, case["case_id"]) == []
    meta = chat.stored()[-1]["meta"]
    assert "record_pending" in meta and meta["decision"]["route"] in ("person", "nothing_owed")
