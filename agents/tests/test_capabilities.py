"""each capability on its own, without a database: the input guard, identity, intake's rules and checks, triage,
the order status template, the promotions investigator's outcomes, the settings file."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from hivemind_agents import settings
from hivemind_agents.backend_client import NotFound
from hivemind_agents.capabilities import intake, order_status, promotions_investigator, triage
from hivemind_agents.capabilities.identity import CASE_AUDIENCE, IdentityVerifier, NotSignedIn
from hivemind_agents.capabilities.input_guard import PiiVault, guard, luhn_ok
from hivemind_agents.contracts import IntentHint, Issue, OrderRef, Turn
from hivemind_agents.wording import clock, day_with_year, rupees, when
from scripted import draft, issue

UTC = timezone.utc


# c2 input guard


@pytest.mark.parametrize("text,expected,found", [
    ("call me on 9876543210", "call me on <PHONE_1>", {"PHONE": 1}),
    ("my number is +91 98765 43210.", "my number is <PHONE_1>.", {"PHONE": 1}),
    ("ring 098765-43210 or 9123456780", "ring <PHONE_1> or <PHONE_2>", {"PHONE": 2}),
    ("mail me at Anu.K+shop@example.co.in please", "mail me at <EMAIL_1> please", {"EMAIL": 1}),
    ("refund to anu92@okbank", "refund to <UPI_1>", {"UPI": 1}),
    ("my card 4111 1111 1111 1111 was charged", "my card <CARD_1> was charged", {"CARD": 1}),
    ("card ending 4242 was charged twice", "card ending <CARD_LAST4_1> was charged twice", {"CARD_LAST4": 1}),
    ("paid with XXXX-XXXX-XXXX-9010", "paid with <CARD_LAST4_1>", {"CARD_LAST4": 1}),
    ("deliver to pin code 560034", "deliver to pin code <PINCODE_1>", {"PINCODE": 1}),
])
def test_guard_replaces_personal_data(text, expected, found):
    result = guard(text, PiiVault())
    assert result.sanitised_text == expected
    assert result.pii_found == found


@pytest.mark.parametrize("text", [
    "where is my order ORD-100123",          # an order id is not personal data
    "order 100123 has not come",             # nor a bare order number
    "I paid ₹1,299 for 2 items at 8:15 PM",
    "the coupon SAVE10 gave 10% off",
    "1234567890123456 is not a card",        # 16 digits that fail the card checksum
    "it cost 499 and I am 30 minutes late",
])
def test_guard_leaves_other_text_alone(text):
    result = guard(text, PiiVault())
    assert result.sanitised_text == text and result.pii_found == {} and result.flags == []


def test_guard_flags_a_full_card_number():
    assert guard("4111-1111-1111-1111", PiiVault()).flags == ["contains_payment_credential"]
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")


def test_same_value_same_token_and_the_vault_restores_it():
    vault = PiiVault()
    first = guard("call 9876543210, or mail a@b.com", vault)
    second = guard("i said 9876543210! not 9000000001", vault)
    assert first.sanitised_text == "call <PHONE_1>, or mail <EMAIL_1>"
    assert second.sanitised_text == "i said <PHONE_1>! not <PHONE_2>"
    assert vault.restore(second.sanitised_text) == "i said 9876543210! not 9000000001"
    assert set(vault.new) == {"<PHONE_1>", "<EMAIL_1>", "<PHONE_2>"}


def test_guard_strips_markup_so_only_our_tokens_use_angle_brackets():
    vault = PiiVault({"<PHONE_1>": ("PHONE", "9876543210")})
    result = guard("<script>alert(1)</script>hello <b>there</b> <PHONE_1> 5 < 6\x00\x07 <<PHONE_1>>", vault)
    assert result.sanitised_text == "alert(1)hello there  5 ‹ 6 ‹›"  # a typed token goes like any other tag
    assert vault.restore(result.sanitised_text) == result.sanitised_text


def test_guard_cuts_a_very_long_message():
    assert len(guard("a" * 5000, PiiVault()).sanitised_text) == 2000


# c1 identity

SESSION_KEY, CASE_KEY = "s" * 40, "c" * 40


def session_token(key=SESSION_KEY, **overrides) -> str:
    now = datetime.now(UTC)
    claims = {"iss": "hivemart-app", "aud": "hivemind-support", "sub": "CUST-00042", "sid": "SES-000042",
              "iat": now, "exp": now + timedelta(hours=1), **overrides}
    return jwt.encode(claims, key, algorithm="HS256")


def test_identity_accepts_a_good_session_and_mints_a_short_case_token():
    verifier = IdentityVerifier(SESSION_KEY, CASE_KEY)
    assert verifier.verify_session(session_token()) == ("CUST-00042", "SES-000042")
    identity, token = verifier.mint_case_token("CUST-00042", "SES-000042", "conv_1")
    claims = jwt.decode(token, CASE_KEY, algorithms=["HS256"], audience=CASE_AUDIENCE)
    assert claims["sub"] == "CUST-00042" and claims["conv"] == "conv_1" and claims["scopes"] == identity.scopes
    assert 0 < claims["exp"] - claims["iat"] <= 30 * 60
    assert "payments:read:self" not in identity.scopes  # nothing built so far reads payments


@pytest.mark.parametrize("token", [
    "garbage",
    session_token(key="x" * 40),                                   # signed by someone else
    session_token(exp=datetime.now(UTC) - timedelta(seconds=5)),   # expired
    session_token(aud="another-service"),
    session_token(iss="someone-else"),
    jwt.encode({"sub": "CUST-00042"}, SESSION_KEY, algorithm="HS256"),  # no expiry, no audience
])
def test_identity_refuses_a_bad_session(token):
    with pytest.raises(NotSignedIn):
        IdentityVerifier(SESSION_KEY, CASE_KEY).verify_session(token)


def test_a_case_token_is_not_a_session_token():
    verifier = IdentityVerifier(SESSION_KEY, SESSION_KEY)  # even with one shared key
    _, token = verifier.mint_case_token("CUST-00042", "SES-000042", "conv_1")
    with pytest.raises(NotSignedIn):
        verifier.verify_session(token)


# c3 intake: rules first


@pytest.mark.parametrize("text", [
    "Where is my order?", "where's my order", "WHERE IS MY DELIVERY", "track my order", "order status?",
    "when will my order arrive", "how long until my order gets here", "my order has not arrived",
    "status of my order ORD-100123",
])
def test_plain_where_is_my_order_needs_no_model(text):
    result = intake.from_rules(text)
    assert result is not None and result.source == "rule" and result.scope == "in_scope"
    assert [i.case_type for i in result.issues] == ["where_is_my_order"]


def test_the_rule_picks_up_one_named_order():
    assert intake.from_rules("where is my order ord 100123?").issues[0].order_id == "ORD-100123"
    assert intake.from_rules("where is my order ORD-100123 or ORD-100124").issues[0].order_id is None


@pytest.mark.parametrize("text", [
    "where is my order? I want a refund",                    # a second topic
    "where is my order and why was I charged twice",
    "my order has not arrived and the last one had no bread",
    "where is my refund",                                    # not about an order's whereabouts
    "the delivery partner was rude",
    "my coupon did not apply",
    "where is my order " + "please " * 20,                   # long: let the model read it
])
def test_anything_less_plain_goes_to_the_model(text):
    assert intake.from_rules(text) is None


@pytest.mark.parametrize("text,scope", [("hi", "small_talk"), ("Hello!!", "small_talk"), ("thank you so much", "small_talk"),
                                        ("ok", "small_talk")])
def test_greetings_and_thanks_need_no_model(text, scope):
    assert intake.from_rules(text).scope == scope


def test_a_button_needs_no_model():
    hint = IntentHint(case_type="offers_coupons", order_id="ORD-100123", coupon_code="SAVE10")
    result = intake.intake("anything", hint, None, [], [], llm=None)
    assert result.source == "button" and result.issues == [
        Issue(case_type="offers_coupons", order_id="ORD-100123", coupon_code="SAVE10")]


def test_an_answer_to_our_question_carries_the_case_over():
    asks_order = {"asks": "order", "case_type": "offers_coupons", "coupon_code": "SAVE10"}
    result = intake.from_pending(asks_order, "it was ORD-100123")
    assert result.issues == [Issue(case_type="offers_coupons", order_id="ORD-100123", coupon_code="SAVE10")]
    asks_code = {"asks": "coupon_code", "case_type": "offers_coupons", "order_id": "ORD-100123"}
    assert intake.from_pending(asks_code, " save10 ").issues[0].coupon_code == "SAVE10"
    # anything that is not a clear answer is read afresh
    assert intake.from_pending(asks_order, "forget it, where is my order") is None
    assert intake.from_pending(asks_code, "I don't remember the code") is None
    assert intake.from_pending(None, "ORD-100123") is None


# c3 intake: checks on what the model returns


def checked(model_answer: dict, text: str, history=()) -> intake.IntakeV1:
    return intake.check_draft(intake.IntakeDraft.model_validate(model_answer), text, list(history))


def test_an_order_id_the_customer_never_wrote_is_dropped():
    answer = draft(issues=[issue("offers_coupons", order_id="ORD-999999", coupon_code="save10")])
    result = checked(answer, "my coupon save10 failed")
    assert result.issues[0].order_id is None and result.issues[0].coupon_code == "SAVE10"
    kept = checked(draft(issues=[issue("offers_coupons", order_id="ord-100123")]), "coupon failed on ORD-100123")
    assert kept.issues[0].order_id == "ORD-100123"
    earlier = [Turn(role="customer", text="it is about ORD-100123"), Turn(role="assistant", text="ORD-555555?")]
    assert checked(draft(issues=[issue("refund", order_id="ORD-100123")]), "refund it", earlier).issues[0].order_id
    assert checked(draft(issues=[issue("refund", order_id="ORD-555555")]), "refund it", earlier).issues[0].order_id is None


def test_one_problem_and_one_written_order_id_go_together_when_the_model_leaves_it_out():
    one = checked(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]), "SAVE10 failed on ORD-100123")
    assert one.issues[0].order_id == "ORD-100123"
    # two problems, or two order ids: no telling which goes with which
    two = draft(issues=[issue("item_missing_or_wrong"), issue("where_is_my_order")])
    assert [i.order_id for i in checked(two, "bread missing on ORD-100123, where is the rest?").issues] == [None, None]
    both = checked(draft(issues=[issue("refund")]), "refund ORD-100123 or ORD-100124")
    assert both.issues[0].order_id is None


def test_a_bad_code_or_amount_is_dropped_and_rupees_become_paise():
    result = checked(draft(issues=[issue("offers_coupons", coupon_code="my coupon!", amount=49.5)]), "x")
    assert result.issues[0].coupon_code is None and result.issues[0].claimed_amount_paise == 4950
    assert checked(draft(issues=[issue("refund", amount=-5)]), "x").issues[0].claimed_amount_paise is None


def test_issues_only_count_inside_the_case_list():
    result = checked(draft(scope="out_of_scope", issues=[issue("refund")]), "x")
    assert result.issues == [] and not result.needs_clarification
    assert checked(draft(scope="in_scope"), "x").needs_clarification  # in scope but no issue named
    assert checked(draft(issues=[issue("refund")], language=" EN "), "x").language == "en"


def test_the_prompt_holds_the_orders_the_turns_and_the_message():
    orders = [OrderRef(order_id="ORD-100123", status="DELIVERED", placed_at=datetime(2026, 10, 5, 14, 0, tzinfo=UTC))]
    prompt = intake.build_prompt("coupon <PHONE_1> failed", [Turn(role="assistant", text="Hi")], orders)
    assert "- ORD-100123, placed 2026-10-05 19:30 IST, DELIVERED" in prompt
    assert "assistant: Hi" in prompt and 'coupon <PHONE_1> failed\n"""' in prompt


# triage


class Shop:
    """a stand-in for the backend client, answering from fixed rows."""

    def __init__(self, orders=(), redemptions=(), now="2026-10-05T14:30:00Z", order=None, trip=None, evaluation=None):
        self.orders, self.redemptions, self.now = list(orders), list(redemptions), now
        self._order, self._trip, self._evaluation = order, trip, evaluation

    def my_orders(self, limit=5):
        return {"orders": self.orders[:limit]}

    def my_redemptions(self, limit=20):
        return {"redemptions": self.redemptions}

    def order(self, order_id):
        if self._order and self._order["order_id"] == order_id:
            return self._order
        if any(o["order_id"] == order_id for o in self.orders):
            return {"order_id": order_id}
        raise NotFound("getOrder")

    def trip(self, order_id):
        return self._trip

    def clock(self):
        return {"now": self.now}

    def evaluate_promotion(self, order_id, code):
        return self._evaluation


def row(order_id, status="DELIVERED", minutes_ago=60):
    placed = datetime(2026, 10, 5, 14, 30, tzinfo=UTC) - timedelta(minutes=minutes_ago)
    return {"order_id": order_id, "status": status, "placed_at": placed.isoformat().replace("+00:00", "Z")}


def attempt(order_id, code="SAVE10", status="rejected"):
    return {"order_id": order_id, "code": code, "status": status}


def run_triage(issue_, shop):
    return triage.triage(issue_, triage.recent_orders(shop), shop)


def test_where_is_my_order_takes_the_newest_order_on_its_way():
    shop = Shop([row("ORD-000003"), row("ORD-000002", "OUT_FOR_DELIVERY"), row("ORD-000001", "PACKING")])
    assert run_triage(Issue(case_type="where_is_my_order"), shop).issue.order_id == "ORD-000002"
    all_delivered = Shop([row("ORD-000003"), row("ORD-000002")])
    assert run_triage(Issue(case_type="where_is_my_order"), all_delivered).issue.order_id == "ORD-000003"
    named = run_triage(Issue(case_type="where_is_my_order", order_id="ORD-000003"), shop)
    assert named.issue.order_id == "ORD-000003" and named.ask is None


def test_an_order_that_is_not_the_customers_is_asked_about_never_used():
    shop = Shop([row("ORD-000002")])
    result = run_triage(Issue(case_type="where_is_my_order", order_id="ORD-777777"), shop)
    assert result.ask == "order" and result.unknown_order == "ORD-777777" and result.issue.order_id is None
    assert [o.order_id for o in result.choices] == ["ORD-000002"]


def test_no_orders_at_all():
    assert run_triage(Issue(case_type="where_is_my_order"), Shop()).no_orders


@pytest.mark.parametrize("orders,attempts,given,expect", [
    # one attempted order among the recent ones: that order, with its code
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000001")], {}, ("ORD-000001", "SAVE10", None)),
    # two attempted orders: ask which
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000002"), attempt("ORD-000001", "BIG100")], {}, (None, None, "order")),
    # ... unless the code the customer named fits only one
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000002"), attempt("ORD-000001", "BIG100")],
     {"coupon_code": "BIG100"}, ("ORD-000001", "BIG100", None)),
    # no attempt on record, several orders: ask which order
    (["ORD-000002", "ORD-000001"], [], {"coupon_code": "SAVE10"}, (None, "SAVE10", "order")),
    # no attempt, a single order, a code named: go on with it
    (["ORD-000001"], [], {"coupon_code": "SAVE10"}, ("ORD-000001", "SAVE10", None)),
    # no attempt, a single order, no code: ask for the code
    (["ORD-000001"], [], {}, ("ORD-000001", None, "coupon_code")),
    # an order named, one attempt on it: its code
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000002", "FREEDEL")], {"order_id": "ORD-000002"},
     ("ORD-000002", "FREEDEL", None)),
    # an order named, nothing tried on it, no code: ask for the code
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000001")], {"order_id": "ORD-000002"},
     ("ORD-000002", None, "coupon_code")),
    # an attempt on an order older than the recent five does not count
    (["ORD-000002", "ORD-000001"], [attempt("ORD-000000")], {}, (None, None, "order")),
])
def test_coupon_triage(orders, attempts, given, expect):
    result = run_triage(Issue(case_type="offers_coupons", **given), Shop([row(o) for o in orders], attempts))
    assert (result.issue.order_id, result.issue.coupon_code, result.ask) == expect


def test_other_case_types_are_recorded_as_they_are():
    shop = Shop([row("ORD-000002"), row("ORD-000001")])
    assert run_triage(Issue(case_type="refund"), shop).issue.order_id is None
    assert run_triage(Issue(case_type="refund"), Shop([row("ORD-000001")])).issue.order_id == "ORD-000001"


# the where-is-my-order template


def order_row(status, **extra):
    base = {"order_id": "ORD-000001", "status": status, "promised_by": "2026-10-05T14:40:00Z", "delivered_at": None,
            "cancelled_at": None, "evidence_id": "ev_order"}
    return {**base, **extra}


def trip_row(**extra):
    return {"eta_at": "2026-10-05T14:40:00Z", "picked_up_at": "2026-10-05T14:20:00Z", "evidence_id": "ev_trip",
            **extra}


@pytest.mark.parametrize("status,trip,now,expected", [
    ("CONFIRMED", None, "2026-10-05T14:30:00Z",
     "Your order ORD-000001 is confirmed and the store will start packing it shortly. It should reach you by 8:10 PM."),
    ("PACKING", None, "2026-10-05T14:30:00Z",
     "Your order ORD-000001 is being packed at the store right now. It should reach you by 8:10 PM."),
    ("PACKED", trip_row(picked_up_at=None), "2026-10-05T14:30:00Z",
     "Your order ORD-000001 is packed and waiting for a delivery partner to pick it up. It should reach you by "
     "8:10 PM."),
    ("OUT_FOR_DELIVERY", trip_row(), "2026-10-05T14:30:00Z",
     "Your order ORD-000001 is on its way: the delivery partner picked it up at 7:50 PM. It should reach you by "
     "8:10 PM."),
    ("OUT_FOR_DELIVERY", trip_row(), "2026-10-05T14:52:00Z",
     "Your order ORD-000001 is on its way: the delivery partner picked it up at 7:50 PM. It was due by 8:10 PM and "
     "is running about 12 minutes late. I'm sorry for the wait."),
    ("PACKING", None, "2026-10-05T14:41:00Z",
     "Your order ORD-000001 is being packed at the store right now. It was due by 8:10 PM and is running about 1 "
     "minute late. I'm sorry for the wait."),
])
def test_order_status_wording_for_an_order_on_its_way(status, trip, now, expected):
    text, facts = order_status.order_status("ORD-000001", Shop(order=order_row(status), trip=trip, now=now))
    assert text == expected
    assert facts["evidence_refs"] == ["ev_order"] + (["ev_trip"] if trip else [])


def test_order_status_wording_for_a_finished_order():
    delivered = order_row("DELIVERED", delivered_at="2026-10-04T13:05:00Z")
    text, _ = order_status.order_status("ORD-000001", Shop(order=delivered, trip=trip_row()))
    assert text == "Your order ORD-000001 was delivered at 6:35 PM on 4 October."  # another day: the day is said
    cancelled = order_row("CANCELLED", cancelled_at="2026-10-05T14:02:00Z")
    text, _ = order_status.order_status("ORD-000001", Shop(order=cancelled))
    assert text == "Your order ORD-000001 was cancelled at 7:32 PM, so it will not be delivered."


# c12 promotions investigator


def evaluation(eligible, due, applied, failed=None, checkout="rejected", rules=True):
    return {
        "order_id": "ORD-000001", "code": "SAVE10", "eligible": eligible, "due_discount_paise": due,
        "applied_discount_paise": applied, "difference_paise": due - applied, "failed_rule": failed,
        "rules": {"kind": "percent", "value": 10, "max_discount_paise": 10000, "min_cart_paise": 49900,
                  "first_order_only": False, "valid_from": "2026-08-01T00:00:00Z",
                  "valid_to": "2026-12-01T00:00:00Z"} if rules else None,
        "order": {"item_total_paise": 46200, "delivery_fee_paise": 0, "placed_at": "2026-10-05T14:00:00Z",
                  "first_order": False},
        "checkout": {"status": checkout, "discount_paise": applied, "reason": None} if checkout else None,
        "evidence_id": "ev_eval",
    }


@pytest.mark.parametrize("result,outcome,owed", [
    (evaluation(False, 0, 0, "min_cart"), "not_eligible", False),
    (evaluation(False, 0, 0, "unknown_code", checkout=None, rules=False), "unknown_code", False),
    (evaluation(True, 6000, 6000, checkout="applied"), "applied", False),
    (evaluation(True, 6000, 3000, checkout="applied"), "applied_too_little", True),
    (evaluation(True, 6000, 0, checkout="rejected"), "wrongly_rejected", True),
    (evaluation(True, 6000, 0, checkout=None), "not_entered", False),
    (evaluation(False, 0, 2500, "min_cart", checkout="applied"), "applied", False),  # got more than the rules give
])
def test_promotion_outcomes(result, outcome, owed):
    finding = promotions_investigator.investigate("ORD-000001", "SAVE10", Shop(evaluation=result))
    assert finding.outcome == outcome and finding.owed is owed
    assert finding.evidence_refs == ["ev_eval"]
    assert finding.failed_rule == (result["failed_rule"] if outcome == "not_eligible" else None)


# wording and settings


def test_money_and_times_are_written_the_indian_way():
    assert [rupees(p) for p in (46200, 129950, 5, 0, -3700)] == ["₹462", "₹1,299.50", "₹0.05", "₹0", "-₹37"]
    at = datetime(2026, 10, 5, 18, 35, tzinfo=UTC)  # 12:05 am the next day in india
    assert clock(at) == "12:05 AM" and day_with_year(at) == "6 October 2026"
    assert when(at, at + timedelta(hours=1)) == "12:05 AM"
    assert when(at, at + timedelta(days=1)) == "12:05 AM on 6 October"


def test_every_job_has_a_model_and_names_are_in_one_file():
    models = settings.load_models()
    assert models.job("intake").model == "gemini-3.5-flash-lite"
    assert models.job("communicator").model == "gemini-3.5-flash-lite"
    assert models.key_envs == {"google": "GEMINI_API_KEY"}
    with pytest.raises(KeyError):
        models.job("case_lead")  # no model until the phase that builds it
    with pytest.raises(ValueError):
        settings.load_models('[jobs.x]\nprovider = "nobody"\nmodel = "m"\n')
