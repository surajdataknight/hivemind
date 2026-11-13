"""each capability on its own, without a database: the input guard, identity, intake's rules and checks, triage,
the order status template, the promotions investigator's outcomes, the settings file."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from hivemind_agents import settings, wording
from hivemind_agents.backend_client import NotFound
from hivemind_agents.capabilities import intake, items, order_status, promotions_investigator, triage
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
    assert {"payments:read:self", "inventory:read", "history:read:self", "risk:read"} <= set(identity.scopes)
    assert "media:read:self" not in identity.scopes  # the media analyst comes in phase 4


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


LINES = ["Amul Butter (100 g)", "Amul Taaza Milk (1 l)", "Soya Milk (1 l)", "Fresh Paneer (200 g)", "Brown Bread"]


@pytest.mark.parametrize("answer, named", [
    ("the paneer", ["Fresh Paneer (200 g)"]),
    ("it was the paneer that never came", ["Fresh Paneer (200 g)"]),
    ("Paneer and butter", ["Fresh Paneer (200 g)", "Amul Butter (100 g)"]),
    ("amul milk", ["Amul Taaza Milk (1 l)"]),
    ("panner, bred", ["Fresh Paneer (200 g)", "Brown Bread"]),  # typos
    ("Brown Bread", ["Brown Bread"]),  # the button's own words, typed
    ("milk", None),  # fits two lines equally: not clean
    ("the eggs", None),  # no such line
    ("paneer and the eggs", None),  # one part names no line
    ("no idea", None),
    ("", None),
])
def test_an_answer_to_which_item_names_lines_or_is_not_clean(answer, named):
    assert items.named_in_answer(answer, LINES) == named


ASKS_ITEM = {"asks": "items", "case_type": "item_missing_or_wrong", "order_id": "ORD-100123", "lines": LINES,
             "queue": [{"case_type": "where_is_my_order", "order_id": None}], "question": "Which item was missing?"}


class NoModel:
    """a model the test does not expect to be asked."""

    def structured(self, *args, **kwargs):
        raise AssertionError("the model was asked")


class OneDraft:
    def __init__(self, answer):
        self.answer, self.asked = answer, 0

    def structured(self, job, system, prompt, shape):
        self.asked += 1
        return shape.model_validate(self.answer)


def test_an_answer_to_which_item_carries_the_case_and_the_waiting_issues_over():
    typed = intake.intake("the paneer and the butter", None, ASKS_ITEM, [], [], NoModel())
    assert [(i.case_type, i.order_id, i.items_mentioned) for i in typed.issues] == [
        ("item_missing_or_wrong", "ORD-100123", ["Fresh Paneer (200 g)", "Amul Butter (100 g)"]),
        ("where_is_my_order", None, [])]
    hint = IntentHint(case_type="item_missing_or_wrong", order_id="ORD-100123", items=["Brown Bread"])
    pressed = intake.intake("Brown Bread", hint, ASKS_ITEM, [], [], NoModel())
    assert pressed.source == "button" and pressed.issues[0].items_mentioned == ["Brown Bread"]
    assert [i.case_type for i in pressed.issues] == ["item_missing_or_wrong", "where_is_my_order"]


@pytest.mark.parametrize("pending", [
    ASKS_ITEM,
    {"asks": "order", "case_type": "offers_coupons", "coupon_code": "SAVE10", "queue": []},
    {"asks": "coupon_code", "case_type": "offers_coupons", "order_id": "ORD-100123", "queue": []},
])
def test_a_message_that_is_not_an_answer_gets_the_question_once_more_and_no_model_reads_it(pending):
    result = intake.intake("hmm, not sure, maybe the eggs?", None, pending, [], [], NoModel())
    assert result.ask_again and result.issues == [] and result.source == "rule"


def test_after_the_question_was_asked_again_a_message_is_read_as_new_and_nothing_waiting_is_dropped():
    again = {**ASKS_ITEM, "asked_again": True}
    # a new problem: worked first; the unanswered one and the one queued behind it come after it
    model = OneDraft({"scope": "in_scope", "issues": [
        {"case_type": "payment_failed", "order_id": None, "items_mentioned": [], "claimed_amount_rupees": 250,
         "coupon_code": None, "about_latest_order": False, "order_day": None,
         "money_to": "not_said"}],
        "sentiment": "negative", "urgency": "normal", "language": "en", "needs_clarification": False})
    result = intake.intake("forget that, I was charged twice for something", None, again, [], [], model)
    assert model.asked == 1 and not result.ask_again
    assert [(i.case_type, i.order_id) for i in result.issues] == [
        ("payment_failed", None), ("item_missing_or_wrong", "ORD-100123"), ("where_is_my_order", None)]
    # the same problem raised again in the message replaces the unanswered one, never twice
    rule = intake.intake("where is my order?", None, {**again, "case_type": "where_is_my_order", "queue": []},
                         [], [], NoModel())
    assert [i.case_type for i in rule.issues] == ["where_is_my_order"]
    # a message that gets a fixed reply: the question follows it
    thanks = intake.intake("thanks", None, again, [], [], NoModel())
    assert thanks.scope == "small_talk" and thanks.still_waiting


def test_a_button_for_another_problem_is_read_at_once_and_keeps_the_open_question():
    pressed = intake.intake("Where is my order?", IntentHint(case_type="where_is_my_order"),
                            {**ASKS_ITEM, "queue": []}, [], [], NoModel())
    assert [(i.case_type, i.order_id) for i in pressed.issues] == [
        ("where_is_my_order", None), ("item_missing_or_wrong", "ORD-100123")]


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


def test_the_last_order_counts_only_when_no_order_id_goes_with_the_problem():
    said = checked(draft(issues=[issue("item_missing_or_wrong", latest=True)]), "my last order had no bread")
    assert said.issues[0].latest_order and said.issues[0].order_id is None
    written = checked(draft(issues=[issue("item_missing_or_wrong", latest=True)]), "my last order ORD-100123")
    assert written.issues[0].order_id == "ORD-100123" and not written.issues[0].latest_order


def test_a_bad_code_or_amount_is_dropped_and_rupees_become_paise():
    result = checked(draft(issues=[issue("offers_coupons", coupon_code="my coupon!", amount=49.5)]), "x")
    assert result.issues[0].coupon_code is None and result.issues[0].claimed_amount_paise == 4950
    assert checked(draft(issues=[issue("refund", amount=-5)]), "x").issues[0].claimed_amount_paise is None


def test_money_back_for_a_problem_is_part_of_it_not_a_second_issue():
    """seen 2026-11-13: "the tomatoes did not come in my order ORD-100115. please give my money back" became two
    issues, so the several-issues rule sent a plain refund ask to a person."""
    text = "The Green Valley Tomatoes did not come in my order ORD-100115. Please give my money back."
    split = draft(issues=[issue("item_missing_or_wrong", order_id="ORD-100115", items=["tomatoes"]),
                          issue("refund", order_id="ORD-100115", amount=40, money_to="card_or_bank")])
    (one,) = checked(split, text).issues
    assert (one.case_type, one.order_id, one.items_mentioned, one.claimed_amount_paise, one.money_to) == (
        "item_missing_or_wrong", "ORD-100115", ["tomatoes"], 4000, "card_or_bank")
    # the refund names no order: it goes with the only money problem
    no_order = draft(issues=[issue("quality", order_id="ORD-100115", items=["eggs"]), issue("refund")])
    assert [i.case_type for i in checked(no_order, "Broken eggs in ORD-100115. Refund please").issues] == ["quality"]


@pytest.mark.parametrize("issues, kept", [
    # another order: a problem of its own
    ([issue("item_missing_or_wrong", order_id="ORD-100115"), issue("refund", order_id="ORD-100116")],
     ["item_missing_or_wrong", "refund"]),
    # a refund beside a problem that pays nothing (where is my order) stays
    ([issue("where_is_my_order", order_id="ORD-100115"), issue("refund", order_id="ORD-100115")],
     ["where_is_my_order", "refund"]),
    # no order named, and two money problems: no telling which it goes with
    ([issue("quality", order_id="ORD-100115"), issue("item_missing_or_wrong", order_id="ORD-100116"),
      issue("refund")], ["quality", "item_missing_or_wrong", "refund"]),
    # the refund is the only problem
    ([issue("refund", order_id="ORD-100115")], ["refund"]),
])
def test_a_refund_that_is_not_about_a_listed_money_problem_stays_an_issue(issues, kept):
    result = checked(draft(issues=issues), "ORD-100115 ORD-100116")
    assert [i.case_type for i in result.issues] == kept


@pytest.mark.parametrize("said, text, money_to", [
    ("wallet", "please add it to my wallet", "wallet"),
    ("wallet", "Put it as HiveMart credit", "wallet"),
    ("wallet", "please refund it", None),  # the model read a wallet the customer never named
    ("card_or_bank", "back to my card please", "card_or_bank"),
    ("not_said", "give my money back", None),
])
def test_the_wallet_only_when_the_customers_words_name_it(said, text, money_to):
    result = checked(draft(issues=[issue("item_missing_or_wrong", money_to=said)]), text)
    assert result.issues[0].money_to == money_to


def test_a_wallet_named_in_an_earlier_turn_counts():
    earlier = [Turn(role="customer", text="the bread was missing, wallet is fine")]
    result = checked(draft(issues=[issue("item_missing_or_wrong", money_to="wallet")]), "ORD-100123", earlier)
    assert result.issues[0].money_to == "wallet"


def test_the_way_to_pay_waits_with_a_question():
    pending = {"asks": "order", "case_type": "item_missing_or_wrong", "queue": [], "question": "which order?",
               "items_mentioned": ["bread"], "claimed_amount_paise": None, "money_to": "wallet"}
    answer = intake.intake("ORD-100123", None, pending, [], [], None)
    assert answer.issues[0].money_to == "wallet" and answer.issues[0].order_id == "ORD-100123"


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
            return {"order_id": order_id, "lines": []}
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


MIXED = [row("ORD-000005", "OUT_FOR_DELIVERY"), row("ORD-000004"), row("ORD-000003", "CANCELLED"),
         row("ORD-000002", "PARTIALLY_FULFILLED"), row("ORD-000001", "PACKING")]


@pytest.mark.parametrize("case_type,latest,orders,expect", [
    # "my last order": the newest order this kind of problem can be about
    ("item_missing_or_wrong", True, MIXED, ("ORD-000004", None, [])),
    ("quality", True, MIXED[2:], ("ORD-000002", None, [])),
    ("delivery_partner_behaviour", True, MIXED, ("ORD-000005", None, [])),
    ("cancellation", True, MIXED[1:], ("ORD-000001", None, [])),
    ("refund", True, MIXED, ("ORD-000005", None, [])),
    ("payment_failed", True, MIXED, ("ORD-000005", None, [])),
    # no order said: only one it can be about, that one; several, ask among them
    ("item_missing_or_wrong", False, MIXED[2:], ("ORD-000002", None, [])),
    ("item_missing_or_wrong", False, MIXED, (None, "order", ["ORD-000004", "ORD-000002"])),
    ("cancellation", False, MIXED, (None, "order", ["ORD-000005", "ORD-000001"])),
    ("refund", False, MIXED[:2], (None, "order", ["ORD-000005", "ORD-000004"])),
    # none it can be about: ask among all of them (it may be an older one, typed in)
    ("item_missing_or_wrong", False, [row("ORD-000002", "PACKING"), row("ORD-000001", "CANCELLED")],
     (None, "order", ["ORD-000002", "ORD-000001"])),
    ("item_missing_or_wrong", True, [row("ORD-000002", "PACKING"), row("ORD-000001", "CANCELLED")],
     (None, "order", ["ORD-000002", "ORD-000001"])),
    # a failed payment may be for an order that was never placed: recorded as it is, never asked
    ("payment_failed", False, MIXED, (None, None, [])),
    ("payment_failed", False, MIXED[:1], ("ORD-000005", None, [])),
])
def test_which_order_a_problem_is_about(case_type, latest, orders, expect):
    result = run_triage(Issue(case_type=case_type, latest_order=latest), Shop(orders))
    assert (result.issue.order_id, result.ask, [o.order_id for o in result.choices]) == expect


def placed(order_id, ist, status="DELIVERED", delivered=None):
    """an order placed at `ist` (india time, "2026-10-04 23:50"), delivered at `delivered` if given."""
    utc = lambda s: (datetime.fromisoformat(s).replace(tzinfo=wording.IST).astimezone(UTC)
                     .isoformat().replace("+00:00", "Z"))
    return {"order_id": order_id, "status": status, "placed_at": utc(ist),
            "delivered_at": utc(delivered) if delivered else None}


# the business's clock: 2026-10-05 20:00 in india, a monday
DAYS = [placed("ORD-000005", "2026-10-05 19:40", "OUT_FOR_DELIVERY"), placed("ORD-000004", "2026-10-05 09:10"),
        placed("ORD-000003", "2026-10-04 23:50", delivered="2026-10-05 00:05"),
        placed("ORD-000002", "2026-10-04 13:00"), placed("ORD-000001", "2026-10-02 18:00")]


@pytest.mark.parametrize("day,orders,expect", [
    ("today", DAYS, (None, "order", ["ORD-000004", "ORD-000003"])),  # 0003 was delivered after midnight
    ("yesterday", DAYS[:1] + DAYS[3:], ("ORD-000002", None, [])),
    ("yesterday", DAYS, (None, "order", ["ORD-000003", "ORD-000002"])),  # two of that day: ask among those
    ("friday", DAYS, ("ORD-000001", None, [])),
    ("monday", DAYS[:2], ("ORD-000004", None, [])),  # today is a monday; the order still on its way does not fit
    ("2026-10-02", DAYS, ("ORD-000001", None, [])),
    ("day_before_yesterday", DAYS, (None, "order", ["ORD-000004", "ORD-000003", "ORD-000002", "ORD-000001"])),
    ("2026-10-09", DAYS[1:], (None, "order", ["ORD-000004", "ORD-000003", "ORD-000002", "ORD-000001"])),  # future
])
def test_an_order_named_by_a_day_is_picked_by_code_on_the_business_clock(day, orders, expect):
    result = run_triage(Issue(case_type="item_missing_or_wrong", order_day=day), Shop(orders))
    assert (result.issue.order_id, result.ask, [o.order_id for o in result.choices]) == expect


@pytest.mark.parametrize("said,kept", [("Yesterday", "yesterday"), ("day before yesterday", "day_before_yesterday"),
                                       ("2026-10-04", "2026-10-04"), ("last week", None), ("ORD-000001", None),
                                       (None, None)])
def test_only_a_day_the_form_allows_is_kept(said, kept):
    assert intake.day_word(said) == kept


def test_an_order_named_is_kept_whatever_its_state():
    named = run_triage(Issue(case_type="item_missing_or_wrong", order_id="ORD-000005"), Shop(MIXED))
    assert (named.issue.order_id, named.ask) == ("ORD-000005", None)


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
    assert models.key_envs == {"google": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
    assert models.job("case_lead").model == "openai/gpt-oss-120b"  # the case lead (013, since 2026-11-13)
    with pytest.raises(KeyError):
        models.job("no_such_job")
    with pytest.raises(ValueError):
        settings.load_models('[jobs.x]\nprovider = "nobody"\nmodel = "m"\n')


def test_a_problem_with_items_but_none_named_asks_which_item_on_its_order():
    lines = [{"line_no": 1, "name": "Brown Bread"}, {"line_no": 2, "name": "Fresh Paneer (200 g)"},
             {"line_no": 3, "name": "Brown Bread"}]
    shop = Shop([row("ORD-000005", "DELIVERED")], order={"order_id": "ORD-000005", "lines": lines})
    for case_type in ("item_missing_or_wrong", "quality"):
        asked = run_triage(Issue(case_type=case_type, order_id="ORD-000005"), shop)
        assert (asked.ask, asked.lines, asked.issue.order_id) == (
            "items", ["Brown Bread", "Fresh Paneer (200 g)"], "ORD-000005")  # each name once
    named = run_triage(Issue(case_type="item_missing_or_wrong", order_id="ORD-000005", items_mentioned=["bread"]),
                       shop)
    assert named.ask is None
    for other in ("refund", "delivery_partner_behaviour"):  # not about particular items: never asked
        assert run_triage(Issue(case_type=other, order_id="ORD-000005"), shop).ask is None
