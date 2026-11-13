"""the customer communicator: the amount, number and date check, and what happens to a reply that fails it."""

from datetime import datetime, timezone

import pytest

from hivemind_agents.capabilities import communicator
from hivemind_agents.contracts import PolicySection, PromotionFinding
from hivemind_agents.llm import LLMUnavailable

UTC = timezone.utc
TERMS = PolicySection(policy_id="offer_terms/SAVE10", version="ab12cd34", title="10% off up to ₹100",
                      text="10% off the item total, up to ₹100. Minimum cart ₹499. Delivery fee not included.",
                      evidence_ref="ev_offer")


def finding(outcome="not_eligible", failed_rule="min_cart", due=0, applied=0, **extra) -> PromotionFinding:
    base = dict(order_id="ORD-100123", code="SAVE10", outcome=outcome, failed_rule=failed_rule,
                item_total_paise=46200, min_cart_paise=49900, due_discount_paise=due, applied_discount_paise=applied,
                difference_paise=due - applied, offer_kind="percent", offer_value=10, max_discount_paise=10000,
                valid_from=datetime(2026, 8, 1, tzinfo=UTC), valid_to=datetime(2026, 9, 30, 18, 30, tzinfo=UTC),
                placed_at=datetime(2026, 10, 5, 14, 0, tzinfo=UTC), evidence_refs=["ev_eval"])
    return PromotionFinding(**{**base, **extra})


FACTS = communicator.facts_sheet(finding(), [TERMS])


# the check


@pytest.mark.parametrize("reply", [
    "Your cart came to ₹462, which is below the ₹499 minimum for SAVE10, so it could not apply to ORD-100123.",
    "The cart was ₹462.00 and the offer needs ₹499; you were short by Rs. 37.",
    "SAVE10 gives 10% off, up to ₹100, on carts of 499 rupees or more.",
    "I'm sorry about that. The coupon needs a bigger cart.",
])
def test_a_reply_that_only_uses_the_facts_passes(reply):
    assert communicator.check_reply(reply, FACTS) == []


@pytest.mark.parametrize("reply,problem", [
    ("Your cart came to ₹452, below the ₹499 minimum.", "amount ₹452 is not in the facts"),
    ("I have refunded ₹50 to you.", "amount ₹50 is not in the facts"),
    ("You were short by just 73 rupees.", "amount ₹73 is not in the facts"),
    ("You will hear from us within 24 hours.", "number 24 is not in the facts"),
    ("SAVE10 gives 15% off.", "number 15 is not in the facts"),
    ("The offer ended on 30 October.", "date 30 oct is not in the facts"),
    ("We will fix it by 5:30 PM.", "time 5:30 pm is not in the facts"),
    ("We will fix it tomorrow.", "'tomorrow' is a date the facts do not give"),
    ("Order ORD-100124 was short.", "number 100124 is not in the facts"),
])
def test_a_reply_that_states_anything_else_fails(reply, problem):
    assert problem in communicator.check_reply(reply, FACTS)


def test_dates_must_pair_the_same_day_and_month():
    facts = communicator.facts_sheet(finding(failed_rule="expired"), [])
    assert "30 September 2026" in facts and "5 October 2026" in facts
    assert communicator.check_reply("The offer ended on 30 September 2026.", facts) == []
    assert communicator.check_reply("The offer ended on Sep 30, before your order of 5 October.", facts) == []
    # both numbers are in the facts, but not as this date
    assert communicator.check_reply("The offer ended on 5 September.", facts) == ["date 5 sep is not in the facts"]


# the facts sheet and the template


@pytest.mark.parametrize("f,words,owed", [
    (finding(), ("₹462", "below the offer's minimum of ₹499", "short by ₹37"), False),
    (finding(failed_rule="first_order_only"), ("only for the first order",), False),
    (finding(failed_rule="expired"), ("ended on 30 September 2026", "placed on 5 October 2026"), False),
    (finding(failed_rule="delivery_already_free"), ("already free",), False),
    (finding("unknown_code", None), ("no offer with the code SAVE10",), False),
    (finding("applied", None, due=6000, applied=6000), ("₹60 was taken off", "full discount"), False),
    (finding("applied_too_little", None, due=9120, applied=4560), ("₹91.20", "₹45.60", "difference is ₹45.60"), True),
    (finding("wrongly_rejected", None, due=5510), ("should have been accepted", "₹55.10"), True),
    (finding("not_entered", None, due=5510), ("no record of the coupon SAVE10 being entered", "₹55.10"), False),
])
def test_the_finding_in_words_and_the_template_pass_their_own_check(f, words, owed):
    facts = communicator.facts_sheet(f, [TERMS])
    template = communicator.template_reply(f, [TERMS])
    for word in words:
        assert word in facts and word in template
    assert ("recorded this for a person" in template) is owed
    assert ("you cannot correct the amount yourself" in facts) is owed
    assert communicator.check_reply(template, facts) == []
    assert "ev_" not in facts and "risk" not in facts.lower()  # nothing internal is put in front of the model


# composing: stream, check, retry, template


class Router:
    """stands in for the model router: each call to stream() plays the next queued reply."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def stream(self, job, system, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for start in range(0, len(reply), 5):
            yield reply[start:start + 5]


def compose(*replies):
    events = []
    router = Router(*replies)
    text, record = communicator.compose("my coupon failed!", finding(), [TERMS], router, events.append)
    return text, record, events, router


GOOD = "I'm sorry about the coupon. Your cart came to ₹462, below the ₹499 minimum for SAVE10. Nothing more is owed."
BAD = "I'm sorry about the coupon. I have refunded ₹50 to your account. It will arrive tomorrow."


def test_a_good_reply_streams_sentence_by_sentence():
    text, record, events, router = compose(GOOD)
    assert text == GOOD and record["source"] == "llm" and record["tries"] == 1
    assert [e["type"] for e in events] == ["token", "token", "token"]
    assert "".join(e["text"] for e in events).strip() == GOOD
    assert events[0]["text"] == "I'm sorry about the coupon. "
    assert "my coupon failed!" in router.prompts[0] and "what the check found:" in router.prompts[0]


def test_a_wrong_amount_is_never_shown_and_the_second_try_replaces_the_first():
    text, record, events, router = compose(BAD, GOOD)
    assert text == GOOD and record["tries"] == 2
    assert events[0] == {"type": "token", "text": "I'm sorry about the coupon. "}  # passed before the bad sentence
    assert events[-1] == {"type": "replace", "text": GOOD} and len(events) == 2
    assert all("₹50" not in e["text"] and "tomorrow" not in e["text"] for e in events)
    assert "amount ₹50 is not in the facts" in router.prompts[1]  # the second try is told why


def test_two_bad_replies_fall_back_to_the_template():
    text, record, events, _ = compose(BAD, "You get ₹5,000 back.")
    assert record["source"] == "template" and record["tries"] == 2 and len(record["rejected"]) == 2
    assert text == communicator.template_reply(finding(), [TERMS])
    assert events[-1] == {"type": "replace", "text": text}
    assert all("5,000" not in e["text"] and "₹50 " not in e["text"] for e in events)


def test_no_model_means_the_template_straight_away():
    text, record, events, _ = compose(LLMUnavailable("quota used up"))
    assert record["source"] == "template" and events == [{"type": "token", "text": text}]
    assert text == communicator.template_reply(finding(), [TERMS])


def test_an_empty_reply_is_retried():
    text, record, _, _ = compose("   ", GOOD)
    assert text == GOOD and record["tries"] == 2


def test_a_reply_that_leaves_out_the_amount_owed_is_retried():
    owed = finding("wrongly_rejected", None, due=5510)
    vague = "I'm sorry that SAVE10 was not accepted. I've recorded this for a person in customer care to review."
    full = "SAVE10 should have taken ₹55.10 off your order. I've recorded this for a person in customer care to review."
    events = []
    router = Router(vague, full)
    text, record = communicator.compose("SAVE10 was rejected", owed, [TERMS], router, events.append)
    assert text == full and record["tries"] == 2
    assert events[-1] == {"type": "replace", "text": full}
    assert "it does not give the amount ₹55.10" in router.prompts[1]
    # too little: both what was due and what was taken
    too_little = communicator.must_state(finding("applied_too_little", None, due=9120, applied=4560))
    assert communicator.missing_from("₹91.20 was due.", too_little) == ["it does not give the amount ₹45.60"]
    assert communicator.missing_from("Nothing is owed.", communicator.must_state(finding())) == []


def test_a_refund_reply_may_name_its_order():
    """seen 2026-11-13: a wallet reply naming ORD-100345 was thrown away, as the order was not in its facts."""
    made = {"order_id": "ORD-100345", "amount_paise": 46100, "destination": "wallet"}
    parts = [{"part": "line:4", "name": "Sunrise Farms Salted Butter (100 g)", "amount_paise": 46100,
              "basis": "proven"}]
    facts = communicator.refund_facts(made, parts, ["Butter: the records show it was not delivered."])
    reply = "₹461 for the butter in order ORD-100345 has been added to your HiveMart wallet."
    assert communicator.check_reply(reply, facts) == []
