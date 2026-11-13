"""the fixed replies: wording that never comes from a model."""

from datetime import datetime

from hivemind_agents.contracts import Chip, IntentHint, OrderRef
from hivemind_agents.settings import BUSINESS_NAME
from hivemind_agents.wording import rupees, when

CAN_HELP_WITH = ("where it is, a missing or wrong item, the quality of an item, a cancellation, a refund, a failed "
                 "payment, how the delivery partner behaved, or an offer or coupon")

WELCOME = (f"Hi, I'm HiveMind, an AI assistant for {BUSINESS_NAME} support, not a person. I can help with a problem "
           f"on an order: {CAN_HELP_WITH}. What do you need help with?")

OUT_OF_SCOPE = (f"I can only help with a problem on an order: {CAN_HELP_WITH}. For anything else, please use Help "
                f"in the {BUSINESS_NAME} app.")

SAFETY = ("I'm sorry this happened. I'm an AI and can't handle a safety matter in this chat. If anyone is in danger, "
          f"please call 112 right away. To report it to {BUSINESS_NAME}, please use Help in the {BUSINESS_NAME} app "
          "so a person can take it up.")

LEGAL = (f"I'm an AI and can't handle a legal complaint in this chat. Please write to {BUSINESS_NAME}'s grievance "
         f"officer; the contact details are under Help in the {BUSINESS_NAME} app.")

GREETING = (f"Hello! I'm HiveMind, {BUSINESS_NAME}'s AI support assistant. Tell me what went wrong with an order, or "
            "ask me where an order is.")

THANKS = "You're welcome. If anything else comes up with an order, just tell me."

ENGLISH_ONLY = "Sorry, I can only help in English for now. Please write your message in English."

TROUBLE = ("Sorry, I'm having trouble on my side right now and couldn't work on your message. Please try again in a "
           "little while.")

NO_ORDERS = "I couldn't find any orders on your account, so there is nothing for me to check."

ALREADY_WITH_PERSON = ("This is already recorded for a person in customer care to review. You'll get the update "
                       "here in this chat.")

CLARIFY = (f"I want to get this right. Is it about an order: {CAN_HELP_WITH}? Tell me a little more about what "
           "happened.")

CASE_BUSY = "I'm already looking into this. Please ask me again in a couple of minutes."

# between the parts of one reply that answers several problems
PART_BREAK = "\n\n"

CASE_LABELS = {
    "where_is_my_order": "where your order is",
    "item_missing_or_wrong": "a missing or wrong item",
    "quality": "the quality of an item",
    "cancellation": "cancelling an order",
    "refund": "a refund",
    "payment_failed": "a failed payment",
    "delivery_partner_behaviour": "how the delivery partner behaved",
    "offers_coupons": "an offer or coupon",
}

STATUS_LABELS = {
    "PLACED": "placed", "CONFIRMED": "confirmed", "PACKING": "being packed", "PACKED": "packed",
    "OUT_FOR_DELIVERY": "on its way", "DELIVERED": "delivered", "CANCELLED": "cancelled",
    "PARTIALLY_FULFILLED": "delivered in part",
}


def waiting_for_person(case_type: str) -> str:
    return (f"I understand this is about {CASE_LABELS[case_type]}. I can't settle this kind of problem by myself yet, "
            "so I've recorded it for a person in customer care to review. You'll get the update here in this chat.")


def with_a_person(case_type: str) -> str:
    """a case support may not settle alone: neutral words, never why (blueprint s09)."""
    return (f"I understand this is about {CASE_LABELS[case_type]}. I've recorded it for a person in customer care to "
            "review, and you'll get the update here in this chat.")


PENDING_SYSTEM = ("I've asked for this to be put right, but our system hasn't confirmed it yet. Please check back here "
                  "in a little while; nothing will be done twice.")


def cancelled(order_id: str, refund: dict | None) -> str:
    text = f"I've cancelled order {order_id}."
    if refund:
        text += (f" The {rupees(refund['amount_paise'])} you paid is being refunded to your original payment method; "
                 f"it should reach you in {refund['expected_settlement']}.")
    return text


def cannot_cancel(order_id: str) -> str:
    return (f"Order {order_id} is already being packed, so I can't cancel it myself. I've recorded your request for a "
            "person in customer care to review, and you'll get the update here in this chat.")


NOTHING_TO_CANCEL = {
    "delivered": "was already delivered", "cancelled": "is already cancelled",
    "partially_fulfilled": "was already delivered", "refunded": "is already refunded",
}


def nothing_to_cancel(order_id: str, why: str) -> str:
    state = NOTHING_TO_CANCEL.get(why.removeprefix("order_"), "can no longer be cancelled")
    return f"Order {order_id} {state}, so there is nothing to cancel."


def already_refunded(order_id: str, amount_paise: int) -> str:
    return (f"This was already refunded on order {order_id}: {rupees(amount_paise)} in all. Nothing more is owed for "
            "it, and nothing will be refunded twice.")


def nothing_owed_payment(order_id: str) -> str:
    return (f"I checked the payments on order {order_id}. They show no money taken that should not have been: no "
            "double charge, and no failed payment that kept your money. So there's nothing to refund.")


def order_not_found(order_id: str) -> str:
    return f"I couldn't find order {order_id} on your account. Which order is this about?"


def which_order(case_type: str | None = None) -> str:
    """the question alone, or, after the parts about other problems in the same reply, naming its problem."""
    if case_type is None:
        return "Which order is this about?"
    return f"About {CASE_LABELS[case_type]}: which order is it?"


def which_coupon(order_id: str) -> str:
    return f"Which coupon code did you use on order {order_id}? Please type the code."


ITEM_QUESTIONS = {
    "item_missing_or_wrong": "Which item was missing or wrong",
    "quality": "Which item had the problem",
}


def which_item(case_type: str, order_id: str) -> str:
    return (f"{ITEM_QUESTIONS[case_type]} in order {order_id}? Tap it below, or type the names if there was more "
            "than one.")


def ask_again(question: str) -> str:
    """our question once more, when the customer's message did not answer it."""
    return f"Sorry, I couldn't tell the answer from that. {question}"


def item_chips(case_type: str, order_id: str, names: list[str]) -> list[Chip]:
    return [Chip(label=name, text=name, intent=IntentHint(case_type=case_type, order_id=order_id, items=[name]))
            for name in names]


def order_chips(case_type: str, orders: list[OrderRef], now: datetime, coupon_code: str | None = None) -> list[Chip]:
    return [Chip(label=f"{o.order_id} · {when(o.placed_at, now)} · {STATUS_LABELS.get(o.status, o.status)}",
                 text=o.order_id,
                 intent=IntentHint(case_type=case_type, order_id=o.order_id, coupon_code=coupon_code))
            for o in orders]
