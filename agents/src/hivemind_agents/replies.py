"""the fixed replies: wording that never comes from a model."""

from datetime import datetime

from hivemind_agents.contracts import Chip, IntentHint, OrderRef
from hivemind_agents.settings import BUSINESS_NAME
from hivemind_agents.wording import when

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

MORE_ISSUES = (" You also mentioned something else. Please send it as a separate message so I can look at it on "
               "its own.")

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


def order_not_found(order_id: str) -> str:
    return f"I couldn't find order {order_id} on your account. Which order is this about?"


def which_order() -> str:
    return "Which order is this about?"


def which_coupon(order_id: str) -> str:
    return f"Which coupon code did you use on order {order_id}? Please type the code."


def order_chips(case_type: str, orders: list[OrderRef], now: datetime, coupon_code: str | None = None) -> list[Chip]:
    return [Chip(label=f"{o.order_id} · {when(o.placed_at, now)} · {STATUS_LABELS.get(o.status, o.status)}",
                 text=o.order_id,
                 intent=IntentHint(case_type=case_type, order_id=o.order_id, coupon_code=coupon_code))
            for o in orders]
