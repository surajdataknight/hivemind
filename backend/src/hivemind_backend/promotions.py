"""the promotion engine: what an offer's rules give for one order. the rules are code, so the answer to "should
this coupon have applied" never comes from a model."""

from datetime import datetime
from typing import Literal

FailedRule = Literal["unknown_code", "not_started", "expired", "first_order_only", "min_cart", "delivery_already_free"]


def evaluate(offer: dict | None, item_total_paise: int, delivery_fee_paise: int, first_order: bool,
             placed_at: datetime) -> tuple[int, FailedRule | None]:
    """the discount the offer gives for this order, or 0 and the rule that stopped it."""
    if offer is None:
        return 0, "unknown_code"
    if placed_at < offer["valid_from"]:
        return 0, "not_started"
    if placed_at >= offer["valid_to"]:
        return 0, "expired"
    if offer["first_order_only"] and not first_order:
        return 0, "first_order_only"
    if item_total_paise < offer["min_cart_paise"]:
        return 0, "min_cart"
    if offer["kind"] == "flat":
        return min(offer["value"], item_total_paise), None
    if offer["kind"] == "percent":
        return min(item_total_paise * offer["value"] // 100, offer["max_discount_paise"]), None
    if delivery_fee_paise == 0:
        return 0, "delivery_already_free"
    return delivery_fee_paise, None
