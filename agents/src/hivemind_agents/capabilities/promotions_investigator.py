"""c12 promotions investigator: asks the business's promotion engine whether a coupon should have applied to an
order, and reports the difference and the rule that failed. plain code; no model decides this."""

from datetime import datetime

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.contracts import PromotionFinding


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def investigate(order_id: str, code: str, backend: BackendClient) -> PromotionFinding:
    result = backend.evaluate_promotion(order_id, code)
    rules = result["rules"]
    checkout = result["checkout"]
    due, applied = result["due_discount_paise"], result["applied_discount_paise"]

    if rules is None:
        outcome = "unknown_code"
    elif applied > 0 and applied >= due:
        outcome = "applied"
    elif not result["eligible"]:
        outcome = "not_eligible"
    elif applied > 0:
        outcome = "applied_too_little"
    elif checkout is not None:
        outcome = "wrongly_rejected"
    else:
        outcome = "not_entered"

    return PromotionFinding(
        order_id=order_id, code=code, outcome=outcome,
        failed_rule=result["failed_rule"] if outcome == "not_eligible" else None,
        item_total_paise=result["order"]["item_total_paise"],
        min_cart_paise=rules["min_cart_paise"] if rules else None,
        due_discount_paise=due, applied_discount_paise=applied, difference_paise=result["difference_paise"],
        offer_kind=rules["kind"] if rules else None, offer_value=rules["value"] if rules else None,
        max_discount_paise=rules["max_discount_paise"] if rules else None,
        valid_from=_time(rules["valid_from"]) if rules else None, valid_to=_time(rules["valid_to"]) if rules else None,
        placed_at=_time(result["order"]["placed_at"]),
        checkout_status=checkout["status"] if checkout else None,
        checkout_reason=checkout["reason"] if checkout else None,
        evidence_refs=[result["evidence_id"]],
    )
