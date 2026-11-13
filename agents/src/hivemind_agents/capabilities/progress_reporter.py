"""c25 progress reporter: tells the customer what is being looked at while the checks run (blueprint c25).

no model. a message goes out only when a check actually starts or finishes, and its words come from a fixed list:
nothing a check found, and no word of the risk check (it shows the same neutral words as the history check).
"""

from typing import Literal

LABELS = {
    "order": "Checking your order details",
    "delivery": "Reviewing delivery information",
    "inventory_store": "Checking how your order was packed",
    "payment": "Checking your payment",
    "customer_history": "Reviewing your request",
    "risk": "Reviewing your request",
    "decision": "Working out the next step",
    "action": "Putting this right",
}
OTHER = "Looking into this"
STARTING = "Looking into your request"   # once the plan is made, before the first check


def label(capability: str) -> str:
    return LABELS.get(capability, OTHER)


def event(capability: str | None, state: Literal["started", "finished"]) -> dict:
    """a `progress` event for the screen. it says only which kind of check, never what it found."""
    return {"type": "progress", "state": state, "text": label(capability) if capability else STARTING}
