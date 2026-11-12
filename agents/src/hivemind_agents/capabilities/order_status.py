"""the "where is my order" short path: reads the order and its delivery trip, answers from a template. no model."""

from datetime import datetime

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.wording import minutes_between, when

LIVE = ("PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY")


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def order_status(order_id: str, backend: BackendClient) -> tuple[str, dict]:
    """(the reply text, what it was built from)."""
    order = backend.order(order_id)
    trip = backend.trip(order_id)
    now = _time(backend.clock()["now"])
    status = order["status"]
    due = _time(trip["eta_at"]) if trip else _time(order["promised_by"])
    late_by = minutes_between(due, now) if status in LIVE and now > due else 0
    facts = {"order_id": order_id, "status": status, "due_by": due.isoformat(), "late_by_min": late_by,
             "evidence_refs": [order["evidence_id"]] + ([trip["evidence_id"]] if trip else [])}

    if status in ("PLACED", "CONFIRMED"):
        text = f"Your order {order_id} is confirmed and the store will start packing it shortly."
    elif status == "PACKING":
        text = f"Your order {order_id} is being packed at the store right now."
    elif status == "PACKED":
        text = f"Your order {order_id} is packed and waiting for a delivery partner to pick it up."
    elif status == "OUT_FOR_DELIVERY":
        picked_up = _time(trip["picked_up_at"]) if trip else None
        text = f"Your order {order_id} is on its way."
        if picked_up:
            text = f"Your order {order_id} is on its way: the delivery partner picked it up at {when(picked_up, now)}."
    elif status == "DELIVERED":
        delivered = _time(order["delivered_at"])
        facts["delivered_at"] = delivered.isoformat()
        return f"Your order {order_id} was delivered at {when(delivered, now)}.", facts
    elif status == "CANCELLED":
        cancelled = _time(order["cancelled_at"])
        facts["cancelled_at"] = cancelled.isoformat()
        return f"Your order {order_id} was cancelled at {when(cancelled, now)}, so it will not be delivered.", facts
    else:  # PARTIALLY_FULFILLED
        return (f"Your order {order_id} was delivered in part: at least one item was not available at the "
                "store."), facts

    if late_by:
        text += (f" It was due by {when(due, now)} and is running about {late_by} "
                 f"minute{'s' if late_by != 1 else ''} late. I'm sorry for the wait.")
    else:
        text += f" It should reach you by {when(due, now)}."
    return text, facts
