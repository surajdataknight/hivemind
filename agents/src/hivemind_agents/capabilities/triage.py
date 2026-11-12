"""c3's triage half: which order, and which coupon, an issue is about. rules only.

an order id the customer named is checked against their own orders. when the rules cannot tell which order is
meant, the customer is asked to choose; nothing is guessed.
"""

from dataclasses import dataclass, field
from datetime import datetime

from hivemind_agents.backend_client import BackendClient, NotFound
from hivemind_agents.contracts import Issue, OrderRef

LIVE = ("PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY")
RECENT_ORDERS = 5


@dataclass
class Triage:
    issue: Issue
    ask: str | None = None  # "order", "coupon_code", or None when the issue is ready
    choices: list[OrderRef] = field(default_factory=list)
    unknown_order: str | None = None
    no_orders: bool = False


def recent_orders(backend: BackendClient) -> list[OrderRef]:
    rows = backend.my_orders(RECENT_ORDERS)["orders"]
    return [OrderRef(order_id=r["order_id"], status=r["status"],
                     placed_at=datetime.fromisoformat(r["placed_at"].replace("Z", "+00:00"))) for r in rows]


def _owned(order_id: str, recent: list[OrderRef], backend: BackendClient) -> bool:
    if any(o.order_id == order_id for o in recent):
        return True
    try:
        backend.order(order_id)
        return True
    except NotFound:
        return False


def triage(issue: Issue, recent: list[OrderRef], backend: BackendClient) -> Triage:
    if issue.order_id and not _owned(issue.order_id, recent, backend):
        return Triage(issue.model_copy(update={"order_id": None}), ask="order", choices=recent,
                      unknown_order=issue.order_id)
    if not recent and not issue.order_id:
        return Triage(issue, no_orders=True)
    if issue.case_type == "where_is_my_order":
        return _where_is_my_order(issue, recent)
    if issue.case_type == "offers_coupons":
        return _coupon(issue, recent, backend)
    if issue.order_id is None and len(recent) == 1:
        issue = issue.model_copy(update={"order_id": recent[0].order_id})
    return Triage(issue)  # other case types are recorded as they are, with or without an order


def _where_is_my_order(issue: Issue, recent: list[OrderRef]) -> Triage:
    """the order named; else the newest one still on its way; else the newest one."""
    if issue.order_id:
        return Triage(issue)
    live = [o for o in recent if o.status in LIVE]
    chosen = live[0] if live else recent[0]
    return Triage(issue.model_copy(update={"order_id": chosen.order_id}))


def _coupon(issue: Issue, recent: list[OrderRef], backend: BackendClient) -> Triage:
    attempts = backend.my_redemptions()["redemptions"]
    if issue.order_id:
        codes = [a["code"] for a in attempts if a["order_id"] == issue.order_id]
        if issue.coupon_code is None and len(set(codes)) == 1:
            issue = issue.model_copy(update={"coupon_code": codes[0]})
        return Triage(issue, ask=None if issue.coupon_code else "coupon_code")

    recent_ids = [o.order_id for o in recent]
    tried = [a for a in attempts if a["order_id"] in recent_ids
             and (issue.coupon_code is None or a["code"] == issue.coupon_code)]
    orders_tried = [o for o in recent if any(a["order_id"] == o.order_id for a in tried)]
    if len(orders_tried) == 1:
        order_id = orders_tried[0].order_id
        code = issue.coupon_code or next(a["code"] for a in tried if a["order_id"] == order_id)
        return Triage(issue.model_copy(update={"order_id": order_id, "coupon_code": code}))
    if len(orders_tried) > 1:
        return Triage(issue, ask="order", choices=orders_tried)
    if len(recent) == 1:
        issue = issue.model_copy(update={"order_id": recent[0].order_id})
        return Triage(issue, ask=None if issue.coupon_code else "coupon_code")
    return Triage(issue, ask="order", choices=recent)
