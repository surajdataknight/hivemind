"""c3's triage half: which order, and which coupon, an issue is about. rules only.

an order id the customer named is checked against their own orders. an order named by a day ("yesterday's order")
is the one order of that day this problem can be about, by the business's clock in india; the model only reads the
day, code picks the order. when the rules cannot tell which order is meant, the customer is asked to choose;
nothing is guessed.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from hivemind_agents.backend_client import BackendClient, NotFound
from hivemind_agents.contracts import Issue, OrderRef
from hivemind_agents.wording import IST

LIVE = ("PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY")
DELIVERED = ("DELIVERED", "PARTIALLY_FULFILLED")
RECENT_ORDERS = 5

# the orders a problem of each type can be about; None: any order. these case types need an order before they are
# worked, so when the rules cannot tell which one, the customer is asked
FITS: dict[str, tuple[str, ...] | None] = {
    "item_missing_or_wrong": DELIVERED,
    "quality": DELIVERED,
    "delivery_partner_behaviour": ("OUT_FOR_DELIVERY",) + DELIVERED,
    "cancellation": LIVE,
    "refund": None,
}


# the case types that are about particular items of the order: when the customer named none, they are asked which
NAMES_ITEMS = ("item_missing_or_wrong", "quality")
MAX_ITEM_CHIPS = 12  # buttons shown; a typed answer may name any line


@dataclass
class Triage:
    issue: Issue
    ask: str | None = None  # "order", "coupon_code", "items", or None when the issue is ready
    choices: list[OrderRef] = field(default_factory=list)
    unknown_order: str | None = None
    no_orders: bool = False
    lines: list[str] = field(default_factory=list)  # the order's line names, when asking which item


def recent_orders(backend: BackendClient) -> list[OrderRef]:
    rows = backend.my_orders(RECENT_ORDERS)["orders"]
    return [OrderRef(order_id=r["order_id"], status=r["status"], placed_at=_at(r["placed_at"]),
                     delivered_at=_at(r.get("delivered_at"))) for r in rows]


def _at(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _owned(order_id: str, recent: list[OrderRef], backend: BackendClient) -> bool:
    if any(o.order_id == order_id for o in recent):
        return True
    try:
        backend.order(order_id)
        return True
    except NotFound:
        return False


def triage(issue: Issue, recent: list[OrderRef], backend: BackendClient) -> Triage:
    """settles the order (and coupon); then, for a problem with particular items where the customer named none,
    asks which, with the order's lines to choose from."""
    result = _settle_order(issue, recent, backend)
    settled = result.issue
    if result.ask or result.no_orders or not settled.order_id or settled.case_type not in NAMES_ITEMS:
        return result
    if settled.items_mentioned:
        return result
    names = list(dict.fromkeys(line["name"] for line in backend.order(settled.order_id)["lines"]))
    return Triage(settled, ask="items", lines=names) if names else result


def _settle_order(issue: Issue, recent: list[OrderRef], backend: BackendClient) -> Triage:
    if issue.order_id and not _owned(issue.order_id, recent, backend):
        return Triage(issue.model_copy(update={"order_id": None}), ask="order", choices=recent,
                      unknown_order=issue.order_id)
    if not recent and not issue.order_id:
        return Triage(issue, no_orders=True)
    if issue.case_type == "where_is_my_order":
        return _where_is_my_order(issue, recent)
    if issue.case_type == "offers_coupons":
        return _coupon(issue, recent, backend)
    if issue.order_id:
        return Triage(issue)
    fitting = [o for o in recent if _fits(issue.case_type, o)]
    if issue.latest_order and fitting:  # "my last order": the newest one this problem can be about
        return Triage(issue.model_copy(update={"order_id": fitting[0].order_id}))
    if issue.order_day and fitting:  # "yesterday's order": the orders it fits placed or delivered that day
        day = resolve_day(issue.order_day, business_today(backend))
        on_day = [o for o in fitting if day is not None and day in _days(o)]
        if len(on_day) == 1:
            return Triage(issue.model_copy(update={"order_id": on_day[0].order_id}))
        if on_day:
            return Triage(issue, ask="order", choices=on_day)
        # no order of that day among the newest: asked among them below (it may be an older one, typed in)
    if issue.case_type not in FITS:  # a failed payment may have no order at all: recorded as it is
        if len(recent) == 1:
            issue = issue.model_copy(update={"order_id": recent[0].order_id})
        return Triage(issue)
    choices = fitting or recent
    if len(choices) == 1:
        return Triage(issue.model_copy(update={"order_id": choices[0].order_id}))
    return Triage(issue, ask="order", choices=choices)


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def business_today(backend: BackendClient) -> date:
    """today on the business's clock (seeded time, never this computer's), as a day in india."""
    return datetime.fromisoformat(backend.clock()["now"].replace("Z", "+00:00")).astimezone(IST).date()


def resolve_day(word: str, today: date) -> date | None:
    """the date a day word names, on or before today: a weekday is the latest one up to today; a later date is
    none."""
    if word in ("today", "yesterday", "day_before_yesterday"):
        return today - timedelta(days=("today", "yesterday", "day_before_yesterday").index(word))
    if word in WEEKDAYS:
        return today - timedelta(days=(today.weekday() - WEEKDAYS.index(word)) % 7)
    try:
        named = date.fromisoformat(word)
    except ValueError:
        return None
    return named if named <= today else None


def _days(order: OrderRef) -> set[date]:
    """the days in india an order was placed and delivered: "yesterday's order" can mean either."""
    return {at.astimezone(IST).date() for at in (order.placed_at, order.delivered_at) if at is not None}


def _fits(case_type: str, order: OrderRef) -> bool:
    statuses = FITS.get(case_type)  # a type not in FITS (a failed payment) can be about any order
    return statuses is None or order.status in statuses


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
