"""response shapes of the read apis. these models are the source of the openapi contract."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

# every time leaves the api as rfc 3339 utc ("...Z")
UtcDatetime = Annotated[datetime, AfterValidator(lambda d: d.astimezone(timezone.utc))]
Paise = Annotated[int, Field(description="money in integer paise (1 rupee = 100 paise)")]


class Problem(BaseModel):
    """rfc 9457 problem details."""

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


class Evidenced(BaseModel):
    evidence_id: str = Field(description="ev_ + the first 32 hex characters of the sha-256 of the body without this field")


# orders


class OrderLine(BaseModel):
    line_no: int
    sku: str
    name: str
    qty: int
    unit_paise: Paise
    status: Literal["ORDERED", "UNAVAILABLE", "SUBSTITUTED"]


class StatusChange(BaseModel):
    seq: int
    status: str
    at: UtcDatetime


OrderStatus = Literal[
    "PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY", "DELIVERED", "CANCELLED", "PARTIALLY_FULFILLED"
]


class Order(BaseModel):
    order_id: str
    customer_id: str
    store_id: str
    address_id: str
    status: OrderStatus
    item_total_paise: Paise
    discount_paise: Paise
    delivery_fee_paise: Paise
    total_paise: Paise
    coupon_code: str | None
    placed_at: UtcDatetime
    promised_by: UtcDatetime
    delivered_at: UtcDatetime | None
    cancelled_at: UtcDatetime | None
    lines: list[OrderLine]
    status_history: list[StatusChange]


class OrderResponse(Order, Evidenced):
    pass


class OrderSummary(BaseModel):
    order_id: str
    store_id: str
    status: OrderStatus
    total_paise: Paise
    placed_at: UtcDatetime
    delivered_at: UtcDatetime | None


class OrderHistory(BaseModel):
    customer_id: str
    orders: list[OrderSummary]


class OrderHistoryResponse(OrderHistory, Evidenced):
    pass


# store & inventory


class Scan(BaseModel):
    scan_id: str
    sku: str
    event: Literal["picked", "packed"]
    qty: int
    bag: int | None
    picker: str
    at: UtcDatetime


class MissingScan(BaseModel):
    sku: str
    line_no: int
    expected: Literal["picked", "packed"]


class Scans(BaseModel):
    order_id: str
    store_id: str
    scans: list[Scan]
    missing_scans: list[MissingScan] = Field(
        description="ordered lines without the scan their stage needs; empty until the order is packed"
    )


class ScansResponse(Scans, Evidenced):
    pass


class Incident(BaseModel):
    incident_id: str
    type: Literal["out_of_stock", "damaged_stock", "power_cut", "staff_shortage"]
    sku: str | None
    description: str
    at: UtcDatetime


class Incidents(BaseModel):
    store_id: str
    since: UtcDatetime
    until: UtcDatetime
    incidents: list[Incident]


class IncidentsResponse(Incidents, Evidenced):
    pass


class StockEvent(BaseModel):
    event_id: str
    sku: str
    event_type: Literal["restock", "stock_out", "adjust"]
    qty_delta: int
    at: UtcDatetime


class StockEvents(BaseModel):
    store_id: str
    sku: str
    since: UtcDatetime
    until: UtcDatetime
    events: list[StockEvent]


class StockEventsResponse(StockEvents, Evidenced):
    pass


# delivery


class GpsPoint(BaseModel):
    seq: int
    at: UtcDatetime
    lat: float
    lng: float


class Trip(BaseModel):
    order_id: str
    trip_id: str
    rider_id: str
    store_id: str
    status: Literal["ASSIGNED", "PICKED_UP", "DELIVERED", "CANCELLED"]
    assigned_at: UtcDatetime
    picked_up_at: UtcDatetime | None
    delivered_at: UtcDatetime | None
    eta_at: UtcDatetime
    distance_m: int
    gps_points: list[GpsPoint]


class TripResponse(Trip, Evidenced):
    pass


class ProofOfDelivery(BaseModel):
    order_id: str
    trip_id: str
    otp_verified: bool
    bags_handed: int
    handed_to: Literal["customer", "family_member", "security"]
    delivered_at: UtcDatetime


class ProofOfDeliveryResponse(ProofOfDelivery, Evidenced):
    pass


class PartnerNote(BaseModel):
    note_id: str
    trip_id: str
    text: str = Field(description="free text from the delivery partner; untrusted input")
    at: UtcDatetime


class PartnerNotes(BaseModel):
    order_id: str
    notes: list[PartnerNote]


class PartnerNotesResponse(PartnerNotes, Evidenced):
    pass


# payments and refunds


class Capture(BaseModel):
    capture_id: str
    amount_paise: Paise
    gateway_ref: str
    captured_at: UtcDatetime


class GatewayEvent(BaseModel):
    event_id: str
    type: Literal["debit_success", "debit_failed", "capture", "reversal"]
    amount_paise: Paise
    gateway_ref: str
    at: UtcDatetime


class PaymentIntent(BaseModel):
    intent_id: str
    method: Literal["upi", "card", "wallet"]
    instrument_ref: str
    amount_paise: Paise
    status: Literal["created", "succeeded", "failed"]
    created_at: UtcDatetime
    captures: list[Capture]
    gateway_events: list[GatewayEvent]


class Payments(BaseModel):
    order_id: str
    intents: list[PaymentIntent]


class PaymentsResponse(Payments, Evidenced):
    pass


class Refund(BaseModel):
    refund_id: str
    amount_paise: Paise
    destination: Literal["source", "wallet"]
    status: Literal["INITIATED", "SUBMITTED_TO_GATEWAY", "COMPLETED", "FAILED", "RETRYING", "COMPENSATED"]
    reason: str
    case_id: str | None
    created_at: UtcDatetime


class Refunds(BaseModel):
    order_id: str
    refunds: list[Refund]


class RefundsResponse(Refunds, Evidenced):
    pass


# clock


class Clock(BaseModel):
    now: UtcDatetime = Field(description="the simulated business's current time: the moment its made-up data ends")


class ClockResponse(Clock, Evidenced):
    pass


# identity: the fake sign-in


class SignInOption(BaseModel):
    customer_id: str
    name: str
    hints: list[str] = Field(description="what this customer's recent orders hold, to pick one for a demo")


class SignInOptions(BaseModel):
    customers: list[SignInOption]


class SessionRequest(BaseModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$", examples=["CUST-00042"])


class Session(BaseModel):
    session_token: str = Field(description="hs256 jwt; send it to hivemind, which checks it")
    customer_id: str
    name: str
    expires_at: UtcDatetime


class Me(BaseModel):
    customer_id: str
    name: str
    language: str


class MeResponse(Me, Evidenced):
    pass


# promotions


class Offer(BaseModel):
    code: str
    title: str
    kind: Literal["flat", "percent", "free_delivery"]
    value: int = Field(description="flat: paise off; percent: percent off the item total; free_delivery: unused")
    max_discount_paise: Paise | None
    min_cart_paise: Paise
    first_order_only: bool
    valid_from: UtcDatetime
    valid_to: UtcDatetime
    terms: str


class OfferResponse(Offer, Evidenced):
    pass


class Redemption(BaseModel):
    redemption_id: str
    code: str
    order_id: str
    status: Literal["applied", "rejected"]
    discount_paise: Paise
    reason: str | None
    at: UtcDatetime


class Redemptions(BaseModel):
    customer_id: str
    redemptions: list[Redemption]


class RedemptionsResponse(Redemptions, Evidenced):
    pass


class OfferRules(BaseModel):
    kind: Literal["flat", "percent", "free_delivery"]
    value: int
    max_discount_paise: Paise | None
    min_cart_paise: Paise
    first_order_only: bool
    valid_from: UtcDatetime
    valid_to: UtcDatetime


class OrderSnapshot(BaseModel):
    item_total_paise: Paise
    delivery_fee_paise: Paise
    placed_at: UtcDatetime
    first_order: bool


class Checkout(BaseModel):
    status: Literal["applied", "rejected"]
    discount_paise: Paise
    reason: str | None


class PromotionEvaluation(BaseModel):
    order_id: str
    code: str
    rules: OfferRules | None = Field(description="null when no offer has this code")
    order: OrderSnapshot
    eligible: bool
    due_discount_paise: Paise = Field(description="what the offer's rules give this order; 0 when not eligible")
    failed_rule: Literal["unknown_code", "not_started", "expired", "first_order_only", "min_cart",
                         "delivery_already_free"] | None
    checkout: Checkout | None = Field(description="what happened to this code at checkout; null when never tried")
    applied_discount_paise: Paise = Field(description="the discount the order really got from this code")
    difference_paise: int = Field(description="due minus applied; above 0 the customer got less than the rules give")


class PromotionEvaluationResponse(PromotionEvaluation, Evidenced):
    pass


# support: cases


IssueType = Literal["where_is_my_order", "item_missing_or_wrong", "quality", "cancellation", "refund",
                    "payment_failed", "delivery_partner_behaviour", "offers_coupons"]
CaseState = Literal["OPEN", "INVESTIGATING", "DECIDING", "AWAITING_CUSTOMER", "AWAITING_HUMAN", "PENDING_SYSTEM",
                    "ACTIONING", "ESCALATED", "RESOLVED", "CLOSED", "REOPENED"]


class CaseOpenRequest(BaseModel):
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{6}$")
    issue_type: IssueType


class Case(BaseModel):
    case_id: str
    customer_id: str
    order_id: str | None
    issue_type: IssueType
    state: CaseState
    created_at: UtcDatetime
    updated_at: UtcDatetime


class OpenedCase(Case):
    opened: bool = Field(description="true when this call opened the case; false when it was already open")


class CaseStateRequest(BaseModel):
    state: CaseState
    reason: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,59}$", examples=["reply_sent"],
                               description="why the case moves, as a short code; kept in the case's history")
