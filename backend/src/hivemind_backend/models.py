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
