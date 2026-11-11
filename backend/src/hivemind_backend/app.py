"""the simulated business's read apis (phase 0). writes come with the action gateway in phase 3."""

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from hivemind_backend import db, models
from hivemind_backend.evidence import evidence_id

API_VERSION = "0.1.0"
MAX_WINDOW = timedelta(days=7)
PACKED_OR_LATER = {"PACKED", "OUT_FOR_DELIVERY", "DELIVERED", "PARTIALLY_FULFILLED"}

app = FastAPI(
    title="HiveMind simulated business",
    version=API_VERSION,
    description=(
        "read apis of the simulated quick-commerce grocery business. money is integer paise; times are rfc 3339 utc; "
        "errors are rfc 9457 problem details. every body carries an evidence_id."
    ),
)

ERRORS = {404: {"model": models.Problem}, 422: {"model": models.Problem}}


def get_conn() -> Iterator[psycopg.Connection]:
    with db.connect("backend", row_factory=dict_row, autocommit=True) as conn:
        yield conn


DbConn = Annotated[psycopg.Connection, Depends(get_conn)]
OrderId = Annotated[str, Path(pattern=r"^ORD-\d{6}$", examples=["ORD-100123"])]
StoreId = Annotated[str, Path(pattern=r"^DS-[A-Z]{3}-\d{2}$", examples=["DS-BLR-01"])]


# errors as problem details


def _problem(status: int, type_: str, title: str, detail: str | None, instance: str) -> JSONResponse:
    body = models.Problem(type=type_, title=title, status=status, detail=detail, instance=instance)
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, media_type="application/problem+json")


@app.exception_handler(HTTPException)
async def http_problem(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 404:
        return _problem(404, "/problems/not-found", "not found", exc.detail, request.url.path)
    return _problem(exc.status_code, "/problems/http-error", "request failed", str(exc.detail), request.url.path)


@app.exception_handler(RequestValidationError)
async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return _problem(422, "/problems/invalid-request", "invalid request", detail, request.url.path)


def not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=what)


def invalid(detail: str) -> RequestValidationError:
    return RequestValidationError([{"loc": ("query",), "msg": detail, "type": "value_error"}])


def evidenced(body_model: type[BaseModel], response_model: type[BaseModel], data: dict) -> BaseModel:
    body = body_model.model_validate(data).model_dump(mode="json")
    return response_model.model_validate({**body, "evidence_id": evidence_id(body)})


def check_window(since: datetime, until: datetime) -> None:
    if since.tzinfo is None or until.tzinfo is None:
        raise invalid("since and until need a timezone, for example 2026-10-02T10:00:00Z")
    if until <= since:
        raise invalid("until must be after since")
    if until - since > MAX_WINDOW:
        raise invalid("the window may be at most 7 days")


# orders module


def fetch_order(conn: psycopg.Connection, order_id: str) -> dict | None:
    return conn.execute(
        "select order_id, customer_id, store_id, address_id, status, item_total_paise, discount_paise, "
        "delivery_fee_paise, total_paise, coupon_code, placed_at, promised_by, delivered_at, cancelled_at "
        "from orders.orders where order_id = %s",
        (order_id,),
    ).fetchone()


def fetch_lines(conn: psycopg.Connection, order_id: str) -> list[dict]:
    return conn.execute(
        "select line_no, sku, name, qty, unit_paise, status from orders.order_lines where order_id = %s "
        "order by line_no",
        (order_id,),
    ).fetchall()


def require_order(conn: psycopg.Connection, order_id: str) -> dict:
    order = fetch_order(conn, order_id)
    if order is None:
        raise not_found(f"order {order_id} does not exist")
    return order


@app.get("/v1/orders/{order_id}", response_model=models.OrderResponse, responses=ERRORS, tags=["orders"],
         operation_id="getOrder", summary="one order with its lines and status history")
def get_order(order_id: OrderId, conn: DbConn):
    order = require_order(conn, order_id)
    order["lines"] = fetch_lines(conn, order_id)
    order["status_history"] = conn.execute(
        "select seq, status, at from orders.order_status_history where order_id = %s order by seq", (order_id,)
    ).fetchall()
    return evidenced(models.Order, models.OrderResponse, order)


@app.get("/v1/customers/{customer_id}/orders", response_model=models.OrderHistoryResponse, responses=ERRORS,
         tags=["orders"], operation_id="listCustomerOrders", summary="a customer's orders, newest first")
def list_customer_orders(
    customer_id: Annotated[str, Path(pattern=r"^CUST-\d{5}$", examples=["CUST-00042"])],
    conn: DbConn,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    rows = conn.execute(
        "select order_id, store_id, status, total_paise, placed_at, delivered_at from orders.orders "
        "where customer_id = %s order by placed_at desc, order_id desc limit %s",
        (customer_id, limit),
    ).fetchall()
    return evidenced(models.OrderHistory, models.OrderHistoryResponse, {"customer_id": customer_id, "orders": rows})


# store & inventory module


@app.get("/v1/orders/{order_id}/scans", response_model=models.ScansResponse, responses=ERRORS,
         tags=["store_inventory"], operation_id="getOrderScans", summary="pick and pack scans for an order")
def get_order_scans(order_id: OrderId, conn: DbConn):
    order = require_order(conn, order_id)
    scans = conn.execute(
        "select scan_id, sku, event, qty, bag, picker, at from store_inventory.pick_pack_scans "
        "where order_id = %s order by at, scan_id",
        (order_id,),
    ).fetchall()
    missing = []
    if order["status"] in PACKED_OR_LATER:
        seen = {(s["sku"], s["event"]) for s in scans}
        for line in fetch_lines(conn, order_id):
            if line["status"] != "ORDERED":
                continue
            if (line["sku"], "picked") not in seen:
                missing.append({"sku": line["sku"], "line_no": line["line_no"], "expected": "picked"})
            elif (line["sku"], "packed") not in seen:
                missing.append({"sku": line["sku"], "line_no": line["line_no"], "expected": "packed"})
    data = {"order_id": order_id, "store_id": order["store_id"], "scans": scans, "missing_scans": missing}
    return evidenced(models.Scans, models.ScansResponse, data)


def require_store(conn: psycopg.Connection, store_id: str) -> None:
    if conn.execute("select 1 from store_inventory.stores where store_id = %s", (store_id,)).fetchone() is None:
        raise not_found(f"store {store_id} does not exist")


@app.get("/v1/stores/{store_id}/incidents", response_model=models.IncidentsResponse, responses=ERRORS,
         tags=["store_inventory"], operation_id="listStoreIncidents", summary="store incidents in a time window")
def list_store_incidents(
    store_id: StoreId,
    since: Annotated[datetime, Query(description="rfc 3339, inclusive")],
    until: Annotated[datetime, Query(description="rfc 3339, exclusive; at most 7 days after since")],
    conn: DbConn,
):
    check_window(since, until)
    require_store(conn, store_id)
    rows = conn.execute(
        "select incident_id, type, sku, description, at from store_inventory.incidents "
        "where store_id = %s and at >= %s and at < %s order by at, incident_id",
        (store_id, since, until),
    ).fetchall()
    data = {"store_id": store_id, "since": since, "until": until, "incidents": rows}
    return evidenced(models.Incidents, models.IncidentsResponse, data)


@app.get("/v1/stores/{store_id}/stock-events", response_model=models.StockEventsResponse, responses=ERRORS,
         tags=["store_inventory"], operation_id="listStockEvents", summary="stock events for one sku in a window")
def list_stock_events(
    store_id: StoreId,
    sku: Annotated[str, Query(pattern=r"^SKU-[A-Z]{3}-\d{3}$", examples=["SKU-DAI-001"])],
    since: Annotated[datetime, Query(description="rfc 3339, inclusive")],
    until: Annotated[datetime, Query(description="rfc 3339, exclusive; at most 7 days after since")],
    conn: DbConn,
):
    check_window(since, until)
    require_store(conn, store_id)
    rows = conn.execute(
        "select event_id, sku, event_type, qty_delta, at from store_inventory.stock_events "
        "where store_id = %s and sku = %s and at >= %s and at < %s order by at, event_id",
        (store_id, sku, since, until),
    ).fetchall()
    data = {"store_id": store_id, "sku": sku, "since": since, "until": until, "events": rows}
    return evidenced(models.StockEvents, models.StockEventsResponse, data)


# delivery module


@app.get("/v1/orders/{order_id}/trip", response_model=models.TripResponse, responses=ERRORS, tags=["delivery"],
         operation_id="getOrderTrip", summary="the delivery trip and its gps points")
def get_order_trip(order_id: OrderId, conn: DbConn):
    require_order(conn, order_id)
    trip = conn.execute(
        "select order_id, trip_id, rider_id, store_id, status, assigned_at, picked_up_at, delivered_at, eta_at, "
        "distance_m from delivery.trips where order_id = %s",
        (order_id,),
    ).fetchone()
    if trip is None:
        raise not_found(f"order {order_id} has no delivery trip yet")
    trip["gps_points"] = conn.execute(
        "select seq, at, lat, lng from delivery.gps_points where trip_id = %s order by seq", (trip["trip_id"],)
    ).fetchall()
    return evidenced(models.Trip, models.TripResponse, trip)


@app.get("/v1/orders/{order_id}/pod", response_model=models.ProofOfDeliveryResponse, responses=ERRORS,
         tags=["delivery"], operation_id="getProofOfDelivery", summary="proof of delivery: otp, bags handed over")
def get_proof_of_delivery(order_id: OrderId, conn: DbConn):
    require_order(conn, order_id)
    pod = conn.execute(
        "select order_id, trip_id, otp_verified, bags_handed, handed_to, delivered_at from delivery.pod "
        "where order_id = %s",
        (order_id,),
    ).fetchone()
    if pod is None:
        raise not_found(f"order {order_id} has no proof of delivery")
    return evidenced(models.ProofOfDelivery, models.ProofOfDeliveryResponse, pod)


@app.get("/v1/orders/{order_id}/partner-notes", response_model=models.PartnerNotesResponse, responses=ERRORS,
         tags=["delivery"], operation_id="listPartnerNotes", summary="free-text notes from the delivery partner")
def list_partner_notes(order_id: OrderId, conn: DbConn):
    require_order(conn, order_id)
    notes = conn.execute(
        "select note_id, trip_id, text, at from delivery.partner_notes where order_id = %s order by at, note_id",
        (order_id,),
    ).fetchall()
    return evidenced(models.PartnerNotes, models.PartnerNotesResponse, {"order_id": order_id, "notes": notes})


# payments and refunds modules


@app.get("/v1/orders/{order_id}/payments", response_model=models.PaymentsResponse, responses=ERRORS,
         tags=["payments"], operation_id="listOrderPayments", summary="payment attempts, captures, gateway events")
def list_order_payments(order_id: OrderId, conn: DbConn):
    require_order(conn, order_id)
    intents = conn.execute(
        "select intent_id, method, instrument_ref, amount_paise, status, created_at from payments.payment_intents "
        "where order_id = %s order by created_at, intent_id",
        (order_id,),
    ).fetchall()
    for intent in intents:
        intent["captures"] = conn.execute(
            "select capture_id, amount_paise, gateway_ref, captured_at from payments.captures "
            "where intent_id = %s order by captured_at, capture_id",
            (intent["intent_id"],),
        ).fetchall()
        intent["gateway_events"] = conn.execute(
            "select event_id, type, amount_paise, gateway_ref, at from payments.gateway_events "
            "where intent_id = %s order by at, event_id",
            (intent["intent_id"],),
        ).fetchall()
    return evidenced(models.Payments, models.PaymentsResponse, {"order_id": order_id, "intents": intents})


@app.get("/v1/orders/{order_id}/refunds", response_model=models.RefundsResponse, responses=ERRORS,
         tags=["refunds_wallet"], operation_id="listOrderRefunds", summary="refunds made for an order")
def list_order_refunds(order_id: OrderId, conn: DbConn):
    require_order(conn, order_id)
    refunds = conn.execute(
        "select refund_id, amount_paise, destination, status, reason, case_id, created_at "
        "from refunds_wallet.refunds where order_id = %s order by created_at, refund_id",
        (order_id,),
    ).fetchall()
    return evidenced(models.Refunds, models.RefundsResponse, {"order_id": order_id, "refunds": refunds})


# the contract: error bodies are application/problem+json


def openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    spec = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    for path in spec["paths"].values():
        for operation in path.values():
            for status, response in operation.get("responses", {}).items():
                content = response.get("content", {})
                if status.startswith("4") and "application/json" in content:
                    content["application/problem+json"] = content.pop("application/json")
    app.openapi_schema = spec
    return spec


app.openapi = openapi
