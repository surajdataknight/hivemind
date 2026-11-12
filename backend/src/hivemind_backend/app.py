"""the simulated business's apis: reads, the fake sign-in, and opening a support case. money and order changes
come with the action gateway."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated
from uuid import uuid4

import psycopg
from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from hivemind_backend import auth, db, models, promotions
from hivemind_backend.auth import Caller, scope
from hivemind_backend.evidence import evidence_id

API_VERSION = "0.2.0"
MAX_WINDOW = timedelta(days=7)
PACKED_OR_LATER = {"PACKED", "OUT_FOR_DELIVERY", "DELIVERED", "PARTIALLY_FULFILLED"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # refuse to start without the two signing keys, rather than fail on the first sign-in
    auth.signing_key("HM_SESSION_SECRET")
    auth.signing_key("HM_CASE_TOKEN_SECRET")
    yield


app = FastAPI(
    title="HiveMind simulated business",
    version=API_VERSION,
    lifespan=lifespan,
    description=(
        "apis of the simulated quick-commerce grocery business. money is integer paise; times are rfc 3339 utc; "
        "errors are rfc 9457 problem details. every read carries an evidence_id. every api except the fake sign-in "
        "needs a case token (`Authorization: Bearer ...`); the customer comes from that token, never from the url."
    ),
)

PROBLEM = {"model": models.Problem}
ERRORS = {401: PROBLEM, 403: PROBLEM, 404: PROBLEM, 422: PROBLEM}
OPEN_ERRORS = {404: PROBLEM, 422: PROBLEM}  # the fake sign-in takes no token
LIVE = ("PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY")


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
    known = {
        401: ("/problems/unauthenticated", "no valid case token"),
        403: ("/problems/forbidden", "not allowed"),
        404: ("/problems/not-found", "not found"),
        409: ("/problems/conflict", "conflict"),
        503: ("/problems/no-data", "no data loaded"),
    }
    if exc.status_code in known:
        type_, title = known[exc.status_code]
        return _problem(exc.status_code, type_, title, exc.detail, request.url.path)
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


def require_order(conn: psycopg.Connection, order_id: str, who: Caller) -> dict:
    """the order, only if it belongs to the token's customer. another customer's order looks like no order."""
    order = fetch_order(conn, order_id)
    if order is None or order["customer_id"] != who.customer_id:
        raise not_found(f"order {order_id} does not exist")
    return order


def sim_now(conn: psycopg.Connection) -> datetime:
    row = conn.execute("select now from public.sim_clock").fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="the made-up data is not loaded; run `hm-seed`")
    return row["now"]


# clock


@app.get("/v1/clock", response_model=models.ClockResponse, responses={401: PROBLEM}, tags=["clock"],
         operation_id="getClock", summary="the simulated business's current time")
def get_clock(conn: DbConn, who: Annotated[Caller, Depends(auth.caller)]):
    return evidenced(models.Clock, models.ClockResponse, {"now": sim_now(conn)})


# identity module: the fake sign-in; a real app would check a password or an otp here


@app.get("/v1/identity/sign-in-options", response_model=models.SignInOptions, tags=["identity"],
         operation_id="listSignInOptions", summary="fake sign-in: a few customers to sign in as")
def list_sign_in_options(conn: DbConn):
    picks: dict[str, list[str]] = {}
    queries = [
        ("an order on its way",
         "select customer_id from orders.orders where status = any(%s) group by 1 "
         "order by max(placed_at) desc, 1 limit 4", (list(LIVE),)),
        ("a coupon that was not accepted",
         "select customer_id from promotions.redemptions where status = 'rejected' group by 1 "
         "order by max(at) desc, 1 limit 4", ()),
        ("a coupon that was applied",
         "select customer_id from promotions.redemptions where status = 'applied' group by 1 "
         "order by max(at) desc, 1 limit 4", ()),
    ]
    for hint, query, params in queries:
        for row in conn.execute(query, params).fetchall():
            picks.setdefault(row["customer_id"], []).append(hint)
    names = {r["customer_id"]: r["name"] for r in conn.execute(
        "select customer_id, name from customer.customers where customer_id = any(%s)", (list(picks),)).fetchall()}
    return {"customers": [{"customer_id": c, "name": names[c], "hints": hints} for c, hints in picks.items()]}


@app.post("/v1/identity/sessions", response_model=models.Session, responses=OPEN_ERRORS, tags=["identity"],
          operation_id="signIn", summary="fake sign-in: a session token for a customer, no password asked")
def sign_in(body: models.SessionRequest, conn: DbConn):
    row = conn.execute(
        "select c.name, s.session_id from customer.customers c join identity.sessions s using (customer_id) "
        "where c.customer_id = %s order by s.created_at desc, s.session_id limit 1",
        (body.customer_id,),
    ).fetchone()
    if row is None:
        raise not_found(f"customer {body.customer_id} does not exist")
    token, expires = auth.issue_session_token(body.customer_id, row["session_id"])
    return {"session_token": token, "customer_id": body.customer_id, "name": row["name"], "expires_at": expires}


@app.get("/v1/me", response_model=models.MeResponse, responses=ERRORS, tags=["identity"], operation_id="getMe",
         summary="the token's customer")
def get_me(conn: DbConn, who: Caller = scope("profile:read:self")):
    row = conn.execute(
        "select c.customer_id, c.name, p.language from customer.customers c join customer.preferences p "
        "using (customer_id) where c.customer_id = %s",
        (who.customer_id,),
    ).fetchone()
    if row is None:
        raise not_found(f"customer {who.customer_id} does not exist")
    return evidenced(models.Me, models.MeResponse, row)


@app.get("/v1/orders/{order_id}", response_model=models.OrderResponse, responses=ERRORS, tags=["orders"],
         operation_id="getOrder", summary="one order with its lines and status history")
def get_order(order_id: OrderId, conn: DbConn, who: Caller = scope("orders:read:self")):
    order = require_order(conn, order_id, who)
    order["lines"] = fetch_lines(conn, order_id)
    order["status_history"] = conn.execute(
        "select seq, status, at from orders.order_status_history where order_id = %s order by seq", (order_id,)
    ).fetchall()
    return evidenced(models.Order, models.OrderResponse, order)


@app.get("/v1/me/orders", response_model=models.OrderHistoryResponse, responses=ERRORS, tags=["orders"],
         operation_id="listMyOrders", summary="the token's customer's orders, newest first")
def list_my_orders(
    conn: DbConn,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    who: Caller = scope("orders:read:self"),
):
    rows = conn.execute(
        "select order_id, store_id, status, total_paise, placed_at, delivered_at from orders.orders "
        "where customer_id = %s order by placed_at desc, order_id desc limit %s",
        (who.customer_id, limit),
    ).fetchall()
    data = {"customer_id": who.customer_id, "orders": rows}
    return evidenced(models.OrderHistory, models.OrderHistoryResponse, data)


# store & inventory module


@app.get("/v1/orders/{order_id}/scans", response_model=models.ScansResponse, responses=ERRORS,
         tags=["store_inventory"], operation_id="getOrderScans", summary="pick and pack scans for an order")
def get_order_scans(order_id: OrderId, conn: DbConn, who: Caller = scope("inventory:read")):
    order = require_order(conn, order_id, who)
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
    who: Caller = scope("inventory:read"),
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
    who: Caller = scope("inventory:read"),
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
def get_order_trip(order_id: OrderId, conn: DbConn, who: Caller = scope("delivery:read:self")):
    require_order(conn, order_id, who)
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
def get_proof_of_delivery(order_id: OrderId, conn: DbConn, who: Caller = scope("delivery:read:self")):
    require_order(conn, order_id, who)
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
def list_partner_notes(order_id: OrderId, conn: DbConn, who: Caller = scope("delivery:read:self")):
    require_order(conn, order_id, who)
    notes = conn.execute(
        "select note_id, trip_id, text, at from delivery.partner_notes where order_id = %s order by at, note_id",
        (order_id,),
    ).fetchall()
    return evidenced(models.PartnerNotes, models.PartnerNotesResponse, {"order_id": order_id, "notes": notes})


# payments and refunds modules


@app.get("/v1/orders/{order_id}/payments", response_model=models.PaymentsResponse, responses=ERRORS,
         tags=["payments"], operation_id="listOrderPayments", summary="payment attempts, captures, gateway events")
def list_order_payments(order_id: OrderId, conn: DbConn, who: Caller = scope("payments:read:self")):
    require_order(conn, order_id, who)
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
def list_order_refunds(order_id: OrderId, conn: DbConn, who: Caller = scope("payments:read:self")):
    require_order(conn, order_id, who)
    refunds = conn.execute(
        "select refund_id, amount_paise, destination, status, reason, case_id, created_at "
        "from refunds_wallet.refunds where order_id = %s order by created_at, refund_id",
        (order_id,),
    ).fetchall()
    return evidenced(models.Refunds, models.RefundsResponse, {"order_id": order_id, "refunds": refunds})


# promotions module

OfferCode = Annotated[str, Path(pattern=r"^[A-Z0-9]{3,20}$", examples=["SAVE10"])]
OFFER_COLUMNS = ("code, title, kind, value, max_discount_paise, min_cart_paise, first_order_only, valid_from, "
                 "valid_to, terms")


def fetch_offer(conn: psycopg.Connection, code: str) -> dict | None:
    return conn.execute(f"select {OFFER_COLUMNS} from promotions.offers where code = %s", (code,)).fetchone()


def is_first_order(conn: psycopg.Connection, order: dict) -> bool:
    """no earlier order, on an account younger than the order history the business holds."""
    row = conn.execute(
        "select c.persona = 'new' and not exists (select 1 from orders.orders o where o.customer_id = c.customer_id "
        "and (o.placed_at, o.order_id) < (%s, %s)) as first from customer.customers c where c.customer_id = %s",
        (order["placed_at"], order["order_id"], order["customer_id"]),
    ).fetchone()
    return bool(row and row["first"])


@app.get("/v1/promotions/offers/{code}", response_model=models.OfferResponse, responses=ERRORS,
         tags=["promotions"], operation_id="getOffer", summary="one offer with its terms")
def get_offer(code: OfferCode, conn: DbConn, who: Caller = scope("promotions:read:self")):
    offer = fetch_offer(conn, code)
    if offer is None:
        raise not_found(f"offer {code} does not exist")
    return evidenced(models.Offer, models.OfferResponse, offer)


@app.get("/v1/me/redemptions", response_model=models.RedemptionsResponse, responses=ERRORS, tags=["promotions"],
         operation_id="listMyRedemptions", summary="the token's customer's coupon attempts, newest first")
def list_my_redemptions(
    conn: DbConn,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    who: Caller = scope("promotions:read:self"),
):
    rows = conn.execute(
        "select redemption_id, code, order_id, status, discount_paise, reason, at from promotions.redemptions "
        "where customer_id = %s order by at desc, redemption_id desc limit %s",
        (who.customer_id, limit),
    ).fetchall()
    data = {"customer_id": who.customer_id, "redemptions": rows}
    return evidenced(models.Redemptions, models.RedemptionsResponse, data)


@app.get("/v1/orders/{order_id}/promotion-evaluation", response_model=models.PromotionEvaluationResponse,
         responses=ERRORS, tags=["promotions"], operation_id="evaluatePromotion",
         summary="the offer's rules run again on the order: what it should have given, and what it gave")
def evaluate_promotion(
    order_id: OrderId,
    code: Annotated[str, Query(pattern=r"^[A-Z0-9]{3,20}$", examples=["SAVE10"])],
    conn: DbConn,
    who: Caller = scope("promotions:read:self"),
):
    order = require_order(conn, order_id, who)
    offer = fetch_offer(conn, code)
    first = is_first_order(conn, order)
    due, failed = promotions.evaluate(offer, order["item_total_paise"], order["delivery_fee_paise"], first,
                                      order["placed_at"])
    checkout = conn.execute(
        "select status, discount_paise, reason from promotions.redemptions where order_id = %s and code = %s "
        "order by at desc, redemption_id desc limit 1",
        (order_id, code),
    ).fetchone()
    applied = order["discount_paise"] if order["coupon_code"] == code else 0
    data = {
        "order_id": order_id, "code": code, "rules": offer,
        "order": {"item_total_paise": order["item_total_paise"], "delivery_fee_paise": order["delivery_fee_paise"],
                  "placed_at": order["placed_at"], "first_order": first},
        "eligible": failed is None, "due_discount_paise": due, "failed_rule": failed, "checkout": checkout,
        "applied_discount_paise": applied, "difference_paise": due - applied,
    }
    return evidenced(models.PromotionEvaluation, models.PromotionEvaluationResponse, data)


# support module: cases; the full state machine comes with the full case manager, these are the moves the
# short paths make

CaseId = Annotated[str, Path(pattern=r"^CASE-[0-9A-F]{12}$", examples=["CASE-0123456789AB"])]
CASE_COLUMNS = "case_id, customer_id, order_id, issue_type, state, created_at, updated_at"
TRANSITIONS = {"OPEN": {"INVESTIGATING"}, "INVESTIGATING": {"RESOLVED", "AWAITING_HUMAN"}}
CASE_ERRORS = {**ERRORS, 409: PROBLEM}


@app.post("/v1/cases", response_model=models.OpenedCase, responses=ERRORS, tags=["support"],
          operation_id="openCase", summary="the open case for this customer, order and issue; opened if there is none")
def open_case(body: models.CaseOpenRequest, conn: DbConn, who: Caller = scope("cases:write:self")):
    if body.order_id is not None:
        require_order(conn, body.order_id, who)
    now = sim_now(conn)
    with conn.transaction():
        # one open at a time per customer, so a double submit cannot open two cases
        conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (who.customer_id,))
        row = conn.execute(
            f"select {CASE_COLUMNS} from support.cases where customer_id = %s and order_id is not distinct from %s "
            "and issue_type = %s and state not in ('RESOLVED', 'CLOSED')",
            (who.customer_id, body.order_id, body.issue_type),
        ).fetchone()
        if row is not None:
            return {**row, "opened": False}
        row = conn.execute(
            "insert into support.cases (case_id, customer_id, order_id, issue_type, state, created_at, updated_at) "
            f"values (%s, %s, %s, %s, 'OPEN', %s, %s) returning {CASE_COLUMNS}",
            ("CASE-" + uuid4().hex[:12].upper(), who.customer_id, body.order_id, body.issue_type, now, now),
        ).fetchone()
        return {**row, "opened": True}


@app.post("/v1/cases/{case_id}/state", response_model=models.Case, responses=CASE_ERRORS, tags=["support"],
          operation_id="moveCase", summary="move a case to its next state; the same state again changes nothing")
def move_case(case_id: CaseId, body: models.CaseStateRequest, conn: DbConn,
              who: Caller = scope("cases:write:self")):
    now = sim_now(conn)
    with conn.transaction():
        row = conn.execute(
            f"select {CASE_COLUMNS} from support.cases where case_id = %s and customer_id = %s for update",
            (case_id, who.customer_id),
        ).fetchone()
        if row is None:
            raise not_found(f"case {case_id} does not exist")
        if row["state"] == body.state:
            return row
        if body.state not in TRANSITIONS.get(row["state"], set()):
            raise HTTPException(status_code=409, detail=f"a case cannot move from {row['state']} to {body.state}")
        return conn.execute(
            f"update support.cases set state = %s, updated_at = %s where case_id = %s returning {CASE_COLUMNS}",
            (body.state, now, case_id),
        ).fetchone()


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
