"""l1 contract tests: every api answers as backend/openapi.json says, on the seeded test database."""

import json
import os
from datetime import datetime, timedelta, timezone

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from hivemind_backend import auth, db, promotions
from hivemind_backend.app import app
from hivemind_backend.cli import CONTRACT, contract_text
from hivemind_backend.evidence import evidence_id
from hivemind_datagen.generate import SMALL, iso

SPEC = json.loads(CONTRACT.read_text(encoding="utf-8"))


# the contract file itself (no database needed)


def test_contract_file_matches_the_code():
    assert CONTRACT.read_text(encoding="utf-8") == contract_text(), "run `uv run hm-db export-openapi`"


def test_contract_is_openapi_3_1():
    assert SPEC["openapi"].startswith("3.1")


def test_every_operation_has_a_case():
    operations = {(method.upper(), path) for path, item in SPEC["paths"].items() for method in item}
    assert operations == set(CASES)


def test_error_bodies_are_problem_json():
    for path, item in SPEC["paths"].items():
        for operation in item.values():
            for status, response in operation["responses"].items():
                if status.startswith("4"):
                    assert list(response["content"]) == ["application/problem+json"], (path, status)


# helpers


ALL_SCOPES = ["profile:read:self", "orders:read:self", "delivery:read:self", "payments:read:self",
              "inventory:read", "promotions:read:self", "cases:write:self", "policy:read", "history:read:self",
              "risk:read"]


def case_token(customer_id: str, scopes=ALL_SCOPES, minutes: float = 10, age_minutes: float = 0,
               key_name: str = "HM_CASE_TOKEN_SECRET") -> str:
    """a case token as hivemind mints it."""
    issued = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    claims = {"iss": auth.CASE_ISSUER, "aud": auth.CASE_AUDIENCE, "sub": customer_id, "sid": "SES-000001",
              "scopes": scopes, "iat": issued, "exp": issued + timedelta(minutes=minutes)}
    return jwt.encode(claims, os.environ[key_name], algorithm="HS256")


def bearer(customer_id: str, **kwargs) -> dict:
    return {"Authorization": f"Bearer {case_token(customer_id, **kwargs)}"}


def owner(tables, order_id: str) -> str:
    return next(o["customer_id"] for o in tables["orders.orders"] if o["order_id"] == order_id)


def schema_for(path: str, status: int, content_type: str, method: str = "get") -> Draft202012Validator:
    operation = SPEC["paths"][path][method]
    schema = operation["responses"][str(status)]["content"][content_type]["schema"]
    return Draft202012Validator({**schema, "components": SPEC["components"]})


def check_body(path: str, response, method: str = "get") -> dict:
    """status 200, body valid against the contract, evidence id right (on reads), money integers, times utc."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    errors = list(schema_for(path, 200, "application/json", method).iter_errors(body))
    assert not errors, errors[0].message
    if "evidence_id" in body:
        rest = {k: v for k, v in body.items() if k != "evidence_id"}
        assert body["evidence_id"] == evidence_id(rest)
    walk(body)
    return body


def walk(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            walk(v, k)
    elif isinstance(value, list):
        for v in value:
            walk(v, key)
    elif key.endswith("_paise"):
        assert isinstance(value, int), key
    elif key.endswith("_at") or key in ("at", "since", "until", "promised_by", "now", "valid_from", "valid_to"):
        assert value is None or (isinstance(value, str) and value.endswith("Z")), (key, value)


def check_problem(path: str, response, status: int, method: str = "get") -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    errors = list(schema_for(path, status, "application/problem+json", method).iter_errors(body))
    assert not errors, errors[0].message
    assert body["status"] == status and body["type"].startswith("/problems/")
    return body


def fault(tables, fault_type, variant=None) -> dict:
    return next(f for f in tables["truth.order_faults"]
                if f["fault_type"] == fault_type and (variant is None or f["variant"] == variant))


def window(tables, at):
    return {"since": iso(at - timedelta(hours=1)), "until": iso(at + timedelta(hours=1))}


# one case per operation: the url to call, whose token to send (None: no token), and a body for a post


def of_order(order_id_of, suffix=""):
    def case(t):
        order_id = order_id_of(t)
        return f"/v1/orders/{order_id}{suffix}", owner(t, order_id), None
    return case


def any_customer(t) -> str:
    return t["orders.orders"][0]["customer_id"]


def ring_member(t) -> str:
    return t["truth.fraud_rings"][0]["members"][0]


def rejected(t) -> dict:
    return next(r for r in t["promotions.redemptions"] if r["status"] == "rejected")


CASES = {
    ("GET", "/v1/clock"): lambda t: ("/v1/clock", any_customer(t), None),
    ("GET", "/v1/identity/sign-in-options"): lambda t: ("/v1/identity/sign-in-options", None, None),
    ("POST", "/v1/identity/sessions"): lambda t: ("/v1/identity/sessions", None, {"customer_id": any_customer(t)}),
    ("GET", "/v1/me"): lambda t: ("/v1/me", any_customer(t), None),
    ("GET", "/v1/orders/{order_id}"): of_order(lambda t: t["orders.orders"][0]["order_id"]),
    ("GET", "/v1/me/orders"): lambda t: ("/v1/me/orders", any_customer(t), None),
    ("GET", "/v1/orders/{order_id}/scans"): of_order(
        lambda t: fault(t, "missing_item", "store_miss")["order_id"], "/scans"),
    ("GET", "/v1/stores/{store_id}/incidents"): lambda t: (
        "/v1/stores/{}/incidents".format(t["store_inventory.incidents"][0]["store_id"]), any_customer(t), None),
    ("GET", "/v1/stores/{store_id}/stock-events"): lambda t: (
        "/v1/stores/{}/stock-events".format(t["store_inventory.stock_events"][0]["store_id"]), any_customer(t), None),
    ("GET", "/v1/orders/{order_id}/trip"): of_order(lambda t: fault(t, "delay", "rider_delay")["order_id"], "/trip"),
    ("GET", "/v1/orders/{order_id}/pod"): of_order(
        lambda t: fault(t, "missing_item", "rider_handover")["order_id"], "/pod"),
    ("GET", "/v1/orders/{order_id}/partner-notes"): of_order(
        lambda t: t["delivery.partner_notes"][0]["order_id"], "/partner-notes"),
    ("GET", "/v1/orders/{order_id}/payments"): of_order(
        lambda t: fault(t, "duplicate_capture")["order_id"], "/payments"),
    ("GET", "/v1/orders/{order_id}/refunds"): of_order(
        lambda t: t["refunds_wallet.refunds"][0]["order_id"], "/refunds"),
    ("GET", "/v1/promotions/offers/{code}"): lambda t: ("/v1/promotions/offers/SAVE10", any_customer(t), None),
    ("GET", "/v1/policy/sections"): lambda t: ("/v1/policy/sections", any_customer(t), None),
    ("GET", "/v1/me/redemptions"): lambda t: ("/v1/me/redemptions", rejected(t)["customer_id"], None),
    ("GET", "/v1/orders/{order_id}/promotion-evaluation"): of_order(
        lambda t: rejected(t)["order_id"], "/promotion-evaluation"),
    ("GET", "/v1/me/support-profile"): lambda t: ("/v1/me/support-profile", ring_member(t), None),
    ("GET", "/v1/me/risk-signals"): lambda t: ("/v1/me/risk-signals", ring_member(t), None),
    ("POST", "/v1/cases"): lambda t: ("/v1/cases", any_customer(t), {
        "issue_type": "where_is_my_order", "order_id": t["orders.orders"][0]["order_id"]}),
    ("POST", "/v1/cases/{case_id}/state"): lambda t: (None, any_customer(t), {"state": "INVESTIGATING"}),
}
PARAMS = {
    "/v1/stores/{store_id}/incidents": lambda t: window(t, t["store_inventory.incidents"][0]["at"]),
    "/v1/stores/{store_id}/stock-events": lambda t: {"sku": t["store_inventory.stock_events"][0]["sku"],
                                                     **window(t, t["store_inventory.stock_events"][0]["at"])},
    "/v1/orders/{order_id}/promotion-evaluation": lambda t: {"code": rejected(t)["code"]},
    "/v1/policy/sections": lambda t: {"issue_type": "item_missing_or_wrong"},
}


@pytest.fixture(scope="module")
def client(test_db):
    return TestClient(app)


# every operation against the contract


def call(client, operation, tables):
    """runs one operation's case and returns the response."""
    method, path = operation
    url, customer, body = CASES[operation](tables)
    headers = bearer(customer) if customer else {}
    if url is None:  # a case has to be opened before it can be moved
        opened = client.post("/v1/cases", headers=headers, json={"issue_type": "refund"}).json()
        url = f"/v1/cases/{opened['case_id']}/state"
    params = PARAMS.get(path, lambda t: {})(tables)
    return client.request(method, url, params=params, json=body, headers=headers)


@pytest.mark.db
@pytest.mark.parametrize("operation", sorted(CASES), ids=lambda op: f"{op[0]} {op[1]}")
def test_operation_matches_contract(operation, client, test_db):
    method, path = operation
    body = check_body(path, call(client, operation, test_db), method.lower())
    if "evidence_id" in body:
        assert call(client, operation, test_db).json()["evidence_id"] == body["evidence_id"]


PROTECTED = [op for op in sorted(CASES) if "/identity/" not in op[1]]
ORDER_PATHS = [p for (_, p) in CASES if p.startswith("/v1/orders/{order_id}")]


def params_for(path: str, tables) -> dict:
    return PARAMS.get(path, lambda t: {})(tables)


@pytest.mark.db
@pytest.mark.parametrize("path", ORDER_PATHS)
def test_unknown_order_is_404_problem(path, client, test_db):
    response = client.get(path.replace("{order_id}", "ORD-999999"), params=params_for(path, test_db),
                          headers=bearer(any_customer(test_db)))
    body = check_problem(path, response, 404)
    assert body["type"] == "/problems/not-found"


@pytest.mark.db
@pytest.mark.parametrize("path", ORDER_PATHS)
def test_malformed_order_id_is_422_problem(path, client, test_db):
    response = client.get(path.replace("{order_id}", "order-1"), params=params_for(path, test_db),
                          headers=bearer(any_customer(test_db)))
    check_problem(path, response, 422)


@pytest.mark.db
def test_unknown_store_is_404(client, test_db):
    path = "/v1/stores/{store_id}/incidents"
    response = client.get("/v1/stores/DS-BLR-99/incidents", params=window(test_db, SMALL.start),
                          headers=bearer(any_customer(test_db)))
    check_problem(path, response, 404)


@pytest.mark.db
@pytest.mark.parametrize("since,until", [
    ("2026-10-03T10:00:00Z", "2026-10-03T09:00:00Z"),   # until before since
    ("2026-10-01T00:00:00Z", "2026-10-09T00:00:00Z"),   # longer than 7 days
    ("2026-10-03T10:00:00", "2026-10-03T11:00:00"),     # no timezone
])
def test_bad_window_is_422(since, until, client, test_db):
    path = "/v1/stores/{store_id}/incidents"
    response = client.get("/v1/stores/DS-BLR-01/incidents", params={"since": since, "until": until},
                          headers=bearer(any_customer(test_db)))
    check_problem(path, response, 422)


# the apis return what the data holds


@pytest.mark.db
def test_order_matches_generated_rows(client, test_db):
    order = test_db["orders.orders"][5]
    body = client.get(f"/v1/orders/{order['order_id']}", headers=bearer(order["customer_id"])).json()
    lines = [l for l in test_db["orders.order_lines"] if l["order_id"] == order["order_id"]]
    assert body["total_paise"] == order["total_paise"]
    assert [l["sku"] for l in body["lines"]] == [l["sku"] for l in lines]
    assert body["placed_at"] == iso(order["placed_at"])
    assert body["status_history"][-1]["status"] == order["status"]


@pytest.mark.db
def test_history_is_newest_first(client, test_db):
    customer = test_db["orders.orders"][0]["customer_id"]
    orders = client.get("/v1/me/orders", params={"limit": 100}, headers=bearer(customer)).json()["orders"]
    expected = sorted((o for o in test_db["orders.orders"] if o["customer_id"] == customer),
                      key=lambda o: (o["placed_at"], o["order_id"]), reverse=True)
    assert [o["order_id"] for o in orders] == [o["order_id"] for o in expected]


@pytest.mark.db
def test_store_miss_shows_as_missing_scan(client, test_db):
    f = fault(test_db, "missing_item", "store_miss")
    body = client.get(f"/v1/orders/{f['order_id']}/scans", headers=bearer(f["customer_id"])).json()
    sku = f["facts"]["missing"][0]["sku"]
    assert [m for m in body["missing_scans"] if m["sku"] == sku and m["expected"] == "picked"]
    assert all(s["sku"] != sku for s in body["scans"])


@pytest.mark.db
def test_rider_handover_shows_in_pod(client, test_db):
    f = fault(test_db, "missing_item", "rider_handover")
    who = bearer(f["customer_id"])
    pod = client.get(f"/v1/orders/{f['order_id']}/pod", headers=who).json()
    packed_bags = {s["bag"] for s in client.get(f"/v1/orders/{f['order_id']}/scans", headers=who).json()["scans"]
                   if s["event"] == "packed"}
    assert pod["bags_handed"] == len(packed_bags) - 1


@pytest.mark.db
def test_payment_faults_show_in_payments(client, test_db):
    dup = fault(test_db, "duplicate_capture")
    intents = client.get(f"/v1/orders/{dup['order_id']}/payments",
                         headers=bearer(dup["customer_id"])).json()["intents"]
    assert len([c for i in intents for c in i["captures"]]) == 2
    failed = fault(test_db, "failed_debit_not_reversed")
    intents = client.get(f"/v1/orders/{failed['order_id']}/payments",
                         headers=bearer(failed["customer_id"])).json()["intents"]
    bad = next(i for i in intents if i["intent_id"] == failed["facts"]["intent_id"])
    assert bad["status"] == "failed" and [e["type"] for e in bad["gateway_events"]] == ["debit_success"]


@pytest.mark.db
def test_conflicting_incident_is_found_in_its_window(client, test_db):
    f = fault(test_db, "conflicting_records")
    order = next(o for o in test_db["orders.orders"] if o["order_id"] == f["order_id"])
    body = client.get(f"/v1/stores/{order['store_id']}/incidents", params=window(test_db, order["placed_at"]),
                      headers=bearer(f["customer_id"])).json()
    assert [i for i in body["incidents"] if i["sku"] == f["facts"]["sku"] and i["type"] == "out_of_stock"]


@pytest.mark.db
def test_live_order_has_no_pod_yet(client, test_db):
    live = next(o for o in test_db["orders.orders"] if o["status"] not in ("DELIVERED", "CANCELLED"))
    who = bearer(live["customer_id"])
    assert client.get(f"/v1/orders/{live['order_id']}", headers=who).json()["delivered_at"] is None
    check_problem("/v1/orders/{order_id}/pod", client.get(f"/v1/orders/{live['order_id']}/pod", headers=who), 404)


# who may read what


@pytest.mark.db
@pytest.mark.parametrize("role,table,allowed", [
    ("backend", "orders.orders", True),
    ("backend", "truth.order_faults", False),
    ("backend", "agent.findings", False),
    ("agent", "agent.findings", True),
    ("agent", "orders.orders", False),
    ("agent", "truth.order_faults", False),
    ("eval", "truth.order_faults", True),
    ("eval", "orders.orders", True),
])
def test_role_access(role, table, allowed, test_db):
    with db.connect(role) as conn:
        try:
            conn.execute(f"select count(*) from {table}").fetchone()
            ok = True
        except psycopg.errors.InsufficientPrivilege:
            ok = False
    assert ok is allowed


# who is calling: the case token


@pytest.mark.db
@pytest.mark.parametrize("operation", PROTECTED, ids=lambda op: f"{op[0]} {op[1]}")
def test_no_token_is_401(operation, client, test_db):
    method, path = operation
    url = path.replace("{order_id}", "ORD-100001").replace("{store_id}", "DS-BLR-01").replace(
        "{code}", "SAVE10").replace("{case_id}", "CASE-0123456789AB")
    body = check_problem(path, client.request(method, url, json={}), 401, method.lower())
    assert body["type"] == "/problems/unauthenticated"


@pytest.mark.db
@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer not-a-token"},
    {"Authorization": "Basic abc"},
], ids=["garbage", "not bearer"])
def test_bad_token_is_401(headers, client, test_db):
    check_problem("/v1/me", client.get("/v1/me", headers=headers), 401)


@pytest.mark.db
def test_expired_or_long_lived_or_wrongly_signed_token_is_401(client, test_db):
    customer = any_customer(test_db)
    for headers in (bearer(customer, minutes=5, age_minutes=6),      # expired a minute ago
                    bearer(customer, minutes=31),                     # lives longer than 30 minutes
                    bearer(customer, key_name="HM_SESSION_SECRET")):  # signed with the wrong key
        check_problem("/v1/me", client.get("/v1/me", headers=headers), 401)


@pytest.mark.db
def test_session_token_is_not_a_case_token(client, test_db):
    session = client.post("/v1/identity/sessions", json={"customer_id": any_customer(test_db)}).json()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    check_problem("/v1/me", client.get("/v1/me", headers=headers), 401)


@pytest.mark.db
def test_missing_scope_is_403(client, test_db):
    order = test_db["orders.orders"][0]
    headers = bearer(order["customer_id"], scopes=["delivery:read:self"])
    body = check_problem("/v1/orders/{order_id}", client.get(f"/v1/orders/{order['order_id']}", headers=headers), 403)
    assert body["type"] == "/problems/forbidden"


@pytest.mark.db
@pytest.mark.parametrize("path", ORDER_PATHS)
def test_another_customers_order_looks_like_no_order(path, client, test_db):
    order = test_db["orders.orders"][0]
    other = next(o["customer_id"] for o in test_db["orders.orders"] if o["customer_id"] != order["customer_id"])
    response = client.get(path.replace("{order_id}", order["order_id"]), params=params_for(path, test_db),
                          headers=bearer(other))
    body = check_problem(path, response, 404)
    assert body["detail"] == f"order {order['order_id']} does not exist"


@pytest.mark.db
def test_my_orders_are_the_tokens_customers(client, test_db):
    customer = any_customer(test_db)
    body = client.get("/v1/me/orders", params={"limit": 100}, headers=bearer(customer)).json()
    mine = {o["order_id"] for o in test_db["orders.orders"] if o["customer_id"] == customer}
    assert body["customer_id"] == customer and {o["order_id"] for o in body["orders"]} == mine


# the fake sign-in


@pytest.mark.db
def test_sign_in_gives_a_session_token_hivemind_can_check(client, test_db):
    customer = any_customer(test_db)
    session = client.post("/v1/identity/sessions", json={"customer_id": customer}).json()
    claims = jwt.decode(session["session_token"], os.environ["HM_SESSION_SECRET"], algorithms=["HS256"],
                        audience=auth.SESSION_AUDIENCE, issuer=auth.SESSION_ISSUER)
    assert claims["sub"] == customer
    assert claims["sid"] in {r["session_id"] for r in test_db["identity.sessions"] if r["customer_id"] == customer}
    name = next(c["name"] for c in test_db["customer.customers"] if c["customer_id"] == customer)
    assert session["name"] == name


@pytest.mark.db
def test_sign_in_unknown_customer_is_404(client, test_db):
    response = client.post("/v1/identity/sessions", json={"customer_id": "CUST-99999"})
    check_problem("/v1/identity/sessions", response, 404, "post")


@pytest.mark.db
def test_sign_in_options_name_real_customers_with_a_live_order(client, test_db):
    customers = client.get("/v1/identity/sign-in-options").json()["customers"]
    live = {o["customer_id"] for o in test_db["orders.orders"] if o["status"] not in ("DELIVERED", "CANCELLED")}
    assert customers and len({c["customer_id"] for c in customers}) == len(customers)
    with_live = [c for c in customers if "an order on its way" in c["hints"]]
    assert with_live and all(c["customer_id"] in live for c in with_live)


@pytest.mark.db
def test_clock_is_where_the_data_ends(client, test_db):
    body = client.get("/v1/clock", headers=bearer(any_customer(test_db))).json()
    assert body["now"] == iso(SMALL.as_of)


# promotions: the engine against the data


def evaluation(client, tables, order_id: str, code: str) -> dict:
    response = client.get(f"/v1/orders/{order_id}/promotion-evaluation", params={"code": code},
                          headers=bearer(owner(tables, order_id)))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.db
def test_engine_agrees_with_every_untouched_checkout(client, test_db):
    """for every coupon attempt no fault touched, running the rules again gives what checkout gave."""
    faulty = {f["order_id"] for f in test_db["truth.order_faults"] if f["fault_type"] == "coupon_misapplied"}
    checked = 0
    for r in test_db["promotions.redemptions"]:
        if r["order_id"] in faulty:
            continue
        body = evaluation(client, test_db, r["order_id"], r["code"])
        assert body["eligible"] is (r["status"] == "applied"), r
        assert body["due_discount_paise"] == r["discount_paise"], r
        assert body["difference_paise"] == 0, r
        assert body["checkout"] == {"status": r["status"], "discount_paise": r["discount_paise"],
                                    "reason": r["reason"]}
        checked += 1
    assert checked > 20


@pytest.mark.db
def test_engine_names_the_rule_that_failed(client, test_db):
    seen = set()
    words = {"min_cart": "cart below minimum", "first_order_only": "only for the first order",
             "expired": "expired", "delivery_already_free": "already free"}
    faulty = {f["order_id"] for f in test_db["truth.order_faults"] if f["fault_type"] == "coupon_misapplied"}
    for r in test_db["promotions.redemptions"]:
        if r["status"] != "rejected" or r["order_id"] in faulty:
            continue
        body = evaluation(client, test_db, r["order_id"], r["code"])
        assert words[body["failed_rule"]] in r["reason"], (body["failed_rule"], r["reason"])
        seen.add(body["failed_rule"])
    assert {"min_cart", "expired"} <= seen


@pytest.mark.db
def test_engine_finds_the_coupon_faults(client, test_db):
    wrong = fault(test_db, "coupon_misapplied", "wrong_amount")
    body = evaluation(client, test_db, wrong["order_id"], wrong["facts"]["code"])
    assert body["eligible"] and body["due_discount_paise"] == wrong["facts"]["due_paise"]
    assert body["applied_discount_paise"] == wrong["facts"]["applied_paise"]
    assert body["difference_paise"] == wrong["expected"]["owed_paise"]

    rejected_wrongly = fault(test_db, "coupon_misapplied", "wrongly_rejected")
    body = evaluation(client, test_db, rejected_wrongly["order_id"], rejected_wrongly["facts"]["code"])
    assert body["eligible"] and body["checkout"]["status"] == "rejected" and body["applied_discount_paise"] == 0
    assert body["difference_paise"] == rejected_wrongly["expected"]["owed_paise"]


@pytest.mark.db
def test_unknown_code_is_not_eligible(client, test_db):
    body = evaluation(client, test_db, test_db["orders.orders"][0]["order_id"], "NOSUCH99")
    assert body["rules"] is None and body["failed_rule"] == "unknown_code" and not body["eligible"]
    response = client.get("/v1/promotions/offers/NOSUCH99", headers=bearer(any_customer(test_db)))
    check_problem("/v1/promotions/offers/{code}", response, 404)


def test_engine_rules():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    offer = {"kind": "percent", "value": 10, "max_discount_paise": 10000, "min_cart_paise": 49900,
             "first_order_only": False, "valid_from": start, "valid_to": start + timedelta(days=30)}
    inside = start + timedelta(days=1)
    assert promotions.evaluate(offer, 60000, 0, False, inside) == (6000, None)
    assert promotions.evaluate(offer, 250000, 0, False, inside) == (10000, None)          # capped
    assert promotions.evaluate(offer, 49899, 0, False, inside) == (0, "min_cart")
    assert promotions.evaluate(offer, 60000, 0, False, start - timedelta(seconds=1)) == (0, "not_started")
    assert promotions.evaluate(offer, 60000, 0, False, start + timedelta(days=30)) == (0, "expired")
    assert promotions.evaluate({**offer, "first_order_only": True}, 60000, 0, False, inside) == (0, "first_order_only")
    flat = {**offer, "kind": "flat", "value": 5000, "min_cart_paise": 0}
    assert promotions.evaluate(flat, 3000, 0, False, inside) == (3000, None)               # never more than the cart
    free = {**offer, "kind": "free_delivery", "value": 0, "min_cart_paise": 0}
    assert promotions.evaluate(free, 60000, 2500, False, inside) == (2500, None)
    assert promotions.evaluate(free, 60000, 0, False, inside) == (0, "delivery_already_free")
    assert promotions.evaluate(None, 60000, 0, False, inside) == (0, "unknown_code")


# support: cases


@pytest.mark.db
def test_a_case_opens_once_and_moves_only_along_its_path(client, test_db):
    order = test_db["orders.orders"][3]
    who = bearer(order["customer_id"])
    ask = {"issue_type": "offers_coupons", "order_id": order["order_id"]}
    first = client.post("/v1/cases", headers=who, json=ask).json()
    assert first["opened"] is True and first["state"] == "OPEN" and first["customer_id"] == order["customer_id"]
    assert first["created_at"] == iso(SMALL.as_of)  # the business's clock, not the wall clock
    again = client.post("/v1/cases", headers=who, json=ask).json()
    assert again["opened"] is False and again["case_id"] == first["case_id"]

    move = f"/v1/cases/{first['case_id']}/state"
    path = "/v1/cases/{case_id}/state"
    body = check_problem(path, client.post(move, headers=who, json={"state": "RESOLVED"}), 409, "post")
    assert body["type"] == "/problems/conflict"
    assert client.post(move, headers=who, json={"state": "INVESTIGATING"}).json()["state"] == "INVESTIGATING"
    assert client.post(move, headers=who, json={"state": "INVESTIGATING"}).json()["state"] == "INVESTIGATING"
    assert client.post(move, headers=who, json={"state": "RESOLVED"}).json()["state"] == "RESOLVED"
    check_problem(path, client.post(move, headers=who, json={"state": "INVESTIGATING"}), 409, "post")

    reopened = client.post("/v1/cases", headers=who, json=ask).json()  # the old one is resolved: a new case
    assert reopened["opened"] is True and reopened["case_id"] != first["case_id"]


@pytest.mark.db
def test_a_case_belongs_to_its_customer(client, test_db):
    order = test_db["orders.orders"][4]
    other = next(o["customer_id"] for o in test_db["orders.orders"] if o["customer_id"] != order["customer_id"])
    ask = {"issue_type": "refund", "order_id": order["order_id"]}
    check_problem("/v1/cases", client.post("/v1/cases", headers=bearer(other), json=ask), 404, "post")
    case = client.post("/v1/cases", headers=bearer(order["customer_id"]), json=ask).json()
    response = client.post(f"/v1/cases/{case['case_id']}/state", headers=bearer(other), json={"state": "INVESTIGATING"})
    check_problem("/v1/cases/{case_id}/state", response, 404, "post")


@pytest.mark.db
def test_a_case_without_an_order_also_opens_once(client, test_db):
    who = bearer(test_db["orders.orders"][6]["customer_id"])
    first = client.post("/v1/cases", headers=who, json={"issue_type": "payment_failed"}).json()
    again = client.post("/v1/cases", headers=who, json={"issue_type": "payment_failed"}).json()
    assert first["order_id"] is None and first["opened"] and not again["opened"]
    assert again["case_id"] == first["case_id"]


# support: the case state machine and its history


def case_events(case_id: str) -> list[dict]:
    from psycopg.rows import dict_row

    with db.connect("admin", row_factory=dict_row) as conn:
        return conn.execute("select seq, type, from_state, to_state, reason from support.case_events "
                            "where case_id = %s order by seq", (case_id,)).fetchall()


def force_state(case_id: str, state: str) -> None:
    """sets a case's state directly, as test setup; the api never does this."""
    with db.connect("admin") as conn:
        conn.execute("update support.cases set state = %s where case_id = %s", (state, case_id))


def test_the_state_machine_names_the_contract_states():
    from typing import get_args

    from hivemind_backend import cases, models

    assert set(cases.STATES) == set(get_args(models.CaseState)) == set(cases.TRANSITIONS)
    assert all(to in cases.STATES for moves in cases.TRANSITIONS.values() for to in moves)
    assert cases.TRANSITIONS["CLOSED"] == set()


def test_every_state_but_closed_can_be_left_and_reached():
    from hivemind_backend import cases

    reachable, todo = {"OPEN"}, ["OPEN"]
    while todo:
        for to in cases.TRANSITIONS[todo.pop()]:
            if to not in reachable:
                reachable.add(to)
                todo.append(to)
    assert reachable == set(cases.STATES)
    # from anywhere a case can still end: resolved or closed
    for state in cases.STATES:
        seen, todo = {state}, [state]
        while todo:
            for to in cases.TRANSITIONS[todo.pop()]:
                if to not in seen:
                    seen.add(to)
                    todo.append(to)
        assert "CLOSED" in seen, state


@pytest.mark.db
def test_every_move_between_two_states_is_allowed_or_refused_by_the_table(client, test_db):
    """all 121 pairs: a legal move changes the state and is kept in the history; any other is refused (409) and
    changes nothing; the same state again changes nothing either."""
    from hivemind_backend import cases

    order = test_db["orders.orders"][8]
    who = bearer(order["customer_id"])
    case = client.post("/v1/cases", headers=who, json={"issue_type": "quality", "order_id": order["order_id"]}).json()
    path = "/v1/cases/{case_id}/state"
    move = f"/v1/cases/{case['case_id']}/state"
    for start in cases.STATES:
        for to in cases.STATES:
            force_state(case["case_id"], start)
            before = len(case_events(case["case_id"]))
            response = client.post(move, headers=who, json={"state": to, "reason": "test_move"})
            if start == to:
                assert check_body(path, response, "post")["state"] == start
                assert len(case_events(case["case_id"])) == before
            elif cases.can_move(start, to):
                assert check_body(path, response, "post")["state"] == to, (start, to)
                last = case_events(case["case_id"])[-1]
                assert (last["type"], last["from_state"], last["to_state"], last["reason"]) == (
                    "moved", start, to, "test_move")
            else:
                check_problem(path, response, 409, "post")
                assert len(case_events(case["case_id"])) == before, (start, to)
    force_state(case["case_id"], "CLOSED")


@pytest.mark.db
def test_a_case_keeps_its_history(client, test_db):
    order = test_db["orders.orders"][9]
    who = bearer(order["customer_id"])
    ask = {"issue_type": "refund", "order_id": order["order_id"]}
    case = client.post("/v1/cases", headers=who, json=ask).json()
    client.post("/v1/cases", headers=who, json=ask)  # a second message about the same problem
    move = f"/v1/cases/{case['case_id']}/state"
    client.post(move, headers=who, json={"state": "INVESTIGATING"})
    client.post(move, headers=who, json={"state": "INVESTIGATING"})  # a repeat: no new line
    client.post(move, headers=who, json={"state": "AWAITING_HUMAN", "reason": "no_path_yet"})
    client.post(move, headers=who, json={"state": "OPEN"})  # refused: no line
    assert [(e["seq"], e["type"], e["from_state"], e["to_state"], e["reason"]) for e in case_events(case["case_id"])] == [
        (1, "opened", None, "OPEN", None),
        (2, "attached", "OPEN", "OPEN", None),
        (3, "moved", "OPEN", "INVESTIGATING", None),
        (4, "moved", "INVESTIGATING", "AWAITING_HUMAN", "no_path_yet"),
    ]
    bad = client.post(move, headers=who, json={"state": "RESOLVED", "reason": "Not A Code"})
    check_problem("/v1/cases/{case_id}/state", bad, 422, "post")


@pytest.mark.db
def test_a_resolved_case_reopens_only_when_no_other_case_about_it_is_open(client, test_db):
    order = test_db["orders.orders"][10]
    who = bearer(order["customer_id"])
    ask = {"issue_type": "cancellation", "order_id": order["order_id"]}
    first = client.post("/v1/cases", headers=who, json=ask).json()
    move = f"/v1/cases/{first['case_id']}/state"
    for state in ("INVESTIGATING", "RESOLVED"):
        client.post(move, headers=who, json={"state": state})
    second = client.post("/v1/cases", headers=who, json=ask).json()
    assert second["opened"] and second["case_id"] != first["case_id"]
    body = check_problem("/v1/cases/{case_id}/state", client.post(move, headers=who, json={"state": "REOPENED"}),
                         409, "post")
    assert second["case_id"] in body["detail"]
    force_state(second["case_id"], "CLOSED")
    assert client.post(move, headers=who, json={"state": "REOPENED"}).json()["state"] == "REOPENED"


# support profile and risk signals (phase 2.4)


def claim_cases_of(customer_id):
    """the customer's closed claim cases as the database holds them: the generated ones, and any a test resolved."""
    with db.connect("admin", dbname=os.environ["HM_DB_NAME"], row_factory=psycopg.rows.dict_row) as conn:
        return conn.execute("select * from support.cases where customer_id = %s and issue_type = any(%s) "
                            "and state in ('RESOLVED', 'CLOSED')",
                            (customer_id, ["item_missing_or_wrong", "quality", "refund"])).fetchall()


@pytest.mark.db
def test_support_profile_counts_the_generated_claims(client, test_db):
    for ring in test_db["truth.fraud_rings"]:
        for member in ring["members"]:
            body = check_body("/v1/me/support-profile", client.get("/v1/me/support-profile", headers=bearer(member)))
            mine = claim_cases_of(member)
            assert body["claims"] == len(mine) >= 2
            assert sum(body["claims_by_type"].values()) == body["claims"]
            assert (body["refunded_by_rules"] + body["turned_down"] + body["escalated"]) == body["claims"]
            assert body["escalations_approved"] + body["escalations_rejected"] == body["escalated"]
            refunds = [r for r in test_db["refunds_wallet.refunds"] if r["case_id"] in {c["case_id"] for c in mine}]
            assert body["refunded_paise"] == sum(r["amount_paise"] for r in refunds)
            delivered = [o for o in test_db["orders.orders"] if o["customer_id"] == member
                         and o["status"] == "DELIVERED"]
            assert body["delivered_orders"] == len(delivered)
            assert len(body["last_resolutions"]) == min(3, len(mine))
            stamps = [r["closed_at"] for r in body["last_resolutions"]]
            assert stamps == sorted(stamps, reverse=True)


@pytest.mark.db
def test_a_customer_with_no_claims_has_an_empty_profile(client, test_db):
    claimed = {c["customer_id"] for c in test_db["support.cases"]}
    customer = next(o["customer_id"] for o in test_db["orders.orders"] if o["customer_id"] not in claimed)
    body = check_body("/v1/me/support-profile", client.get("/v1/me/support-profile", headers=bearer(customer)))
    assert body["claims"] == body["refunded_paise"] == body["escalated"] == 0
    assert body["last_resolutions"] == [] and body["orders"] >= 1


@pytest.mark.db
def test_risk_signals_find_each_ring_and_only_counts(client, test_db):
    for ring in test_db["truth.fraud_rings"]:
        for member in ring["members"]:
            response = client.get("/v1/me/risk-signals", headers=bearer(member))
            body = check_body("/v1/me/risk-signals", response)
            links = {row["link_type"]: row for row in body["links"]}
            assert set(links) == {"device", "address", "upi"}
            shared = links[ring["link_type"]]
            assert shared["accounts"] == len(ring["members"]) - 1
            assert shared["accounts_with_claims"] == len(ring["members"]) - 1 and shared["claims"] >= 2
            others = set(ring["members"]) - {member}
            assert not any(other in response.text for other in others)  # counts only, never another id


@pytest.mark.db
def test_an_account_outside_the_rings_shares_nothing(client, test_db):
    ring = {m for r in test_db["truth.fraud_rings"] for m in r["members"]}
    customer = next(c["customer_id"] for c in test_db["customer.customers"] if c["customer_id"] not in ring)
    body = check_body("/v1/me/risk-signals", client.get("/v1/me/risk-signals", headers=bearer(customer)))
    assert all(row["accounts"] == 0 for row in body["links"])


@pytest.mark.db
@pytest.mark.parametrize("path,needed", [("/v1/me/support-profile", "history:read:self"),
                                         ("/v1/me/risk-signals", "risk:read")])
def test_profile_and_signals_need_their_scope(path, needed, client, test_db):
    member = ring_member(test_db)
    others = [s for s in ALL_SCOPES if s != needed]
    check_problem(path, client.get(path, headers=bearer(member, scopes=others)), 403)
    check_problem(path, client.get(path, params={"window_days": 0}, headers=bearer(member)), 422)
