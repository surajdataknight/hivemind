"""l1 contract tests: every read api answers as backend/openapi.json says, on the seeded test database."""

import json
from datetime import timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from hivemind_backend import db
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


def schema_for(path: str, status: int, content_type: str) -> Draft202012Validator:
    operation = SPEC["paths"][path]["get"]
    schema = operation["responses"][str(status)]["content"][content_type]["schema"]
    return Draft202012Validator({**schema, "components": SPEC["components"]})


def check_body(path: str, response) -> dict:
    """status 200, body valid against the contract, evidence id right, money integers, times utc."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    errors = list(schema_for(path, 200, "application/json").iter_errors(body))
    assert not errors, errors[0].message
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
    elif key.endswith("_at") or key in ("at", "since", "until", "promised_by"):
        assert value is None or (isinstance(value, str) and value.endswith("Z")), (key, value)


def check_problem(path: str, response, status: int) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    errors = list(schema_for(path, status, "application/problem+json").iter_errors(body))
    assert not errors, errors[0].message
    assert body["status"] == status and body["type"].startswith("/problems/")
    return body


def fault(tables, fault_type, variant=None) -> dict:
    return next(f for f in tables["truth.order_faults"]
                if f["fault_type"] == fault_type and (variant is None or f["variant"] == variant))


def window(tables, at):
    return {"since": iso(at - timedelta(hours=1)), "until": iso(at + timedelta(hours=1))}


# one case per operation: the url to call, built from the generated data

CASES = {
    ("GET", "/v1/orders/{order_id}"): lambda t: f"/v1/orders/{t['orders.orders'][0]['order_id']}",
    ("GET", "/v1/customers/{customer_id}/orders"): lambda t: f"/v1/customers/{t['orders.orders'][0]['customer_id']}/orders",
    ("GET", "/v1/orders/{order_id}/scans"): lambda t: f"/v1/orders/{fault(t, 'missing_item', 'store_miss')['order_id']}/scans",
    ("GET", "/v1/stores/{store_id}/incidents"): lambda t: "/v1/stores/{}/incidents".format(
        t["store_inventory.incidents"][0]["store_id"]),
    ("GET", "/v1/stores/{store_id}/stock-events"): lambda t: "/v1/stores/{}/stock-events".format(
        t["store_inventory.stock_events"][0]["store_id"]),
    ("GET", "/v1/orders/{order_id}/trip"): lambda t: f"/v1/orders/{fault(t, 'delay', 'rider_delay')['order_id']}/trip",
    ("GET", "/v1/orders/{order_id}/pod"): lambda t: f"/v1/orders/{fault(t, 'missing_item', 'rider_handover')['order_id']}/pod",
    ("GET", "/v1/orders/{order_id}/partner-notes"): lambda t: "/v1/orders/{}/partner-notes".format(
        t["delivery.partner_notes"][0]["order_id"]),
    ("GET", "/v1/orders/{order_id}/payments"): lambda t: f"/v1/orders/{fault(t, 'duplicate_capture')['order_id']}/payments",
    ("GET", "/v1/orders/{order_id}/refunds"): lambda t: f"/v1/orders/{t['refunds_wallet.refunds'][0]['order_id']}/refunds",
}
PARAMS = {
    "/v1/stores/{store_id}/incidents": lambda t: window(t, t["store_inventory.incidents"][0]["at"]),
    "/v1/stores/{store_id}/stock-events": lambda t: {"sku": t["store_inventory.stock_events"][0]["sku"],
                                                     **window(t, t["store_inventory.stock_events"][0]["at"])},
}


@pytest.fixture(scope="module")
def client(test_db):
    return TestClient(app)


# every operation against the contract


@pytest.mark.db
@pytest.mark.parametrize("operation", sorted(CASES), ids=lambda op: op[1])
def test_operation_matches_contract(operation, client, test_db):
    path = operation[1]
    url = CASES[operation](test_db)
    params = PARAMS.get(path, lambda t: {})(test_db)
    body = check_body(path, client.get(url, params=params))
    again = client.get(url, params=params).json()
    assert again["evidence_id"] == body["evidence_id"]


ORDER_PATHS = [p for (_, p) in CASES if p.startswith("/v1/orders/{order_id}")]


@pytest.mark.db
@pytest.mark.parametrize("path", ORDER_PATHS)
def test_unknown_order_is_404_problem(path, client, test_db):
    body = check_problem(path, client.get(path.replace("{order_id}", "ORD-999999")), 404)
    assert body["type"] == "/problems/not-found"


@pytest.mark.db
@pytest.mark.parametrize("path", ORDER_PATHS)
def test_malformed_order_id_is_422_problem(path, client, test_db):
    check_problem(path, client.get(path.replace("{order_id}", "order-1")), 422)


@pytest.mark.db
def test_unknown_store_is_404(client, test_db):
    path = "/v1/stores/{store_id}/incidents"
    check_problem(path, client.get("/v1/stores/DS-BLR-99/incidents", params=window(test_db, SMALL.start)), 404)


@pytest.mark.db
@pytest.mark.parametrize("since,until", [
    ("2026-10-03T10:00:00Z", "2026-10-03T09:00:00Z"),   # until before since
    ("2026-10-01T00:00:00Z", "2026-10-09T00:00:00Z"),   # longer than 7 days
    ("2026-10-03T10:00:00", "2026-10-03T11:00:00"),     # no timezone
])
def test_bad_window_is_422(since, until, client, test_db):
    path = "/v1/stores/{store_id}/incidents"
    check_problem(path, client.get("/v1/stores/DS-BLR-01/incidents", params={"since": since, "until": until}), 422)


# the apis return what the data holds


@pytest.mark.db
def test_order_matches_generated_rows(client, test_db):
    order = test_db["orders.orders"][5]
    body = client.get(f"/v1/orders/{order['order_id']}").json()
    lines = [l for l in test_db["orders.order_lines"] if l["order_id"] == order["order_id"]]
    assert body["total_paise"] == order["total_paise"]
    assert [l["sku"] for l in body["lines"]] == [l["sku"] for l in lines]
    assert body["placed_at"] == iso(order["placed_at"])
    assert body["status_history"][-1]["status"] == order["status"]


@pytest.mark.db
def test_history_is_newest_first(client, test_db):
    customer = test_db["orders.orders"][0]["customer_id"]
    orders = client.get(f"/v1/customers/{customer}/orders", params={"limit": 100}).json()["orders"]
    expected = sorted((o for o in test_db["orders.orders"] if o["customer_id"] == customer),
                      key=lambda o: (o["placed_at"], o["order_id"]), reverse=True)
    assert [o["order_id"] for o in orders] == [o["order_id"] for o in expected]


@pytest.mark.db
def test_store_miss_shows_as_missing_scan(client, test_db):
    f = fault(test_db, "missing_item", "store_miss")
    body = client.get(f"/v1/orders/{f['order_id']}/scans").json()
    sku = f["facts"]["missing"][0]["sku"]
    assert [m for m in body["missing_scans"] if m["sku"] == sku and m["expected"] == "picked"]
    assert all(s["sku"] != sku for s in body["scans"])


@pytest.mark.db
def test_rider_handover_shows_in_pod(client, test_db):
    f = fault(test_db, "missing_item", "rider_handover")
    pod = client.get(f"/v1/orders/{f['order_id']}/pod").json()
    packed_bags = {s["bag"] for s in client.get(f"/v1/orders/{f['order_id']}/scans").json()["scans"]
                   if s["event"] == "packed"}
    assert pod["bags_handed"] == len(packed_bags) - 1


@pytest.mark.db
def test_payment_faults_show_in_payments(client, test_db):
    dup = fault(test_db, "duplicate_capture")
    intents = client.get(f"/v1/orders/{dup['order_id']}/payments").json()["intents"]
    assert len([c for i in intents for c in i["captures"]]) == 2
    failed = fault(test_db, "failed_debit_not_reversed")
    intents = client.get(f"/v1/orders/{failed['order_id']}/payments").json()["intents"]
    bad = next(i for i in intents if i["intent_id"] == failed["facts"]["intent_id"])
    assert bad["status"] == "failed" and [e["type"] for e in bad["gateway_events"]] == ["debit_success"]


@pytest.mark.db
def test_conflicting_incident_is_found_in_its_window(client, test_db):
    f = fault(test_db, "conflicting_records")
    order = next(o for o in test_db["orders.orders"] if o["order_id"] == f["order_id"])
    body = client.get(f"/v1/stores/{order['store_id']}/incidents", params=window(test_db, order["placed_at"])).json()
    assert [i for i in body["incidents"] if i["sku"] == f["facts"]["sku"] and i["type"] == "out_of_stock"]


@pytest.mark.db
def test_live_order_has_no_pod_yet(client, test_db):
    live = next(o for o in test_db["orders.orders"] if o["status"] not in ("DELIVERED", "CANCELLED"))
    assert client.get(f"/v1/orders/{live['order_id']}").json()["delivered_at"] is None
    check_problem("/v1/orders/{order_id}/pod", client.get(f"/v1/orders/{live['order_id']}/pod"), 404)


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
