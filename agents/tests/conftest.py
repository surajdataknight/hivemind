"""test setup for hivemind itself: the real backend in-process, the real test database, and a scripted model.

`--live-llm` swaps the scripted model for the real ones in models.toml (needs GEMINI_API_KEY, sends the made-up
chat text to google) and also sends the traces to phoenix.
"""

import time
from collections import defaultdict
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hivemind_agents import settings, tracing
from hivemind_agents.capabilities.identity import IdentityVerifier
from hivemind_agents.chat import ChatService, Deps
from hivemind_agents.llm import ModelRouter
from hivemind_agents.store import ChatStore
from hivemind_backend import db
from hivemind_backend.app import app as backend_app
from scripted import ScriptedModel

LIVE_STATUSES = ("PLACED", "CONFIRMED", "PACKING", "PACKED", "OUT_FOR_DELIVERY")


@pytest.fixture(scope="session")
def backend_http(test_db):
    return TestClient(backend_app)


@pytest.fixture(scope="session")
def span_exporter():
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer(span_exporter, live):
    provider = tracing.make_provider(span_exporter, SimpleSpanProcessor)
    endpoint = settings.traces_endpoint()
    if live and endpoint:
        provider.add_span_processor(SimpleSpanProcessor(tracing.otlp_exporter(endpoint)))
    return tracing.Tracer(provider)


@pytest.fixture
def spans(span_exporter):
    """the spans of this test only."""
    span_exporter.clear()
    return span_exporter


@pytest.fixture
def model(live):
    return ScriptedModel()


@pytest.fixture
def clean_chat_state(test_db, test_db_name):
    """no cases or chats left over from another test."""
    with db.connect("admin", dbname=test_db_name) as conn:
        conn.execute("truncate support.cases cascade")
        conn.execute("truncate agent.pii_vault, agent.messages, agent.conversations")


@pytest.fixture
def service(request, clean_chat_state, backend_http, tracer, model, live, live_pause, test_db_name):
    """the chat service on the scripted model. with --live-llm, tests marked `live_capable` get the real models;
    every other test keeps the scripted one, since it scripts what the model says."""
    use_real_models = live and request.node.get_closest_marker("live_capable") is not None
    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"),
                                settings.signing_key("HM_CASE_TOKEN_SECRET"))
    providers = None if use_real_models else {"google": lambda: model}
    deps = Deps(http=backend_http, llm=ModelRouter(settings.load_models(), tracer, providers),
                store=ChatStore(settings.agent_conninfo()), tracer=tracer, verifier=verifier)
    yield ChatService(deps)
    if use_real_models:
        time.sleep(live_pause)


class Chat:
    """one signed-in customer's chat, driven the way the screen drives it."""

    def __init__(self, service: ChatService, backend_http, customer_id: str):
        self.service = service
        self.customer_id = customer_id
        session = backend_http.post("/v1/identity/sessions", json={"customer_id": customer_id})
        assert session.status_code == 200, session.text
        self.token = session.json()["session_token"]
        opened = service.start_conversation(self.token)
        self.conversation_id = opened["conversation_id"]
        self.welcome = opened["messages"][0]
        self.events: list[dict] = []

    def say(self, text: str, intent=None, client_msg_id: str | None = None) -> dict:
        """sends one message; returns the `done` event. the events of this turn are in `self.events`."""
        self.events = []
        return self.service.handle_message(self.token, self.conversation_id, text,
                                           client_msg_id or uuid4().hex, intent, self.events.append)

    def press(self, chip: dict) -> dict:
        from hivemind_agents.contracts import IntentHint

        return self.say(chip["text"], IntentHint(**chip["intent"]))

    def shown(self) -> list[dict]:
        return self.service.conversation(self.token, self.conversation_id)["messages"]

    def stored(self) -> list[dict]:
        return self.service.deps.store.messages(self.conversation_id)

    def streamed_text(self) -> str:
        """what the screen shows from the token and replace events alone."""
        text = ""
        for event in self.events:
            if event["type"] == "token":
                text += event["text"]
            elif event["type"] == "replace":
                text = event["text"]
        return text.strip()


@pytest.fixture
def chat_as(service, backend_http):
    return lambda customer_id: Chat(service, backend_http, customer_id)


class Data:
    """lookups over the generated rows, to pick a customer or an order that fits a script."""

    def __init__(self, tables):
        self.t = tables
        self.orders = sorted(tables["orders.orders"], key=lambda o: (o["placed_at"], o["order_id"]), reverse=True)
        self.by_customer = defaultdict(list)
        for order in self.orders:
            self.by_customer[order["customer_id"]].append(order)
        self.redemptions = tables["promotions.redemptions"]
        self.fault_orders = {f["order_id"] for f in tables["truth.order_faults"]}

    def order(self, order_id: str) -> dict:
        return next(o for o in self.orders if o["order_id"] == order_id)

    def live_order(self, status: str) -> dict:
        """a customer's newest order, still on its way, in this status."""
        return next(o for o in self.orders if o["status"] == status and self.by_customer[o["customer_id"]][0] is o)

    def customer_without_live_order(self) -> str:
        return next(c for c, orders in sorted(self.by_customer.items())
                    if all(o["status"] == "DELIVERED" for o in orders[:5]))

    def customer_without_orders(self) -> str:
        return next(c["customer_id"] for c in self.t["customer.customers"]
                    if c["customer_id"] not in self.by_customer)

    def order_with_status(self, status: str) -> dict:
        return next(o for o in self.orders if o["status"] == status)

    def recent_attempts(self, customer_id: str) -> list[dict]:
        recent = {o["order_id"] for o in self.by_customer[customer_id][:5]}
        return [r for r in self.redemptions if r["order_id"] in recent]

    def sole_attempt(self, reason_part: str | None = None, status: str = "rejected", code: str | None = None) -> dict:
        """a coupon attempt that is the only one among its customer's five newest orders, untouched by a fault."""
        for r in self.redemptions:
            if r["status"] != status or r["order_id"] in self.fault_orders:
                continue
            if reason_part and reason_part not in (r["reason"] or ""):
                continue
            if code and r["code"] != code:
                continue
            if self.recent_attempts(r["customer_id"]) == [r]:
                return r
        raise LookupError(f"no sole attempt with {reason_part!r} / {status} / {code}")

    def any_attempt(self, reason_part: str) -> dict:
        return next(r for r in self.redemptions if reason_part in (r["reason"] or "")
                    and r["order_id"] not in self.fault_orders)

    def customer_with_two_attempted_orders(self) -> str:
        for customer in sorted(self.by_customer):
            if len({r["order_id"] for r in self.recent_attempts(customer)}) == 2:
                return customer
        raise LookupError("no customer with two attempted orders")

    def fault(self, fault_type: str, variant: str) -> dict:
        return next(f for f in self.t["truth.order_faults"]
                    if f["fault_type"] == fault_type and f["variant"] == variant)

    def case_rows(self, dbname: str) -> list[dict]:
        from psycopg.rows import dict_row

        with db.connect("admin", dbname=dbname, row_factory=dict_row) as conn:
            return conn.execute("select * from support.cases order by created_at, case_id").fetchall()


@pytest.fixture(scope="session")
def data(test_db):
    return Data(test_db)


@pytest.fixture
def cases(data, test_db_name):
    return lambda: data.case_rows(test_db_name)
