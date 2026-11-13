"""runs a scenario's chat against hivemind and the simulated business, both in this process, on the evaluation's own
database (never the main one): rebuilt from the seed before each run, so every run starts from the same data.

hivemind is used through its chat service exactly as the screen uses it (sign in, start a chat, send messages,
press buttons); nothing here reaches into its insides.
"""

import os
import time
from dataclasses import dataclass, field
from uuid import uuid4

from hivemind_agents import settings, tracing
from hivemind_agents.capabilities.identity import IdentityVerifier
from hivemind_agents.chat import ChatService, Deps
from hivemind_agents.contracts import IntentHint
from hivemind_agents.llm import ModelRouter
from hivemind_agents.store import ChatStore
from hivemind_evals.customer import MAX_TURNS, Customer
from hivemind_evals.scenarios import Scenario

EVAL_DB = "hivemind_eval"
SEED = 42
ASKS = ("ask_order", "ask_coupon_code", "ask_items", "ask_again", "clarify")  # replies that ask the customer something


def prepare(dbname: str = EVAL_DB, seed: int = SEED) -> None:
    """the evaluation database, rebuilt: every migration, then the small generated dataset."""
    from hivemind_backend.migrate import migrate
    from hivemind_datagen.dataset import generate
    from hivemind_datagen.generate import SMALL
    from hivemind_datagen.load import load

    migrate(dbname, reset=True)
    load(generate(seed, SMALL), dbname)


def make_service(shop, llm: ModelRouter, case_lead: bool = True, tracer: tracing.Tracer | None = None) -> ChatService:
    tracer = tracer or tracing.Tracer(tracing.make_provider(None))
    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"),
                                settings.signing_key("HM_CASE_TOKEN_SECRET"))
    return ChatService(Deps(http=shop, llm=llm, store=ChatStore(settings.agent_conninfo()), tracer=tracer,
                            verifier=verifier, case_lead=case_lead))


def business(dbname: str = EVAL_DB):
    """the simulated business's api in this process, on the evaluation database."""
    os.environ["HM_DB_NAME"] = dbname
    from fastapi.testclient import TestClient
    from hivemind_backend.app import app

    return TestClient(app)


@dataclass
class Transcript:
    scenario_id: str
    conversation_id: str = ""
    turns: list[dict] = field(default_factory=list)  # role, text, kind, buttons, asks, llm_calls, llm_tokens
    seconds: float = 0.0
    error: str | None = None
    customer_error: str | None = None  # the simulated customer's own model failed: not hivemind's doing

    @property
    def hivemind_calls(self) -> int:
        return sum(t.get("llm_calls", 0) for t in self.turns if t["role"] == "assistant")

    @property
    def hivemind_tokens(self) -> int:
        return sum(t.get("llm_tokens", 0) for t in self.turns if t["role"] == "assistant")

    @property
    def questions(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "assistant" and t.get("asks"))


def run_chat(scenario: Scenario, customer: Customer, service: ChatService, shop,
             max_turns: int = MAX_TURNS) -> Transcript:
    out = Transcript(scenario.scenario_id)
    started = time.monotonic()
    try:
        token = shop.post("/v1/identity/sessions", json={"customer_id": scenario.customer_id}).json()["session_token"]
        out.conversation_id = service.start_conversation(token)["conversation_id"]
        chips: list[dict] = []
        for _ in range(max_turns):
            try:
                turn = customer.next(out.turns)
            except Exception as exc:
                out.customer_error = f"{type(exc).__name__}: {exc}"
                break
            if turn.done and not turn.press:
                break
            chip = next((c for c in chips if turn.press and turn.press in (c["text"], c["label"])), None)
            text = chip["text"] if chip else (turn.message or turn.press or "").strip()
            if not text:
                break
            out.turns.append({"role": "customer", "text": text, "pressed": chip is not None})
            done = service.handle_message(token, out.conversation_id, text, uuid4().hex,
                                          IntentHint(**chip["intent"]) if chip else None, lambda event: None)
            message = done["message"]
            chips = message.get("chips", [])
            out.turns.append({"role": "assistant", "text": message["text"], "kind": message["kind"],
                              "buttons": [c["text"] for c in chips],
                              "asks": message["kind"] in ASKS or bool(chips),  # buttons come only with a question
                              "llm_calls": done.get("llm_calls", 0), "llm_tokens": done.get("llm_tokens", 0)})
    except Exception as exc:  # a run that breaks is graded as failed, with the reason kept
        out.error = f"{type(exc).__name__}: {exc}"
    out.seconds = round(time.monotonic() - started, 2)
    return out
