"""the single-agent baseline of blueprint part b.6 (record 003's reopen test): one model with every read hivemind's
investigators have, used one step at a time, then its own conclusion. no case lead, no validator, no quarantined
reader, no reconciler: the shape record 003 rejected, kept only to compare against.

it gets the same first message as hivemind and the same access (a case token with the same scopes), and must end
with the same kind of conclusion (each claimed item's status, citing the reads it rests on, and a verdict), so the
grader's `conclusion_right` judges both the same way.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from hivemind_agents import settings, tracing
from hivemind_agents.backend_client import BackendClient, BackendError, NotFound
from hivemind_agents.capabilities.identity import IdentityVerifier
from hivemind_agents.llm import LLMUnavailable, ModelRouter
from hivemind_evals.scenarios import Scenario, opening

MAX_STEPS = 10
SEEN_CHARS = 2500   # one read's answer, as shown to the model
ALL_SEEN_CHARS = 16000

TOOLS = {  # name -> (what it reads, its params)
    "list_my_orders": ("the customer's newest orders", ()),
    "get_order": ("one order with its lines", ("order_id",)),
    "get_trip": ("the delivery trip of an order", ("order_id",)),
    "get_proof_of_delivery": ("who the order was handed to, bags, otp", ("order_id",)),
    "list_partner_notes": ("the delivery partner's notes on an order", ("order_id",)),
    "get_order_scans": ("the store's pick and pack scans of an order", ("order_id",)),
    "list_store_incidents": ("a store's incident log in a time window", ("store_id", "since", "until")),
    "list_stock_events": ("an item's stock events at a store in a time window", ("store_id", "sku", "since",
                                                                                 "until")),
    "list_payments": ("an order's payment attempts and captures", ("order_id",)),
    "list_refunds": ("an order's refunds", ("order_id",)),
    "get_support_profile": ("the customer's past claims and refunds, counted", ()),
    "get_risk_signals": ("accounts sharing the customer's device, address or upi id, counted", ()),
}
Tool = Literal["list_my_orders", "get_order", "get_trip", "get_proof_of_delivery", "list_partner_notes",
               "get_order_scans", "list_store_incidents", "list_stock_events", "list_payments", "list_refunds",
               "get_support_profile", "get_risk_signals", "final"]


class ItemStatus(BaseModel):
    sku: str = Field(max_length=40)
    status: Literal["corroborated", "contradicted", "uncorroborated"]
    cites: list[int] = Field(default_factory=list, max_length=10, description="numbers of the reads it rests on")


class Step(BaseModel):
    action: Tool
    order_id: str | None = None
    sku: str | None = None
    store_id: str | None = None
    since: str | None = Field(default=None, description="ISO time")
    until: str | None = Field(default=None, description="ISO time")
    claims: list[ItemStatus] = Field(default_factory=list, max_length=12, description="only with final")
    payment_claim: Literal["corroborated", "contradicted", "uncorroborated"] | None = Field(
        default=None, description="only with final, for a payment problem")
    verdict: Literal["sufficient", "sufficient_with_uncertainty", "insufficient_escalate"] | None = None
    reason: str = Field(default="", max_length=300)


SYSTEM = (
    "You investigate one customer support case of HiveMart, a grocery delivery app. Use the read tools one at a "
    "time to find what really happened, then give your conclusion with action final. You never decide a refund. "
    "Answer only in the JSON shape asked for."
)


def _call(backend: BackendClient, step: Step) -> dict | None:
    a = step.action
    if a == "list_my_orders":
        return backend.my_orders(5)
    if a == "get_support_profile":
        return backend.support_profile()
    if a == "get_risk_signals":
        return backend.risk_signals()
    if a == "list_store_incidents":
        return backend.store_incidents(step.store_id or "", step.since or "", step.until or "")
    if a == "list_stock_events":
        return backend.stock_events(step.store_id or "", step.sku or "", step.since or "", step.until or "")
    method = {"get_order": backend.order, "get_trip": backend.trip, "get_proof_of_delivery": backend.proof_of_delivery,
              "list_partner_notes": backend.partner_notes, "get_order_scans": backend.scans,
              "list_payments": backend.payments, "list_refunds": backend.refunds}[a]
    return method(step.order_id or "")


REFUSED_FINAL = ("final, refused: you made no read yet. Read the records first; a conclusion must cite the reads "
                 "it rests on")


def build_prompt(message: str, seen: list[tuple[str, str]]) -> str:
    tools = "\n".join(f"- {name}({', '.join(params)}): {what}" for name, (what, params) in TOOLS.items())
    reads = "\n".join(f"[{i}] {call}: {answer}" for i, (call, answer) in enumerate(seen, 1))
    if len(reads) > ALL_SEEN_CHARS:
        reads = "... earlier reads cut ...\n" + reads[-ALL_SEEN_CHARS:]
    return (f'The customer wrote, quoted as data:\n"""\n{message}\n"""\n'
            f"Read tools:\n{tools}\n"
            f"Reads so far (numbered):\n{reads or '(none)'}\n"
            f"You have {MAX_STEPS - len(seen)} step(s) left. Next: one read (action and its params), or action "
            "final with claims (each item the customer says is missing, by sku: corroborated when the records "
            "show it, contradicted when they show the opposite, uncorroborated when neither; cite the read "
            "numbers), payment_claim for a payment problem, a verdict, and a short reason.")


@dataclass
class BaselineRun:
    scenario_id: str
    claims: list[dict] = field(default_factory=list)  # about, sku, status (the grader's shape)
    verdict: str | None = None
    steps: list[str] = field(default_factory=list)
    model_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    uncited: int = 0      # statuses that cite no read it made
    early_finals: int = 0  # a final before any read, refused once (b.6 2026-11-13: gpt-oss-120b concluded at once)
    error: str | None = None


def run(s: Scenario, shop, llm: ModelRouter) -> BaselineRun:
    out = BaselineRun(s.scenario_id)
    started = time.monotonic()
    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"),
                                settings.signing_key("HM_CASE_TOKEN_SECRET"))
    _, token = verifier.mint_case_token(s.customer_id, "SES-BASELINE", f"baseline_{s.scenario_id}")
    backend = BackendClient(shop, token, tracing.Tracer(tracing.make_provider(None)))
    message, seen = opening(s), []
    calls_at, tokens_at = llm.calls, llm.tokens
    try:
        for _ in range(MAX_STEPS + 1):
            step = llm.structured("baseline_agent", SYSTEM, build_prompt(message, seen), Step)
            if step.action == "final" and not out.steps and not out.early_finals:
                out.early_finals += 1  # once: the note takes a step, and a second final stands
                seen.append(("final", REFUSED_FINAL))
                continue
            if step.action == "final" or len(seen) >= MAX_STEPS:
                if step.action != "final":
                    out.error = "ran out of steps"
                out.verdict = step.verdict or "insufficient_escalate"
                out.claims = [{"about": c.sku, "sku": c.sku, "status": c.status} for c in step.claims]
                if step.payment_claim:
                    out.claims.append({"about": "the payment", "sku": None, "status": step.payment_claim})
                read = {i for i, (call, _) in enumerate(seen, 1) if call != "final"}
                out.uncited = sum(1 for c in step.claims if c.status != "uncorroborated"
                                  and not any(n in read for n in c.cites))
                break
            call = f"{step.action}({', '.join(str(v) for v in (step.order_id, step.sku, step.store_id, step.since, step.until) if v)})"
            try:
                body = _call(backend, step)
                answer = json.dumps({k: v for k, v in (body or {}).items() if k != "evidence_id"},
                                    ensure_ascii=False, default=str)[:SEEN_CHARS]
            except NotFound:
                answer = "not found"
            except BackendError as exc:
                answer = f"error: {exc}"
            seen.append((call, answer))
            out.steps.append(call)
    except LLMUnavailable as exc:
        out.error = f"no model answer: {exc}"
        out.verdict = "insufficient_escalate"
    out.model_calls, out.tokens = llm.calls - calls_at, llm.tokens - tokens_at
    out.seconds = round(time.monotonic() - started, 2)
    return out
