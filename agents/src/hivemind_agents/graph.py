"""the case graph: one run per customer message, from the cleaned message to the reply.

    input_guard -> intake -> fixed_reply                                   (outside the case list, or unclear)
                          -> triage -> case_manager -> order_status        (where is my order: template, no model)
                               ^                    -> promotions_investigator -> policy_provider -> communicator
                               |                    -> waiting_for_person  (a case type with no path yet)
                               |     then close_case moves the case to where its part of the reply left it
                               +---- the next issue in the message, until none is left; then reply

each issue in a message gets its own case and its own part of one reply (blueprint s21). an issue that needs a
question (which order? which coupon?) waits for the answer; the reply ends with one question, and the other issues
that need one wait in it.

rules choose the path; no model does. a node that cannot reach the backend or the model raises, and the chat
service answers with the "having trouble" reply instead of a guess.
"""

from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from hivemind_agents import replies
from hivemind_agents.backend_client import BackendClient
from hivemind_agents.capabilities import (case_manager, communicator, input_guard, intake, order_status,
                                          policy_provider, promotions_investigator, triage)
from hivemind_agents.case_store import CaseStore
from hivemind_agents.contracts import Issue, Reply, RunState
from hivemind_agents.llm import ModelRouter
from hivemind_agents.reply_stream import ReplyStream
from hivemind_agents.tracing import Tracer, set_output

# a few nodes per issue; this is far above any real message and still stops a run that never ends
RECURSION_LIMIT = 200


@dataclass
class RunContext:
    """what one run works with. none of it is kept after the run."""

    backend: BackendClient
    llm: ModelRouter
    vault: input_guard.PiiVault
    tracer: Tracer
    cases: CaseStore
    out: ReplyStream = field(default_factory=ReplyStream)


Run = Runtime[RunContext]


def guard_node(state: RunState, runtime: Run) -> dict:
    with runtime.context.tracer.capability("input_guard") as span:
        result = input_guard.guard(state["raw_text"], runtime.context.vault)
        set_output(span, result.model_dump())
        return {"guard": result}


def intake_node(state: RunState, runtime: Run) -> dict:
    ctx = runtime.context
    with ctx.tracer.capability("intake_triage", uses_llm=True) as span:
        recent = triage.recent_orders(ctx.backend)
        result = intake.intake(state["guard"].sanitised_text, state.get("hint"), state.get("pending"),
                               state.get("history", []), recent, ctx.llm)
        span.set_attribute("hivemind.intake.source", result.source)
        set_output(span, result.model_dump())
        return {"recent_orders": recent, "intake": result, "todo": list(result.issues)}


def after_intake(state: RunState) -> str:
    result = state["intake"]
    if result.language != "en" or result.scope != "in_scope" or result.needs_clarification:
        return "fixed_reply"
    return "triage"


def fixed_reply_node(state: RunState, runtime: Run) -> dict:
    result = state["intake"]
    with runtime.context.tracer.capability("fixed_reply") as span:
        if result.language != "en":
            kind, text = "english_only", replies.ENGLISH_ONLY
        elif result.scope == "out_of_scope":
            kind, text = "out_of_scope", replies.OUT_OF_SCOPE
        elif result.scope == "safety":
            kind, text = "safety", replies.SAFETY
        elif result.scope == "legal":
            kind, text = "legal", replies.LEGAL
        elif result.scope == "small_talk":
            thanks = intake.THANKS.match(state["guard"].sanitised_text)
            kind, text = ("thanks", replies.THANKS) if thanks else ("greeting", replies.GREETING)
        else:
            kind, text = "clarify", replies.CLARIFY
        span.set_attribute("hivemind.reply.kind", kind)
        return {"reply": Reply(kind=kind, text=text)}


def _question(result: triage.Triage, issue: Issue, backend: BackendClient) -> Reply:
    if result.ask == "order":
        now = order_status._time(backend.clock()["now"])
        text = replies.order_not_found(result.unknown_order) if result.unknown_order else replies.which_order()
        pending = {"asks": "order", "case_type": issue.case_type, "coupon_code": issue.coupon_code, "queue": []}
        chips = replies.order_chips(issue.case_type, result.choices, now, issue.coupon_code)
        return Reply(kind="ask_order", text=text, chips=chips, meta={"pending": pending})
    pending = {"asks": "coupon_code", "case_type": issue.case_type, "order_id": issue.order_id, "queue": []}
    return Reply(kind="ask_coupon_code", text=replies.which_coupon(issue.order_id), meta={"pending": pending})


def triage_node(state: RunState, runtime: Run) -> dict:
    """takes the next issue and settles its order (and coupon). an issue that needs a question waits for it; the
    same problem twice in one message is worked once."""
    ctx = runtime.context
    issue, rest = state["todo"][0], state["todo"][1:]
    update: dict = {"todo": rest, "issue": None, "case": None, "finding": None, "policy": None, "piece": None}
    with ctx.tracer.capability("triage") as span:
        result = triage.triage(issue, state["recent_orders"], ctx.backend)
        issue = result.issue
        span.set_attribute("hivemind.case_type", issue.case_type)
        if result.no_orders:
            if not any(p.kind == "no_orders" for p in state.get("pieces", [])):
                update["pieces"] = [Reply(kind="no_orders", text=replies.NO_ORDERS)]
            return update
        if result.ask:
            ask = state.get("ask")
            if ask is None:
                update["ask"] = _question(result, issue, ctx.backend)
            else:  # one question at a time: this issue waits in the first one
                ask = ask.model_copy(deep=True)
                ask.meta["pending"]["queue"].append(issue.model_dump())
                update["ask"] = ask
            span.set_attribute("hivemind.triage.asks", result.ask)
            return update
        key = f"{issue.case_type}:{issue.order_id}"
        if key in state.get("seen", []):
            span.set_attribute("hivemind.triage.same_as_before", True)
            return update
        set_output(span, issue.model_dump())
        return {**update, "issue": issue, "seen": state.get("seen", []) + [key]}


def after_triage(state: RunState) -> str:
    if state.get("issue") is not None:
        return "case_manager"
    return "triage" if state.get("todo") else "reply"


def case_manager_node(state: RunState, runtime: Run) -> dict:
    ctx = runtime.context
    with ctx.tracer.capability("case_manager") as span:
        case = case_manager.open_case(state["issue"], ctx.backend)
        span.set_attribute("hivemind.case_id", case.case_id)
        span.set_attribute("hivemind.case.opened", case.opened)
        if case.state in case_manager.WITH_A_PERSON:
            return {"case": case, "piece": Reply(kind="already_with_person", text=replies.ALREADY_WITH_PERSON)}
        lease = None if case.state in case_manager.IN_HAND else ctx.cases.acquire(case.case_id)
        if lease is None:
            span.set_attribute("hivemind.case.busy", True)
            return {"case": case, "piece": Reply(kind="case_busy", text=replies.CASE_BUSY)}
        span.set_attribute("hivemind.case.fence", lease.fence)
        return {"case": case_manager.start(case, ctx.backend)}


def after_case_manager(state: RunState) -> str:
    if state.get("piece") is not None:
        return "close_case"
    case_type = state["case"].case_type
    if case_type == "where_is_my_order":
        return "order_status"
    if case_type == "offers_coupons":
        return "promotions_investigator"
    return "waiting_for_person"


def order_status_node(state: RunState, runtime: Run) -> dict:
    ctx = runtime.context
    with ctx.tracer.capability("order_status") as span:
        text, facts = order_status.order_status(state["issue"].order_id, ctx.backend)
        set_output(span, facts)
        return {"piece": Reply(kind="order_status", text=text, case_state="RESOLVED", meta={"facts": facts})}


def promotions_node(state: RunState, runtime: Run) -> dict:
    ctx, issue = runtime.context, state["issue"]
    with ctx.tracer.capability("promotions_investigator") as span:
        finding = promotions_investigator.investigate(issue.order_id, issue.coupon_code, ctx.backend)
        set_output(span, finding.model_dump())
        return {"finding": finding}


def policy_node(state: RunState, runtime: Run) -> dict:
    ctx, issue = runtime.context, state["issue"]
    with ctx.tracer.capability("policy_knowledge_provider") as span:
        sections = policy_provider.sections_for(issue.case_type, ctx.backend, coupon_code=issue.coupon_code)
        set_output(span, [s.ref for s in sections])
        return {"policy": sections}


def communicator_node(state: RunState, runtime: Run) -> dict:
    ctx, finding, policy = runtime.context, state["finding"], state["policy"]
    with ctx.tracer.capability("customer_communicator", uses_llm=True) as span:
        text, record = communicator.compose(state["guard"].sanitised_text, finding, policy, ctx.llm,
                                            ctx.out.writer(runtime.stream_writer))
        span.set_attribute("hivemind.reply.source", record["source"])
        set_output(span, text)
        meta = {"finding": finding.model_dump(mode="json"), "policy_refs": [s.ref for s in policy],
                "evidence_refs": finding.evidence_refs + [s.evidence_ref for s in policy], "communicator": record}
        return {"piece": Reply(kind=f"coupon_{finding.outcome}", text=text,
                               case_state="AWAITING_HUMAN" if finding.owed else "RESOLVED", meta=meta)}


def waiting_node(state: RunState, runtime: Run) -> dict:
    with runtime.context.tracer.capability("fixed_reply") as span:
        span.set_attribute("hivemind.reply.kind", "waiting_for_person")
        text = replies.waiting_for_person(state["case"].case_type)
        return {"piece": Reply(kind="waiting_for_person", text=text, case_state="AWAITING_HUMAN")}


def _reason(piece: Reply) -> str:
    if piece.kind == "waiting_for_person":
        return "no_path_yet"
    return "money_owed" if piece.case_state == "AWAITING_HUMAN" else "answered"


def close_case_node(state: RunState, runtime: Run) -> dict:
    """the case manager moves the case to where its part of the reply left it and lets go of it; the part carries
    the case with it."""
    ctx, case, piece = runtime.context, state["case"], state["piece"]
    with ctx.tracer.capability("case_manager") as span:
        if piece.case_state and piece.case_state != case.state:
            case = case_manager.move(case, piece.case_state, ctx.backend, _reason(piece))
        case_manager.finish(case, ctx.backend)
        ctx.cases.release(case.case_id)
        span.set_attribute("hivemind.case_id", case.case_id)
        span.set_attribute("hivemind.case.state", case.state)
        ctx.out.part_done(piece.text)
        return {"case": case, "pieces": [piece.model_copy(update={"case_id": case.case_id, "case_state": case.state})]}


def reply_node(state: RunState, runtime: Run) -> dict:
    """one reply: the parts in the order of the issues, then the question, if any."""
    ctx = runtime.context
    parts = list(state.get("pieces", []))
    ask = state.get("ask")
    if not parts and ask is None:  # cannot happen: every issue adds a part or a question; kept as a guard
        return {"reply": Reply(kind="clarify", text=replies.CLARIFY)}
    if len(parts) + (ask is not None) == 1:
        return {"reply": parts[0] if parts else ask}
    with ctx.tracer.capability("reply") as span:
        cases = [p for p in parts if p.case_id]
        meta: dict = {"parts": [{"kind": p.kind, "case_id": p.case_id, "case_state": p.case_state, "meta": p.meta}
                                for p in parts]}
        if ask is not None:
            if ask.text == replies.which_order():  # after other parts, say which problem the question is about
                ask = ask.model_copy(update={"text": replies.which_order(ask.meta["pending"]["case_type"])})
            meta["pending"] = ask.meta["pending"]
            meta["parts"].append({"kind": ask.kind, "case_id": None, "case_state": None, "meta": {}})
        text = replies.PART_BREAK.join([p.text for p in parts] + ([ask.text] if ask else []))
        ctx.out.finish(runtime.stream_writer, [ask.text] if ask else [])
        span.set_attribute("hivemind.reply.parts", len(meta["parts"]))
        return {"reply": Reply(kind="several", text=text, chips=ask.chips if ask else [],
                               case_id=cases[0].case_id if cases else None,
                               case_state=cases[0].case_state if cases else None, meta=meta)}


def build_graph():
    graph = StateGraph(RunState, context_schema=RunContext)
    graph.add_node("input_guard", guard_node)
    graph.add_node("intake", intake_node)
    graph.add_node("fixed_reply", fixed_reply_node)
    graph.add_node("triage", triage_node)
    graph.add_node("case_manager", case_manager_node)
    graph.add_node("order_status", order_status_node)
    graph.add_node("promotions_investigator", promotions_node)
    graph.add_node("policy_provider", policy_node)
    graph.add_node("communicator", communicator_node)
    graph.add_node("waiting_for_person", waiting_node)
    graph.add_node("close_case", close_case_node)
    graph.add_node("reply", reply_node)

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "intake")
    graph.add_conditional_edges("intake", after_intake, ["fixed_reply", "triage"])
    graph.add_edge("fixed_reply", END)
    graph.add_conditional_edges("triage", after_triage, ["case_manager", "triage", "reply"])
    graph.add_conditional_edges("case_manager", after_case_manager,
                                ["close_case", "order_status", "promotions_investigator", "waiting_for_person"])
    graph.add_edge("order_status", "close_case")
    graph.add_edge("promotions_investigator", "policy_provider")
    graph.add_edge("policy_provider", "communicator")
    graph.add_edge("communicator", "close_case")
    graph.add_edge("waiting_for_person", "close_case")
    graph.add_conditional_edges("close_case", lambda s: "triage" if s.get("todo") else "reply", ["triage", "reply"])
    graph.add_edge("reply", END)
    return graph.compile()
