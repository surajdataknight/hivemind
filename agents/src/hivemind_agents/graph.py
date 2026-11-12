"""the case graph: one run per customer message, from the cleaned message to the reply.

    input_guard -> intake -> fixed_reply                                   (outside the case list, or unclear)
                          -> triage -> case_manager -> order_status        (where is my order: template, no model)
                                                    -> promotions_investigator -> policy_provider -> communicator
                                                    -> waiting_for_person  (a case type with no path yet)
                             then close_case moves the case to where the reply left it.

rules choose the path; no model does. a node that cannot reach the backend or the model raises, and the chat
service answers with the "having trouble" reply instead of a guess.
"""

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from hivemind_agents import replies
from hivemind_agents.backend_client import BackendClient
from hivemind_agents.capabilities import (case_manager, communicator, input_guard, intake, order_status,
                                          policy_provider, promotions_investigator, triage)
from hivemind_agents.contracts import SHORT_PATHS, Reply, RunState
from hivemind_agents.llm import ModelRouter
from hivemind_agents.tracing import Tracer, set_output


@dataclass
class RunContext:
    """what one run works with. none of it is kept after the run."""

    backend: BackendClient
    llm: ModelRouter
    vault: input_guard.PiiVault
    tracer: Tracer


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
        return {"recent_orders": recent, "intake": result}


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


def triage_node(state: RunState, runtime: Run) -> dict:
    """picks the issue to work on (the first one with a path, else the first) and settles its order."""
    ctx = runtime.context
    issues = state["intake"].issues
    with ctx.tracer.capability("triage") as span:
        primary = next((i for i in issues if i.case_type in SHORT_PATHS), issues[0])
        result = triage.triage(primary, state["recent_orders"], ctx.backend)
        issue = result.issue
        span.set_attribute("hivemind.case_type", issue.case_type)
        if result.no_orders:
            return {"reply": Reply(kind="no_orders", text=replies.NO_ORDERS)}
        if result.ask == "order":
            now = order_status._time(ctx.backend.clock()["now"])
            text = replies.order_not_found(result.unknown_order) if result.unknown_order else replies.which_order()
            pending = {"asks": "order", "case_type": issue.case_type, "coupon_code": issue.coupon_code}
            chips = replies.order_chips(issue.case_type, result.choices, now, issue.coupon_code)
            return {"reply": Reply(kind="ask_order", text=text, chips=chips, meta={"pending": pending})}
        if result.ask == "coupon_code":
            pending = {"asks": "coupon_code", "case_type": issue.case_type, "order_id": issue.order_id}
            return {"reply": Reply(kind="ask_coupon_code", text=replies.which_coupon(issue.order_id),
                                   meta={"pending": pending})}
        set_output(span, issue.model_dump())
        return {"issue": issue}


def after_triage(state: RunState) -> str:
    return END if "reply" in state else "case_manager"


def case_manager_node(state: RunState, runtime: Run) -> dict:
    ctx = runtime.context
    with ctx.tracer.capability("case_manager") as span:
        case = case_manager.open_case(state["issue"], ctx.backend)
        span.set_attribute("hivemind.case_id", case.case_id)
        span.set_attribute("hivemind.case.opened", case.opened)
        if case.state == "AWAITING_HUMAN":
            return {"case": case, "reply": Reply(kind="already_with_person", text=replies.ALREADY_WITH_PERSON)}
        return {"case": case_manager.start_investigating(case, ctx.backend)}


def after_case_manager(state: RunState) -> str:
    if "reply" in state:
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
        return {"reply": Reply(kind="order_status", text=text, case_state="RESOLVED", meta={"facts": facts})}


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
                                            runtime.stream_writer)
        span.set_attribute("hivemind.reply.source", record["source"])
        set_output(span, text)
        meta = {"finding": finding.model_dump(mode="json"), "policy_refs": [s.ref for s in policy],
                "evidence_refs": finding.evidence_refs + [s.evidence_ref for s in policy], "communicator": record}
        return {"reply": Reply(kind=f"coupon_{finding.outcome}", text=text,
                               case_state="AWAITING_HUMAN" if finding.owed else "RESOLVED", meta=meta)}


def waiting_node(state: RunState, runtime: Run) -> dict:
    with runtime.context.tracer.capability("fixed_reply") as span:
        span.set_attribute("hivemind.reply.kind", "waiting_for_person")
        text = replies.waiting_for_person(state["case"].case_type)
        return {"reply": Reply(kind="waiting_for_person", text=text, case_state="AWAITING_HUMAN")}


def close_case_node(state: RunState, runtime: Run) -> dict:
    """the case manager moves the case to where the reply left it, and the reply carries the case with it."""
    ctx, case, reply = runtime.context, state["case"], state["reply"]
    with ctx.tracer.capability("case_manager") as span:
        if reply.case_state and reply.case_state != case.state:
            case = case_manager.move(case, reply.case_state, ctx.backend)
        span.set_attribute("hivemind.case_id", case.case_id)
        span.set_attribute("hivemind.case.state", case.state)
        text = reply.text + (replies.MORE_ISSUES if len(state["intake"].issues) > 1 else "")
        return {"case": case, "reply": reply.model_copy(update={"text": text, "case_id": case.case_id,
                                                                "case_state": case.state})}


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

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "intake")
    graph.add_conditional_edges("intake", after_intake, ["fixed_reply", "triage"])
    graph.add_edge("fixed_reply", END)
    graph.add_conditional_edges("triage", after_triage, ["case_manager", END])
    graph.add_conditional_edges("case_manager", after_case_manager,
                                ["close_case", "order_status", "promotions_investigator", "waiting_for_person"])
    graph.add_edge("order_status", "close_case")
    graph.add_edge("promotions_investigator", "policy_provider")
    graph.add_edge("policy_provider", "communicator")
    graph.add_edge("communicator", "close_case")
    graph.add_edge("waiting_for_person", "close_case")
    graph.add_edge("close_case", END)
    return graph.compile()
