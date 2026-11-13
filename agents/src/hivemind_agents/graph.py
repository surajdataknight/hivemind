"""the case graph: one run per customer message, from the cleaned message to the reply.

    input_guard -> intake -> ask_again                                     (not an answer to our open question)
                          -> fixed_reply                                   (outside the case list, or unclear)
                          -> triage -> case_manager -> order_status        (where is my order: template, no model)
                               ^                    -> promotions_investigator -> policy_provider -> communicator
                               |                    -> plan -> schedule <-> investigate   (T2 types: case lead, c5-c7)
                               |                         -> finish_plan -> reconcile -> replan -> schedule  (c17's
                               |                                                        follow-ups, one round)
                               |                                                     -> decide
                               |                    -> decide   (cancellation: the short path, no check, no model)
                               |                         decide (c18 rules, c21 router, c19 when acting alone)
                               |                           -> act (c22: refund, wallet credit or cancel, under a key)
                               |                           -> to_person (a person decides; the decision is kept)
                               |                           -> explain (nothing owed, or already refunded)
                               |                    -> waiting_for_person  (a case type with no path yet, or a plan
                               |                                           whose minimum checks cannot run)
                               |     then close_case moves the case to where its part of the reply left it
                               +---- the next issue in the message, until none is left; then reply

each issue in a message gets its own case and its own part of one reply (blueprint s21). an issue that needs a
question (which order? which coupon? which item?) waits for the answer; the reply ends with one question, and the
other issues that need one wait in it.

rules choose the path; no model does, except the case lead, which only proposes checks (the validator decides what
runs), and the evidence reconciler, whose follow-ups go back to the case lead. a node that cannot reach the backend or the model raises, and the chat service answers with the "having
trouble" reply instead of a guess.
"""

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from hivemind_agents import board, replies
from hivemind_agents.backend_client import ActionRefused, BackendClient
from hivemind_agents.capabilities import (action_gateway, approval_router, case_manager, communicator, eligibility,
                                          input_guard, intake, order_status, planner, policy_provider,
                                          progress_reporter, promotions_investigator, reconciler, resolver, triage)
from hivemind_agents.capabilities import registry as check_registry
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents import scheduler
from hivemind_agents.case_store import CaseStore, LeaseLost
from hivemind_agents.contracts import Chip, Decision, Finding, Issue, Plan, Reply, RunState, TaskOutcome
from hivemind_agents.llm import ModelRouter
from hivemind_agents.reply_stream import ReplyStream
from hivemind_agents.tracing import Tracer, set_output

# a few nodes per issue; this is far above any real message and still stops a run that never ends
RECURSION_LIMIT = 200

# whether the plan's stop condition is met by the findings so far (c17's stop judge, in code)
StopJudge = Callable[[Plan, list[Finding]], bool]


@dataclass
class RunContext:
    """what one run works with. none of it is kept after the run."""

    backend: BackendClient
    llm: ModelRouter
    vault: input_guard.PiiVault
    tracer: Tracer
    cases: CaseStore
    out: ReplyStream = field(default_factory=ReplyStream)
    checks: dict[str, Capability] = field(default_factory=lambda: check_registry.CAPABILITIES)  # c5, c6
    retry_sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)
    stop_judge: StopJudge | None = reconciler.stop_judge  # none: every planned check runs
    case_lead: bool = True  # false: the minimum plan only, never revised (an evaluation run, blueprint b.6)
    _stops: dict[str, threading.Event] = field(default_factory=dict)
    _seen: dict[str, list[Finding]] = field(default_factory=dict)  # grounded findings so far, for the stop judge
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def stop_signal(self, case_id: str) -> threading.Event:
        with self._lock:
            return self._stops.setdefault(case_id, threading.Event())

    def judge(self, case_id: str, plan: Plan, board: list[Finding], new: list[Finding]) -> bool:
        """adds a check's grounded findings to what the stop judge has seen for the case, and asks it."""
        if self.stop_judge is None:
            return False
        with self._lock:
            seen = self._seen.setdefault(case_id, list(board))
            seen.extend(new)
            met = self.stop_judge(plan, list(seen))
        if met:
            self.stop_signal(case_id).set()
        return met


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
    if result.ask_again:
        return "ask_again"
    if result.language != "en" or result.scope != "in_scope" or result.needs_clarification:
        return "fixed_reply"
    return "triage"


def question_of(pending: dict) -> str:
    """the words of our open question (a question asked before they were kept is worded again)."""
    if pending.get("question"):
        return pending["question"]
    if pending.get("asks") == "coupon_code":
        return replies.which_coupon(pending.get("order_id"))
    if pending.get("asks") == "items":
        return replies.which_item(pending["case_type"], pending.get("order_id"))
    return replies.which_order(pending["case_type"])


def _open_question(state: RunState) -> tuple[list[Chip], dict]:
    return [Chip(**chip) for chip in state.get("pending_chips") or []], {"pending": state["pending"]}


def ask_again_node(state: RunState, runtime: Run) -> dict:
    """the message does not answer our question: the same question once more, with its buttons; the next message
    that does not answer it is read as a new one (intake.keep_waiting)."""
    with runtime.context.tracer.capability("fixed_reply") as span:
        span.set_attribute("hivemind.reply.kind", "ask_again")
        chips, meta = _open_question(state)
        meta["pending"] = {**meta["pending"], "asked_again": True}
        return {"reply": Reply(kind="ask_again", text=replies.ask_again(question_of(state["pending"])),
                               chips=chips, meta=meta)}


def fixed_reply_node(state: RunState, runtime: Run) -> dict:
    result = state["intake"]
    with runtime.context.tracer.capability("fixed_reply") as span:
        reply = _fixed_reply(state)
        if result.still_waiting and state.get("pending"):  # our question went unanswered: it follows the reply
            chips, meta = _open_question(state)
            reply = Reply(kind=reply.kind, text=reply.text + replies.PART_BREAK + question_of(state["pending"]),
                          chips=chips, meta=meta)
            span.set_attribute("hivemind.reply.question_follows", True)
        span.set_attribute("hivemind.reply.kind", reply.kind)
        return {"reply": reply}


def _fixed_reply(state: RunState) -> Reply:
    result = state["intake"]
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
    return Reply(kind=kind, text=text)


def _claim_words(issue: Issue) -> dict:
    """what the customer said about the problem itself, kept with a question so the answer does not lose it."""
    return {"items_mentioned": issue.items_mentioned, "claimed_amount_paise": issue.claimed_amount_paise,
            "money_to": issue.money_to}


def _question(result: triage.Triage, issue: Issue, backend: BackendClient) -> Reply:
    """the question an issue waits on. its pending record keeps the issue and the question's own words, so an
    answer that is not clean gets the same question again."""
    if result.ask == "order":
        now = order_status._time(backend.clock()["now"])
        text = replies.order_not_found(result.unknown_order) if result.unknown_order else replies.which_order()
        pending = {"asks": "order", "case_type": issue.case_type, "coupon_code": issue.coupon_code, "queue": [],
                   "question": text, **_claim_words(issue)}
        chips = replies.order_chips(issue.case_type, result.choices, now, issue.coupon_code)
        return Reply(kind="ask_order", text=text, chips=chips, meta={"pending": pending})
    if result.ask == "items":
        text = replies.which_item(issue.case_type, issue.order_id)
        pending = {"asks": "items", "case_type": issue.case_type, "order_id": issue.order_id,
                   "coupon_code": issue.coupon_code, "lines": result.lines, "queue": [], "question": text,
                   **_claim_words(issue)}
        chips = replies.item_chips(issue.case_type, issue.order_id, result.lines[:triage.MAX_ITEM_CHIPS])
        return Reply(kind="ask_items", text=text, chips=chips, meta={"pending": pending})
    text = replies.which_coupon(issue.order_id)
    pending = {"asks": "coupon_code", "case_type": issue.case_type, "order_id": issue.order_id, "queue": [],
               "question": text, **_claim_words(issue)}
    return Reply(kind="ask_coupon_code", text=text, meta={"pending": pending})


def triage_node(state: RunState, runtime: Run) -> dict:
    """takes the next issue and settles its order (and coupon). an issue that needs a question waits for it; the
    same problem twice in one message is worked once."""
    ctx = runtime.context
    issue, rest = state["todo"][0], state["todo"][1:]
    update: dict = {"todo": rest, "issue": None, "case": None, "finding": None, "policy": None, "piece": None,
                    "fact_sheet": None, "decision": None}
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
    if case_type in planner.MINIMUM:
        return "plan"
    if case_type == "cancellation":  # the short path: the order's state decides, no check and no model
        return "decide"
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
        sheet = state.get("fact_sheet")
        meta = {"fact_sheet": {"round": sheet.round, "verdict": sheet.verdict, "source": sheet.source}} if sheet else {}
        return {"piece": Reply(kind="waiting_for_person", text=text, case_state="AWAITING_HUMAN", meta=meta)}


# the case lead, the scheduler and the checks (c5, c6, c7)


def plan_node(state: RunState, runtime: Run) -> dict:
    ctx, case = runtime.context, state["case"]
    usage = (ctx.llm.calls, ctx.llm.tokens)  # the case's model budget counts from here, the case lead included
    started = time.monotonic()
    with ctx.tracer.capability("case_lead", uses_llm=True) as span:
        plan = planner.plan_for(case.case_type, state["issue"], state["guard"].sanitised_text, ctx.checks,
                                set(state["identity"].scopes), ctx.llm, use_case_lead=ctx.case_lead)
        span.set_attribute("hivemind.plan.source", plan.source)
        span.set_attribute("hivemind.plan.runnable", plan.runnable)
        span.set_attribute("hivemind.plan.checks", len(plan.tasks))
        if plan.note:
            span.set_attribute("hivemind.plan.note", plan.note)
        set_output(span, plan.model_dump(mode="json"))
    _keep_plan(state, ctx, plan, {})
    if plan.runnable:
        runtime.stream_writer(progress_reporter.event(None, "started"))
    return {"plan": plan, "plan_started": started, "plan_usage": usage, "follow_ups": None, "fact_sheet": None}


def _lease(ctx: RunContext, case_id: str):
    lease = ctx.cases.leases.get(case_id)
    if lease is None:
        raise LeaseLost(case_id)
    return lease


def _keep_plan(state: RunState, ctx: RunContext, plan: Plan, outcomes: dict[str, TaskOutcome]) -> None:
    """the plan and what became of each check so far, for anyone watching the case (the scenario panel)."""
    case = state["case"]
    about = {"case_type": case.case_type, "order_id": case.order_id, "customer_id": state["identity"].customer_id,
             "conversation_id": state["conversation_id"]}
    ctx.cases.save_plan(_lease(ctx, case.case_id), about, plan.model_dump(mode="json"),
                        {tid: o.model_dump(mode="json", exclude={"findings"}) for tid, o in outcomes.items()})


def after_plan(state: RunState) -> str:
    return "schedule" if state["plan"].runnable else "waiting_for_person"


def _outcomes_of(state: RunState, case_id: str) -> dict[str, TaskOutcome]:
    prefix = f"{case_id}/"
    return {key[len(prefix):]: outcome for key, outcome in state.get("outcomes", {}).items() if key.startswith(prefix)}


def _budget_stop(state: RunState, ctx: RunContext) -> str | None:
    calls_at, tokens_at = state.get("plan_usage") or (ctx.llm.calls, ctx.llm.tokens)
    return scheduler.over_budget(state.get("plan_started"), time.monotonic(), ctx.llm.calls - calls_at,
                                 ctx.llm.tokens - tokens_at)


def schedule_node(state: RunState, runtime: Run) -> dict:
    """after each wave: the findings of the checks that just finished go on the board (one writer, under the case's
    lease), and the checks they ask for go through the validator into the plan. checks that cannot run any more
    are recorded as blocked, as not run when the case is over a budget, or as cancelled when the stop condition is
    met (never a minimum check)."""
    ctx, plan, case = runtime.context, state["plan"], state["case"]
    with ctx.tracer.capability("scheduler") as span:
        decided = _outcomes_of(state, case.case_id)
        on_board = list(state.get("boards", {}).get(case.case_id, []))
        update: dict = {}
        fresh = {tid: o for tid, o in decided.items() if o.status == "done" and not o.posted}
        if fresh:
            findings = [f for o in fresh.values() for f in o.findings]
            posted = board.post(case.case_id, findings, ctx.cases)
            span.set_attribute("hivemind.findings.dropped", len(posted.dropped))
            update["boards"] = posted.update()["boards"]
            on_board += posted.kept
            update["outcomes"] = {scheduler.key(case.case_id, tid): o.model_copy(update={"posted": True})
                                  for tid, o in fresh.items()}
            decided.update({tid: o.model_copy(update={"posted": True}) for tid, o in fresh.items()})
            by_id = {t.task_id: t for t in plan.tasks}
            for tid, outcome in fresh.items():
                if outcome.info_requests:
                    plan = planner.admit(state["issue"], plan, by_id[tid], outcome.info_requests, ctx.checks,
                                         set(state["identity"].scopes))
            if plan is not state["plan"]:
                update["plan"] = plan
                span.set_attribute("hivemind.plan.checks", len(plan.tasks))
        stop = _budget_stop(state, ctx)
        if stop:
            span.set_attribute("hivemind.plan.stopped", stop)
            for task in plan.tasks:
                if task.task_id not in decided:
                    decided[task.task_id] = TaskOutcome(task_id=task.task_id, status="not_run", error=stop)
                    update.setdefault("outcomes", {})[scheduler.key(case.case_id, task.task_id)] = decided[task.task_id]
        signal = ctx.stop_signal(case.case_id)
        if not signal.is_set() and fresh and ctx.stop_judge is not None and ctx.stop_judge(plan, on_board):
            signal.set()
        if signal.is_set():
            span.set_attribute("hivemind.plan.stop_condition_met", True)
            for task in plan.tasks:
                if task.task_id not in decided and not task.mandatory:
                    decided[task.task_id] = TaskOutcome(task_id=task.task_id, status="cancelled",
                                                        error="the stop condition was met")
                    update.setdefault("outcomes", {})[scheduler.key(case.case_id, task.task_id)] = decided[task.task_id]
        while True:
            stuck = scheduler.blocked(plan.tasks, decided)
            if not stuck:
                break
            for task in stuck:
                decided[task.task_id] = TaskOutcome(task_id=task.task_id, status="blocked",
                                                    error="a check it needs did not finish")
                update.setdefault("outcomes", {})[scheduler.key(case.case_id, task.task_id)] = decided[task.task_id]
        span.set_attribute("hivemind.checks.decided", len(decided))
    if update:
        _keep_plan(state, ctx, update.get("plan", plan), decided)
    return update


def fan_out(state: RunState):
    """the wave: every check whose checks are done, at most MAX_PARALLEL, side by side (LangGraph Send)."""
    case = state["case"]
    ready_now = scheduler.ready(state["plan"].tasks, _outcomes_of(state, case.case_id))[:scheduler.MAX_PARALLEL]
    if not ready_now:
        return "finish_plan"
    on_board = list(state.get("boards", {}).get(case.case_id, []))
    return [Send("investigate", {"task": task, "case": case, "plan": state["plan"], "board": on_board,
                                 "started": state.get("plan_started"), "issue": state.get("issue")})
            for task in ready_now]


def investigate_node(payload: dict, runtime: Run) -> dict:
    """one check, within its own time limit and the case's clock. a check that is not a minimum check stops when
    the stop condition is met, by a sibling's findings too; this check's grounded findings go to the stop judge."""
    ctx, task, case = runtime.context, payload["task"], payload["case"]
    capability = ctx.checks[task.capability]
    started = payload.get("started")
    runtime.stream_writer(progress_reporter.event(task.capability, "started"))
    with ctx.tracer.capability(f"check {task.capability}") as span:
        check = CheckInput(case, ctx.backend, issue=payload.get("issue"), board=tuple(payload["board"]), llm=ctx.llm)
        outcome = scheduler.run_task(task, capability, check, ctx.retry_sleep, ctx.rng,
                                     deadline=lambda: scheduler.seconds_left(started, time.monotonic()),
                                     stop=None if task.mandatory else ctx.stop_signal(case.case_id))
        span.set_attribute("hivemind.check.status", outcome.status)
        span.set_attribute("hivemind.check.attempts", outcome.attempts)
        if outcome.status == "done" and ctx.stop_judge is not None:
            proven = board.grounded(case.case_id, outcome.findings, ctx.cases.evidence_ids(case.case_id)).kept
            span.set_attribute("hivemind.check.stop_condition_met",
                               ctx.judge(case.case_id, payload["plan"], payload["board"], proven))
        set_output(span, outcome.model_dump(mode="json"))
    runtime.stream_writer(progress_reporter.event(task.capability, "finished"))
    return {"outcomes": {scheduler.key(case.case_id, task.task_id): outcome}}


def finish_plan_node(state: RunState, runtime: Run) -> dict:
    """no check is ready: whatever is still undecided waits on a check that did not finish, so it is blocked."""
    ctx, case, plan = runtime.context, state["case"], state["plan"]
    decided = _outcomes_of(state, case.case_id)
    blocked = {t.task_id: TaskOutcome(task_id=t.task_id, status="blocked", error="a check it needs did not finish")
               for t in plan.tasks if t.task_id not in decided}
    with ctx.tracer.capability("scheduler") as span:
        span.set_attribute("hivemind.checks.done", sum(1 for o in decided.values() if o.status == "done"))
    return {"outcomes": {scheduler.key(case.case_id, tid): o for tid, o in blocked.items()}} if blocked else {}


def reconcile_node(state: RunState, runtime: Run) -> dict:
    """c17: the fact sheet of the board, kept per round with the plan and what became of each check. its follow-ups
    go to the case lead only while the plan may still be revised, the stop condition is not met and the case is
    within its budget; over the budget, no model is asked and the sheet sends the case to a person."""
    ctx, case, plan = runtime.context, state["case"], state["plan"]
    decided = _outcomes_of(state, case.case_id)
    found = list(state.get("boards", {}).get(case.case_id, []))
    issues = len(state["intake"].issues)
    over = _budget_stop(state, ctx)
    may_follow_up = (ctx.case_lead and plan.replans < planner.replans_allowed(issues) and over is None
                     and not ctx.stop_signal(case.case_id).is_set())
    with ctx.tracer.capability("evidence_reconciler", uses_llm=over is None) as span:
        if over is None:
            planned = {t.capability for t in plan.tasks}
            scopes = set(state["identity"].scopes)
            addable = [c for c in ctx.checks.values() if c.name not in planned and c.scopes <= scopes]
            sheet = reconciler.reconcile(case.case_id, state["issue"], found, plan, decided, ctx.llm,
                                         complex_case=issues > 1, may_follow_up=may_follow_up, addable=addable)
        else:
            sheet = reconciler.from_code(found, plan, decided, reconciler.cross_checks(found), plan.replans + 1,
                                         case.case_id, f"no model asked: the case is {over}")
        span.set_attribute("hivemind.fact_sheet.verdict", sheet.verdict)
        span.set_attribute("hivemind.fact_sheet.source", sheet.source)
        span.set_attribute("hivemind.fact_sheet.round", sheet.round)
        span.set_attribute("hivemind.plan.follow_ups", len(sheet.follow_ups))
        span.set_attribute("hivemind.fact_sheet.dropped", len(sheet.dropped))
        set_output(span, sheet.model_dump(mode="json"))
    _keep_plan(state, ctx, plan, decided)
    ctx.cases.save_fact_sheet(_lease(ctx, case.case_id), sheet)
    return {"fact_sheet": sheet, "follow_ups": sheet.follow_ups or None}


def after_finish(state: RunState) -> str:
    plan = state["plan"]
    if state.get("follow_ups") and plan.replans < planner.replans_allowed(len(state["intake"].issues)):
        return "replan"
    return "decide"


def replan_node(state: RunState, runtime: Run) -> dict:
    """the case lead revises the plan for the reconciler's follow-ups; the validator lets in only new checks."""
    ctx, case, plan = runtime.context, state["case"], state["plan"]
    with ctx.tracer.capability("case_lead", uses_llm=True) as span:
        revised = planner.replan(case.case_type, state["issue"], plan, _outcomes_of(state, case.case_id),
                                 list(state.get("boards", {}).get(case.case_id, [])), state["follow_ups"],
                                 ctx.checks, set(state["identity"].scopes), ctx.llm,
                                 planner.replans_allowed(len(state["intake"].issues)))
        span.set_attribute("hivemind.plan.replans", revised.replans)
        span.set_attribute("hivemind.plan.checks", len(revised.tasks))
        if revised.note:
            span.set_attribute("hivemind.plan.note", revised.note)
        set_output(span, revised.model_dump(mode="json"))
    return {"plan": revised, "follow_ups": None}


# deciding and acting: the rules (c18), the router (c21), the resolution planner (c19), the gateway (c22)


def decide_node(state: RunState, runtime: Run) -> dict:
    """what the business's rules allow on the case, from the fact sheet (none for a cancellation), and how the case
    goes. the resolution planner proposes the way to pay only when support may act alone."""
    ctx, case, issue = runtime.context, state["case"], state["issue"]
    sheet = state.get("fact_sheet")
    board_now = list(state.get("boards", {}).get(case.case_id, []))
    several = len(state["intake"].issues) > 1
    runtime.stream_writer(progress_reporter.event("decision", "started"))
    with ctx.tracer.capability("policy_eligibility_engine") as span:
        if case.state != "DECIDING":
            case = case_manager.move(case, "DECIDING", ctx.backend, "checks_done" if sheet else "short_path")
        inputs = eligibility.inputs_for(issue, sheet, board_now)
        answer = eligibility.evaluate(case.case_id, inputs, ctx.backend)
        span.set_attribute("hivemind.rules_version", answer["rules_version"])
        set_output(span, answer)
    with ctx.tracer.capability("approval_router") as span:
        route, why = approval_router.route(answer, sheet, several, issue.money_to)
        span.set_attribute("hivemind.route", route)
        set_output(span, why)
    proposal = None
    if route == "auto":
        with ctx.tracer.capability("resolution_planner") as span:
            proposal = resolver.propose(answer, issue.money_to)
            span.set_attribute("hivemind.proposal.source", proposal.source)
            set_output(span, proposal.model_dump(mode="json"))
    runtime.stream_writer(progress_reporter.event("decision", "finished"))
    decision = Decision(route=route, why=why, inputs=inputs, eligibility=answer, proposal=proposal)
    return {"case": case, "decision": decision}


def after_decide(state: RunState) -> str:
    route = state["decision"].route
    return {"auto": "act", "person": "to_person"}.get(route, "explain")


def _decision_meta(state: RunState) -> dict:
    d = state["decision"]
    meta: dict = {"decision": {"route": d.route, "why": d.why, "rules_version": d.eligibility.get("rules_version"),
                               "parts": [{k: p.get(k) for k in ("part", "outcome", "max_amount_paise", "basis",
                                                                 "requires", "rule", "why")}
                                         for p in d.eligibility.get("parts", [])]}}
    if d.proposal is not None:
        meta["decision"]["proposal"] = d.proposal.model_dump(mode="json")
    sheet = state.get("fact_sheet")
    if sheet is not None:
        meta["fact_sheet"] = {"round": sheet.round, "verdict": sheet.verdict, "source": sheet.source}
    return meta


def _record(state: RunState) -> dict:
    return action_gateway.record(state["case"], state["issue"], state.get("fact_sheet"), state["decision"])


def _keep_outcome(state: RunState, runtime: Run, outcome: str, body: dict) -> dict:
    """the decision record of a case decided without moving money. when the business refuses it or does not answer,
    the reply still goes out and the decision stays with it in the chat's own record (what to add to its meta)."""
    ctx, case = runtime.context, state["case"]
    try:
        kept = action_gateway.keep_outcome(case.case_id, outcome, body, ctx.backend, ctx.rng, ctx.retry_sleep)
        return {"decision_record_id": kept["decision_record_id"]}
    except ActionRefused as exc:
        return {"record_refused": exc.detail}
    except action_gateway.StillPending as exc:
        return {"record_pending": str(exc)[:200]}


def _with_a_person(state: RunState, runtime: Run, extra: dict | None = None) -> dict:
    """the decision is kept, and the case waits for a person: neutral words, never why."""
    case = state["case"]
    kept = _keep_outcome(state, runtime, "to_a_person", {**_record(state), **(extra or {})})
    meta = {**_decision_meta(state), **kept}
    if extra:
        meta.update(extra)
    parts = state["decision"].eligibility.get("parts", [])
    if case.case_type == "cancellation" and any(p["rule"] == "cancel.packing_started" for p in parts):
        text = replies.cannot_cancel(case.order_id)
    else:
        text = replies.with_a_person(case.case_type)
    return {"piece": Reply(kind="waiting_for_person", text=text, case_state="AWAITING_HUMAN", meta=meta)}


def to_person_node(state: RunState, runtime: Run) -> dict:
    with runtime.context.tracer.capability("approval_router") as span:
        span.set_attribute("hivemind.reply.kind", "waiting_for_person")
        return _with_a_person(state, runtime)


def explain_node(state: RunState, runtime: Run) -> dict:
    """nothing is owed, or it was already refunded: the decision is kept and the reply says so, from a template."""
    ctx, case, decision = runtime.context, state["case"], state["decision"]
    parts = decision.eligibility.get("parts", [])
    outcome = "already_refunded" if decision.route == "already_refunded" else "nothing_owed"
    with ctx.tracer.capability("customer_communicator") as span:
        kept = _keep_outcome(state, runtime, outcome, _record(state))
        if outcome == "already_refunded":
            text = replies.already_refunded(case.order_id, sum(p["max_amount_paise"] for p in parts))
        elif case.case_type == "cancellation":
            text = replies.nothing_to_cancel(case.order_id, parts[0]["why"] if parts else "")
        else:
            text = replies.nothing_owed_payment(case.order_id)
        span.set_attribute("hivemind.reply.kind", outcome)
        return {"piece": Reply(kind=outcome, text=text, case_state="RESOLVED", meta={**_decision_meta(state), **kept})}


def act_node(state: RunState, runtime: Run) -> dict:
    """the action, under its key; then the reply says what was done. the business's own re-check can refuse it (a
    person decides), and a business that does not answer leaves the case waiting on the system."""
    ctx, case, decision = runtime.context, state["case"], state["decision"]
    runtime.stream_writer(progress_reporter.event("action", "started"))
    with ctx.tracer.capability("action_gateway") as span:
        case = case_manager.move(case, "ACTIONING", ctx.backend, "acting")
        state = {**state, "case": case}
        try:
            made = action_gateway.act(case.case_id, decision, _record(state), ctx.backend, ctx.retry_sleep)
        except ActionRefused as exc:
            span.set_attribute("hivemind.action.refused", exc.detail[:200])
            return {"case": case, **_with_a_person(state, runtime, {"refused": exc.detail})}
        except action_gateway.StillPending as exc:
            span.set_attribute("hivemind.action.pending", str(exc)[:200])
            return {"case": case, "piece": Reply(kind="pending_system", text=replies.PENDING_SYSTEM,
                                                 case_state="PENDING_SYSTEM", meta=_decision_meta(state))}
        span.set_attribute("hivemind.action.replayed", made["replayed"])
        set_output(span, made)
    runtime.stream_writer(progress_reporter.event("action", "finished"))
    meta = {**_decision_meta(state), "action": made, "decision_record_id": made["decision_record_id"]}
    if decision.proposal.kind == "cancel":
        return {"case": case, "piece": Reply(kind="cancelled", text=replies.cancelled(case.order_id, made["refund"]),
                                             case_state="RESOLVED", meta=meta)}
    parts = [p.model_dump() for p in decision.proposal.parts]
    with ctx.tracer.capability("customer_communicator", uses_llm=True) as span:
        text, how = communicator.compose_refund(state["guard"].sanitised_text, made, parts,
                                                decision.proposal.explanation_points, ctx.llm,
                                                ctx.out.writer(runtime.stream_writer))
        span.set_attribute("hivemind.reply.source", how["source"])
        set_output(span, text)
    kind = "wallet_credited" if made["destination"] == "wallet" else "refunded"
    return {"case": case, "piece": Reply(kind=kind, text=text, case_state="RESOLVED",
                                         meta={**meta, "communicator": how})}


def _reason(piece: Reply) -> str:
    if piece.kind == "waiting_for_person":
        if "decision" in piece.meta:  # the rules or the router sent it to a person
            return "refused_at_action" if "refused" in piece.meta else "decided_a_person"
        return "fact_sheet_" + piece.meta["fact_sheet"]["verdict"] if "fact_sheet" in piece.meta else "no_path_yet"
    if piece.kind in ("refunded", "wallet_credited", "cancelled"):
        return piece.kind
    if piece.kind in ("nothing_owed", "already_refunded", "pending_system"):
        return piece.kind
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
                ask = ask.model_copy(update={"text": replies.which_order(ask.meta["pending"]["case_type"])},
                                     deep=True)
                ask.meta["pending"]["question"] = ask.text
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
    graph.add_node("ask_again", ask_again_node)
    graph.add_node("triage", triage_node)
    graph.add_node("case_manager", case_manager_node)
    graph.add_node("order_status", order_status_node)
    graph.add_node("promotions_investigator", promotions_node)
    graph.add_node("policy_provider", policy_node)
    graph.add_node("communicator", communicator_node)
    graph.add_node("waiting_for_person", waiting_node)
    graph.add_node("plan", plan_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("finish_plan", finish_plan_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("replan", replan_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)
    graph.add_node("to_person", to_person_node)
    graph.add_node("explain", explain_node)
    graph.add_node("close_case", close_case_node)
    graph.add_node("reply", reply_node)

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "intake")
    graph.add_conditional_edges("intake", after_intake, ["ask_again", "fixed_reply", "triage"])
    graph.add_edge("fixed_reply", END)
    graph.add_edge("ask_again", END)
    graph.add_conditional_edges("triage", after_triage, ["case_manager", "triage", "reply"])
    graph.add_conditional_edges("case_manager", after_case_manager,
                                ["close_case", "order_status", "promotions_investigator", "plan", "decide",
                                 "waiting_for_person"])
    graph.add_conditional_edges("plan", after_plan, ["schedule", "waiting_for_person"])
    graph.add_conditional_edges("schedule", fan_out, ["investigate", "finish_plan"])
    graph.add_edge("investigate", "schedule")
    graph.add_edge("finish_plan", "reconcile")
    graph.add_conditional_edges("reconcile", after_finish, ["replan", "decide"])
    graph.add_edge("replan", "schedule")
    graph.add_conditional_edges("decide", after_decide, ["act", "to_person", "explain"])
    graph.add_edge("act", "close_case")
    graph.add_edge("to_person", "close_case")
    graph.add_edge("explain", "close_case")
    graph.add_edge("order_status", "close_case")
    graph.add_edge("promotions_investigator", "policy_provider")
    graph.add_edge("policy_provider", "communicator")
    graph.add_edge("communicator", "close_case")
    graph.add_edge("waiting_for_person", "close_case")
    graph.add_conditional_edges("close_case", lambda s: "triage" if s.get("todo") else "reply", ["triage", "reply"])
    graph.add_edge("reply", END)
    return graph.compile()
