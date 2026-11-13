"""the dag scheduler (c7) on its own: which checks are ready or blocked, the case's time budget, the retries for
read-only checks, and the waves of at most five checks side by side."""

import random

import pytest
from langgraph.types import Send

from hivemind_agents import graph, scheduler
from hivemind_agents.backend_client import BackendError
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CaseRef, Finding, IntakeV1, Issue, Plan, Task, TaskOutcome


def task(i: int, deps=()) -> Task:
    return Task(task_id=f"t{i}", capability="order", params={"order_id": "ORD-100001"}, depends_on=list(deps),
                mandatory=False)


def done(task_id: str) -> TaskOutcome:
    return TaskOutcome(task_id=task_id, status="done", attempts=1)


def failed(task_id: str) -> TaskOutcome:
    return TaskOutcome(task_id=task_id, status="error", error="http 500", attempts=3)


CASE = CaseRef(case_id="CASE-1", case_type="item_missing_or_wrong", order_id="ORD-100001", state="INVESTIGATING",
               opened=False)


def check_input() -> CheckInput:
    return CheckInput(case=CASE, backend=None)


def capability(run, read_only=True) -> Capability:
    return Capability(name="order", description="reads the order", params=("order_id",),
                      scopes=frozenset({"orders:read:self"}), run=run, read_only=read_only)


def finding(for_task: Task, case_id: str = "CASE-1") -> Finding:
    return Finding(case_id=case_id, task_id=for_task.task_id, capability="order", type="order.checked",
                   claim="the order was read", evidence_refs=["ev_1"], confidence="high")


# which checks are ready, and which are blocked


def test_the_first_wave_is_the_checks_with_nothing_to_wait_for():
    tasks = [task(1), task(2, ["t1"]), task(3)]
    assert [t.task_id for t in scheduler.ready(tasks, {})] == ["t1", "t3"]


def test_a_check_waits_until_every_check_it_needs_is_done():
    tasks = [task(1), task(2, ["t1"])]
    assert [t.task_id for t in scheduler.ready(tasks, {"t1": done("t1")})] == ["t2"]
    assert scheduler.ready(tasks, {"t1": TaskOutcome(task_id="t1", status="blocked")}) == []


def test_a_check_after_a_failed_one_is_blocked_and_the_chain_follows():
    tasks = [task(1), task(2, ["t1"]), task(3, ["t2"])]
    outcomes = {"t1": failed("t1")}
    assert [t.task_id for t in scheduler.blocked(tasks, outcomes)] == ["t2"]
    outcomes["t2"] = TaskOutcome(task_id="t2", status="blocked", error="x")
    assert [t.task_id for t in scheduler.blocked(tasks, outcomes)] == ["t3"]


def test_a_check_not_yet_decided_is_not_blocked_just_because_it_waits():
    tasks = [task(1), task(2, ["t1"])]
    assert scheduler.blocked(tasks, {}) == []


# the case's time budget


def test_a_case_is_out_of_time_only_after_its_budget():
    assert not scheduler.out_of_time(0.0, scheduler.CASE_SECONDS)
    assert scheduler.out_of_time(0.0, scheduler.CASE_SECONDS + 0.5)
    assert not scheduler.out_of_time(None, 10_000.0)


def test_the_budget_stops_further_waves_for_time_calls_or_tokens():
    assert scheduler.over_budget(0.0, 1.0, scheduler.MAX_MODEL_CALLS, scheduler.MAX_TOKENS) is None
    assert scheduler.over_budget(0.0, scheduler.CASE_SECONDS + 1, 0, 0) == "out of time"
    assert "model calls" in scheduler.over_budget(0.0, 1.0, scheduler.MAX_MODEL_CALLS + 1, 0)
    assert "tokens" in scheduler.over_budget(0.0, 1.0, 0, scheduler.MAX_TOKENS + 1)


# retries


@pytest.mark.parametrize("seed", range(20))
def test_a_retry_waits_with_jitter_that_doubles(seed):
    rng = random.Random(seed)
    assert 0.25 <= scheduler.backoff(1, rng) <= 0.75
    assert 0.5 <= scheduler.backoff(2, rng) <= 1.5


def test_a_read_that_keeps_failing_is_tried_three_times_then_recorded():
    calls, waits = [], []

    def run(t, check):
        calls.append(t.task_id)
        raise BackendError("order: http 500")

    outcome = scheduler.run_task(task(1), capability(run), check_input(), waits.append, random.Random(0))
    assert outcome.status == "error" and outcome.attempts == scheduler.READ_TRIES
    assert len(calls) == scheduler.READ_TRIES and len(waits) == scheduler.READ_TRIES - 1


def test_a_read_that_recovers_is_done():
    attempts = []

    def run(t, check):
        attempts.append(1)
        if len(attempts) == 1:
            raise BackendError("order: http 503")
        return [finding(t)]

    outcome = scheduler.run_task(task(1), capability(run), check_input(), lambda s: None, random.Random(0))
    assert outcome.status == "done" and outcome.attempts == 2 and len(outcome.findings) == 1


def test_a_check_that_is_not_read_only_is_not_retried():
    waits = []

    def run(t, check):
        raise BackendError("order: http 500")

    outcome = scheduler.run_task(task(1), capability(run, read_only=False), check_input(), waits.append,
                                 random.Random(0))
    assert outcome.status == "error" and outcome.attempts == 1 and waits == []


def test_a_bug_in_a_check_is_not_swallowed():
    def run(t, check):
        raise RuntimeError("a bug")

    with pytest.raises(RuntimeError):
        scheduler.run_task(task(1), capability(run), check_input(), lambda s: None, random.Random(0))


def test_a_finding_for_another_check_or_case_is_refused():
    def run(t, check):
        return [finding(task(9))]

    outcome = scheduler.run_task(task(1), capability(run), check_input(), lambda s: None, random.Random(0))
    assert outcome.status == "error" and "another check or case" in outcome.error


# the waves: at most five checks side by side


def plan_of(tasks) -> Plan:
    return Plan(source="planner", tasks=tasks)


def test_a_wave_is_at_most_five_checks_and_the_next_wave_takes_the_rest():
    tasks = [task(i) for i in range(1, 8)]
    state = {"plan": plan_of(tasks), "case": CASE, "outcomes": {}}
    first = graph.fan_out(state)
    assert len(first) == scheduler.MAX_PARALLEL
    assert all(isinstance(s, Send) and s.node == "investigate" for s in first)

    state["outcomes"] = {f"CASE-1/{s.arg['task'].task_id}": done(s.arg["task"].task_id) for s in first}
    second = graph.fan_out(state)
    assert [s.arg["task"].task_id for s in second] == ["t6", "t7"]


def test_when_no_check_is_ready_the_plan_finishes():
    tasks = [task(1), task(2, ["t1"])]
    state = {"plan": plan_of(tasks), "case": CASE, "outcomes": {"CASE-1/t1": failed("t1")}}
    assert graph.fan_out(state) == "finish_plan"


# after the checks: a revision of the plan, or waiting for a person


def intake_of(issues: int) -> IntakeV1:
    return IntakeV1(scope="in_scope", source="llm",
                    issues=[Issue(case_type="item_missing_or_wrong", order_id="ORD-100001")] * issues)


@pytest.mark.parametrize("issues, replans, expected", [
    (1, 0, "replan"), (1, 1, "waiting_for_person"),
    (2, 1, "replan"), (2, 2, "waiting_for_person"),
])
def test_a_standard_case_is_revised_once_a_complex_one_twice(issues, replans, expected):
    state = {"plan": Plan(source="planner", replans=replans), "follow_ups": ["more on the bag"],
             "intake": intake_of(issues)}
    assert graph.after_finish(state) == expected


def test_with_no_follow_ups_the_case_waits_for_a_person():
    state = {"plan": Plan(source="planner"), "follow_ups": None, "intake": intake_of(1)}
    assert graph.after_finish(state) == "waiting_for_person"
