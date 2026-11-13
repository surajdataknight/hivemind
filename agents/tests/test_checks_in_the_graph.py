"""the case lead, the scheduler and the checks inside the case graph, on the real backend and the test database. a
missing-item case runs its checks side by side and posts their findings with proof; it still waits for a person,
since no decision step exists until phase 3. the checks are stand-ins registered for the test (the registry is
empty in production until phase 2.4)."""

import pytest

from hivemind_agents import scheduler, settings
from hivemind_agents.backend_client import BackendError
from hivemind_agents.capabilities import registry as check_registry
from hivemind_agents.capabilities.planner import MINIMUM
from hivemind_agents.capabilities.registry import Capability
from hivemind_agents.case_store import CaseStore
from hivemind_agents.contracts import CheckResult, Finding, InfoRequest
from scripted import calls_of, draft, issue

pytestmark = [pytest.mark.db]

MISSING = MINIMUM["item_missing_or_wrong"]
NO_EXTRA_CHECKS = {"tasks": [], "stop_condition": "", "rationale_summary": "the minimum checks"}


def reading(name: str, fail: bool = False, asks=(), slow: bool = False) -> Capability:
    """a check that reads the case's order, and says what it read, with the read as its proof. `asks` names the
    checks it asks for; a slow check waits until it is told its answer is not wanted (or 5 s) before it reads."""

    def run(task, check):
        if fail:
            raise BackendError(f"{name}: http 500")
        if slow and check.stop.wait(5):
            return []
        body = check.backend.order(check.case.order_id)
        found = [Finding(case_id=check.case.case_id, task_id=task.task_id, capability=name,
                         type=f"{name}.checked", claim=f"{name} read the order", evidence_refs=[body["evidence_id"]],
                         confidence="high")]
        return CheckResult(findings=found, info_requests=[InfoRequest(capability=a, reason="need it") for a in asks])

    return Capability(name=name, description=f"reads the order for {name}", params=("order_id",),
                      scopes=frozenset({"orders:read:self"}), run=run)


@pytest.fixture
def checks(monkeypatch):
    def install(failing=(), extra=(), asks=None, slow=()):
        for name in [*MISSING, *extra]:
            monkeypatch.setitem(check_registry.CAPABILITIES, name,
                                reading(name, fail=name in failing, asks=(asks or {}).get(name, ()),
                                        slow=name in slow))

    return install


def findings_of(case_id: str) -> list[Finding]:
    store = CaseStore(settings.agent_conninfo())
    try:
        return store.findings(case_id)
    finally:
        store.close()


def say_missing_item(chat_as, data, model, plan, then=()):
    """the customer reports a missing item; the model answers the intake, then the plan, then `then` in order."""
    fault = data.fault("missing_item", "store_miss")
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=["butter"])],
                               sentiment="negative"))
    model.intakes.append(plan)
    model.intakes.extend(then)
    done = chat_as(fault["customer_id"]).say(f"Butter was missing from my order {fault['order_id']}. Refund please.")
    return done, fault


def span_named(spans, name):
    return [s for s in spans.get_finished_spans() if s.name == name]


def test_the_minimum_checks_run_and_their_findings_are_kept(chat_as, data, model, cases, checks, spans, live):
    checks()
    done, fault = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS)
    assert done["message"]["kind"] == "waiting_for_person"
    assert done["llm_calls"] == 3  # the intake, the case lead, then the reconciler
    (case,) = cases()
    assert (case["state"], case["order_id"]) == ("AWAITING_HUMAN", fault["order_id"])
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted(MISSING)
    assert span_named(spans, "case_lead")[0].attributes["hivemind.plan.checks"] == len(MISSING)


def test_a_check_that_keeps_failing_blocks_only_the_check_that_needs_it(chat_as, data, model, cases, checks, spans,
                                                                       monkeypatch, live):
    monkeypatch.setattr(scheduler, "backoff", lambda attempt, rng: 0.0)
    checks(failing=("customer_history",))
    done, _ = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS)
    assert done["message"]["kind"] == "waiting_for_person"
    (case,) = cases()
    assert {f.capability for f in findings_of(case["case_id"])} == {"order", "inventory_store", "delivery"}
    (history,) = span_named(spans, "check customer_history")
    assert history.attributes["hivemind.check.status"] == "error"
    assert history.attributes["hivemind.check.attempts"] == scheduler.READ_TRIES
    assert span_named(spans, "check risk") == []  # risk needs the history: blocked, never run
    assert span_named(spans, "scheduler")[-1].attributes["hivemind.checks.done"] == 3


def test_a_case_over_its_token_budget_runs_no_further_checks(chat_as, data, model, cases, checks, spans,
                                                              monkeypatch, live):
    monkeypatch.setattr(scheduler, "MAX_TOKENS", 100)  # the case lead's own call (460 tokens) already passes it
    checks()
    done, _ = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS)
    assert done["message"]["kind"] == "waiting_for_person"
    (case,) = cases()
    assert case["state"] == "AWAITING_HUMAN"
    assert findings_of(case["case_id"]) == []
    assert not [s for s in spans.get_finished_spans() if s.name.startswith("check ")]
    stopped = [s.attributes.get("hivemind.plan.stopped") for s in span_named(spans, "scheduler")]
    assert "over the budget of 100 tokens" in stopped


def test_a_model_plan_with_a_loop_and_another_order_adds_nothing_that_runs(chat_as, data, model, cases, checks,
                                                                           spans, live):
    checks(extra=("extra_a", "extra_b"))
    plan = {"tasks": [
        {"task_id": "t1", "capability": "extra_a", "params": {}, "depends_on": ["t2"], "reason": "a"},
        {"task_id": "t2", "capability": "extra_b", "params": {}, "depends_on": ["t1"], "reason": "b"},
        {"task_id": "t3", "capability": "order", "params": {"order_id": "ORD-999999"}, "depends_on": [],
         "reason": "someone else's order"},
    ], "stop_condition": "", "rationale_summary": "loop and another order"}
    done, _ = say_missing_item(chat_as, data, model, plan)
    assert done["message"]["kind"] == "waiting_for_person"
    assert done["llm_calls"] == 3
    planned = span_named(spans, "case_lead")[0].attributes
    assert planned["hivemind.plan.checks"] == len(MISSING)  # the loop and the other order are not run
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted(MISSING)


def outcome_spans(spans, name):
    return [s.attributes["hivemind.check.status"] for s in span_named(spans, f"check {name}")]


def test_a_check_may_ask_for_another_and_the_validator_lets_it_in_once(chat_as, data, model, cases, checks, spans,
                                                                       live):
    checks(extra=("bags",), asks={"delivery": ("bags", "order"), "inventory_store": ("bags",)})
    done, _ = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS)
    assert done["message"]["kind"] == "waiting_for_person"
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted([*MISSING, "bags"])
    assert outcome_spans(spans, "bags") == ["done"]  # asked for twice, run once; "order" again is not run
    assert len(span_named(spans, "check order")) == 1


def test_a_check_with_no_answer_in_time_is_recorded_and_the_rest_go_on(chat_as, data, model, cases, checks, spans,
                                                                       monkeypatch, live):
    monkeypatch.setattr(scheduler, "TASK_SECONDS", 0.3)
    monkeypatch.setattr(scheduler, "backoff", lambda attempt, rng: 0.0)
    checks(slow=("delivery",))
    done, _ = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS)
    assert done["message"]["kind"] == "waiting_for_person"
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted(set(MISSING) - {"delivery"})
    (delivery,) = span_named(spans, "check delivery")
    assert delivery.attributes["hivemind.check.status"] == "timeout"
    assert delivery.attributes["hivemind.check.attempts"] == scheduler.READ_TRIES


def test_the_stop_condition_cancels_extra_checks_never_a_minimum_check(chat_as, data, model, cases, checks, spans,
                                                                       service, live):
    checks(extra=("extra_slow", "extra_late"), slow=("extra_slow",))
    service.deps.stop_judge = lambda plan, found: any(f.capability == "order" for f in found)
    plan = {"tasks": [
        {"task_id": "t1", "capability": "extra_slow", "params": {}, "depends_on": [], "reason": "runs beside"},
        {"task_id": "t2", "capability": "extra_late", "params": {}, "depends_on": ["m5"], "reason": "after risk"},
    ], "stop_condition": "the order read settles it", "rationale_summary": "two extra checks"}
    done, _ = say_missing_item(chat_as, data, model, plan)
    assert done["message"]["kind"] == "waiting_for_person"
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted(MISSING)  # every minimum check ran
    assert outcome_spans(spans, "extra_slow") == ["cancelled"]  # stopped while running, by a sibling's finding
    assert span_named(spans, "check extra_late") == []           # never started
    assert any(s.attributes.get("hivemind.plan.stop_condition_met") for s in span_named(spans, "scheduler"))


def test_the_case_lead_revises_the_plan_once_for_the_reconcilers_follow_ups(chat_as, data, model, cases, checks,
                                                                            spans, live):
    checks(extra=("bags",))
    model.reconciles.append({"facts": [{"statement": "the order was read", "cites": ["F1"]}], "claims": [],
                             "contradictions": [], "gaps": [{"about": "butter", "why": "nothing shows the bag"}],
                             "follow_ups": [{"check": "bags", "why": "was the butter in the bag at handover?"}],
                             "verdict": "sufficient_with_uncertainty"})
    revision = {"tasks": [{"task_id": "t1", "capability": "bags", "params": {}, "depends_on": [],
                           "reason": "count the bags", "expects": "one bag short"}],
                "stop_condition": "", "rationale_summary": "the gap on butter"}
    done, _ = say_missing_item(chat_as, data, model, NO_EXTRA_CHECKS, then=[revision])
    assert done["message"]["kind"] == "waiting_for_person"
    # the intake, the case lead, the reconciler, one revision, the reconciler again (no follow-ups allowed now)
    assert done["llm_calls"] == 5
    first, second = calls_of(model, "reconcile")
    assert "you may ask for up to 3 more checks" in first["prompt"] and "- bags:" in first["prompt"]
    assert "no follow-ups are allowed now" in second["prompt"]
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted([*MISSING, "bags"])
    assert "bags: was the butter in the bag at handover?" in calls_of(model, "generate")[2]["prompt"]
    assert [s.attributes.get("hivemind.plan.replans") for s in span_named(spans, "case_lead")] == [None, 1]
    sheets = sheets_of(case["case_id"])
    assert [(x.round, x.follow_ups) for x in sheets] == [(1, ["bags: was the butter in the bag at handover?"]),
                                                         (2, [])]


def sheets_of(case_id: str):
    store = CaseStore(settings.agent_conninfo())
    try:
        return store.fact_sheets(case_id)
    finally:
        store.close()


def test_with_the_case_lead_switched_off_only_the_minimum_plan_runs(chat_as, data, model, cases, checks, spans,
                                                                     service):
    """blueprint b.6's "minimum plan only" run: no case lead call, no revision, the same minimum checks."""
    checks()
    service.deps.case_lead = False
    model.reconciles.append({"facts": [], "claims": [], "contradictions": [], "gaps": [],
                             "follow_ups": [{"check": "payment", "why": "ignored: no revision without the case lead"}],
                             "verdict": "sufficient_with_uncertainty"})
    fault = data.fault("missing_item", "store_miss")
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=["butter"])]))
    done = chat_as(fault["customer_id"]).say(f"Butter was missing from my order {fault['order_id']}.")
    assert done["message"]["kind"] == "waiting_for_person"
    assert done["llm_calls"] == 2  # the intake and the reconciler
    (lead,) = span_named(spans, "case_lead")
    assert lead.attributes["hivemind.plan.source"] == "minimum"
    (case,) = cases()
    assert sorted(f.capability for f in findings_of(case["case_id"])) == sorted(MISSING)
    (sheet,) = sheets_of(case["case_id"])
    assert sheet.follow_ups == []
