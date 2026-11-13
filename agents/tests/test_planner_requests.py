"""the validator on what comes after the first plan (c5, c6): a check asking for another (info requests: depth,
loops, duplicates, the budget) and the case lead revising the plan for the reconciler's follow-ups (only new checks,
at most one revision, two in a complex case). no database; the model is the scripted one."""

import random
from graphlib import TopologicalSorter

import pytest

from hivemind_agents import settings
from hivemind_agents.capabilities import planner
from hivemind_agents.capabilities.identity import SCOPES
from hivemind_agents.capabilities.registry import Capability
from hivemind_agents.contracts import Finding, InfoRequest, Issue, PlanV1, TaskOutcome, TaskSpec
from hivemind_agents.llm import ModelRouter
from scripted import ScriptedModel

ORDER = "ORD-100001"
MINIMUM_NAMES = list(planner.MINIMUM["item_missing_or_wrong"])
SCOPE = set(SCOPES)


def issue() -> Issue:
    return Issue(case_type="item_missing_or_wrong", order_id=ORDER, items_mentioned=["butter"])


def cap(name, scopes=("orders:read:self",), cost="low") -> Capability:
    return Capability(name=name, description=f"checks {name}", params=("order_id",), scopes=frozenset(scopes),
                      run=lambda task, check: [], cost=cost)


def registry(*extra: str) -> dict[str, Capability]:
    return {name: cap(name) for name in [*MINIMUM_NAMES, *extra]}


def minimum_plan(reg=None):
    return planner.validate("item_missing_or_wrong", issue(), None, reg or registry("bags", "weights"), SCOPE)


def asks(*names: str) -> list[InfoRequest]:
    return [InfoRequest(capability=n, reason=f"need {n}") for n in names]


def task_of(plan, task_id):
    return next(t for t in plan.tasks if t.task_id == task_id)


def router(model: ScriptedModel, tracer) -> ModelRouter:
    return ModelRouter(settings.load_models(), tracer, {"google": lambda: model, "groq": lambda: model}, busy_waits=(),
                       sleep=lambda seconds: None)


class NoModel:
    def structured(self, *args, **kwargs):
        raise AssertionError("the model was asked, but no revision is left")


# the task envelope


def test_the_plans_checks_are_depth_one_and_say_who_asked_for_them(model, tracer):
    model.intakes.append({"tasks": [{"task_id": "t1", "capability": "bags", "params": {}, "depends_on": [],
                                     "reason": "count the bags", "expects": "fewer bags than packed"}],
                          "stop_condition": "", "rationale_summary": ""})
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", registry("bags"), SCOPE, router(model, tracer))
    assert {t.requested_by for t in plan.tasks if t.mandatory} == {"minimum"}
    added = task_of(plan, "t1")
    assert (added.requested_by, added.depth, added.expects) == ("case_lead", 1, "fewer bags than packed")
    assert all(t.depth == 1 for t in plan.tasks)
    assert added.dedupe_key == f"bags:order_id={ORDER}"


def test_the_case_lead_sees_each_checks_cost():
    reg = registry("bags")
    reg["photos"] = cap("photos", cost="medium")
    prompt = planner.build_prompt("item_missing_or_wrong", issue(), "text", reg, minimum_plan(reg).tasks)
    assert "photos: checks photos (params: order_id; cost: medium)" in prompt
    assert "expects" in prompt


# info requests


def test_a_check_may_ask_for_another_and_it_runs_one_level_deeper():
    plan = minimum_plan()
    delivery = task_of(plan, "m3")
    grown = planner.admit(issue(), plan, delivery, asks("bags"), registry("bags", "weights"), SCOPE)
    (new,) = [t for t in grown.tasks if t.task_id not in {t.task_id for t in plan.tasks}]
    assert (new.task_id, new.capability, new.params) == ("m3.r1", "bags", {"order_id": ORDER})
    assert (new.requested_by, new.depth, new.mandatory, new.depends_on) == ("m3", 2, False, [])


def test_a_check_that_was_asked_for_cannot_ask_for_more():
    reg = registry("bags", "weights")
    plan = planner.admit(issue(), minimum_plan(), task_of(minimum_plan(), "m3"), asks("bags"), reg, SCOPE)
    deeper = planner.admit(issue(), plan, task_of(plan, "m3.r1"), asks("weights"), reg, SCOPE)
    assert len(deeper.tasks) == len(plan.tasks)
    assert deeper.dropped[-1] == ("m3.r1 asked for weights",
                                  f"asked for by a check that was itself asked for (deeper than {planner.MAX_DEPTH})")


def test_a_check_cannot_ask_for_itself():
    plan = minimum_plan(registry("bags"))
    grown = planner.admit(issue(), plan, task_of(plan, "m3"), asks("delivery"), registry("bags"), SCOPE)
    assert grown.tasks == plan.tasks
    assert "a loop" in grown.dropped[-1][1]


def test_a_check_already_in_the_plan_is_not_run_twice():
    plan = minimum_plan()
    grown = planner.admit(issue(), plan, task_of(plan, "m3"), asks("order"), registry("bags", "weights"), SCOPE)
    assert grown.tasks == plan.tasks
    assert grown.dropped[-1] == ("m3 asked for order", "same check as m1: its findings are the case's already")


def test_two_checks_asking_for_the_same_check_run_it_once():
    reg = registry("bags", "weights")
    plan = planner.admit(issue(), minimum_plan(), task_of(minimum_plan(), "m3"), asks("bags"), reg, SCOPE)
    again = planner.admit(issue(), plan, task_of(plan, "m1"), asks("bags"), reg, SCOPE)
    assert [t.capability for t in again.tasks].count("bags") == 1
    assert "same check as m3.r1" in again.dropped[-1][1]


@pytest.mark.parametrize("name, reg, why", [
    ("ghost", registry(), "not a registered check"),
    ("pays", {**registry(), "pays": cap("pays", scopes=("media:read:self",))}, "needs a scope"),
])
def test_a_request_the_rules_refuse_is_kept_with_its_reason(name, reg, why):
    plan = minimum_plan(reg)
    grown = planner.admit(issue(), plan, task_of(plan, "m1"), asks(name), reg, SCOPE)
    assert grown.tasks == plan.tasks
    assert why in grown.dropped[-1][1]


def test_requests_stop_at_the_budget_of_twelve_checks():
    extra = [f"x{i}" for i in range(10)]
    reg = registry(*extra)
    plan = minimum_plan(reg)  # 5 minimum checks
    grown = planner.admit(issue(), plan, task_of(plan, "m1"), asks(*extra), reg, SCOPE)
    assert len(grown.tasks) == planner.MAX_TASKS
    assert sum(1 for _, why in grown.dropped if "over the budget of 12" in why) == 3


# the case lead revises the plan


def board() -> list[Finding]:
    return [Finding(case_id="CASE-1", task_id="m2", capability="inventory_store", type="inventory.scan",
                    claim="butter was scanned at packing", evidence_refs=["ev_1"], confidence="high")]


def outcomes(plan) -> dict[str, TaskOutcome]:
    return {t.task_id: TaskOutcome(task_id=t.task_id, status="done") for t in plan.tasks}


def revise(plan, llm, reg=None, allowed=1):
    return planner.replan("item_missing_or_wrong", issue(), plan, outcomes(plan), board(),
                          ["was butter in the bag at handover?"], reg or registry("bags", "weights"), SCOPE, llm,
                          allowed)


def test_a_revision_adds_new_checks_and_keeps_every_check_already_there(model, tracer):
    plan = minimum_plan()
    model.intakes.append({"tasks": [{"task_id": "t1", "capability": "bags", "params": {}, "depends_on": ["m3"],
                                     "reason": "bag count", "expects": "one bag short"}],
                          "stop_condition": "bags agree", "rationale_summary": "gap on butter"})
    revised = revise(plan, router(model, tracer))
    assert revised.tasks[:len(plan.tasks)] == plan.tasks
    new = task_of(revised, "t1")
    assert (new.capability, new.requested_by, new.depth, new.depends_on) == ("bags", "replan", 1, ["m3"])
    assert (revised.replans, revised.stop_condition) == (1, "bags agree")


def test_the_revision_prompt_quotes_the_findings_and_the_follow_ups_as_data(model, tracer):
    plan = minimum_plan()
    model.intakes.append({"tasks": [], "stop_condition": "", "rationale_summary": ""})
    revise(plan, router(model, tracer))
    prompt = model.calls[-1]["prompt"]
    assert "- m1: order (done)" in prompt
    assert '"""\n- inventory_store: butter was scanned at packing (confidence high)\n"""' in prompt
    assert '"""\n- was butter in the bag at handover?\n"""' in prompt
    assert "- bags: checks bags" in prompt and "- order: checks order" not in prompt  # only checks not in the plan


def test_a_revision_cannot_remove_change_or_repeat_a_check(model, tracer):
    first = PlanV1(tasks=[TaskSpec(task_id="t1", capability="bags")])
    plan = planner.validate("item_missing_or_wrong", issue(), first, registry("bags", "weights"), SCOPE)
    model.intakes.append({"tasks": [
        {"task_id": "t1", "capability": "weights", "params": {}, "depends_on": [], "reason": "a taken id"},
        {"task_id": "t2", "capability": "order", "params": {}, "depends_on": [], "reason": "again"},
        {"task_id": "t3", "capability": "weights", "params": {"order_id": "ORD-999999"}, "depends_on": [],
         "reason": "another order"},
    ], "stop_condition": "", "rationale_summary": ""})
    revised = revise(plan, router(model, tracer))
    assert revised.tasks == plan.tasks
    reasons = dict(revised.dropped)
    assert reasons["t1"] == "repeats the id of an earlier check"
    assert reasons["t2"] == "same check as m1"
    assert reasons["t3"] == "another order than the one in this issue"


def test_over_its_revisions_the_plan_stays_as_it_is_with_no_model_call():
    plan = minimum_plan().model_copy(update={"replans": 1})
    revised = revise(plan, NoModel(), allowed=1)
    assert revised.tasks == plan.tasks and revised.replans == 1
    assert revised.dropped[-1] == ("replan", "over the budget of 1 revisions")


def test_a_revision_with_no_usable_answer_keeps_the_plan_and_says_why(model, tracer):
    plan = minimum_plan()
    model.intakes.extend(["not json", "still not json"])
    revised = revise(plan, router(model, tracer))
    assert revised.tasks == plan.tasks and revised.replans == 1
    assert revised.note.startswith("no usable plan from the case lead")


def test_a_complex_case_may_be_revised_twice_a_standard_one_once():
    assert planner.replans_allowed(1) == 1
    assert planner.replans_allowed(2) == 2
    assert planner.replans_allowed(3) == 2


def test_a_revision_keeps_within_twelve_checks(model, tracer):
    extra = [f"x{i}" for i in range(10)]
    reg = registry(*extra)
    plan = minimum_plan(reg)
    model.intakes.append({"tasks": [{"task_id": f"t{i + 1}", "capability": n, "params": {}, "depends_on": [],
                                     "reason": ""} for i, n in enumerate(extra)],
                          "stop_condition": "", "rationale_summary": ""})
    revised = revise(plan, router(model, tracer), reg=reg)
    assert len(revised.tasks) == planner.MAX_TASKS


# every sequence of requests and revisions (seeded)


@pytest.mark.parametrize("seed", range(200))
def test_random_requests_and_revisions_never_break_the_rules(seed, tracer):
    rng = random.Random(seed)
    extra = [f"x{i}" for i in range(8)]
    reg = registry(*extra)
    reg["pays"] = cap("pays", scopes=("media:read:self",))
    pool = list(reg) + ["ghost"]
    plan = minimum_plan(reg)
    for _ in range(rng.randint(1, 8)):
        if rng.random() < 0.7:
            requester = rng.choice(plan.tasks)
            plan = planner.admit(issue(), plan, requester, asks(*rng.sample(pool, rng.randint(1, 4))), reg, SCOPE)
        else:
            model = ScriptedModel()
            ids = [t.task_id for t in plan.tasks]
            model.intakes.append(PlanV1(tasks=[
                TaskSpec(task_id=f"t{rng.randint(1, 15)}", capability=rng.choice(pool),
                         depends_on=[rng.choice(ids + ["t1", "t2", "t9"]) for _ in range(rng.randint(0, 2))])
                for _ in range(rng.randint(0, 6))]).model_dump())
            plan = planner.replan("item_missing_or_wrong", issue(), plan, {}, [], ["more"], reg, SCOPE,
                                  router(model, tracer), allowed=2)

    ids = [t.task_id for t in plan.tasks]
    by_id = {t.task_id: t for t in plan.tasks}
    assert len(plan.tasks) <= planner.MAX_TASKS
    assert len(set(ids)) == len(ids)
    assert plan.replans <= 2
    assert len({t.dedupe_key for t in plan.tasks}) == len(plan.tasks)       # no check twice
    assert {t.capability for t in plan.tasks if t.mandatory} == set(MINIMUM_NAMES)  # the minimum stays whole
    for task in plan.tasks:
        assert task.capability in reg and task.capability != "pays"
        assert task.params == {"order_id": ORDER}
        assert set(task.depends_on) <= set(ids)
        assert 1 <= task.depth <= planner.MAX_DEPTH
        if task.depth == 2:
            assert by_id[task.requested_by].depth == 1                        # asked for by a plan check
            assert by_id[task.requested_by].capability != task.capability    # never itself
    TopologicalSorter({t.task_id: set(t.depends_on) for t in plan.tasks}).prepare()  # raises on a loop


# the fallback model


class BusyOnce(ScriptedModel):
    """says busy on the case lead's own model, then answers on any other."""

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        if model == settings.load_models().job("case_lead").model:
            from hivemind_agents.llm import ProviderBusy

            self.calls.append({"model": model})
            raise ProviderBusy("ServerError: 503 UNAVAILABLE high demand")
        return super().generate(model, system, prompt, temperature, schema, timeout_s)


def test_a_plan_from_the_fallback_model_is_used_and_says_so(tracer):
    model = BusyOnce()
    model.intakes.append({"tasks": [{"task_id": "t1", "capability": "bags", "params": {}, "depends_on": [],
                                     "reason": "count the bags"}], "stop_condition": "", "rationale_summary": ""})
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", registry("bags"), SCOPE, router(model, tracer))
    assert plan.source == "planner" and "bags" in [t.capability for t in plan.tasks]
    lead = settings.load_models().job("case_lead")
    assert [c["model"] for c in model.calls] == [lead.model, lead.fallback]
    assert plan.note == (f"the case lead's plan came from {lead.fallback_provider} {lead.fallback}: "
                         f"{lead.model} was busy")


def test_a_plan_from_the_case_leads_own_model_has_no_note(model, tracer):
    model.intakes.append({"tasks": [], "stop_condition": "", "rationale_summary": ""})
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", registry(), SCOPE, router(model, tracer))
    assert plan.source == "planner" and plan.note == ""
