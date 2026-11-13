"""the case lead and the plan validator (c5, c6), on their own: the minimum checks, what the model may and may not
change, every refusal and its reason, the budget, loops, and the fallbacks when the model gives no usable plan.
no database; the model is the scripted one (or a stand-in that must not be asked)."""

import random
from graphlib import TopologicalSorter

import pytest

from hivemind_agents import settings
from hivemind_agents.capabilities import planner
from hivemind_agents.capabilities.identity import SCOPES
from hivemind_agents.capabilities.registry import Capability
from hivemind_agents.contracts import Issue, PlanV1, TaskParams, TaskSpec
from hivemind_agents.llm import ModelRouter
from scripted import ScriptedModel

ORDER = "ORD-100001"
MINIMUM_NAMES = list(planner.MINIMUM["item_missing_or_wrong"])  # order, inventory_store, delivery, history, risk
SCOPE = set(SCOPES)


def issue(order_id=ORDER, coupon_code=None) -> Issue:
    return Issue(case_type="item_missing_or_wrong", order_id=order_id, items_mentioned=["butter"],
                 coupon_code=coupon_code)


def cap(name, params=("order_id",), scopes=("orders:read:self",)) -> Capability:
    return Capability(name=name, description=f"checks {name}", params=params, scopes=frozenset(scopes),
                      run=lambda task, check: [])


def registry(*extra: str) -> dict[str, Capability]:
    reg = {name: cap(name) for name in MINIMUM_NAMES}
    reg.update({name: cap(name) for name in extra})
    return reg


def names(plan) -> list[str]:
    return [t.capability for t in plan.tasks]


def refused(plan, why: str) -> list[str]:
    return [task_id for task_id, reason in plan.dropped if why in reason]


class NoModel:
    """stands in for the model where it must not be asked."""

    def structured(self, *args, **kwargs):
        raise AssertionError("the model was asked, but the minimum checks cannot all run")


def router(model: ScriptedModel, tracer) -> ModelRouter:
    return ModelRouter(settings.load_models(), tracer, {"google": lambda: model, "groq": lambda: model}, busy_waits=(),
                       sleep=lambda seconds: None)


# the minimum plan


def test_the_minimum_plan_runs_with_no_model():
    plan = planner.validate("item_missing_or_wrong", issue(), None, registry(), SCOPE)
    assert plan.source == "minimum" and plan.runnable
    assert names(plan) == MINIMUM_NAMES
    assert all(t.mandatory and t.params == {"order_id": ORDER} for t in plan.tasks)


def test_risk_waits_for_the_customer_history_and_the_order():
    plan = planner.validate("item_missing_or_wrong", issue(), None, registry(), SCOPE)
    history = next(t for t in plan.tasks if t.capability == "customer_history")
    order = next(t for t in plan.tasks if t.capability == "order")
    risk = next(t for t in plan.tasks if t.capability == "risk")
    assert risk.depends_on == [history.task_id, order.task_id]


def test_without_an_order_check_risk_waits_for_the_history_only():
    reg = {**registry(), "payment": cap("payment")}
    plan = planner.validate("payment_failed", issue(), None, reg, SCOPE)
    risk = next(t for t in plan.tasks if t.capability == "risk")
    assert [next(t.capability for t in plan.tasks if t.task_id == d) for d in risk.depends_on] == ["customer_history"]


def test_the_model_adds_checks():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="promo", params={"order_id": ORDER})])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("promo"), SCOPE)
    assert plan.source == "planner"
    assert names(plan) == MINIMUM_NAMES + ["promo"]
    assert not plan.tasks[-1].mandatory


def test_the_model_cannot_remove_a_minimum_check():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="delivery", params={"order_id": ORDER})])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry(), SCOPE)
    assert names(plan) == MINIMUM_NAMES  # the model's copy of a minimum check is the same check, not an extra


def test_an_empty_model_plan_still_runs_every_minimum_check():
    plan = planner.validate("item_missing_or_wrong", issue(), PlanV1(), registry(), SCOPE)
    assert names(plan) == MINIMUM_NAMES and plan.source == "planner"


# what the model may not change


@pytest.mark.parametrize("task, why", [
    (TaskSpec(task_id="t1", capability="ghost"), "not a registered check"),
    (TaskSpec(task_id="t1", capability="promo", params={"order_id": "ORD-999999"}), "another order"),
    (TaskSpec(task_id="t1", capability="pays"), "needs a scope the case token does not carry"),
])
def test_a_refused_check_stays_on_the_plan_with_its_reason(task, why):
    reg = registry("promo")
    reg["pays"] = cap("pays", scopes=("media:read:self",))
    plan = planner.validate("item_missing_or_wrong", issue(), PlanV1(tasks=[task]), reg, SCOPE)
    assert refused(plan, why) == ["t1"]
    assert "t1" not in [t.task_id for t in plan.tasks]
    assert names(plan) == MINIMUM_NAMES


def test_a_param_other_than_order_and_code_cannot_even_be_written():
    assert TaskSpec(task_id="t1", capability="promo", params={"amount": "500"}).params == TaskParams()


def test_an_order_id_in_a_check_is_always_the_issues_order():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="promo", params={"order_id": ORDER})])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("promo"), SCOPE)
    promo = next(t for t in plan.tasks if t.capability == "promo")
    assert promo.params == {"order_id": ORDER}


def test_a_code_always_comes_from_the_issue_not_the_model():
    reg = registry()
    reg["coupon"] = cap("coupon", params=("order_id", "code"))
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="coupon", params={"code": "EVIL"})])
    plan = planner.validate("item_missing_or_wrong", issue(coupon_code="SAVE10"), proposed, reg, SCOPE)
    assert next(t for t in plan.tasks if t.capability == "coupon").params == {"order_id": ORDER, "code": "SAVE10"}


def test_a_check_with_no_code_in_the_issue_is_refused():
    reg = registry()
    reg["coupon"] = cap("coupon", params=("order_id", "code"))
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="coupon", params={"code": "EVIL"})])
    plan = planner.validate("item_missing_or_wrong", issue(coupon_code=None), proposed, reg, SCOPE)
    assert refused(plan, "no code in this issue") == ["t1"]


def test_a_missing_order_in_the_issue_makes_the_plan_not_runnable():
    plan = planner.validate("item_missing_or_wrong", issue(order_id=None), None, registry(), SCOPE)
    assert not plan.runnable
    assert set(plan.missing) == set(MINIMUM_NAMES)


def test_a_repeated_task_id_is_refused_so_two_checks_never_share_an_outcome():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="a"), TaskSpec(task_id="t1", capability="b")])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("a", "b"), SCOPE)
    assert names(plan) == MINIMUM_NAMES + ["a"]
    assert refused(plan, "repeats the id of an earlier check") == ["t1"]


def test_a_model_check_may_wait_for_a_minimum_check():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="promo", depends_on=["m1"])])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("promo"), SCOPE)
    assert next(t for t in plan.tasks if t.capability == "promo").depends_on == ["m1"]


def test_the_prompt_names_the_checks_that_run_and_offers_only_the_others():
    minimum = planner.validate("item_missing_or_wrong", issue(), None, registry("promo"), SCOPE).tasks
    prompt = planner.build_prompt("item_missing_or_wrong", issue(), "text", registry("promo"), minimum)
    running, offered = prompt.split("checks you may add:")
    assert "- m1: order" in running and "- m5: risk (after m4, m1)" in running
    assert "- promo:" in offered
    assert not any(f"- {name}:" in offered for name in MINIMUM_NAMES)


def test_a_duplicate_of_a_minimum_check_is_merged_into_it():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="order", params={"order_id": ORDER})])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry(), SCOPE)
    assert names(plan) == MINIMUM_NAMES
    assert refused(plan, "same check as m1") == ["t1"]


# loops, unknown dependencies, the budget


def test_a_loop_is_cut_and_the_checks_that_need_it_go_with_it():
    reg = registry("a", "b", "c")
    proposed = PlanV1(tasks=[
        TaskSpec(task_id="t1", capability="a", depends_on=["t2"]),
        TaskSpec(task_id="t2", capability="b", depends_on=["t1"]),
        TaskSpec(task_id="t3", capability="c", depends_on=["t1"]),
    ])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, reg, SCOPE)
    assert names(plan) == MINIMUM_NAMES
    assert refused(plan, "loop") == ["t1", "t2"]
    assert refused(plan, "needs a check that did not run") == ["t3"]


def test_a_check_that_depends_on_itself_is_a_loop():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="promo", depends_on=["t1"])])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("promo"), SCOPE)
    assert refused(plan, "loop") == ["t1"]


def test_a_check_that_needs_an_unknown_check_is_dropped():
    proposed = PlanV1(tasks=[TaskSpec(task_id="t1", capability="promo", depends_on=["t9"])])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, registry("promo"), SCOPE)
    assert refused(plan, "needs a check that did not run") == ["t1"]


def test_the_budget_is_twelve_checks_with_the_minimum_inside_it():
    reg = registry(*[f"x{i}" for i in range(20)])
    proposed = PlanV1(tasks=[TaskSpec(task_id=f"t{i}", capability=f"x{i - 1}") for i in range(1, 21)])
    plan = planner.validate("item_missing_or_wrong", issue(), proposed, reg, SCOPE)
    assert len(plan.tasks) == planner.MAX_TASKS
    assert set(MINIMUM_NAMES) <= set(names(plan))
    assert len(refused(plan, "over the budget")) == 20 - (planner.MAX_TASKS - len(MINIMUM_NAMES))


# when the minimum cannot run, or the model fails


def test_a_minimum_check_not_registered_means_no_model_call():
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", {}, SCOPE, NoModel())
    assert not plan.runnable and set(plan.missing) == set(MINIMUM_NAMES)


def test_a_minimum_check_the_token_cannot_use_means_no_model_call():
    reg = {name: cap(name) for name in MINIMUM_NAMES}
    reg["risk"] = cap("risk", scopes=("media:read:self",))
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", reg, SCOPE, NoModel())
    assert plan.missing == ["risk"]


def test_a_case_type_with_no_minimum_checks_is_not_runnable():
    plan = planner.plan_for("cancellation", issue(), "text", registry(), SCOPE, NoModel())
    assert not plan.runnable


def test_a_plan_the_model_gives_is_used(model: ScriptedModel, tracer):
    model.intakes.append({"tasks": [{"task_id": "t1", "capability": "promo", "params": {"order_id": ORDER},
                                     "depends_on": [], "reason": "a second source"}],
                          "stop_condition": "two sources agree", "rationale_summary": "short"})
    llm = router(model, tracer)
    plan = planner.plan_for("item_missing_or_wrong", issue(), "butter was missing", registry("promo"), SCOPE, llm)
    assert (llm.calls, llm.tokens) == (1, 460)  # the scripted model reports 400 in and 60 out
    assert plan.source == "planner" and "promo" in names(plan)
    assert plan.stop_condition == "two sources agree"
    prompt = model.calls[-1]["prompt"]
    assert "checks you may add" in prompt and ORDER in prompt
    assert '"""butter was missing"""' in prompt  # the customer's words, quoted as data


@pytest.mark.parametrize("answers", [["not json", "still not json"], []])
def test_a_model_plan_that_is_no_use_leaves_the_minimum_plan(model: ScriptedModel, tracer, answers):
    model.intakes.extend(answers)  # two bad answers, or none at all: the model call fails
    plan = planner.plan_for("item_missing_or_wrong", issue(), "text", registry("promo"), SCOPE, router(model, tracer))
    assert plan.source == "minimum" and plan.runnable
    assert names(plan) == MINIMUM_NAMES
    assert plan.note.startswith("no usable plan from the case lead")  # never a silent fallback


# every plan the model could write (seeded, so the same plans every run)


@pytest.mark.parametrize("seed", range(200))
def test_random_plans_never_break_the_rules(seed):
    rng = random.Random(seed)
    case_type = sorted(planner.MINIMUM)[seed % len(planner.MINIMUM)]  # every planned case type, 50 plans each
    minimum_names = set(planner.MINIMUM[case_type])
    reg = registry(*[f"x{i}" for i in range(6)], "payment")
    reg["pays"] = cap("pays", scopes=("media:read:self",))
    pool = list(reg) + ["ghost", "order", "risk"]
    case = issue(coupon_code=rng.choice([None, "SAVE10"]))
    tasks = []
    for i in range(rng.randint(0, 20)):
        params = {}
        if rng.random() < 0.3:
            params["order_id"] = rng.choice([ORDER, "ORD-999999"])
        if rng.random() < 0.2:
            params["amount"] = "5"
        if rng.random() < 0.3:
            params["code"] = "SAVE10"
        task_id = f"t{rng.randint(1, 20)}" if rng.random() < 0.2 else f"t{i + 1}"  # sometimes an id twice
        tasks.append(TaskSpec(task_id=task_id, capability=rng.choice(pool), params=params,
                              depends_on=[f"t{rng.randint(1, 20)}" for _ in range(rng.randint(0, 3))]))

    plan = planner.validate(case_type, case, PlanV1(tasks=tasks), reg, SCOPE)

    ids = [t.task_id for t in plan.tasks]
    assert len(plan.tasks) <= planner.MAX_TASKS
    assert len(set(ids)) == len(ids)
    for task in plan.tasks:
        assert set(task.depends_on) <= set(ids)               # nothing needs a check that did not run
        assert task.params.get("order_id", ORDER) == ORDER    # never another order
        assert "amount" not in task.params
        assert task.capability in reg
    TopologicalSorter({t.task_id: set(t.depends_on) for t in plan.tasks}).prepare()  # raises on a loop
    keys = [(t.capability, tuple(sorted(t.params.items()))) for t in plan.tasks]
    assert len(set(keys)) == len(keys)                       # no check twice
    covered = {t.capability for t in plan.tasks if t.mandatory}
    assert covered | set(plan.missing) == minimum_names       # each minimum check runs, or is named as missing
    assert plan.missing == []                                 # here every minimum check is registered and allowed
    assert plan.runnable == (not plan.missing)
