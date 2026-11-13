"""the scenarios, the harness, the grader and the baseline, with scripted models on the test database. a grader that
cannot fail is worth nothing, so each check is also shown failing."""

import pytest

from hivemind_agents.capabilities import triage
from hivemind_evals import baseline, harness
from hivemind_evals.customer import CustomerTurn, ScriptedCustomer, build_prompt
from hivemind_evals.grader import grade
from hivemind_evals.runner import _false_corroboration
from hivemind_evals.scenarios import SMALL, opening
from scripted import NO_EXTRA_CHECKS, draft, issue

pytestmark = pytest.mark.db


def sheet(claims=(), verdict="sufficient_with_uncertainty"):
    return {"facts": [], "claims": list(claims), "contradictions": [], "gaps": [], "follow_ups": [],
            "verdict": verdict}


def chat(s, service, shop, model, reconciled=None, day=None):
    """the scripted customer's chat; the model reads the message as a careful model would."""
    order = None if day else s.order_id
    model.intakes.append(draft(issues=[issue(s.case_type, order_id=order, items=[i["name"] for i in s.items],
                                             day=day)], sentiment="negative"))
    model.intakes.append(NO_EXTRA_CHECKS)
    if reconciled is not None:
        model.reconciles.append(reconciled)
    return harness.run_chat(s, ScriptedCustomer(s), service, shop)


# the scenarios


def test_the_small_set_has_one_scenario_per_fault_kind_one_named_by_its_day(small_set):
    assert [k.removesuffix("/by_day") for k in small_set] == [f"{t}/{v}" for t, v in SMALL]
    (by_day,) = [s for s in small_set.values() if s.names_order_by != "id"]
    assert by_day.scenario_id.endswith("/by_day") and by_day.case_type == "item_missing_or_wrong"
    for s in small_set.values():
        assert s.case_type and s.customer_id and s.order_id
        if s.case_type == "item_missing_or_wrong":
            assert s.items and all(i["name"] for i in s.items)


def test_what_the_customer_is_told_never_holds_the_cause(small_set):
    for s in small_set.values():
        prompt = build_prompt(s, [])
        assert s.facts.get("cause", "none") == "none" or s.facts["cause"] not in prompt
        assert "fraud" not in prompt and "ring" not in prompt.lower().replace("bring", "")


def test_the_day_word_names_the_order_the_way_triage_reads_it(small_set, shop):
    """the scenario's day must lead hivemind's own rule to the truth's order, or the scenario tests nothing."""
    s = next(s for s in small_set.values() if s.names_order_by != "id")
    token = shop.post("/v1/identity/sessions", json={"customer_id": s.customer_id}).json()["session_token"]
    from hivemind_agents import settings, tracing
    from hivemind_agents.backend_client import BackendClient
    from hivemind_agents.capabilities.identity import IdentityVerifier
    from hivemind_agents.contracts import Issue

    verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"), settings.signing_key("HM_CASE_TOKEN_SECRET"))
    customer_id, session_id = verifier.verify_session(token)
    _, case_token = verifier.mint_case_token(customer_id, session_id, "conv_check")
    backend = BackendClient(shop, case_token, tracing.Tracer(tracing.make_provider(None)))
    named = [i["name"] for i in s.items]  # the customer names the items, as the scenario's opening does
    result = triage.triage(Issue(case_type=s.case_type, order_day=s.names_order_by, items_mentioned=named),
                           triage.recent_orders(backend), backend)
    assert (result.issue.order_id, result.ask) == (s.order_id, None)


# the harness and the grader on a run that is right


def test_a_store_miss_found_and_corroborated_passes_every_check(small_set, service, shop, model, records):
    s = small_set["missing_item/store_miss"]
    t = chat(s, service, shop, model, sheet([{"about": "L1", "status": "corroborated", "cites": ["F1"]}]))
    assert t.error is None and t.questions == 0
    g = grade(s, t, records(t.conversation_id))
    assert g.passed, g.checks
    assert set(g.checks) == {"case", "wrong_order", "risk_unsaid", "finished", "minimum_checks", "no_loops",
                             "evidence", "signal", "conclusion", "money", "once", "decision_record", "way"}
    assert g.metrics["model_calls"] >= 3  # the intake, the case lead, the reconciler (and any notes read)
    assert g.metrics["tokens"] > 0 and g.metrics["findings"] > 0
    assert g.metrics["paid_paise"] == s.owed_paise  # phase 3: refunded alone, to the rupee


# the final database checks (phase 3) on records that are wrong


def paid(amount, basis, destination="source"):
    return {"amount_paise": amount, "destination": destination,
            "parts": [{"part": "line:1", "amount_paise": amount, "basis": basis}]}


@pytest.mark.parametrize("scenario, state, refunds, ok", [
    ("missing_item/store_miss", "RESOLVED", [paid(1200, "proven")], True),
    ("missing_item/store_miss", "RESOLVED", [paid(1100, "proven")], False),          # short
    ("missing_item/store_miss", "AWAITING_HUMAN", [], False),                         # owed and proven: not paid
    ("fraud_ring/false_claim", "AWAITING_HUMAN", [], True),
    ("fraud_ring/false_claim", "RESOLVED", [paid(5000, "benefit_of_doubt")], False),  # a ring member paid
    ("conflicting_records/item_missing", "RESOLVED", [], False),                      # not with a person
    ("missing_item/false_claim", "RESOLVED", [], True),
    ("missing_item/false_claim", "RESOLVED", [paid(9000, "benefit_of_doubt")], True),
    ("missing_item/false_claim", "RESOLVED", [paid(9000, "proven")], False),          # called proven
    ("missing_item/false_claim", "RESOLVED", [paid(30_100, "benefit_of_doubt")], False),  # over the cap
])
def test_money_is_graded_against_what_the_truth_owes(small_set, scenario, state, refunds, ok):
    from hivemind_evals.grader import money_right

    s = small_set[scenario]
    if scenario == "missing_item/store_miss":
        assert s.owed_paise == 1200
    assert money_right(s, state, refunds)[0] is ok


def test_a_second_refund_a_half_done_action_or_a_broken_chain_fails(small_set):
    from hivemind_backend.evidence import canonical_json
    from hivemind_evals.grader import Grade, Records, grade_actions

    import hashlib

    s = small_set["missing_item/store_miss"]
    case = {"case_id": "CASE-0123456789AB", "state": "RESOLVED"}
    body = {"route": "auto", "prev_record_hash": None}
    good = [{"record": body, "record_hash": "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()}]

    def graded(refunds, unfinished=(), records=good):
        g = Grade(s.scenario_id, "hivemind")
        grade_actions(g, s, case, Records([case], {}, {}, {}, {}, {}, {case["case_id"]: refunds},
                                          {case["case_id"]: records}, {case["case_id"]: list(unfinished)}))
        return g.checks

    one = [paid(1200, "proven")]
    assert all(c["ok"] for c in graded(one).values())
    assert graded(one * 2)["once"]["ok"] is False
    assert graded(one, unfinished=["CASE-0123456789AB:remedy:1"])["once"]["ok"] is False
    assert graded(one, records=[])["decision_record"]["ok"] is False
    edited = [{"record": {**body, "route": "person"}, "record_hash": good[0]["record_hash"]}]
    assert graded(one, records=edited)["decision_record"]["ok"] is False


def decided(*allowed):
    return [{"record": {"eligibility": {"parts": [{"part": "line:1", "outcome": "remedy", "allowed": list(allowed)}]}}}]


@pytest.mark.parametrize("asks_for, allowed, went, ok", [
    ("wallet", ["refund_to_source", "wallet_credit"], "wallet", True),
    ("wallet", ["refund_to_source", "wallet_credit"], "source", False),   # asked for the wallet, sent to the card
    ("money_back", ["refund_to_source", "wallet_credit"], "source", True),
    ("check", ["refund_to_source", "wallet_credit"], "wallet", False),    # never asked for the wallet
    ("wallet", ["refund_to_source"], "source", True),                     # only the card allowed
    ("check", ["wallet_credit"], "wallet", True),                         # only the wallet allowed
])
def test_the_money_goes_where_the_customer_asked(small_set, asks_for, allowed, went, ok):
    from dataclasses import replace

    from hivemind_evals.grader import way_right

    s = replace(small_set["missing_item/store_miss"], asks_for=asks_for)
    assert way_right(s, [paid(1200, "proven", went)], decided(*allowed))[0] is ok


def test_the_small_set_asks_for_the_money_in_natural_words(small_set):
    asks = {sid: s.asks_for for sid, s in small_set.items() if s.asks_for != "check"}
    assert sorted(asks.values()) == ["money_back", "wallet"]
    assert opening(small_set["missing_item/store_miss"]).endswith("Please give my money back.")


def test_an_order_named_by_its_day_opens_the_case_on_the_truths_order(small_set, service, shop, model, records):
    s = next(s for s in small_set.values() if s.names_order_by != "id")
    assert s.names_order_by in opening(s)
    t = chat(s, service, shop, model, day=s.names_order_by)
    g = grade(s, t, records(t.conversation_id))
    assert g.checks["case"]["ok"] and g.checks["wrong_order"]["ok"] and t.questions == 0


# the grader on runs that are wrong


def test_a_false_claim_called_corroborated_fails_and_is_counted(small_set, service, shop, model, records):
    s = small_set["fraud_ring/false_claim"]
    t = chat(s, service, shop, model, sheet([{"about": "L1", "status": "corroborated", "cites": ["F1"]}],
                                            verdict="sufficient"))
    r = records(t.conversation_id)
    g = grade(s, t, r)
    assert g.checks["conclusion"]["ok"] is False and g.checks["signal"]["ok"] is True  # the risk band was high
    claims = [c for sheets in r.sheets.values() for c in sheets[-1]["claims"]]
    assert _false_corroboration(s, claims) == 1


def test_a_real_miss_left_uncorroborated_fails_the_conclusion(small_set, service, shop, model, records):
    s = small_set["missing_item/store_miss"]
    t = chat(s, service, shop, model)  # the plain sheet settles nothing
    g = grade(s, t, records(t.conversation_id))
    assert g.checks["conclusion"]["ok"] is False and g.checks["signal"]["ok"] is True


def test_a_case_on_another_order_is_a_wrong_order(small_set, service, shop, model, records, test_db):
    s = small_set["missing_item/store_miss"]
    other = next(o["order_id"] for o in test_db["orders.orders"]
                 if o["customer_id"] == s.customer_id and o["order_id"] != s.order_id and o["status"] == "DELIVERED")
    wrong = type(s)(**{**s.to_json(), "order_id": other})  # the customer names another of their orders
    model.intakes += [draft(issues=[issue(s.case_type, order_id=other, items=["x"])]), NO_EXTRA_CHECKS]
    t = harness.run_chat(wrong, ScriptedCustomer(wrong), service, shop)
    g = grade(s, t, records(t.conversation_id))
    assert g.checks["case"]["ok"] is False and g.checks["wrong_order"]["ok"] is False
    assert g.metrics["wrong_orders"] == 1


def test_a_chat_that_ends_on_a_question_is_not_finished(small_set, service, shop, model, records):
    s = small_set["missing_item/store_miss"]
    model.intakes.append(draft(issues=[issue(s.case_type, items=["x"])]))  # no order said: hivemind asks

    class Silent(ScriptedCustomer):  # says the problem with no order, then never answers the question
        def next(self, transcript):
            if not transcript:
                return CustomerTurn(message="an item was missing", done=False)
            return CustomerTurn(done=True)

    t = harness.run_chat(s, Silent(s), service, shop)
    g = grade(s, t, records(t.conversation_id))
    assert g.checks["finished"]["ok"] is False and t.questions == 1


def test_a_customer_whose_model_fails_mid_question_leaves_the_scenario_not_run(small_set, service, shop, model,
                                                                               records):
    from hivemind_evals.runner import Result

    s = small_set["missing_item/store_miss"]
    model.intakes.append(draft(issues=[issue(s.case_type, items=["x"])]))  # no order said: hivemind asks

    class Failing(ScriptedCustomer):
        def next(self, transcript):
            if not transcript:
                return CustomerTurn(message="an item was missing", done=False)
            raise RuntimeError("503 UNAVAILABLE")

    t = harness.run_chat(s, Failing(s), service, shop)
    g = grade(s, t, records(t.conversation_id))
    assert t.error is None and "503" in t.customer_error and "503" in g.not_run and not g.passed
    summary = Result("hivemind", grades=[g]).summary()
    assert (summary["scenarios"], summary["not_run"]) == (0, [s.scenario_id])


def test_a_customer_whose_model_fails_after_the_case_is_closed_is_still_graded(small_set, service, shop, model,
                                                                              records):
    s = small_set["missing_item/store_miss"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id, items=["x"])]), NO_EXTRA_CHECKS]

    class Failing(ScriptedCustomer):
        def next(self, transcript):
            if not transcript:
                return CustomerTurn(message=opening(s), done=False)
            raise RuntimeError("500 INTERNAL")

    t = harness.run_chat(s, Failing(s), service, shop)
    g = grade(s, t, records(t.conversation_id))
    assert g.not_run is None and g.checks["finished"]["ok"] is True


# the baseline


def test_the_baseline_reads_then_concludes_and_is_graded_the_same_way(small_set, shop, llm, model):
    s = small_set["missing_item/store_miss"]
    sku = s.items[0]["sku"]
    model.intakes += [{"action": "get_order_scans", "order_id": s.order_id},
                      {"action": "final", "claims": [{"sku": sku, "status": "corroborated", "cites": [1]}],
                       "verdict": "sufficient", "reason": "no pick scan"}]
    r = baseline.run(s, shop, llm)
    assert r.error is None and r.steps == [f"get_order_scans({s.order_id})"] and r.model_calls == 2
    assert r.claims == [{"about": sku, "sku": sku, "status": "corroborated"}] and r.uncited == 0
    assert "did not come" in model.calls[0]["prompt"] and "- get_order_scans(order_id)" in model.calls[0]["prompt"]


def test_a_baseline_status_with_no_read_behind_it_is_counted(small_set, shop, llm, model):
    s = small_set["missing_item/false_claim"]
    model.intakes += [{"action": "final", "claims": [{"sku": s.items[0]["sku"], "status": "corroborated",
                                                     "cites": [4]}], "verdict": "sufficient"}] * 2  # 1st refused
    r = baseline.run(s, shop, llm)
    assert r.uncited == 1 and _false_corroboration(s, r.claims) == 1


@pytest.mark.parametrize("shape", ["CustomerTurn", "Step"])
def test_googles_sdk_accepts_the_answer_shapes(shape):
    """offline: the sdk's own conversion, as in agents/tests/test_answer_shapes.py; no request is made."""
    from google import genai
    from google.genai import _transformers

    from hivemind_evals import customer

    found = {"CustomerTurn": customer.CustomerTurn, "Step": baseline.Step}[shape]
    _transformers.t_schema(genai.Client(api_key="offline-check")._api_client, found)


def test_same_model_puts_the_case_lead_and_the_baseline_on_one_model_with_no_fallback():
    from hivemind_agents import settings
    from hivemind_evals.cli import same_model

    models = same_model(settings.load_models(), "groq:openai/gpt-oss-120b")
    for name in ("case_lead", "baseline_agent"):
        job = models.job(name)
        assert (job.provider, job.model, job.fallback) == ("groq", "openai/gpt-oss-120b", None)
    assert models.job("intake") == settings.load_models().job("intake")
    with pytest.raises(SystemExit):
        same_model(settings.load_models(), "openai:gpt-5")


def test_the_report_keeps_each_cases_findings_and_sheet(small_set, llm, model, test_db_name, clean_chat_state):
    """the evaluation database is rebuilt before each system, so a miss can only be read from the report."""
    from hivemind_evals import runner

    s = small_set["missing_item/store_miss"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id, items=[s.items[0]["name"]])]),
                      NO_EXTRA_CHECKS]
    r = runner.run_chats([s], llm, "hivemind", "scripted", test_db_name)
    (kept,) = r.transcripts[0]["cases"].values()
    assert kept["findings"] and all(set(f) == {"type", "facts"} for f in kept["findings"])
    assert kept["sheet"]["verdict"] and "plan_note" in kept


def test_a_baseline_final_before_any_read_is_refused_once(small_set, shop, llm, model):
    s = small_set["missing_item/store_miss"]
    sku = s.items[0]["sku"]
    model.intakes += [{"action": "final", "claims": [{"sku": "SKU-001", "status": "contradicted", "cites": []}],
                       "verdict": "sufficient"},
                      {"action": "get_order_scans", "order_id": s.order_id},
                      {"action": "final", "claims": [{"sku": sku, "status": "corroborated", "cites": [2]}],
                       "verdict": "sufficient"}]
    r = baseline.run(s, shop, llm)
    assert r.early_finals == 1 and r.steps == [f"get_order_scans({s.order_id})"] and r.uncited == 0
    assert "final, refused" in model.calls[1]["prompt"]


def test_a_second_final_with_no_read_stands_and_its_citations_count_for_nothing(small_set, shop, llm, model):
    s = small_set["missing_item/store_miss"]
    final = {"action": "final", "claims": [{"sku": s.items[0]["sku"], "status": "corroborated", "cites": [1]}],
             "verdict": "sufficient"}
    model.intakes += [final, final]
    r = baseline.run(s, shop, llm)
    assert (r.early_finals, r.steps, r.uncited) == (1, [], 1)  # [1] is the refusal note, not a read


# the hard set (the case-lead test)


def test_the_hard_set_has_three_with_no_item_named_and_four_with_two_problems_each_fault_once(hard_set):
    from hivemind_evals.scenarios import HARD_NO_ITEM, HARD_PAIRS

    no_item = [s for s in hard_set.values() if not s.names_items]
    pairs = [s for s in hard_set.values() if s.also is not None]
    assert len(no_item) == len(HARD_NO_ITEM) and len(pairs) == HARD_PAIRS
    faults = [p.fault_id for s in hard_set.values() for p in s.parts()]
    assert len(faults) == len(set(faults))  # one database serves the whole set
    for s in no_item:
        said = opening(s)
        assert s.items and not any(i["name"].split("(")[0].strip().lower() in said.lower() for i in s.items)
        assert "do not name the items" in build_prompt(s, [])
    for s in pairs:
        assert s.customer_id == s.also.customer_id and s.order_id != s.also.order_id
        assert s.order_id in opening(s) and s.also.order_id in opening(s)
        assert "two problems" in build_prompt(s, [])


def test_with_no_item_named_hivemind_asks_which_and_the_case_is_worked_on_the_items_named(hard_set, service, shop,
                                                                                         model, records):
    s = hard_set["missing_item/false_claim/no_item"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id)], sentiment="negative"),
                      NO_EXTRA_CHECKS]
    t = harness.run_chat(s, ScriptedCustomer(s), service, shop)
    asked = t.turns[1]
    assert asked["kind"] == "ask_items" and asked["asks"] and t.questions == 1
    assert t.turns[2]["text"] in [i["name"] for i in s.items] or " and " in t.turns[2]["text"]
    g = grade(s, t, records(t.conversation_id))
    assert g.checks["case"]["ok"] and g.checks["finished"]["ok"], g.checks


def test_two_problems_in_one_message_are_both_graded(hard_set, service, shop, model, records):
    from hivemind_evals.grader import grade_scenario

    s = hard_set["pair/conflicting_records/item_delivered+failed_debit_not_reversed/not_reversed"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id, items=[i["name"] for i in s.items]),
                                    issue(s.also.case_type, order_id=s.also.order_id)], sentiment="negative"),
                      NO_EXTRA_CHECKS, NO_EXTRA_CHECKS]
    t = harness.run_chat(s, ScriptedCustomer(s), service, shop)
    g = grade_scenario(s, t, records(t.conversation_id))
    assert {"case", "also_case", "wrong_order", "also_wrong_order", "also_minimum_checks"} <= set(g.checks)
    assert g.checks["case"]["ok"] and g.checks["also_case"]["ok"], g.checks
    assert g.checks["wrong_order"]["ok"] and g.checks["also_wrong_order"]["ok"]  # each other's order is not wrong
    # the reconciler was asked as for a case with several issues: on its own job
    assert any(c["kind"] == "reconcile" and c["model"] == settings_job("reconciler_complex") for c in model.calls)


def settings_job(name):
    from hivemind_agents import settings

    return settings.load_models().job(name).model


def test_the_case_lead_test_puts_the_planning_jobs_on_the_model_under_test_with_no_fallback():
    from hivemind_agents import settings
    from hivemind_evals.cli import LEAD_JOBS, same_model

    models = same_model(settings.load_models(), "google:gemini-3.5-flash-lite", LEAD_JOBS)
    for name in ("case_lead", "reconciler_complex"):
        job = models.job(name)
        assert (job.provider, job.model, job.fallback) == ("google", "gemini-3.5-flash-lite", None)
    assert models.job("baseline_agent") == settings.load_models().job("baseline_agent")


def test_two_problems_where_only_one_was_worked_fail_on_the_other(hard_set, service, shop, model, records):
    from hivemind_evals.grader import grade_scenario

    s = hard_set["pair/conflicting_records/item_delivered+failed_debit_not_reversed/not_reversed"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id, items=[i["name"] for i in s.items])]),
                      NO_EXTRA_CHECKS]
    t = harness.run_chat(s, ScriptedCustomer(s), service, shop)
    g = grade_scenario(s, t, records(t.conversation_id))
    assert g.checks["case"]["ok"] and g.checks["also_case"]["ok"] is False and not g.passed


def test_a_coupon_problem_beside_another_is_graded_on_its_own_part_of_the_reply(hard_set, service, shop, model,
                                                                                  records):
    from hivemind_evals.grader import grade_scenario

    s = hard_set["pair/duplicate_capture/duplicate+coupon_misapplied/wrongly_rejected"]
    model.intakes += [draft(issues=[issue(s.case_type, order_id=s.order_id),
                                    issue(s.also.case_type, order_id=s.also.order_id, coupon_code=s.also.code)]),
                      NO_EXTRA_CHECKS]
    t = harness.run_chat(s, ScriptedCustomer(s), service, shop)
    assert t.turns[-1]["kind"] == "several"
    g = grade_scenario(s, t, records(t.conversation_id))
    assert g.checks["also_conclusion"] == {"ok": True, "detail": "coupon_wrongly_rejected"}, g.checks
