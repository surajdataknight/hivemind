"""runs the small set (record 014) and the b.6 comparison, and sums them up.

- small set: each scenario once, a fresh chat each, on hivemind as built; the customer is simulated (models.toml) or
  scripted.
- b.6 (blueprint part b.6, record 003's reopen test): the same scenarios with the same fixed first message, on three
  systems: hivemind (the case lead plans), "minimum plan only" (the case lead switched off), and the single-agent
  baseline. the evaluation database is rebuilt before each system, so none sees another's cases.
"""

import time
from collections import Counter
from dataclasses import dataclass, field

from hivemind_agents.llm import ModelRouter
from hivemind_backend import db
from hivemind_evals import baseline, harness, scenarios
from hivemind_evals.customer import ScriptedCustomer, SimulatedCustomer
from hivemind_evals.grader import Grade, conclusion_right, grade_scenario, read_records
from hivemind_evals.scenarios import Scenario

SYSTEMS = ("hivemind", "minimum", "baseline")


@dataclass
class Result:
    system: str
    grades: list[Grade] = field(default_factory=list)
    transcripts: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        checks: Counter = Counter()
        failed: Counter = Counter()
        run = [g for g in self.grades if g.not_run is None]
        for g in run:
            for name, c in g.checks.items():
                if c["ok"] is not None:
                    checks[name] += 1
                    failed[name] += c["ok"] is False
        total = lambda key: sum(g.metrics.get(key, 0) or 0 for g in run)
        return {"system": self.system, "scenarios": len(run), "not_run": [g.scenario_id for g in self.grades
                                                                          if g.not_run is not None],
                "passed": sum(g.passed for g in self.grades),
                "checks": {name: f"{checks[name] - failed[name]}/{checks[name]}" for name in sorted(checks)},
                "false_corroborations": sum(g.metrics.get("false_corroboration", 0) for g in run),
                "wrong_orders": total("wrong_orders"), "questions": total("questions"),
                "model_calls": total("model_calls"), "tokens": total("tokens"),
                "seconds": round(total("seconds"), 1)}


def load_scenarios(dbname: str, only: list[tuple[str, str]] | None = None, hard: bool = False) -> list[Scenario]:
    from psycopg.rows import dict_row

    with db.connect("eval", dbname=dbname, row_factory=dict_row) as conn:
        return scenarios.load_hard(conn) if hard else scenarios.load(conn, only)


def _false_corroboration(s: Scenario, claims: list[dict]) -> int:
    """an item the truth says came, called corroborated: the mistake that would pay a false claim."""
    if s.case_type != "item_missing_or_wrong" or s.owed_paise > 0:
        return 0
    skus = {i["sku"] for i in s.items}
    return sum(1 for c in claims if c.get("sku") in skus and c["status"] == "corroborated")


def _kept(records) -> dict:
    """what the report keeps of each case, so a miss can be read after the database is rebuilt: the plan's notes,
    each finding's type and facts (a note's label, never its words), the last fact sheet."""
    if records is None:
        return {}
    return {case_id: {"plan_source": (records.plans.get(case_id) or {}).get("plan", {}).get("source"),
                      "plan_replans": (records.plans.get(case_id) or {}).get("plan", {}).get("replans"),
                      "plan_checks": [t["capability"] for t in
                                      (records.plans.get(case_id) or {}).get("plan", {}).get("tasks", [])],
                      "plan_note": (records.plans.get(case_id) or {}).get("plan", {}).get("note"),
                      "findings": [{"type": f["type"], "facts": f["facts"]} for f in records.findings.get(case_id, [])],
                      "sheet": (records.sheets.get(case_id) or [None])[-1]}
            for case_id in records.findings}


def run_chats(chosen: list[Scenario], llm: ModelRouter, system: str, customer: str, dbname: str,
              customer_llm: ModelRouter | None = None, pause_s: float = 0.0) -> Result:
    from psycopg.rows import dict_row

    shop = harness.business(dbname)
    service = harness.make_service(shop, llm, case_lead=system != "minimum")
    out = Result(system)
    for n, s in enumerate(chosen):
        if n and pause_s:
            time.sleep(pause_s)  # a free per-minute limit (flash-lite: 15 requests a minute)
        who = ScriptedCustomer(s) if customer == "scripted" else SimulatedCustomer(s, customer_llm or llm)
        t = harness.run_chat(s, who, service, shop)
        with db.connect("eval", dbname=dbname, row_factory=dict_row) as conn:
            records = read_records(conn, t.conversation_id) if t.conversation_id else None
        g = grade_scenario(s, t, records, system) if records else Grade(s.scenario_id, system)
        if records is None:
            g.check("finished", False, t.error or "no chat")
        sheet_claims = [c for case_sheets in (records.sheets.values() if records else []) for sheet in
                        case_sheets[-1:] for c in sheet["claims"]]
        g.metrics["false_corroboration"] = sum(_false_corroboration(p, sheet_claims) for p in s.parts())
        out.grades.append(g)
        out.transcripts.append({"scenario_id": s.scenario_id, "conversation_id": t.conversation_id,
                                "turns": t.turns, "error": t.error, "cases": _kept(records)})
    return out


def run_baseline(chosen: list[Scenario], llm: ModelRouter, dbname: str, pause_s: float = 0.0) -> Result:
    shop = harness.business(dbname)
    out = Result("baseline")
    for n, s in enumerate(chosen):
        if n and pause_s:
            time.sleep(pause_s)
        r = baseline.run(s, shop, llm)
        g = Grade(s.scenario_id, "baseline", metrics={"model_calls": r.model_calls, "tokens": r.tokens,
                                                      "seconds": r.seconds, "verdict": r.verdict,
                                                      "reads": len(r.steps), "early_finals": r.early_finals})
        ok, detail = conclusion_right(s, r.claims, r.verdict)
        g.check("conclusion", ok, detail)
        g.check("evidence", r.uncited == 0, f"{r.uncited} status(es) cite no read")
        g.check("finished", r.error is None, r.error or "")
        g.metrics["false_corroboration"] = _false_corroboration(s, r.claims)
        out.grades.append(g)
        out.transcripts.append({"scenario_id": s.scenario_id, "steps": r.steps, "claims": r.claims,
                                "error": r.error})
    return out


def b6_scenarios(chosen: list[Scenario]) -> list[Scenario]:
    """the comparison covers what the case lead plans: the case types with a minimum plan (not coupons)."""
    return [s for s in chosen if s.case_type in ("item_missing_or_wrong", "payment_failed")]
