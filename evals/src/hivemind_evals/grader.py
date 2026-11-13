"""grades one scenario's chat in plain code against the hidden truth (record 014: no model grades facts). phase 2
has no actions yet, so it grades the investigation from hivemind's own records of the case:

- case: one case of the right type, on the order the truth names (an order named by its day included).
- wrong_order: a case of that type on any other order (must never happen).
- minimum_checks: every minimum check of the case type is in the plan, as a check that always runs, and decided.
- no_loops: no check twice with the same inputs, revisions within the limit, the plan within its size.
- evidence: every finding cites only reads made in this case; every fact on the fact sheet cites findings on the
  board and their reads.
- signal: the record that shows what really happened was found (per fault kind).
- conclusion: what hivemind concluded matches the truth: an item really missing is corroborated, an item that came
  is never corroborated, records in conflict are never "sufficient".
- risk_unsaid: no reply mentions the risk check.
- finished: the chat ended without an error and not on a question.
"""

from dataclasses import asdict, dataclass, field

import psycopg

from hivemind_agents.capabilities.planner import MAX_TASKS, MINIMUM, REPLANS
from hivemind_evals.harness import Transcript
from hivemind_evals.scenarios import Scenario

COUPON_KIND = {"wrong_amount": "coupon_applied_too_little", "wrongly_rejected": "coupon_wrongly_rejected"}
NEVER_SAID = ("risk", "fraud")


@dataclass
class Grade:
    scenario_id: str
    system: str
    checks: dict[str, dict] = field(default_factory=dict)  # name -> {"ok": True/False/None, "detail": ...}
    metrics: dict = field(default_factory=dict)
    not_run: str | None = None  # the simulated customer failed before the chat could end: no grade

    def check(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks[name] = {"ok": ok, "detail": detail}

    @property
    def passed(self) -> bool:
        return self.not_run is None and all(c["ok"] is not False for c in self.checks.values())

    def to_json(self) -> dict:
        return {**asdict(self), "passed": self.passed}


@dataclass
class Records:
    """what hivemind kept about the chat's cases, read with the grader's role."""

    cases: list[dict]
    plans: dict[str, dict]           # case id -> its latest run: plan, outcomes
    findings: dict[str, list[dict]]
    ledger: dict[str, set[str]]
    sheets: dict[str, list[dict]]    # case id -> fact sheets, in order
    reply_kinds: dict[str, str] = field(default_factory=dict)  # case id -> the kind of its latest part of a reply


def read_records(conn: psycopg.Connection, conversation_id: str) -> Records:
    ids: list[str] = []
    reply_kinds: dict[str, str] = {}
    for r in conn.execute("select case_id, kind, meta from agent.messages where conversation_id = %s and "
                          "role = 'assistant' order by seq", (conversation_id,)).fetchall():
        # a reply to several problems carries the first case; each part carries its own, with its own kind
        parts = (r["meta"] or {}).get("parts") or [{"case_id": r["case_id"], "kind": r["kind"]}]
        for part in parts:
            case_id = part.get("case_id")
            if case_id:
                reply_kinds[case_id] = part["kind"]
                if case_id not in ids:
                    ids.append(case_id)
    cases = conn.execute("select case_id, issue_type, order_id, state from support.cases where case_id = any(%s) "
                         "order by created_at", (ids,)).fetchall()
    plans = {r["case_id"]: r for r in conn.execute(
        "select distinct on (case_id) case_id, plan, outcomes from agent.case_plans where case_id = any(%s) "
        "order by case_id, fence desc", (ids,)).fetchall()}
    findings: dict[str, list[dict]] = {i: [] for i in ids}
    for r in conn.execute("select case_id, finding_id, capability, type, evidence_refs, facts from agent.findings "
                          "where case_id = any(%s) order by seq", (ids,)).fetchall():
        findings[r["case_id"]].append(r)
    ledger: dict[str, set[str]] = {i: set() for i in ids}
    for r in conn.execute("select case_id, evidence_id from agent.evidence_ledger where case_id = any(%s)",
                          (ids,)).fetchall():
        ledger[r["case_id"]].add(r["evidence_id"])
    sheets: dict[str, list[dict]] = {i: [] for i in ids}
    for r in conn.execute("select case_id, sheet from agent.fact_sheets where case_id = any(%s) "
                          "order by fence, round", (ids,)).fetchall():
        sheets[r["case_id"]].append(r["sheet"])
    return Records(cases, plans, findings, ledger, sheets, reply_kinds)


def _skus(s: Scenario) -> set[str]:
    return {i["sku"] for i in s.items}


def signal_found(s: Scenario, findings: list[dict], sheet: dict | None) -> tuple[bool | None, str]:
    """whether the record that shows what really happened is on the board."""
    skus, kind = _skus(s), (s.fault_type, s.variant)
    rows = (sheet or {}).get("cross_checks", [])

    def has(type_: str, **facts) -> bool:
        return any(f["type"] == type_ and all(f["facts"].get(k) == v for k, v in facts.items()) for f in findings)

    def for_sku(type_: str) -> bool:
        return any(f["type"] == type_ and f["facts"].get("sku") in skus for f in findings)

    if kind == ("missing_item", "store_miss"):
        return for_sku("store.item_not_scanned") or for_sku("store.stock_out_before_pack"), "no scan or stock-out"
    if kind == ("missing_item", "rider_handover"):
        return any(r["signal"] == "bags_short" for r in rows), "bags handed over against bags packed"
    if kind in (("missing_item", "false_claim"), ("conflicting_records", "item_delivered")):
        return for_sku("store.item_scanned"), "the item's packing scan"
    if kind == ("fraud_ring", "false_claim"):
        return has("risk.assessment", band="high"), "risk band high"
    if kind == ("injected_note", "partner_note"):
        return has("delivery.partner_note", label="note_gives_instructions"), "the note labelled as instructions"
    if kind == ("conflicting_records", "item_missing"):
        return (any(r["direction"] == "conflict" for r in rows) or bool((sheet or {}).get("contradictions")),
                "the scan and the store's log in conflict")
    if kind == ("failed_debit_not_reversed", "not_reversed"):
        return has("payment.failed_debit_not_reversed"), "a failed debit not returned"
    if kind == ("duplicate_capture", "duplicate"):
        return has("payment.duplicate_capture"), "a second capture"
    return None, "not graded for this fault"


def conclusion_right(s: Scenario, claims: list[dict], verdict: str | None) -> tuple[bool, str]:
    """what was concluded about the claim, against the truth. `claims`: about, sku, status."""
    status = {c["sku"]: c["status"] for c in claims if c.get("sku")}
    got = ", ".join(f"{k}: {v}" for k, v in sorted(status.items())) or "no item settled"
    if s.fault_type == "conflicting_records":
        return verdict not in (None, "sufficient"), f"verdict {verdict}"
    if s.case_type == "item_missing_or_wrong":
        if s.owed_paise > 0 and s.proven_by_records:
            return all(status.get(k) == "corroborated" for k in _skus(s)), got
        return not any(status.get(k) == "corroborated" for k in _skus(s)), got
    if s.case_type == "payment_failed":
        ok = any(c["status"] == "corroborated" for c in claims)
        return ok, ", ".join(f"{c['about']}: {c['status']}" for c in claims) or "nothing settled"
    return True, ""


def grade_scenario(s: Scenario, t: Transcript, records: Records, system: str = "hivemind") -> Grade:
    """a scenario's grade; with two problems in one message, the second's checks are added as `also_<check>`, and
    each problem's order is no wrong order for the other."""
    g = grade(s, t, records, system, others={p.order_id for p in s.parts()})
    if s.also is not None:
        second = grade(s.also, t, records, system, others={p.order_id for p in s.parts()})
        g.checks.update({f"also_{name}": c for name, c in second.checks.items()})
        g.metrics["wrong_orders"] = g.metrics.get("wrong_orders", 0) + second.metrics.get("wrong_orders", 0)
        g.not_run = g.not_run or second.not_run
    return g


def grade(s: Scenario, t: Transcript, records: Records, system: str = "hivemind",
          others: set[str] | frozenset = frozenset()) -> Grade:
    g = Grade(s.scenario_id, system, metrics={"model_calls": t.hivemind_calls, "tokens": t.hivemind_tokens,
                                              "seconds": t.seconds, "questions": t.questions,
                                              "turns": len(t.turns)})
    of_type = [c for c in records.cases if c["issue_type"] == s.case_type]
    right = [c for c in of_type if c["order_id"] == s.order_id]
    g.check("case", len(right) == 1, f"{len(right)} case(s) of {s.case_type} on {s.order_id}")
    wrong = [c["order_id"] for c in of_type if c["order_id"] != s.order_id and c["order_id"] not in others]
    g.check("wrong_order", not wrong, ", ".join(map(str, wrong)))
    g.metrics["wrong_orders"] = len(wrong)
    replies = [x["text"].lower() for x in t.turns if x["role"] == "assistant"]
    g.check("risk_unsaid", not any(w in r for r in replies for w in NEVER_SAID))
    last = next((x for x in reversed(t.turns) if x["role"] == "assistant"), None)
    if t.customer_error and (last is None or last.get("asks")):  # stopped while hivemind waited on the customer
        g.not_run = t.customer_error
        return g
    g.check("finished", t.error is None and last is not None and not last.get("asks"),
            t.error or (last["kind"] if last else "no reply"))
    if not right:
        return g
    case_id = right[0]["case_id"]

    if s.case_type == "offers_coupons":  # the phase 1 path: the promotions check and its own part of the reply
        want, got = COUPON_KIND.get(s.variant), records.reply_kinds.get(case_id)
        g.check("conclusion", got == want, got or "no reply part")
        return g

    run = records.plans.get(case_id)
    findings = records.findings.get(case_id, [])
    sheets = records.sheets.get(case_id, [])
    sheet = sheets[-1] if sheets else None
    if run is None:
        g.check("minimum_checks", False, "no plan was kept")
        return g
    tasks, outcomes = run["plan"]["tasks"], run["outcomes"]
    mandatory = {t["capability"] for t in tasks if t["mandatory"]}
    missing = [m for m in MINIMUM.get(s.case_type, ()) if m not in mandatory]
    undecided = [t["task_id"] for t in tasks if t["task_id"] not in outcomes]
    g.check("minimum_checks", not missing and not undecided,
            "; ".join(filter(None, [f"missing {missing}" if missing else "",
                                    f"undecided {undecided}" if undecided else ""])))
    keys = [t["capability"] + ":" + ",".join(f"{k}={v}" for k, v in sorted(t["params"].items())) for t in tasks]
    twice = sorted({k for k in keys if keys.count(k) > 1})
    g.check("no_loops", not twice and run["plan"]["replans"] <= REPLANS["complex"] and len(tasks) <= MAX_TASKS,
            f"twice: {twice}" if twice else f"{len(tasks)} checks, {run['plan']['replans']} revision(s)")
    ledger, on_board = records.ledger.get(case_id, set()), {f["finding_id"] for f in findings}
    unproven = [f["finding_id"] for f in findings if not set(f["evidence_refs"]) <= ledger]
    for one in sheets:
        unproven += [x["statement"][:40] for x in one["facts"]
                     if not set(x["finding_ids"]) <= on_board or not set(x["evidence_refs"]) <= ledger]
    g.check("evidence", not unproven, ", ".join(unproven))
    g.metrics["findings"] = len(findings)
    ok, detail = signal_found(s, findings, sheet)
    g.check("signal", ok, detail if ok is not True else "")
    if sheet is None:
        g.check("conclusion", False, "no fact sheet")
        return g
    g.metrics.update(verdict=sheet["verdict"], sheet_source=sheet["source"], rounds=len(sheets))
    ok, detail = conclusion_right(s, sheet["claims"], sheet["verdict"])
    g.check("conclusion", ok, detail)
    return g
