"""hivemind's own records of a case, in postgres (agent schema): the run lease, the evidence ledger and the
findings on the case board. one store per run, on one connection, opened at the first use.

- lease: one run per case (blueprint c4). it runs out by itself, so a run that died does not hold a case for ever;
  the fencing number grows with every new holder, and a finding written with an old number is refused.
- evidence ledger: every read made for a case, under that case, with a hash of what came back.
- findings: the board's findings as they were posted, for anyone reading the case after the run.
- plans and fact sheets: the run's plan with what became of each check, and the evidence reconciler's sheet per
  round; both written under the run's lease.
"""

import functools
import threading
from datetime import timedelta
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from hivemind_agents import evidence
from hivemind_agents.contracts import FactSheet, Finding, Lease

LEASE_TTL = timedelta(minutes=2)  # a run takes seconds; the blueprint's case budget is 60 s
SOURCE = "simulated-business"


class LeaseLost(Exception):
    """this run no longer holds the case: its lease ran out and another run took it, or it never held it."""


def _locked(method):
    """one call at a time on the run's connection: the checks of a wave run side by side, and each may record reads."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class CaseStore:
    def __init__(self, conninfo: str, holder: str | None = None, ttl: timedelta = LEASE_TTL):
        self._conninfo = conninfo
        self._conn: psycopg.Connection | None = None
        self._lock = threading.RLock()
        self.holder = holder or "run_" + uuid4().hex[:16]
        self.ttl = ttl
        self.leases: dict[str, Lease] = {}

    def _db(self) -> psycopg.Connection:
        if self._conn is None:
            self._conn = psycopg.connect(self._conninfo, autocommit=True, row_factory=dict_row)
        return self._conn

    @_locked
    def close(self) -> None:
        """lets go of every case this run holds, then of the connection."""
        try:
            for case_id in list(self.leases):
                self.release(case_id)
        finally:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # lease

    @_locked
    def acquire(self, case_id: str) -> Lease | None:
        """this run's lease on the case, or None while another run holds it."""
        if case_id in self.leases:
            return self.leases[case_id]
        row = self._db().execute(
            "insert into agent.case_leases as l (case_id, holder, fence, expires_at) values (%s, %s, 1, now() + %s) "
            "on conflict (case_id) do update set holder = excluded.holder, fence = l.fence + 1, "
            "expires_at = excluded.expires_at where l.expires_at <= now() returning fence",
            (case_id, self.holder, self.ttl)).fetchone()
        if row is None:
            return None
        lease = Lease(case_id=case_id, holder=self.holder, fence=row["fence"])
        self.leases[case_id] = lease
        return lease

    @_locked
    def release(self, case_id: str) -> None:
        """ends the lease now. the row stays, so the next holder's fencing number is higher."""
        lease = self.leases.pop(case_id, None)
        if lease is not None:
            self._db().execute(
                "update agent.case_leases set expires_at = now() where case_id = %s and holder = %s and fence = %s "
                "and expires_at > now()", (case_id, lease.holder, lease.fence))

    # evidence ledger

    @_locked
    def record_evidence(self, case_id: str, tool: str, body: dict) -> None:
        payload = {k: v for k, v in body.items() if k != "evidence_id"}
        self._db().execute(
            "insert into agent.evidence_ledger (evidence_id, case_id, source, tool, payload, sha256, created_at) "
            "values (%s, %s, %s, %s, %s, %s, now()) on conflict do nothing",
            (body["evidence_id"], case_id, SOURCE, tool, Jsonb(payload), evidence.body_hash(body)))

    @_locked
    def evidence_ids(self, case_id: str) -> set[str]:
        rows = self._db().execute("select evidence_id from agent.evidence_ledger where case_id = %s",
                                  (case_id,)).fetchall()
        return {r["evidence_id"] for r in rows}

    # findings

    @_locked
    def add_findings(self, lease: Lease, findings: list[Finding]) -> None:
        """writes the findings only while the lease still holds; else raises LeaseLost and writes none."""
        conn = self._db()
        with conn.transaction():
            self._held(conn, lease)
            for f in findings:
                conn.execute(
                    "insert into agent.findings (finding_id, case_id, task_id, capability, type, claim, evidence_refs, "
                    "confidence, confidence_basis, facts, fence, created_at) "
                    "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())",
                    (f.finding_id, f.case_id, f.task_id, f.capability, f.type, f.claim, f.evidence_refs, f.confidence,
                     f.confidence_basis, Jsonb(f.facts), lease.fence))

    # the plan and the fact sheet of this run (c6, c7, c17)

    def _held(self, conn: psycopg.Connection, lease: Lease) -> None:
        held = conn.execute(
            "select 1 from agent.case_leases where case_id = %s and holder = %s and fence = %s "
            "and expires_at > now() for update", (lease.case_id, lease.holder, lease.fence)).fetchone()
        if held is None:
            self.leases.pop(lease.case_id, None)
            raise LeaseLost(lease.case_id)

    @_locked
    def save_plan(self, lease: Lease, case: dict, plan: dict, outcomes: dict) -> None:
        """the run's plan and each check's outcome so far; written again as they change, while the lease holds.
        `case`: case_type, order_id, customer_id, conversation_id (for the scenario panel)."""
        conn = self._db()
        with conn.transaction():
            self._held(conn, lease)
            conn.execute(
                "insert into agent.case_plans (case_id, fence, case_type, order_id, customer_id, conversation_id, "
                "plan, outcomes, updated_at) values (%s, %s, %s, %s, %s, %s, %s, %s, now()) on conflict (case_id, "
                "fence) do update set plan = excluded.plan, outcomes = excluded.outcomes, "
                "updated_at = excluded.updated_at",
                (lease.case_id, lease.fence, case["case_type"], case["order_id"], case["customer_id"],
                 case["conversation_id"], Jsonb(plan), Jsonb(outcomes)))

    @_locked
    def save_fact_sheet(self, lease: Lease, sheet: FactSheet) -> None:
        conn = self._db()
        with conn.transaction():
            self._held(conn, lease)
            conn.execute(
                "insert into agent.fact_sheets (case_id, fence, round, verdict, sheet, created_at) values (%s, %s, %s, "
                "%s, %s, now()) on conflict (case_id, fence, round) do update set verdict = excluded.verdict, "
                "sheet = excluded.sheet", (lease.case_id, lease.fence, sheet.round, sheet.verdict,
                                           Jsonb(sheet.model_dump(mode="json"))))

    @_locked
    def fact_sheets(self, case_id: str) -> list[FactSheet]:
        rows = self._db().execute("select sheet from agent.fact_sheets where case_id = %s order by fence, round",
                                  (case_id,)).fetchall()
        return [FactSheet(**r["sheet"]) for r in rows]

    @_locked
    def plans(self, case_id: str) -> list[dict]:
        return self._db().execute("select fence, plan, outcomes from agent.case_plans where case_id = %s "
                                  "order by fence", (case_id,)).fetchall()

    @_locked
    def findings(self, case_id: str) -> list[Finding]:
        rows = self._db().execute(
            "select finding_id, case_id, task_id, capability, type, claim, evidence_refs, confidence, confidence_basis, "
            "facts from agent.findings where case_id = %s order by seq", (case_id,)).fetchall()
        return [Finding(**r) for r in rows]
