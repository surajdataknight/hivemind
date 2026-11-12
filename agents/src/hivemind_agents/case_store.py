"""hivemind's own records of a case, in postgres (agent schema): the run lease, the evidence ledger and the
findings on the case board. one store per run, on one connection, opened at the first use.

- lease: one run per case (blueprint c4). it runs out by itself, so a run that died does not hold a case for ever;
  the fencing number grows with every new holder, and a finding written with an old number is refused.
- evidence ledger: every read made for a case, under that case, with a hash of what came back.
- findings: the board's findings as they were posted, for anyone reading the case after the run.
"""

from datetime import timedelta
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from hivemind_agents import evidence
from hivemind_agents.contracts import Finding, Lease

LEASE_TTL = timedelta(minutes=2)  # a run takes seconds; the blueprint's case budget is 60 s
SOURCE = "simulated-business"


class LeaseLost(Exception):
    """this run no longer holds the case: its lease ran out and another run took it, or it never held it."""


class CaseStore:
    def __init__(self, conninfo: str, holder: str | None = None, ttl: timedelta = LEASE_TTL):
        self._conninfo = conninfo
        self._conn: psycopg.Connection | None = None
        self.holder = holder or "run_" + uuid4().hex[:16]
        self.ttl = ttl
        self.leases: dict[str, Lease] = {}

    def _db(self) -> psycopg.Connection:
        if self._conn is None:
            self._conn = psycopg.connect(self._conninfo, autocommit=True, row_factory=dict_row)
        return self._conn

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

    def release(self, case_id: str) -> None:
        """ends the lease now. the row stays, so the next holder's fencing number is higher."""
        lease = self.leases.pop(case_id, None)
        if lease is not None:
            self._db().execute(
                "update agent.case_leases set expires_at = now() where case_id = %s and holder = %s and fence = %s "
                "and expires_at > now()", (case_id, lease.holder, lease.fence))

    # evidence ledger

    def record_evidence(self, case_id: str, tool: str, body: dict) -> None:
        payload = {k: v for k, v in body.items() if k != "evidence_id"}
        self._db().execute(
            "insert into agent.evidence_ledger (evidence_id, case_id, source, tool, payload, sha256, created_at) "
            "values (%s, %s, %s, %s, %s, %s, now()) on conflict do nothing",
            (body["evidence_id"], case_id, SOURCE, tool, Jsonb(payload), evidence.body_hash(body)))

    def evidence_ids(self, case_id: str) -> set[str]:
        rows = self._db().execute("select evidence_id from agent.evidence_ledger where case_id = %s",
                                  (case_id,)).fetchall()
        return {r["evidence_id"] for r in rows}

    # findings

    def add_findings(self, lease: Lease, findings: list[Finding]) -> None:
        """writes the findings only while the lease still holds; else raises LeaseLost and writes none."""
        conn = self._db()
        with conn.transaction():
            held = conn.execute(
                "select 1 from agent.case_leases where case_id = %s and holder = %s and fence = %s "
                "and expires_at > now() for update", (lease.case_id, lease.holder, lease.fence)).fetchone()
            if held is None:
                self.leases.pop(lease.case_id, None)
                raise LeaseLost(lease.case_id)
            for f in findings:
                conn.execute(
                    "insert into agent.findings (finding_id, case_id, task_id, capability, type, claim, evidence_refs, "
                    "confidence, confidence_basis, fence, created_at) "
                    "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())",
                    (f.finding_id, f.case_id, f.task_id, f.capability, f.type, f.claim, f.evidence_refs, f.confidence,
                     f.confidence_basis, lease.fence))

    def findings(self, case_id: str) -> list[Finding]:
        rows = self._db().execute(
            "select finding_id, case_id, task_id, capability, type, claim, evidence_refs, confidence, confidence_basis "
            "from agent.findings where case_id = %s order by seq", (case_id,)).fetchall()
        return [Finding(**r) for r in rows]
