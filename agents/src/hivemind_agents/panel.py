"""what the scenario panel reads: hivemind's own records of each case (agent schema), never the business's tables
and never the hidden truth. read only.

- cases: the latest run of each case, newest first: its type, order, customer, whether a run holds it now, how many
  findings it has and the reconciler's latest verdict.
- one case: every run's plan with what became of each check, the findings with their proof, the fact sheets, and
  the reads in its evidence ledger (which tool, when; not what came back).
"""

import psycopg
from psycopg.rows import dict_row

MAX_CASES = 100

CASES = """
select p.case_id, p.case_type, p.order_id, p.customer_id, p.conversation_id, p.fence, p.updated_at,
       exists (select 1 from agent.case_leases l where l.case_id = p.case_id and l.expires_at > now()) as running,
       (select count(*) from agent.findings f where f.case_id = p.case_id) as findings,
       (select s.verdict from agent.fact_sheets s where s.case_id = p.case_id
         order by s.fence desc, s.round desc limit 1) as verdict
from (select distinct on (case_id) * from agent.case_plans order by case_id, fence desc) p
order by p.updated_at desc
limit %s
"""


class PanelReader:
    def __init__(self, conninfo: str):
        self.conninfo = conninfo

    def _run(self, query: str, args: tuple) -> list[dict]:
        with psycopg.connect(self.conninfo, row_factory=dict_row) as conn:
            return conn.execute(query, args).fetchall()

    def cases(self, limit: int = 30) -> list[dict]:
        return self._run(CASES, (max(1, min(limit, MAX_CASES)),))

    def case(self, case_id: str) -> dict | None:
        with psycopg.connect(self.conninfo, row_factory=dict_row) as conn:
            runs = conn.execute(
                "select fence, case_type, order_id, customer_id, conversation_id, plan, outcomes, updated_at "
                "from agent.case_plans where case_id = %s order by fence", (case_id,)).fetchall()
            if not runs:
                return None
            findings = conn.execute(
                "select finding_id, task_id, capability, type, claim, evidence_refs, confidence, confidence_basis, "
                "facts, fence, created_at from agent.findings where case_id = %s order by seq", (case_id,)).fetchall()
            sheets = conn.execute(
                "select fence, round, verdict, sheet, created_at from agent.fact_sheets where case_id = %s "
                "order by fence, round", (case_id,)).fetchall()
            reads = conn.execute(
                "select evidence_id, tool, created_at from agent.evidence_ledger where case_id = %s "
                "order by created_at", (case_id,)).fetchall()
            running = conn.execute("select 1 from agent.case_leases where case_id = %s and expires_at > now()",
                                   (case_id,)).fetchone() is not None
        latest = runs[-1]
        return {"case_id": case_id, "case_type": latest["case_type"], "order_id": latest["order_id"],
                "customer_id": latest["customer_id"], "conversation_id": latest["conversation_id"],
                "running": running, "runs": runs, "findings": findings, "fact_sheets": sheets, "reads": reads}
