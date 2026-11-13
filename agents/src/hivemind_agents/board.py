"""the case board: where investigators post findings with their proof (blueprint 2.3, b.5).

no capability talks to another except through the board. a finding is kept only when every evidence id it cites
is a read made in this case, by a tool call, and recorded in the evidence ledger (blueprint 2.3, the evidence rule);
any other is dropped, so a claim without proof never reaches a decision. kept findings go to the run's state (one
append-only list per case) and to `agent.findings`, under this run's lease.
"""

from dataclasses import dataclass, field

from hivemind_agents.case_store import CaseStore, LeaseLost
from hivemind_agents.contracts import Finding


@dataclass
class Posted:
    case_id: str
    kept: list[Finding] = field(default_factory=list)
    dropped: list[tuple[Finding, str]] = field(default_factory=list)  # (finding, why)

    def update(self) -> dict:
        """the state update a graph node returns, for the board's append-only reducer."""
        return {"boards": {self.case_id: self.kept}}


def grounded(case_id: str, findings: list[Finding], ledger: set[str]) -> Posted:
    """sorts findings into those proven by reads in this case's ledger and those that are not. writes nothing."""
    posted = Posted(case_id)
    for finding in findings:
        unproven = [ref for ref in finding.evidence_refs if ref not in ledger]
        if finding.case_id != case_id:
            posted.dropped.append((finding, f"belongs to case {finding.case_id}"))
        elif unproven:
            posted.dropped.append((finding, "no read in this case returned " + ", ".join(unproven)))
        else:
            posted.kept.append(finding)
    return posted


def post(case_id: str, findings: list[Finding], store: CaseStore) -> Posted:
    """checks each finding against this case's ledger and keeps the grounded ones. raises LeaseLost when this run
    no longer holds the case; then nothing is kept."""
    lease = store.leases.get(case_id)
    if lease is None:
        raise LeaseLost(case_id)
    posted = grounded(case_id, findings, store.evidence_ids(case_id))
    store.add_findings(lease, posted.kept)
    return posted
