"""c4 case manager: owns each case's life (blueprint c4).

- one case per issue: the business keeps at most one open case per customer, order and issue type; a new message
  about the same problem attaches to the open case.
- states: the business holds the case and refuses an illegal move (backend `cases.py`); this build makes the moves
  below, each with a reason kept in the case's history.
- one run per case: before a case is worked, this run takes its lease (`CaseStore.acquire`); while another run
  holds it, this one leaves the case alone.
"""

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.contracts import CaseRef, Issue

WITH_A_PERSON = ("AWAITING_HUMAN", "ESCALATED")  # a person decides next; hivemind does not touch it
IN_HAND = ("ACTIONING",)                          # money or an order is being changed; nothing else may start
STARTS_AGAIN = ("OPEN", "REOPENED", "AWAITING_CUSTOMER", "PENDING_SYSTEM")
# INVESTIGATING or DECIDING: a run before this one stopped part-way; this run carries on from there


def open_case(issue: Issue, backend: BackendClient) -> CaseRef:
    """the open case for this issue, opened if there is none. from here on, reads are proof for this case."""
    row = backend.open_case(issue.case_type, issue.order_id)
    backend.case_id = row["case_id"]
    return CaseRef(case_id=row["case_id"], case_type=row["issue_type"], order_id=row["order_id"],
                   state=row["state"], opened=row["opened"])


def move(case: CaseRef, state: str, backend: BackendClient, reason: str | None = None) -> CaseRef:
    row = backend.move_case(case.case_id, state, reason)
    return case.model_copy(update={"state": row["state"]})


def start(case: CaseRef, backend: BackendClient) -> CaseRef:
    """a new case, or one waiting on the customer or on a system, is looked at again."""
    return move(case, "INVESTIGATING", backend) if case.state in STARTS_AGAIN else case


def finish(case: CaseRef, backend: BackendClient) -> None:
    """the run is done with this case: reads after this are not proof for it."""
    if backend.case_id == case.case_id:
        backend.case_id = None
