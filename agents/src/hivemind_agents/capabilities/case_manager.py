"""c4 case manager, basic: opens the case for an issue and moves it along. the business holds the case; a second
message about the same order and issue finds the case that is already open."""

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.contracts import CaseRef, Issue


def open_case(issue: Issue, backend: BackendClient) -> CaseRef:
    row = backend.open_case(issue.case_type, issue.order_id)
    backend.case_id = row["case_id"]
    return CaseRef(case_id=row["case_id"], case_type=row["issue_type"], order_id=row["order_id"],
                   state=row["state"], opened=row["opened"])


def move(case: CaseRef, state: str, backend: BackendClient) -> CaseRef:
    row = backend.move_case(case.case_id, state)
    return case.model_copy(update={"state": row["state"]})


def start_investigating(case: CaseRef, backend: BackendClient) -> CaseRef:
    """a new case starts being looked at; one left mid-way by an earlier run carries on from where it is."""
    return move(case, "INVESTIGATING", backend) if case.state == "OPEN" else case
