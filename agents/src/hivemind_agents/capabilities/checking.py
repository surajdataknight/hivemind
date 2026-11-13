"""what every investigator shares: building a finding for its own check and case, reading times, and money in words.

a finding's facts are copied from what a read returned; its evidence refs are the ids of those reads, which the board
checks against the case's ledger before it keeps the finding.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from hivemind_agents.capabilities.registry import CheckInput
from hivemind_agents.contracts import Finding, Task


def finding(task: Task, check: CheckInput, type_: str, claim: str, refs: list[str],
            confidence: Literal["low", "medium", "high"] = "high", facts: dict[str, Any] | None = None,
            basis: str | None = None) -> Finding:
    return Finding(case_id=check.case.case_id, task_id=task.task_id, capability=task.capability, type=type_,
                   claim=claim, evidence_refs=sorted(set(refs)), confidence=confidence, confidence_basis=basis,
                   facts=facts or {})


def at(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rupees(paise: int) -> str:
    return f"₹{paise // 100}" if paise % 100 == 0 else f"₹{paise / 100:.2f}"


def minutes(a: datetime, b: datetime) -> int:
    """whole minutes from a to b (negative when b is earlier)."""
    return round((b - a).total_seconds() / 60)
