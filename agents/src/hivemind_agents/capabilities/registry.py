"""the capability registry: the checks the case lead may plan and the scheduler may run (blueprint c5, c6).

a check is registered with its name, what it is for, the params it needs, the scopes the case token must carry, its
cost class, and a function that runs it and returns findings (and, if it needs another check, info requests). only
registered checks can run: a plan or a request naming any other is refused.

the investigators (phase 2.4) are registered at the end of this file; a test may swap one for its own.
"""

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.contracts import CaseRef, CheckResult, Finding, Issue, Task

if TYPE_CHECKING:
    from hivemind_agents.llm import ModelRouter


@dataclass(frozen=True)
class CheckInput:
    """what a check runs with: the case it is for (its reads are kept as proof for that case), the backend, and a
    signal that its answer is no longer wanted (a long check may look at it and stop early). also the issue as
    intake read it (the items the customer named), the case board's grounded findings when the check started (the
    only way one check learns what another found), and the model router (a quarantined reader's only use)."""

    case: CaseRef
    backend: BackendClient
    stop: threading.Event = field(default_factory=threading.Event)
    issue: Issue | None = None
    board: tuple[Finding, ...] = ()
    llm: "ModelRouter | None" = None

    def on_board(self, finding_type: str) -> list[Finding]:
        return [f for f in self.board if f.type == finding_type]


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    params: tuple[str, ...]          # the params it needs: order_id, code
    scopes: frozenset[str]           # the scopes the case token must carry
    run: Callable[[Task, CheckInput], list[Finding] | CheckResult]
    read_only: bool = True           # a read-only check may be tried again after a backend error or a timeout
    cost: Literal["low", "medium", "high"] = "low"  # low: reads only; medium: one model call; high: several


CAPABILITIES: dict[str, Capability] = {}


def _register_investigators() -> None:
    from hivemind_agents.capabilities import investigators  # imports this module: done once it is complete

    CAPABILITIES.update(investigators.ALL)


_register_investigators()
