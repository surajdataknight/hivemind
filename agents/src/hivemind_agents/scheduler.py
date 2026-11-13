"""c7 DAG scheduler: which checks are ready, which are blocked, the case's budgets, and how one check runs.

the graph runs the checks in waves (graph.py): each wave is the checks whose checks are all done, at most MAX_PARALLEL
side by side. each check has its own time limit, never past the case's clock. nothing is dropped silently: a check
that cannot run is recorded as blocked (a check it needs did not finish), not_run (the case ran out of its budget),
timeout (no answer in time) or cancelled (the stop condition was met first; never a minimum check).
"""

import contextvars
import dataclasses
import random
import threading
import time
from typing import Callable

from hivemind_agents.backend_client import BackendError
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CheckResult, Task, TaskOutcome

MAX_PARALLEL = 5
TASK_SECONDS = 8.0      # one try of one check (blueprint 2.3, deadline_ms 8000)
# the case's budgets (blueprint c6), counted from the start of its plan, the case lead's own call included
CASE_SECONDS = 60.0     # wall clock for the automated path
MAX_MODEL_CALLS = 25
MAX_TOKENS = 80_000     # input plus output
READ_TRIES = 3          # a read-only check that hits a backend error or a timeout is tried this many times in all
POLL_SECONDS = 0.05     # how often a waiting check looks for the stop signal


def key(case_id: str, task_id: str) -> str:
    return f"{case_id}/{task_id}"


def _done(outcomes: dict[str, TaskOutcome], task_id: str) -> bool:
    outcome = outcomes.get(task_id)
    return outcome is not None and outcome.status == "done"


def ready(tasks: list[Task], outcomes: dict[str, TaskOutcome]) -> list[Task]:
    """checks not decided yet whose checks all finished, in the plan's order."""
    return [t for t in tasks if t.task_id not in outcomes and all(_done(outcomes, d) for d in t.depends_on)]


def blocked(tasks: list[Task], outcomes: dict[str, TaskOutcome]) -> list[Task]:
    """checks not decided yet that wait on a check which did not finish."""
    return [t for t in tasks if t.task_id not in outcomes
            and any(d in outcomes and not _done(outcomes, d) for d in t.depends_on)]


def out_of_time(started: float | None, now: float) -> bool:
    return started is not None and now - started > CASE_SECONDS


def seconds_left(started: float | None, now: float) -> float:
    """the time one try of a check may take: its own limit, never past the case's clock."""
    if started is None:
        return TASK_SECONDS
    return max(0.0, min(TASK_SECONDS, CASE_SECONDS - (now - started)))


def over_budget(started: float | None, now: float, calls_used: int, tokens_used: int) -> str | None:
    """why no further wave may start, or None. checks already running finish; the rest are not run."""
    if out_of_time(started, now):
        return "out of time"
    if calls_used > MAX_MODEL_CALLS:
        return f"over the budget of {MAX_MODEL_CALLS} model calls"
    if tokens_used > MAX_TOKENS:
        return f"over the budget of {MAX_TOKENS} tokens"
    return None


def backoff(attempt: int, rng: random.Random) -> float:
    """seconds to wait before the next try: 0.5 s, then 1 s, with jitter so retries do not come in step."""
    return 0.5 * (2 ** (attempt - 1)) * (0.5 + rng.random())


class _GaveUp(Exception):
    def __init__(self, status: str, why: str):
        super().__init__(why)
        self.status, self.why = status, why


def _one_try(task: Task, capability: Capability, check: CheckInput, seconds: float,
             stop: threading.Event | None) -> CheckResult:
    """one try in a thread of its own, waited for up to `seconds`, or until the stop signal. a try given up on has
    its backend client closed, so a late read adds no proof; its answer is thrown away."""
    client = check.backend.for_case(check.case.case_id) if check.backend is not None else None
    mine = dataclasses.replace(check, backend=client, stop=threading.Event())
    box: dict = {}
    finished = threading.Event()

    def work():
        try:
            box["result"] = capability.run(task, mine)
        except BaseException as exc:  # handed back to the scheduler's thread below
            box["error"] = exc
        finally:
            finished.set()

    context = contextvars.copy_context()  # the check's spans nest under its own span
    threading.Thread(target=context.run, args=(work,), name=f"check-{task.task_id}", daemon=True).start()
    end = time.monotonic() + seconds
    while not finished.wait(min(POLL_SECONDS, max(0.0, end - time.monotonic()))):
        why = None
        if stop is not None and stop.is_set():
            why = ("cancelled", "the stop condition was met")
        elif time.monotonic() >= end:
            why = ("timeout", f"no answer within {seconds:g} s")
        if why:
            mine.stop.set()
            if client is not None:
                client.close()
            raise _GaveUp(*why)
    if "error" in box:
        raise box["error"]
    result = box["result"]
    return result if isinstance(result, CheckResult) else CheckResult(findings=result)


def run_task(task: Task, capability: Capability, check: CheckInput, sleep: Callable[[float], None],
             rng: random.Random, deadline: Callable[[], float] = lambda: TASK_SECONDS,
             stop: threading.Event | None = None) -> TaskOutcome:
    """runs one check. a backend error or a timeout on a read-only check is tried again after a jittered wait, while
    the case has time; any other error is a bug and is raised, so the run ends with the "having trouble" reply rather
    than a guess. `deadline` gives the seconds one try may take; `stop` cancels the check (never a minimum check)."""
    attempts = 0
    while True:
        if stop is not None and stop.is_set():
            return TaskOutcome(task_id=task.task_id, status="cancelled", error="the stop condition was met",
                               attempts=attempts)
        seconds = deadline()
        if seconds <= 0:
            return TaskOutcome(task_id=task.task_id, status="timeout", error="the case ran out of time",
                               attempts=attempts)
        attempts += 1
        try:
            result = _one_try(task, capability, check, seconds, stop)
        except _GaveUp as gave_up:
            if gave_up.status == "cancelled" or not capability.read_only or attempts >= READ_TRIES:
                return TaskOutcome(task_id=task.task_id, status=gave_up.status, error=gave_up.why,
                                   attempts=attempts)
            sleep(backoff(attempts, rng))
            continue
        except BackendError as exc:
            if not capability.read_only or attempts >= READ_TRIES:
                return TaskOutcome(task_id=task.task_id, status="error", error=str(exc), attempts=attempts)
            sleep(backoff(attempts, rng))
            continue
        if any(f.task_id != task.task_id or f.case_id != check.case.case_id for f in result.findings):
            return TaskOutcome(task_id=task.task_id, status="error", attempts=attempts,
                               error="a finding named another check or case")
        return TaskOutcome(task_id=task.task_id, status="done", findings=result.findings,
                           info_requests=result.info_requests, attempts=attempts)
