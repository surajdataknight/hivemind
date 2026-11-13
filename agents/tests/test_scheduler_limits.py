"""the dag scheduler's limits on one check (c7): its own time limit within the case's clock, a timeout retried for
reads, the stop signal that cancels a check, a check's info requests, and a check given up on adding no proof."""

import random
import threading
import time

import pytest

from hivemind_agents import scheduler
from hivemind_agents.backend_client import BackendClient, BackendError
from hivemind_agents.capabilities.registry import Capability, CheckInput
from hivemind_agents.contracts import CaseRef, CheckResult, Finding, InfoRequest, Task

CASE = CaseRef(case_id="CASE-1", case_type="item_missing_or_wrong", order_id="ORD-100001", state="INVESTIGATING",
               opened=False)


def task(i: int = 1) -> Task:
    return Task(task_id=f"t{i}", capability="order", params={"order_id": "ORD-100001"}, depends_on=[],
                mandatory=False)


def capability(run, read_only=True) -> Capability:
    return Capability(name="order", description="reads the order", params=("order_id",),
                      scopes=frozenset({"orders:read:self"}), run=run, read_only=read_only)


def finding(t: Task) -> Finding:
    return Finding(case_id="CASE-1", task_id=t.task_id, capability="order", type="order.checked",
                   claim="the order was read", evidence_refs=["ev_1"], confidence="high")


def check_input() -> CheckInput:
    return CheckInput(case=CASE, backend=None)


def run(run_fn, *, read_only=True, seconds=0.2, stop=None, waits=None):
    return scheduler.run_task(task(), capability(run_fn, read_only), check_input(),
                              (waits if waits is not None else []).append, random.Random(0),
                              deadline=lambda: seconds, stop=stop)


def hang(t, check):
    check.stop.wait(5)  # a check that never answers, until it is told its answer is not wanted
    return []


# the time limit


def test_a_check_with_no_answer_in_time_is_tried_three_times_then_recorded_as_timeout():
    waits = []
    began = time.monotonic()
    outcome = run(hang, seconds=0.1, waits=waits)
    assert outcome.status == "timeout" and outcome.attempts == scheduler.READ_TRIES
    assert outcome.error == "no answer within 0.1 s"
    assert len(waits) == scheduler.READ_TRIES - 1
    assert time.monotonic() - began < 2  # three tries of 0.1 s, not the check's 5 s


def test_a_read_that_answers_after_one_timeout_is_done():
    tries = []

    def slow_once(t, check):
        tries.append(1)
        if len(tries) == 1:
            check.stop.wait(5)
            return []
        return [finding(t)]

    outcome = run(slow_once, seconds=0.1)
    assert outcome.status == "done" and outcome.attempts == 2 and len(outcome.findings) == 1


def test_a_check_that_is_not_read_only_is_not_tried_again_after_a_timeout():
    waits = []
    outcome = run(hang, read_only=False, seconds=0.1, waits=waits)
    assert (outcome.status, outcome.attempts, waits) == ("timeout", 1, [])


def test_a_check_is_told_to_stop_when_it_is_given_up_on():
    told = threading.Event()

    def watch(t, check):
        check.stop.wait(5)
        told.set()
        return []

    run(watch, read_only=False, seconds=0.1)
    assert told.wait(1)


def test_with_no_time_left_on_the_case_a_check_is_not_tried():
    calls = []
    outcome = scheduler.run_task(task(), capability(lambda t, c: calls.append(1) or []), check_input(),
                                 lambda s: None, random.Random(0), deadline=lambda: 0.0)
    assert (outcome.status, outcome.attempts, calls) == ("timeout", 0, [])
    assert outcome.error == "the case ran out of time"


def test_a_try_never_runs_past_the_case_clock():
    assert scheduler.seconds_left(None, 100.0) == scheduler.TASK_SECONDS
    assert scheduler.seconds_left(100.0, 100.0) == scheduler.TASK_SECONDS
    assert scheduler.seconds_left(100.0, 100.0 + scheduler.CASE_SECONDS - 3) == pytest.approx(3)
    assert scheduler.seconds_left(100.0, 100.0 + scheduler.CASE_SECONDS + 5) == 0.0


def test_the_default_time_limit_is_the_blueprints_eight_seconds():
    assert scheduler.TASK_SECONDS == 8.0


# the stop signal


def test_the_stop_signal_cancels_a_check_that_is_running():
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    began = time.monotonic()
    outcome = run(hang, seconds=5, stop=stop)
    assert outcome.status == "cancelled" and outcome.error == "the stop condition was met"
    assert time.monotonic() - began < 1


def test_a_check_cancelled_before_it_starts_is_never_run():
    stop = threading.Event()
    stop.set()
    calls = []
    outcome = run(lambda t, c: calls.append(1) or [], stop=stop)
    assert (outcome.status, outcome.attempts, calls) == ("cancelled", 0, [])


def test_a_cancelled_check_is_not_tried_again():
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()
    waits = []
    outcome = run(hang, seconds=5, stop=stop, waits=waits)
    assert outcome.status == "cancelled" and outcome.attempts == 1 and waits == []


# what a check returns


def test_a_check_may_ask_for_another_check():
    def asks(t, check):
        return CheckResult(findings=[finding(t)], info_requests=[InfoRequest(capability="delivery",
                                                                             reason="bags at handover")])

    outcome = run(asks)
    assert outcome.status == "done" and len(outcome.findings) == 1
    assert [r.capability for r in outcome.info_requests] == ["delivery"]


def test_a_check_may_still_return_only_its_findings():
    outcome = run(lambda t, c: [finding(t)])
    assert outcome.status == "done" and outcome.info_requests == []


def test_a_bug_inside_the_checks_thread_is_raised_in_the_scheduler():
    def broken(t, check):
        raise RuntimeError("a bug")

    with pytest.raises(RuntimeError, match="a bug"):
        run(broken)


def test_a_backend_error_inside_the_checks_thread_is_retried_as_before():
    tries = []

    def flaky(t, check):
        tries.append(1)
        if len(tries) < 3:
            raise BackendError("order: http 503")
        return [finding(t)]

    outcome = run(flaky)
    assert outcome.status == "done" and outcome.attempts == 3


# a check given up on adds no proof


class Http:
    def __init__(self, delay=0.0):
        self.delay = delay

    def request(self, method, url, **kwargs):
        time.sleep(self.delay)
        return Response()


class Response:
    status_code = 200

    @staticmethod
    def json():
        from hivemind_agents import evidence

        body = {"order_id": "ORD-100001"}
        return {**body, "evidence_id": "ev_" + evidence.body_hash(body)[:32]}


def client(tracer, ledger, delay=0.0) -> BackendClient:
    backend = BackendClient(Http(delay), "token", tracer, ledger=lambda case, op, body: ledger.append(case))
    backend.case_id = "CASE-1"
    return backend


def test_a_checks_own_client_keeps_its_case_whatever_the_run_works_on_next(tracer):
    ledger = []
    shared = client(tracer, ledger)
    mine = shared.for_case("CASE-1")
    shared.case_id = "CASE-2"  # the run moved on to the next issue's case
    mine.order("ORD-100001")
    assert ledger == ["CASE-1"]


def test_a_closed_client_reads_nothing_and_adds_no_proof(tracer):
    ledger = []
    mine = client(tracer, ledger).for_case("CASE-1")
    mine.close()
    with pytest.raises(BackendError, match="stopped"):
        mine.order("ORD-100001")
    assert ledger == []


def test_a_check_that_timed_out_mid_read_adds_no_proof_when_its_read_comes_back(tracer):
    ledger = []
    backend = client(tracer, ledger, delay=0.3)  # the read answers after the check's time is up

    def reads(t, check):
        check.backend.order("ORD-100001")
        return []

    outcome = scheduler.run_task(task(), capability(reads, read_only=False), CheckInput(CASE, backend),
                                 lambda s: None, random.Random(0), deadline=lambda: 0.1)
    assert outcome.status == "timeout"
    time.sleep(0.4)  # the late read has come back by now
    assert ledger == []
