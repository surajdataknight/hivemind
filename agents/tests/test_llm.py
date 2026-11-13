"""the model router's waits and retries when the provider is busy, with a fake provider and no network."""

import pytest
from pydantic import BaseModel

from hivemind_agents import settings, tracing
from hivemind_agents.llm import (BUSY_WAITS, Completion, GoogleClient, LLMUnavailable, ModelRouter, ProviderBusy,
                                 ProviderTimeout)

# the case lead's model and its fallback when busy, as models.toml names them (record 013)
LEAD = settings.load_models().job("case_lead").model
FALLBACK = settings.load_models().job("case_lead").fallback



class Answer(BaseModel):
    ok: bool


class Flaky:
    """fails with the queued errors first, then answers."""

    def __init__(self, *errors, pieces=("Hello. ", "Bye.")):
        self.errors = list(errors)
        self.pieces = pieces
        self.requests = 0
        self.timeouts = []
        self.models = []

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        self.requests += 1
        self.timeouts.append(timeout_s)
        self.models.append(model)
        if self.errors:
            raise self.errors.pop(0)
        return Completion('{"ok": true}')

    def stream(self, model, system, prompt, temperature, on_done, timeout_s=None):
        self.requests += 1
        for i, piece in enumerate(self.pieces):
            if self.errors and (i > 0 or not getattr(self.errors[0], "after_first_piece", False)):
                raise self.errors.pop(0)
            yield piece
        on_done(Completion("".join(self.pieces)))


def router(provider):
    waits = []
    r = ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)),
                    {"google": lambda: provider, "groq": lambda: provider}, sleep=waits.append)
    return r, waits


def busy(after_first_piece=False):
    exc = ProviderBusy("ServerError: 503 UNAVAILABLE")
    exc.after_first_piece = after_first_piece
    return exc


def test_a_busy_provider_is_asked_again_after_a_wait():
    provider = Flaky(busy(), busy())
    r, waits = router(provider)
    assert r.structured("intake", "s", "p", Answer).ok
    assert (provider.requests, waits, r.calls, r.retries) == (3, list(BUSY_WAITS), 1, 2)


def test_still_busy_after_two_retries_means_no_answer():
    provider = Flaky(busy(), busy(), busy())
    r, waits = router(provider)
    with pytest.raises(LLMUnavailable, match="busy after 3 tries"):
        r.structured("intake", "s", "p", Answer)
    assert (provider.requests, waits) == (3, list(BUSY_WAITS))


def test_a_job_with_its_own_waits_waits_those():
    provider = Flaky(busy(), busy(), busy(), busy())
    r, waits = router(provider)
    assert r.structured("simulated_customer", "s", "p", Answer).ok
    assert waits == [5, 15, 30, 60]


def test_other_errors_are_not_retried():
    provider = Flaky(ValueError("bad key"))
    r, waits = router(provider)
    with pytest.raises(LLMUnavailable, match="bad key"):
        r.structured("intake", "s", "p", Answer)
    assert (provider.requests, waits) == (1, [])


def test_a_stream_busy_before_its_first_piece_is_asked_again():
    provider = Flaky(busy())
    r, waits = router(provider)
    assert "".join(r.stream("communicator", "s", "p")) == "Hello. Bye."
    assert (provider.requests, waits) == (2, [BUSY_WAITS[0]])


def test_a_stream_busy_after_its_first_piece_is_not_asked_again():
    provider = Flaky(busy(after_first_piece=True))
    r, waits = router(provider)
    got = []
    with pytest.raises(LLMUnavailable):
        for piece in r.stream("communicator", "s", "p"):
            got.append(piece)
    assert (got, provider.requests, waits) == (["Hello. "], 1, [])


class Coded(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


class ReadTimeout(Exception):
    pass


@pytest.mark.parametrize("exc, is_busy", [
    (Coded(503), True), (Coded(429), True), (Coded(500), True), (ReadTimeout("slow"), True),
    (Coded(400), False), (Coded(403), False), (Coded(404), False), (ValueError("x"), False),
])
def test_which_google_errors_count_as_busy(exc, is_busy):
    assert isinstance(GoogleClient._busy(exc), ProviderBusy) == is_busy


@pytest.mark.parametrize("exc, is_timeout", [
    (Coded(504), True), (ReadTimeout("slow"), True), (Coded(503), False), (Coded(429), False),
])
def test_which_google_errors_count_as_a_timeout(exc, is_timeout):
    assert isinstance(GoogleClient._busy(exc), ProviderTimeout) == is_timeout


def google_error(kind, code, status, message):
    from google.genai import errors

    details = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaValue": "20"}]}
    return getattr(errors, kind)(code, {"error": {"code": code, "message": message, "status": status,
                                                  "details": [details]}})


@pytest.mark.parametrize("exc, short", [
    (google_error("ClientError", 429, "RESOURCE_EXHAUSTED",
                  "You exceeded your current quota, please check your plan and billing details. For more information "
                  "on this error, head to: https://example.invalid/rate-limits. \n* Quota exceeded for metric: x, "
                  "limit: 20"),
     "ClientError: 429 RESOURCE_EXHAUSTED: You exceeded your current quota, please check your plan and billing "
     "details"),
    (google_error("ServerError", 503, "UNAVAILABLE", "This model is currently experiencing high demand. Spikes in "
                  "demand are usually temporary. Please try again later."),
     "ServerError: 503 UNAVAILABLE: This model is currently experiencing high demand"),
    (google_error("ServerError", 504, "DEADLINE_EXCEEDED", "Deadline expired before operation could complete."),
     "ServerError: 504 DEADLINE_EXCEEDED: Deadline expired before operation could complete."),
])
def test_a_google_error_becomes_one_short_line(exc, short):
    assert len(str(exc)) > len(short)  # google's own text carries the whole json answer
    assert str(GoogleClient._busy(exc)) == short


def test_a_refused_google_request_reaches_the_note_as_one_short_line():
    provider = Flaky(google_error("ClientError", 400, "INVALID_ARGUMENT", "Minimum allowed deadline is 10s. " * 40))
    r, _ = router(provider)
    with pytest.raises(LLMUnavailable) as raised:
        r.structured("intake", "s", "p", Answer)
    assert str(raised.value) == "intake: ClientError: 400 INVALID_ARGUMENT: Minimum allowed deadline is 10s"
    assert raised.value.__cause__ is not None  # the whole error stays on the cause, for the trace


def test_a_long_error_without_a_message_is_cut_to_one_line():
    provider = Flaky(ValueError("x\n" * 500))
    r, _ = router(provider)
    with pytest.raises(LLMUnavailable) as raised:
        r.structured("intake", "s", "p", Answer)
    assert "\n" not in str(raised.value) and len(str(raised.value)) <= len("intake: ") + 200


def test_the_case_lead_is_not_asked_again_after_a_timeout():
    provider = Flaky(ProviderTimeout("ServerError: 504 DEADLINE_EXCEEDED"))
    r, waits = router(provider)
    with pytest.raises(LLMUnavailable, match="no answer within 45 s"):
        r.structured("case_lead", "s", "p", Answer)
    assert (provider.requests, waits, r.retries) == (1, [], 0)


def test_when_the_case_leads_model_is_busy_its_fallback_is_asked_at_once():
    provider = Flaky(busy())
    r, waits = router(provider)
    assert r.structured("case_lead", "s", "p", Answer).ok
    assert provider.models == [LEAD, FALLBACK]
    assert (waits, r.fallbacks) == ([], 1)


def test_a_busy_fallback_is_asked_again_after_a_wait_and_then_given_up():
    provider = Flaky(busy(), busy(), busy())
    r, waits = router(provider)
    with pytest.raises(LLMUnavailable, match="busy after 3 tries"):
        r.structured("case_lead", "s", "p", Answer)
    assert provider.models == [LEAD, FALLBACK, FALLBACK]
    assert waits == [BUSY_WAITS[1]]


def test_no_fallback_after_a_timeout_or_an_error_that_is_ours():
    timed_out = Flaky(ProviderTimeout("ServerError: 504 DEADLINE_EXCEEDED"))
    r, _ = router(timed_out)
    with pytest.raises(LLMUnavailable):
        r.structured("case_lead", "s", "p", Answer)
    refused = Flaky(ValueError("400 INVALID_ARGUMENT"))
    r2, _ = router(refused)
    with pytest.raises(LLMUnavailable, match="INVALID_ARGUMENT"):
        r2.structured("case_lead", "s", "p", Answer)
    assert timed_out.models == refused.models == [LEAD]
    assert r.fallbacks == r2.fallbacks == 0


def test_the_repair_try_stays_on_the_fallback():
    provider = Flaky(busy())
    texts = ["not json", '{"ok": true}']
    original = provider.generate

    def generate(model, system, prompt, temperature, schema, timeout_s=None):
        if provider.errors:
            return original(model, system, prompt, temperature, schema, timeout_s)
        provider.models.append(model)
        return Completion(texts.pop(0))

    provider.generate = generate
    r, _ = router(provider)
    assert r.structured("case_lead", "s", "p", Answer).ok
    assert provider.models == [LEAD, FALLBACK, FALLBACK]


def test_only_the_planning_jobs_have_a_fallback():
    jobs = settings.load_models().jobs
    assert {name: job.fallback for name, job in jobs.items()} == {"intake": None, "communicator": None,
                                                                  "quarantined_reader": None,
                                                                  "case_lead": "gemini-3.5-flash-lite",
                                                                  "reconciler": None,
                                                                  "reconciler_complex": "gemini-3.5-flash-lite",
                                                                  "simulated_customer": None, "baseline_agent": None}


@pytest.mark.parametrize("job", ["intake", "communicator"])
def test_other_jobs_are_asked_again_after_a_timeout(job):
    provider = Flaky(ProviderTimeout("ReadTimeout: slow"))
    r, waits = router(provider)
    if job == "intake":
        assert r.structured(job, "s", "p", Answer).ok
    else:
        assert "".join(r.stream(job, "s", "p")) == "Hello. Bye."
    assert (provider.requests, waits) == (2, [BUSY_WAITS[0]])


def test_each_job_has_its_own_time_limit():
    jobs = settings.load_models().jobs
    assert {name: job.timeout_s for name, job in jobs.items()} == {"intake": 20, "communicator": 20,
                                                                   "quarantined_reader": 6, "case_lead": 45,
                                                                   "reconciler": 20, "reconciler_complex": 20,
                                                                   "simulated_customer": 20, "baseline_agent": 20}


def test_the_router_gives_the_provider_the_jobs_time_limit():
    provider = Flaky(busy())
    r, _ = timed_router(provider, seconds_per_request=0)
    r.structured("case_lead", "s", "p", Answer)
    r.structured("intake", "s", "p", Answer)
    assert provider.timeouts == [45, 45, 20]  # the case lead's fallback is asked at once, with no wait


# the case lead's shared limit: all its tries together, waits included, within 45 s


class Clock:
    """a fake clock: each request takes `per_request` seconds, and each wait its own length."""

    def __init__(self, per_request: float):
        self.now, self.per_request = 0.0, per_request

    def __call__(self) -> float:
        return self.now


class Timed(Flaky):
    def __init__(self, clock: Clock, *errors, answers=None):
        super().__init__(*errors)
        self.clock = clock
        self.answers = list(answers or [])

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        if self.clock.per_request >= timeout_s:  # the answer would come after the limit: google says 504
            self.clock.now += timeout_s
            self.requests += 1
            self.timeouts.append(timeout_s)
            self.models.append(model)
            raise ProviderTimeout("ServerError: 504 DEADLINE_EXCEEDED")
        self.clock.now += self.clock.per_request
        if self.answers and not self.errors:
            self.requests += 1
            self.timeouts.append(timeout_s)
            self.models.append(model)
            return Completion(self.answers.pop(0))
        return super().generate(model, system, prompt, temperature, schema, timeout_s)


def timed_router(provider, seconds_per_request=10.0):
    clock = provider.clock if isinstance(provider, Timed) else Clock(seconds_per_request)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock.now += seconds

    r = ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)),
                    {"google": lambda: provider, "groq": lambda: provider}, sleep=sleep, clock=clock)
    return r, waits


def test_only_the_case_lead_has_45_seconds_for_all_its_tries():
    jobs = settings.load_models().jobs
    # the quarantined reader has 6 s in all, inside its check's 8 s; the reconciler 20 s, then the code's sheet
    assert {name: job.total_s for name, job in jobs.items()} == {"intake": None, "communicator": None,
                                                                 "quarantined_reader": 6, "case_lead": 45,
                                                                 "reconciler": 20, "reconciler_complex": 20,
                                                                 "simulated_customer": None, "baseline_agent": None}


def test_each_retry_of_the_case_lead_gets_only_the_time_left():
    clock = Clock(per_request=10)
    provider = Timed(clock, busy(), busy())
    r, waits = timed_router(provider)
    assert r.structured("case_lead", "s", "p", Answer).ok
    # 45; the fallback at once with 35 left; then a 5 s wait and 20 left
    assert provider.timeouts == [45, 35, 20]
    assert provider.models == [LEAD, FALLBACK, FALLBACK]
    assert waits == [BUSY_WAITS[1]] and clock.now == 35


def test_busy_answers_never_take_the_case_lead_past_45_seconds():
    clock = Clock(per_request=20)  # google takes 20 s to say it is busy, as on 2026-11-12 (82 s in all)
    provider = Timed(clock, busy(), busy(), busy())
    r, waits = timed_router(provider)
    with pytest.raises(LLMUnavailable, match="no answer within 45 s in all"):
        r.structured("case_lead", "s", "p", Answer)
    assert provider.requests == 2 and waits == []  # the fallback at once; no 5 s wait with no time after it
    assert clock.now <= 45


def test_a_try_is_not_started_with_under_five_seconds_left():
    clock = Clock(per_request=41)
    provider = Timed(clock, busy())
    r, waits = timed_router(provider)
    with pytest.raises(LLMUnavailable, match="in all"):
        r.structured("case_lead", "s", "p", Answer)  # 41 s, then 4 s left: the fallback is not asked
    assert provider.requests == 1 and clock.now == 41 and waits == []


def test_the_repair_try_shares_the_45_seconds():
    clock = Clock(per_request=30)
    provider = Timed(clock, answers=["not json"])
    r, _ = timed_router(provider)
    with pytest.raises(LLMUnavailable, match="no answer within 45 s"):
        r.structured("case_lead", "s", "p", Answer)
    assert provider.timeouts == [45, 15]  # the repair try got only the 15 s left, and timed out
    assert clock.now == 45


def test_jobs_with_no_shared_limit_keep_their_limit_on_every_try():
    clock = Clock(per_request=19)
    provider = Timed(clock, busy(), busy())
    r, _ = timed_router(provider)
    assert r.structured("intake", "s", "p", Answer).ok
    assert provider.timeouts == [20, 20, 20]


def test_google_gets_the_time_limit_with_each_request():
    client = GoogleClient("offline-test-key")
    assert client._config("s", 0.0, None, 45).http_options.timeout == 45000
    assert client._config("s", 0.0, Answer, 20).http_options.timeout == 20000


def test_google_never_gets_a_deadline_under_its_minimum():
    """google refuses one under 10 s (400 invalid argument, seen 2026-11-13 on the note reader's 6 s)."""
    client = GoogleClient("offline-test-key")
    assert client._config("s", 0.0, None, 6).http_options.timeout == 10000


def test_a_limit_under_googles_minimum_is_kept_on_our_side(monkeypatch):
    import threading

    client = GoogleClient("offline-test-key")
    release = threading.Event()

    def slow(**kwargs):
        release.wait(5)
        return None

    monkeypatch.setattr(client._client.models, "generate_content", slow)
    with pytest.raises(ProviderTimeout, match="0.2 s"):
        client.generate("m", "s", "p", 0.0, None, 0.2)
    release.set()


def test_google_gets_the_answer_shape_without_its_limits():
    """google refused the reconciler's full shape with its length limits (400 invalid argument, 2026-11-13); the
    router checks every answer against the full shape itself."""
    from hivemind_agents.capabilities.reconciler import ReconcilerAnswer
    from hivemind_agents.llm import LIMIT_DROPS

    sent = GoogleClient("offline-test-key")._config("s", 0.0, ReconcilerAnswer, 20).response_json_schema

    def keywords(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keywords(value)
        elif isinstance(node, list):
            for value in node:
                yield from keywords(value)

    assert not LIMIT_DROPS & set(keywords(sent))
    assert "facts" in sent["properties"] and "$defs" in sent
