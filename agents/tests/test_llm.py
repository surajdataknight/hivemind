"""the model router's waits and retries when the provider is busy, with a fake provider and no network."""

import pytest
from pydantic import BaseModel

from hivemind_agents import settings, tracing
from hivemind_agents.llm import BUSY_WAITS, Completion, GoogleClient, LLMUnavailable, ModelRouter, ProviderBusy


class Answer(BaseModel):
    ok: bool


class Flaky:
    """fails with the queued errors first, then answers."""

    def __init__(self, *errors, pieces=("Hello. ", "Bye.")):
        self.errors = list(errors)
        self.pieces = pieces
        self.requests = 0

    def generate(self, model, system, prompt, temperature, schema):
        self.requests += 1
        if self.errors:
            raise self.errors.pop(0)
        return Completion('{"ok": true}')

    def stream(self, model, system, prompt, temperature, on_done):
        self.requests += 1
        for i, piece in enumerate(self.pieces):
            if self.errors and (i > 0 or not getattr(self.errors[0], "after_first_piece", False)):
                raise self.errors.pop(0)
            yield piece
        on_done(Completion("".join(self.pieces)))


def router(provider):
    waits = []
    r = ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)),
                    {"google": lambda: provider}, sleep=waits.append)
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
