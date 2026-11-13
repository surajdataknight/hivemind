"""groq, the case lead's fallback provider: the strict answer shape, the request, which answers count as busy or a
timeout, and the move from a busy gemini to groq. offline: a fake http transport, no request leaves the machine."""

import json

import httpx
import pytest

from hivemind_agents import settings, tracing
from hivemind_agents.contracts import PlanV1
from hivemind_agents.llm import (LIMIT_DROPS, Completion, GroqClient, LLMUnavailable, ModelRouter, ProviderBusy,
                                 ProviderTimeout, provider_clients, strict_schema)
from test_answer_shapes import SHAPES

PLAN = {"tasks": [{"task_id": "t1", "capability": "bags", "params": {"order_id": None, "code": None},
                   "depends_on": [], "reason": "count the bags", "expects": "one bag short"}],
        "stop_condition": "two sources agree", "rationale_summary": "short"}


def objects(node):
    """every schema object (a node with properties) inside a schema."""
    if isinstance(node, dict):
        if "properties" in node:
            yield node
        for key, value in node.items():
            yield from objects(list(value.values()) if key in ("properties", "$defs") else value)
    elif isinstance(node, list):
        for item in node:
            yield from objects(item)


def keywords(node):
    """every schema keyword used, not counting property names."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from keywords(list(value.values()) if key in ("properties", "$defs") else value)
    elif isinstance(node, list):
        for item in node:
            yield from keywords(item)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_answer_shape_fits_groqs_strict_mode(name):
    schema = strict_schema(SHAPES[name])
    found = list(objects(schema))
    assert found
    for node in found:
        assert node["required"] == list(node["properties"])  # every property required
        assert node["additionalProperties"] is False
    assert not LIMIT_DROPS & set(keywords(schema))


def test_an_optional_value_stays_nullable_and_a_reference_stands_alone():
    schema = strict_schema(PlanV1)
    params = schema["$defs"]["TaskParams"]["properties"]
    assert {"type": "null"} in params["order_id"]["anyOf"]
    assert schema["$defs"]["TaskSpec"]["properties"]["params"] == {"$ref": "#/$defs/TaskParams"}


def test_a_field_named_like_a_keyword_is_kept():
    from pydantic import BaseModel

    class Odd(BaseModel):
        pattern: str
        title: str = ""

    assert list(strict_schema(Odd)["properties"]) == ["pattern", "title"]


# the request and the answers


def client(handler) -> GroqClient:
    return GroqClient("gsk_test", http=httpx.Client(transport=httpx.MockTransport(handler)))


def answering(content: str, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 120}})

    return handler


def test_the_request_asks_for_the_plan_in_strict_mode():
    seen = []
    done = client(answering(json.dumps(PLAN), seen)).generate("openai/gpt-oss-20b", "sys", "prompt", 0.0, PlanV1, 45)
    (request,) = seen
    body = json.loads(request.content)
    assert request.url == GroqClient.URL and request.headers["Authorization"] == "Bearer gsk_test"
    assert body["model"] == "openai/gpt-oss-20b" and body["temperature"] == 0.0
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == strict_schema(PlanV1)
    assert (done.input_tokens, done.output_tokens) == (900, 120)
    assert PlanV1.model_validate_json(done.text).tasks[0].capability == "bags"


def test_the_time_limit_goes_with_the_request():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client(handler).generate("m", "s", "p", 0.0, None, 12.5)
    assert seen[0]["read"] == 12.5


@pytest.mark.parametrize("code, kind", [(429, ProviderBusy), (503, ProviderBusy), (500, ProviderBusy),
                                        (504, ProviderTimeout)])
def test_busy_and_timeout_answers(code, kind):
    with pytest.raises(kind) as caught:
        client(lambda request: httpx.Response(code, text="slow down")).generate("m", "s", "p", 0.0, None, 5)
    assert caught.type is kind


def test_a_refused_request_is_an_error_not_busy():
    with pytest.raises(RuntimeError, match="groq: http 400"):
        client(lambda request: httpx.Response(400, text="bad schema")).generate("m", "s", "p", 0.0, PlanV1, 5)


def refused_for_its_shape(generation: str):
    return lambda request: httpx.Response(400, json={"error": {
        "message": "Generated JSON does not match the expected schema.", "type": "invalid_request_error",
        "code": "json_validate_failed", "failed_generation": generation}})


def test_an_answer_groq_refuses_for_its_shape_goes_to_the_routers_own_check():
    """seen 2026-11-13: gpt-oss wrote "done": "false"; groq refused it, and the call used to end there."""
    from pydantic import BaseModel

    class Turn(BaseModel):
        done: bool

    done = client(refused_for_its_shape('{"done": "false"}')).generate("m", "s", "p", 0.0, Turn, 5)
    assert Turn.model_validate_json(done.text).done is False


def test_a_wrong_answer_groq_refused_gets_the_routers_one_repair_try():
    from pydantic import BaseModel

    class Turn(BaseModel):
        done: bool

    answers = [refused_for_its_shape('{"done": "maybe"}'),
               lambda request: httpx.Response(200, json={"choices": [{"message": {"content": '{"done": true}'}}]})]
    groq = client(lambda request: answers.pop(0)(request))
    models = settings.load_models()
    r = ModelRouter(models, tracing.Tracer(tracing.make_provider(None)), {"google": lambda: groq, "groq": lambda: groq},
                    sleep=lambda s: None)
    assert r.structured("simulated_customer", "s", "p", Turn).done is True
    assert r.calls == 2  # the answer, then the repair try


def test_no_answer_in_time_is_a_timeout_and_no_connection_is_busy():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ProviderTimeout):
        client(slow).generate("m", "s", "p", 0.0, None, 5)
    with pytest.raises(ProviderBusy) as caught:
        client(down).generate("m", "s", "p", 0.0, None, 5)
    assert not isinstance(caught.value, ProviderTimeout)


def test_a_stream_from_groq_comes_whole():
    seen, finished = [], []
    pieces = list(client(answering("Hello.", seen)).stream("m", "s", "p", 0.3, finished.append, 20))
    assert pieces == ["Hello."] and finished[0].text == "Hello."


# from a busy gemini to groq


class Google:
    def __init__(self):
        self.models = []

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        self.models.append(model)
        raise ProviderBusy("ServerError: 503 UNAVAILABLE high demand")


class Groq:
    def __init__(self):
        self.models = []

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        self.models.append(model)
        return Completion(json.dumps(PLAN), 900, 120)


def test_a_busy_groq_sends_the_case_lead_to_gemini_at_once():
    busy, answers, waits = Google(), Groq(), []  # the fakes: one always busy, one that answers
    r = ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)),
                    {"groq": lambda: busy, "google": lambda: answers}, sleep=waits.append)
    plan = r.structured("case_lead", "s", "p", PlanV1)
    assert plan.tasks[0].capability == "bags"
    assert (busy.models, answers.models, waits, r.fallbacks) == (
        ["openai/gpt-oss-120b"], ["gemini-3.5-flash-lite"], [], 1)


def test_with_no_groq_key_the_case_lead_gives_no_plan_and_says_why(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    google = Google()
    providers = {**provider_clients(settings.load_models()), "google": lambda: google}
    r = ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)), providers,
                    sleep=lambda s: None)
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY is not set"):
        r.structured("case_lead", "s", "p", PlanV1)


def test_the_case_lead_is_groqs_gpt_oss_120b_and_its_fallback_googles_flash_lite():
    for name in ("case_lead", "reconciler_complex"):
        job = settings.load_models().job(name)
        assert (job.provider, job.model, job.fallback_provider, job.fallback) == (
            "groq", "openai/gpt-oss-120b", "google", "gemini-3.5-flash-lite")
    assert settings.load_models().key_envs["groq"] == "GROQ_API_KEY"


def test_a_fallback_provider_models_toml_does_not_list_is_refused():
    text = ('[providers.google]\nkey_env = "K"\n[jobs.x]\nprovider = "google"\nmodel = "m"\n'
            'fallback_provider = "nowhere"\nfallback = "n"\n')
    with pytest.raises(ValueError, match="nowhere"):
        settings.load_models(text)
