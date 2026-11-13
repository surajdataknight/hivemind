"""the model router: every llm call goes through here, by job name. which model a job uses is in models.toml.

a provider client only has to turn (model, system text, prompt) into text; the router adds the job lookup, the
typed-output check with one repair try, and the trace span.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any
from typing import Callable, Iterator, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from hivemind_agents.settings import Job, ModelSettings, env
from hivemind_agents.tracing import Tracer, set_output, set_usage

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """no usable answer: no key, quota used up, provider down, or output that stayed invalid after a repair try."""


class ProviderBusy(Exception):
    """the provider said to try again later (overloaded, per-minute limit, timeout). the router waits and retries."""


class ProviderTimeout(ProviderBusy):
    """the answer did not come within the job's time limit. retried like busy, unless the job says not to."""


MAX_ERROR_CHARS = 200


def short_error(exc: Exception) -> str:
    """a provider's error in one line: for google, its code, status and the first sentence of its message. google's
    str(exc) carries the whole json answer (over 1,500 characters for a quota error), which then filled the plan's
    note on the scenario panel. the whole error stays on the exception's cause, which the trace keeps."""
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        text = f"{getattr(exc, 'code', '')} {getattr(exc, 'status', '')}: {message.split('. ')[0]}"
    else:
        text = str(exc)
    return " ".join(f"{type(exc).__name__}: {text}".split())[:MAX_ERROR_CHARS]


BUSY_WAITS = (2.0, 5.0)  # seconds before each retry; after the last, the call fails
MIN_TRY_S = 5.0  # under a job's shared limit, a try with less time left than this is not started
BUSY_CODES = {429, 500, 502, 503, 504}
GOOGLE_MIN_DEADLINE_S = 10.0  # google refuses a shorter deadline (400 "minimum allowed deadline is 10s")
# keywords left out of the answer shape a provider gets: google refuses a shape with many of them (400 invalid
# argument), groq's strict mode takes none. the router checks every answer against the full shape itself (pydantic)
LIMIT_DROPS = {"default", "title", "maxLength", "minLength", "pattern", "maxItems", "minItems", "examples"}


def loose_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """the answer shape without its length and pattern limits."""

    def fix(node):
        if isinstance(node, list):
            return [fix(n) for n in node]
        if not isinstance(node, dict):
            return node
        return {key: ({name: fix(sub) for name, sub in value.items()} if key in ("properties", "$defs") else fix(value))
                for key, value in node.items() if key not in LIMIT_DROPS}

    return fix(schema.model_json_schema())


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderClient(Protocol):
    def generate(self, model: str, system: str, prompt: str, temperature: float,
                 schema: type[BaseModel] | None, timeout_s: float) -> Completion: ...

    def stream(self, model: str, system: str, prompt: str, temperature: float,
               on_done: Callable[[Completion], None], timeout_s: float) -> Iterator[str]: ...


class GoogleClient:
    """google ai studio through the google-genai sdk."""

    def __init__(self, key: str):
        from concurrent.futures import ThreadPoolExecutor

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="google")  # for limits under google's

    def _config(self, system: str, temperature: float, schema: type[BaseModel] | None, timeout_s: float):
        # the limit goes with each request, so each job keeps its own (google also gets it as its deadline, never
        # under its minimum; a shorter limit is kept on our side, in generate)
        extra = {"response_mime_type": "application/json", "response_json_schema": loose_schema(schema)} if schema else {}
        seconds = max(timeout_s, GOOGLE_MIN_DEADLINE_S)
        return self._types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            http_options=self._types.HttpOptions(timeout=int(seconds * 1000)), **extra)

    @staticmethod
    def _busy(exc: Exception) -> Exception:
        """the same error, or ProviderBusy when google says to try again later."""
        code = getattr(exc, "code", None)
        if code == 504 or "Timeout" in type(exc).__name__:  # google sends 504 when our deadline passes
            return ProviderTimeout(short_error(exc))
        if code in BUSY_CODES:
            return ProviderBusy(short_error(exc))
        return exc

    def generate(self, model, system, prompt, temperature, schema, timeout_s):
        from concurrent.futures import TimeoutError as Late

        def call():
            return self._client.models.generate_content(
                model=model, contents=prompt, config=self._config(system, temperature, schema, timeout_s))

        try:
            if timeout_s >= GOOGLE_MIN_DEADLINE_S:
                response = call()
            else:  # we stop waiting at our own limit; the request itself ends at google's deadline
                response = self._pool.submit(call).result(timeout=timeout_s)
        except Late as exc:
            raise ProviderTimeout(f"no answer within {timeout_s:g} s") from exc
        except Exception as exc:
            raise self._busy(exc) from exc
        usage = response.usage_metadata
        return Completion(response.text or "", getattr(usage, "prompt_token_count", None),
                          getattr(usage, "candidates_token_count", None))

    def stream(self, model, system, prompt, temperature, on_done, timeout_s):
        parts, usage = [], None
        try:
            for chunk in self._client.models.generate_content_stream(
                    model=model, contents=prompt, config=self._config(system, temperature, None, timeout_s)):
                usage = chunk.usage_metadata or usage
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
        except Exception as exc:
            raise self._busy(exc) from exc
        on_done(Completion("".join(parts), getattr(usage, "prompt_token_count", None),
                           getattr(usage, "candidates_token_count", None)))


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """the answer shape as groq's strict mode wants it: every property required (an optional one stays nullable),
    no other properties, and none of the length or pattern limits, which the router checks itself."""

    def fix(node):
        if isinstance(node, list):
            return [fix(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:  # a reference stands alone
            return {"$ref": node["$ref"]}
        out = {}
        for key, value in node.items():
            if key in LIMIT_DROPS:
                continue
            out[key] = {name: fix(sub) for name, sub in value.items()} if key in ("properties", "$defs") else fix(value)
        if out.get("type") == "object" or "properties" in out:
            out["required"] = list(out.get("properties", {}))
            out["additionalProperties"] = False
        return out

    return fix(schema.model_json_schema())


class GroqClient:
    """groq's api (openai-style chat completions) over plain http. answers in a shape use groq's strict mode."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, key: str, http: httpx.Client | None = None):
        self._http = http or httpx.Client()
        self._auth = {"Authorization": f"Bearer {key}"}

    @staticmethod
    def body(model: str, system: str, prompt: str, temperature: float,
             schema: type[BaseModel] | None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "temperature": temperature,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}]}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_schema(schema)}}
        return body

    def generate(self, model, system, prompt, temperature, schema, timeout_s):
        try:
            response = self._http.post(self.URL, json=self.body(model, system, prompt, temperature, schema),
                                       headers=self._auth, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"groq: {type(exc).__name__}") from exc
        except httpx.HTTPError as exc:  # no connection: like busy, worth one more try
            raise ProviderBusy(f"groq: {type(exc).__name__}") from exc
        code = response.status_code
        if code == 504:
            raise ProviderTimeout(f"groq: http {code}")
        if code in BUSY_CODES:
            raise ProviderBusy(f"groq: http {code}: {response.text[:200]}")
        if code == 400 and schema is not None:
            failed = self._failed_generation(response)
            if failed is not None:  # groq refused the model's own answer for its shape: the router checks it
                return Completion(failed)  # and, if it is wrong, asks once more (small set 2026-11-13: "false")
        if code >= 400:
            raise RuntimeError(f"groq: http {code}: {response.text[:300]}")
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(data["choices"][0]["message"].get("content") or "", usage.get("prompt_tokens"),
                          usage.get("completion_tokens"))

    @staticmethod
    def _failed_generation(response: httpx.Response) -> str | None:
        """the model's answer when groq's strict mode refused it (code json_validate_failed), or None."""
        try:
            error = response.json().get("error") or {}
        except ValueError:
            return None
        if error.get("code") != "json_validate_failed" or not isinstance(error.get("failed_generation"), str):
            return None
        return error["failed_generation"]

    def stream(self, model, system, prompt, temperature, on_done, timeout_s):
        """no job streams from groq yet: the answer comes whole, as one piece."""
        done = self.generate(model, system, prompt, temperature, None, timeout_s)
        yield done.text
        on_done(done)


def provider_clients(models: ModelSettings) -> dict[str, Callable[[], ProviderClient]]:
    """one lazy client per provider, made on its first call, so a missing key only fails the calls that need it."""

    def google() -> ProviderClient:
        return GoogleClient(env(models.key_envs["google"]))

    def groq() -> ProviderClient:
        return GroqClient(env(models.key_envs["groq"]))

    return {"google": google, "groq": groq}


class ModelRouter:
    def __init__(self, models: ModelSettings, tracer: Tracer,
                 providers: dict[str, Callable[[], ProviderClient]] | None = None,
                 busy_waits: tuple[float, ...] = BUSY_WAITS, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.models = models
        self._tracer = tracer
        self._factories = providers if providers is not None else provider_clients(models)
        self._clients: dict[str, ProviderClient] = {}
        self._busy_waits = busy_waits
        self._sleep = sleep
        self._clock = clock
        self.calls = 0  # counted for tests and for the reply's own record; retries are not extra calls
        self.tokens = 0  # input plus output tokens, as the provider reports them (the case budget, c6)
        self.retries = 0
        self.fallbacks = 0  # calls that went on to the job's fallback model
        self._count = threading.Lock()  # checks run side by side, and each may call a model

    def _counted_call(self) -> None:
        with self._count:
            self.calls += 1

    def _counted_tokens(self, done: Completion) -> None:
        with self._count:
            self.tokens += (done.input_tokens or 0) + (done.output_tokens or 0)

    def _client(self, provider: str) -> ProviderClient:
        if provider not in self._clients:
            if provider not in self._factories:
                raise LLMUnavailable(f"no client for the provider {provider!r}")
            try:
                self._clients[provider] = self._factories[provider]()
            except Exception as exc:
                raise LLMUnavailable(f"{provider}: {exc}") from exc
        return self._clients[provider]

    def _fall_back(self, job: Job, model: str, span, exc: ProviderBusy) -> tuple[ProviderClient, str] | None:
        """the job's fallback (its client and model), when its own model said it is busy; not after a timeout (no
        time is left), and once only."""
        if job.fallback is None or model == job.fallback or isinstance(exc, ProviderTimeout):
            return None
        provider = job.fallback_provider or job.provider
        span.set_attribute("hivemind.fallback_provider", provider)
        span.set_attribute("hivemind.fallback_model", job.fallback)
        span.set_attribute("hivemind.fallback_reason", str(exc)[:200])
        client = self._client(provider)  # no key for it: LLMUnavailable, and the caller's own fallback applies
        with self._count:
            self.fallbacks += 1
        return client, job.fallback

    def _deadline(self, job: Job) -> float | None:
        return self._clock() + job.total_s if job.total_s is not None else None

    def _try_seconds(self, job: Job, span, deadline: float | None) -> float:
        """the time the next try may take: the job's limit, and never past its shared limit."""
        if deadline is None:
            return job.timeout_s
        left = deadline - self._clock()
        if left < MIN_TRY_S:
            span.set_attribute("hivemind.timed_out", True)
            raise LLMUnavailable(f"{job.name}: no answer within {job.total_s:g} s in all")
        return min(job.timeout_s, left)

    def _wait_or_fail(self, job: Job, span, tries: int, exc: ProviderBusy, deadline: float | None = None) -> None:
        """after a busy answer: wait and let the caller try again, or give up when the waits are used up, or when
        the job's shared limit would leave no time for another try."""
        job_name = job.name
        if isinstance(exc, ProviderTimeout) and not job.retry_timeouts:
            span.set_attribute("hivemind.timed_out", True)
            raise LLMUnavailable(f"{job_name}: no answer within {job.timeout_s:g} s: {exc}") from exc
        waits = job.busy_waits if job.busy_waits is not None else self._busy_waits
        if tries > len(waits):
            raise LLMUnavailable(f"{job_name}: busy after {tries} tries: {exc}") from exc
        wait = waits[tries - 1]
        if deadline is not None and deadline - self._clock() - wait < MIN_TRY_S:
            span.set_attribute("hivemind.timed_out", True)
            raise LLMUnavailable(f"{job_name}: no answer within {job.total_s:g} s in all: {exc}") from exc
        span.set_attribute("hivemind.busy_retries", tries)
        self.retries += 1
        self._sleep(wait)

    def structured(self, job_name: str, system: str, prompt: str, schema: type[T]) -> T:
        """the model's answer as a checked object. invalid output gets one repair try, then LLMUnavailable."""
        job = self.models.job(job_name)
        client = self._client(job.provider)
        deadline = self._deadline(job)  # the repair try shares it too
        model = job.model  # the fallback, once the job's own model said it is busy
        ask = prompt
        for attempt in (1, 2):
            with self._tracer.chat(job_name, job.provider, job.model, job.temperature, ask) as span:
                self._counted_call()
                tries = 0
                while True:
                    tries += 1
                    seconds = self._try_seconds(job, span, deadline)
                    try:
                        done = client.generate(model, system, ask, job.temperature, schema, seconds)
                        break
                    except ProviderBusy as exc:
                        fallback = self._fall_back(job, model, span, exc)
                        if fallback is not None:  # once only: _fall_back gives none when already on it
                            client, model = fallback  # asked at once: a wait would only give the busy model time
                            continue
                        self._wait_or_fail(job, span, tries, exc, deadline)
                    except Exception as exc:
                        raise LLMUnavailable(f"{job_name}: {short_error(exc)}") from exc
                set_usage(span, done.input_tokens, done.output_tokens)
                self._counted_tokens(done)
                set_output(span, done.text)
                try:
                    return schema.model_validate_json(done.text)
                except ValidationError as exc:
                    span.set_attribute("hivemind.invalid_output", True)
                    problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            if attempt == 1:
                ask = (f"{prompt}\n\nYour last answer was not valid ({problems}). "
                       "Answer again with JSON that fits the schema exactly.")
        raise LLMUnavailable(f"{job_name}: the output stayed invalid after one repair try")

    def stream(self, job_name: str, system: str, prompt: str) -> Iterator[str]:
        """the model's answer as text pieces. the span closes when the stream ends or the reader stops."""
        job = self.models.job(job_name)
        client = self._client(job.provider)
        deadline = self._deadline(job)
        model = job.model
        with self._tracer.chat(job_name, job.provider, job.model, job.temperature, prompt) as span:
            self._counted_call()

            def finished(done: Completion) -> None:
                set_usage(span, done.input_tokens, done.output_tokens)
                self._counted_tokens(done)
                set_output(span, done.text)

            tries, sent = 0, False
            while True:
                tries += 1
                seconds = self._try_seconds(job, span, deadline)
                try:
                    for piece in client.stream(model, system, prompt, job.temperature, finished, seconds):
                        sent = True
                        yield piece
                    return
                except GeneratorExit:
                    span.set_attribute("hivemind.stopped_early", True)
                    raise
                except ProviderBusy as exc:
                    if sent:  # the customer already saw part of this answer; a new one would not match it
                        raise LLMUnavailable(f"{job_name}: {exc}") from exc
                    fallback = self._fall_back(job, model, span, exc)
                    if fallback is not None:  # once only: _fall_back gives none when already on it
                        client, model = fallback
                        continue
                    self._wait_or_fail(job, span, tries, exc, deadline)
                except Exception as exc:
                    raise LLMUnavailable(f"{job_name}: {short_error(exc)}") from exc
