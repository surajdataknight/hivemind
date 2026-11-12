"""the model router: every llm call goes through here, by job name. which model a job uses is in models.toml.

a provider client only has to turn (model, system text, prompt) into text; the router adds the job lookup, the
typed-output check with one repair try, and the trace span.
"""

import time
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from hivemind_agents.settings import ModelSettings, env
from hivemind_agents.tracing import Tracer, set_output, set_usage

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """no usable answer: no key, quota used up, provider down, or output that stayed invalid after a repair try."""


class ProviderBusy(Exception):
    """the provider said to try again later (overloaded, per-minute limit, timeout). the router waits and retries."""


BUSY_WAITS = (2.0, 5.0)  # seconds before each retry; after the last, the call fails
BUSY_CODES = {429, 500, 502, 503, 504}


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderClient(Protocol):
    def generate(self, model: str, system: str, prompt: str, temperature: float,
                 schema: type[BaseModel] | None) -> Completion: ...

    def stream(self, model: str, system: str, prompt: str, temperature: float,
               on_done: Callable[[Completion], None]) -> Iterator[str]: ...


class GoogleClient:
    """google ai studio through the google-genai sdk."""

    def __init__(self, key: str, timeout_ms: int = 20000):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_ms))

    def _config(self, system: str, temperature: float, schema: type[BaseModel] | None):
        extra = {"response_mime_type": "application/json", "response_schema": schema} if schema else {}
        return self._types.GenerateContentConfig(system_instruction=system, temperature=temperature, **extra)

    @staticmethod
    def _busy(exc: Exception) -> Exception:
        """the same error, or ProviderBusy when google says to try again later."""
        code = getattr(exc, "code", None)
        if code in BUSY_CODES or "Timeout" in type(exc).__name__:
            return ProviderBusy(f"{type(exc).__name__}: {exc}")
        return exc

    def generate(self, model, system, prompt, temperature, schema):
        try:
            response = self._client.models.generate_content(model=model, contents=prompt,
                                                            config=self._config(system, temperature, schema))
        except Exception as exc:
            raise self._busy(exc) from exc
        usage = response.usage_metadata
        return Completion(response.text or "", getattr(usage, "prompt_token_count", None),
                          getattr(usage, "candidates_token_count", None))

    def stream(self, model, system, prompt, temperature, on_done):
        parts, usage = [], None
        try:
            for chunk in self._client.models.generate_content_stream(model=model, contents=prompt,
                                                                     config=self._config(system, temperature, None)):
                usage = chunk.usage_metadata or usage
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
        except Exception as exc:
            raise self._busy(exc) from exc
        on_done(Completion("".join(parts), getattr(usage, "prompt_token_count", None),
                           getattr(usage, "candidates_token_count", None)))


def provider_clients(models: ModelSettings) -> dict[str, Callable[[], ProviderClient]]:
    """one lazy client per provider, made on its first call, so a missing key only fails the calls that need it."""

    def google() -> ProviderClient:
        return GoogleClient(env(models.key_envs["google"]))

    return {"google": google}


class ModelRouter:
    def __init__(self, models: ModelSettings, tracer: Tracer,
                 providers: dict[str, Callable[[], ProviderClient]] | None = None,
                 busy_waits: tuple[float, ...] = BUSY_WAITS, sleep: Callable[[float], None] = time.sleep):
        self.models = models
        self._tracer = tracer
        self._factories = providers if providers is not None else provider_clients(models)
        self._clients: dict[str, ProviderClient] = {}
        self._busy_waits = busy_waits
        self._sleep = sleep
        self.calls = 0  # counted for tests and for the reply's own record; retries are not extra calls
        self.retries = 0

    def _client(self, provider: str) -> ProviderClient:
        if provider not in self._clients:
            if provider not in self._factories:
                raise LLMUnavailable(f"no client for the provider {provider!r}")
            try:
                self._clients[provider] = self._factories[provider]()
            except Exception as exc:
                raise LLMUnavailable(f"{provider}: {exc}") from exc
        return self._clients[provider]

    def _wait_or_fail(self, job_name: str, span, tries: int, exc: ProviderBusy) -> None:
        """after a busy answer: wait and let the caller try again, or give up when the waits are used up."""
        if tries > len(self._busy_waits):
            raise LLMUnavailable(f"{job_name}: busy after {tries} tries: {exc}") from exc
        span.set_attribute("hivemind.busy_retries", tries)
        self.retries += 1
        self._sleep(self._busy_waits[tries - 1])

    def structured(self, job_name: str, system: str, prompt: str, schema: type[T]) -> T:
        """the model's answer as a checked object. invalid output gets one repair try, then LLMUnavailable."""
        job = self.models.job(job_name)
        client = self._client(job.provider)
        ask = prompt
        for attempt in (1, 2):
            with self._tracer.chat(job_name, job.provider, job.model, job.temperature, ask) as span:
                self.calls += 1
                tries = 0
                while True:
                    tries += 1
                    try:
                        done = client.generate(job.model, system, ask, job.temperature, schema)
                        break
                    except ProviderBusy as exc:
                        self._wait_or_fail(job_name, span, tries, exc)
                    except Exception as exc:
                        raise LLMUnavailable(f"{job_name}: {type(exc).__name__}: {exc}") from exc
                set_usage(span, done.input_tokens, done.output_tokens)
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
        with self._tracer.chat(job_name, job.provider, job.model, job.temperature, prompt) as span:
            self.calls += 1

            def finished(done: Completion) -> None:
                set_usage(span, done.input_tokens, done.output_tokens)
                set_output(span, done.text)

            tries, sent = 0, False
            while True:
                tries += 1
                try:
                    for piece in client.stream(job.model, system, prompt, job.temperature, finished):
                        sent = True
                        yield piece
                    return
                except GeneratorExit:
                    span.set_attribute("hivemind.stopped_early", True)
                    raise
                except ProviderBusy as exc:
                    if sent:  # the customer already saw part of this answer; a new one would not match it
                        raise LLMUnavailable(f"{job_name}: {exc}") from exc
                    self._wait_or_fail(job_name, span, tries, exc)
                except Exception as exc:
                    raise LLMUnavailable(f"{job_name}: {type(exc).__name__}: {exc}") from exc
