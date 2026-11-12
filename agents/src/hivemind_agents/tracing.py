"""traces: one trace per run of the case graph, one span per capability, model call and backend read.

spans carry the opentelemetry gen-ai attribute names and the openinference ones phoenix reads, so the same span
shows its kind, model and token counts there. content in spans is the token form, never raw personal data.
"""

import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

PROJECT = "hivemind"
_propagator = TraceContextTextMapPropagator()


def make_provider(exporter: SpanExporter | None, processor=BatchSpanProcessor) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "hivemind-agents",
                                                        "openinference.project.name": PROJECT}))
    if exporter is not None:
        provider.add_span_processor(processor(exporter))
    return provider


def otlp_exporter(endpoint: str) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint, timeout=5)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


class Tracer:
    def __init__(self, provider: TracerProvider):
        self.provider = provider
        self._tracer = provider.get_tracer("hivemind_agents")

    @contextmanager
    def span(self, name: str, kind: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
        """kind is the openinference span kind: CHAIN, AGENT, LLM or TOOL."""
        with self._tracer.start_as_current_span(name, record_exception=True, set_status_on_exception=True) as span:
            span.set_attribute("openinference.span.kind", kind)
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span
            if span.status.status_code is StatusCode.UNSET:
                span.set_status(Status(StatusCode.OK))

    @contextmanager
    def run(self, conversation_id: str, customer_id: str, text_in: str) -> Iterator[Span]:
        attributes = {"session.id": conversation_id, "user.id": customer_id, "input.value": text_in}
        with self.span("case_run", "CHAIN", attributes) as span:
            yield span

    @contextmanager
    def capability(self, name: str, uses_llm: bool = False, **attributes: Any) -> Iterator[Span]:
        base = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name, **attributes}
        with self.span(name, "AGENT" if uses_llm else "CHAIN", base) as span:
            yield span

    @contextmanager
    def tool(self, operation: str) -> Iterator[Span]:
        attributes = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": operation, "tool.name": operation}
        with self.span(f"execute_tool {operation}", "TOOL", attributes) as span:
            yield span

    @contextmanager
    def chat(self, job: str, provider: str, model: str, temperature: float, prompt: str) -> Iterator[Span]:
        attributes = {
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": provider, "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature, "llm.provider": provider, "llm.model_name": model,
            "hivemind.job": job, "input.value": prompt,
        }
        with self.span(f"chat {model}", "LLM", attributes) as span:
            yield span

    def shutdown(self) -> None:
        self.provider.shutdown()


def set_output(span: Span, value: Any) -> None:
    span.set_attribute("output.value", _text(value))


def set_usage(span: Span, input_tokens: int | None, output_tokens: int | None) -> None:
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("llm.token_count.prompt", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("llm.token_count.completion", output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute("llm.token_count.total", input_tokens + output_tokens)


def trace_headers() -> dict[str, str]:
    """the w3c traceparent of the current span, for calls to the backend."""
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier
