"""OpenTelemetry setup and the per-query stage recorder.

Every pipeline stage runs inside :meth:`StageRecorder.stage`, which (1) opens an OTel span
named after the stage, (2) times it into the Prometheus stage histogram, and (3) stores the
stage's details and latency in the persisted query trace. Without an exporter configured
the OTel API is a no-op, so this costs almost nothing in tests.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

from researchrag.observability.metrics import STAGE_LATENCY

_TRACER = trace.get_tracer("researchrag")
_CONFIGURED = False


def configure_tracing(service_name: str, otlp_endpoint: str = "", console: bool = False) -> bool:
    """Install a tracer provider exporting via OTLP/HTTP and/or console. Returns True if configured."""
    global _CONFIGURED  # noqa: PLW0603 - process-wide, idempotent setup
    if _CONFIGURED or not (otlp_endpoint or console):
        return _CONFIGURED
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: PLC0415

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True
    return True


def current_trace_id() -> str | None:
    """Hex trace id of the active span, if any."""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _attr(value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, bool | int | float | str) for v in value):
        return list(value)
    return str(value)[:500]


@dataclass
class StageRecorder:
    """Collects per-stage latency and details for one query."""

    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def _timed(self, name: str, attributes: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
        details: dict[str, Any] = dict(attributes or {})
        start = time.perf_counter()
        with _TRACER.start_as_current_span(name) as span:
            try:
                yield details
            finally:
                elapsed = time.perf_counter() - start
                STAGE_LATENCY.labels(stage=name).observe(elapsed)
                self.latency_ms[name] = round(self.latency_ms.get(name, 0.0) + elapsed * 1000, 3)
                self.stages[name] = {**self.stages.get(name, {}), **details}
                for key, value in details.items():
                    if not isinstance(value, dict):
                        span.set_attribute(f"rr.{key}", _attr(value))

    @asynccontextmanager
    async def stage(self, name: str, **attributes: Any) -> AsyncIterator[dict[str, Any]]:
        """Run a pipeline stage; mutate the yielded dict to record details."""
        with self._timed(name, attributes) as details:
            yield details

    @contextmanager
    def sync_stage(self, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
        """Synchronous variant of :meth:`stage`."""
        with self._timed(name, attributes) as details:
            yield details

    @property
    def total_ms(self) -> float:
        """Wall-clock time since the recorder was created."""
        return round((time.perf_counter() - self.started) * 1000, 3)
