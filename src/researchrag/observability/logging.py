"""Structured JSON logging with request/trace correlation and PII redaction."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from researchrag.observability.tracing import current_trace_id

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_SECRET_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|(?:api[_-]?key|token|password)=\S+)", re.I
)
_SENSITIVE_KEYS = {"authorization", "x-api-key", "api_key", "password", "secret", "token", "cookie"}


def redact(text: str) -> str:
    """Mask e-mail addresses, phone numbers and credential-looking strings."""
    text = _SECRET_RE.sub("[REDACTED_SECRET]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return _PHONE_RE.sub("[REDACTED_PHONE]", text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else _redact_value(v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_value(v) for v in value]
    return value


def redact_processor(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """structlog processor applying :func:`redact` to every field."""
    for key in list(event):
        event[key] = "[REDACTED]" if key.lower() in _SENSITIVE_KEYS else _redact_value(event[key])
    return event


def add_trace_id(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Attach the active OpenTelemetry trace id (if any)."""
    trace_id = current_trace_id()
    if trace_id and "otel_trace_id" not in event:
        event["otel_trace_id"] = trace_id
    return event


def configure_logging(level: str = "INFO", *, json_logs: bool = True, redact: bool = True) -> None:
    """Configure structlog (JSON to stdout) and route stdlib logging through it."""
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_trace_id,
    ]
    if redact:
        processors.append(redact_processor)
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(colors=False)
    )
    numeric = logging.getLevelName(level.upper())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            numeric if isinstance(numeric, int) else logging.INFO
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(
        level=numeric if isinstance(numeric, int) else logging.INFO, stream=sys.stdout, format="%(message)s"
    )
