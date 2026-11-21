"""Request ids, structured access logs, HTTP metrics, body-size limits and security headers."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from researchrag.api.problems import problem
from researchrag.observability.metrics import REQUEST_LATENCY, REQUESTS

log = structlog.get_logger("researchrag.access")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'"
    ),
}


def _route(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates ``X-Request-ID``, logs access, records metrics and enforces body size."""

    def __init__(self, app: object, *, max_body_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Wrap one request."""
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > self.max_body_bytes:
            response: Response = problem(
                request, 413, "payload_too_large", f"Request body exceeds {self.max_body_bytes} bytes."
            )
        else:
            response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = _route(request)
        REQUESTS.labels(method=request.method, route=route, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(method=request.method, route=route).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        if route not in {"/health", "/metrics"}:
            log.info(
                "http.access",
                method=request.method,
                route=route,
                status=response.status_code,
                latency_ms=round(elapsed * 1000, 2),
                client_key=getattr(request.state, "key_id", None),
            )
        return response
