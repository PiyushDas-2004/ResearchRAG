"""RFC 7807 ``application/problem+json`` error responses."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from researchrag.errors import ResearchRAGError

log = structlog.get_logger("researchrag.api")
PROBLEM_BASE = "https://github.com/PiyushDas-2004/ResearchRAG/blob/main/docs/API.md#"

STATUS_BY_CODE = {
    "validation_failed": 422,
    "upload_rejected": 422,
    "not_found": 404,
    "conflict": 409,
    "provider_error": 502,
    "provider_unavailable": 503,
    "provider_timeout": 504,
    "provider_rate_limited": 503,
    "configuration_error": 500,
    "internal_error": 500,
}
UPLOAD_STATUS = {"too_large": 413, "not_pdf": 415}
TITLES = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    405: "Method not allowed",
    409: "Conflict",
    413: "Payload too large",
    415: "Unsupported media type",
    422: "Unprocessable content",
    429: "Too many requests",
    500: "Internal server error",
    502: "Bad gateway",
    503: "Service unavailable",
    504: "Gateway timeout",
}


def problem(
    request: Request,
    status: int,
    code: str,
    detail: str,
    *,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a problem+json response carrying the request id."""
    body: dict[str, Any] = {
        "type": PROBLEM_BASE + code,
        "title": TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "request_id": getattr(request.state, "request_id", None),
    }
    if extra:
        body.update(extra)
    return JSONResponse(body, status_code=status, media_type="application/problem+json", headers=headers)


def install_problem_handlers(app: FastAPI) -> None:
    """Register handlers so every error leaves the API as problem+json."""

    @app.exception_handler(ResearchRAGError)
    async def _domain(request: Request, exc: ResearchRAGError) -> JSONResponse:
        status = STATUS_BY_CODE.get(exc.code, 500)
        if exc.code == "upload_rejected":
            status = UPLOAD_STATUS.get(str(exc.detail.get("reason")), 422)
        headers = None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            headers = {"Retry-After": str(int(retry_after) + 1)}
        return problem(
            request,
            status,
            exc.code,
            exc.message,
            extra={"context": exc.detail} if exc.detail else None,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "loc": list(err.get("loc", [])),
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
            for err in exc.errors()
        ]
        return problem(
            request, 422, "validation_failed", "The request did not validate.", extra={"errors": errors}
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: "unauthorized",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            429: "rate_limited",
        }.get(exc.status_code, "http_error")
        return problem(request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", path=request.url.path, error_type=type(exc).__name__)
        return problem(request, 500, "internal_error", "An unexpected error occurred.")
