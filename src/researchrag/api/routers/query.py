"""Question answering (JSON and Server-Sent Events) and trace lookup."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from researchrag.api.deps import ContainerDep, PrincipalDep
from researchrag.api.problems import STATUS_BY_CODE
from researchrag.api.schemas import QueryIn
from researchrag.errors import NotFoundError, ResearchRAGError
from researchrag.generation.schema import QueryResult
from researchrag.pipeline import QueryRequest

router = APIRouter(prefix="/v1", tags=["query"])
KEEPALIVE_S = 15.0


def _request(body: QueryIn, key_id: str | None) -> QueryRequest:
    return QueryRequest(
        question=body.question,
        filters=body.filters.to_filters(),
        history=[(t.role, t.content) for t in body.history],
        conversation_id=body.conversation_id,
        api_key_id=key_id,
    )


@router.post("/query", response_model=QueryResult, summary="Answer a question with verified citations")
async def query(body: QueryIn, container: ContainerDep, principal: PrincipalDep) -> QueryResult:
    """Retrieve, generate, verify and return a citation-grounded answer (or an abstention)."""
    return await container.stack.pipeline.answer(_request(body, principal.key_id))


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@router.post(
    "/query/stream",
    summary="Answer a question as Server-Sent Events",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "status*, token*, then final (or error)"}
    },
)
async def query_stream(
    body: QueryIn, request: Request, container: ContainerDep, principal: PrincipalDep
) -> StreamingResponse:
    """Stream ``status`` and ``token`` events, then one ``final`` event with the verified result.

    The final answer can differ from the streamed tokens when verification removed claims;
    clients should replace the streamed text with ``final.answer``.
    """
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    async def on_token(text: str) -> None:
        await queue.put(("token", {"text": text}))

    async def on_event(name: str, data: dict[str, Any]) -> None:
        await queue.put((name, data))

    async def run() -> None:
        try:
            result = await container.stack.pipeline.answer(
                _request(body, principal.key_id), on_token=on_token, on_event=on_event
            )
            await queue.put(("final", result.model_dump(mode="json")))
        except ResearchRAGError as exc:
            await queue.put(
                (
                    "error",
                    {"status": STATUS_BY_CODE.get(exc.code, 500), "code": exc.code, "detail": exc.message},
                )
            )
        except Exception:
            await queue.put(
                (
                    "error",
                    {"status": 500, "code": "internal_error", "detail": "An unexpected error occurred."},
                )
            )
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run())
        try:
            yield _sse(
                "status", {"stage": "accepted", "request_id": getattr(request.state, "request_id", None)}
            )
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                if item is None:
                    break
                yield _sse(*item)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


@router.get("/traces/{trace_id}", summary="Full trace of a query")
async def get_trace(trace_id: str, container: ContainerDep, principal: PrincipalDep) -> dict[str, Any]:
    """Retrieved candidates, packed context, prompt, raw answer, verdicts, latency and cost."""
    trace = await container.traces.get(trace_id)
    if trace is None:
        raise NotFoundError(f"Trace {trace_id} not found")
    return trace
