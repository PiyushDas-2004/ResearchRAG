"""Run PDF parsing in a resource-limited subprocess.

PDFs are untrusted input. MuPDF is robust, but a crafted file can still expand to huge
streams (a "decompression bomb") or take very long to lay out. Parsing in a child process
with an address-space limit and a wall-clock timeout bounds the damage: the child is killed
and the upload is rejected with a clear reason, while the API process stays healthy.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from multiprocessing.connection import Connection
from typing import Any

from researchrag.errors import ResearchRAGError, UploadRejectedError
from researchrag.ingestion.pdf_parser import ParseLimits, parse_pdf
from researchrag.ingestion.types import ParsedDocument


def _child(conn: Connection, data: bytes, limits: ParseLimits, memory_mb: int) -> None:  # pragma: no cover
    # Runs in the child process; coverage is not collected there.
    if memory_mb > 0 and sys.platform.startswith("linux"):
        import resource  # noqa: PLC0415 - POSIX only

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        conn.send(("ok", parse_pdf(data, limits)))
    except ResearchRAGError as exc:
        conn.send(("rejected", (exc.message, exc.detail)))
    except MemoryError:
        conn.send(("rejected", ("The PDF needs too much memory to parse.", {"reason": "memory_limit"})))
    except Exception as exc:
        conn.send(
            ("rejected", (f"The PDF could not be parsed ({type(exc).__name__}).", {"reason": "malformed"}))
        )
    finally:
        conn.close()


def parse_pdf_isolated(
    data: bytes, limits: ParseLimits, *, timeout_s: float = 60.0, memory_mb: int = 1024
) -> ParsedDocument:
    """Parse ``data`` in a subprocess; raise :class:`UploadRejectedError` on timeout or crash."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child_conn, data, limits, memory_mb), daemon=True)
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_s):
            raise UploadRejectedError(
                f"Parsing took longer than {timeout_s:.0f}s.", detail={"reason": "parse_timeout"}
            )
        try:
            status, payload = parent_conn.recv()
        except EOFError as exc:
            raise UploadRejectedError(
                "The parser process crashed (likely a resource limit).", detail={"reason": "parser_crashed"}
            ) from exc
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        parent_conn.close()
    if status == "ok":
        result: Any = payload
        return result  # type: ignore[no-any-return]
    message, detail = payload
    raise UploadRejectedError(message, detail=detail)


def parse_document(
    data: bytes, limits: ParseLimits, *, isolation: str, timeout_s: float, memory_mb: int
) -> ParsedDocument:
    """Parse with the configured isolation mode (``subprocess`` or ``inline``)."""
    if isolation == "subprocess":
        return parse_pdf_isolated(data, limits, timeout_s=timeout_s, memory_mb=memory_mb)
    return parse_pdf(data, limits)
