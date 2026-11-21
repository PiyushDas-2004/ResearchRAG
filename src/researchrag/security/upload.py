"""Validation of uploaded files before they reach the PDF parser.

The file type is decided by magic bytes, never by extension or the client's content type.
Size is checked before parsing. Parsing itself runs in an isolated subprocess with a memory
cap and a wall-clock timeout (see :mod:`researchrag.ingestion.isolation`), which is the
defence against decompression bombs and pathological PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from researchrag.errors import UploadRejectedError

PDF_MAGIC = b"%PDF-"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass(frozen=True)
class UploadPolicy:
    """Limits applied to every upload."""

    max_bytes: int = 25 * 1024 * 1024
    min_bytes: int = 64


def sniff_pdf(data: bytes) -> bool:
    """True if the bytes start like a PDF (the header may follow up to 1 KiB of junk, per spec)."""
    return PDF_MAGIC in data[:1024]


def validate_upload(data: bytes, filename: str | None, policy: UploadPolicy) -> str:
    """Validate raw upload bytes and return a sanitised display filename.

    Raises :class:`UploadRejectedError` with a machine-readable ``reason``.
    """
    if len(data) > policy.max_bytes:
        raise UploadRejectedError(
            f"File is {len(data)} bytes; the limit is {policy.max_bytes}.",
            detail={"reason": "too_large", "limit_bytes": policy.max_bytes},
        )
    if len(data) < policy.min_bytes:
        raise UploadRejectedError("File is empty or truncated.", detail={"reason": "too_small"})
    if not sniff_pdf(data):
        raise UploadRejectedError(
            "Only PDF files are accepted (checked by content, not extension).",
            detail={"reason": "not_pdf"},
        )
    return sanitize_filename(filename)


def sanitize_filename(filename: str | None) -> str:
    """Strip directories and unusual characters from a client-supplied filename."""
    if not filename:
        return "upload.pdf"
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_NAME_RE.sub("_", base).strip(" .")[:200]
    return cleaned or "upload.pdf"
