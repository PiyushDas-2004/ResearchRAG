"""Document upload, listing, metadata editing, deletion and job status."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Query, Response, UploadFile, status

from researchrag.api.deps import ContainerDep, PrincipalDep
from researchrag.api.schemas import DocumentOut, DocumentPage, DocumentPatch, JobOut, UploadAccepted
from researchrag.errors import UploadRejectedError, ValidationFailedError
from researchrag.security.upload import validate_upload

router = APIRouter(prefix="/v1", tags=["documents"])
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await upload.read(_CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            raise UploadRejectedError(
                f"File exceeds the {limit}-byte limit.", detail={"reason": "too_large", "limit_bytes": limit}
            )
    return bytes(data)


@router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadAccepted,
    summary="Upload a PDF for ingestion",
    responses={
        413: {"description": "Too large"},
        415: {"description": "Not a PDF"},
        409: {"description": "Idempotency key reused"},
    },
)
async def upload_document(
    container: ContainerDep,
    principal: PrincipalDep,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF file (validated by content)")],
    collection: Annotated[str | None, Form(max_length=128)] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> UploadAccepted:
    """Validate the upload and enqueue an ingestion job; poll ``job_url`` for progress."""
    if idempotency_key is not None and not _IDEMPOTENCY_RE.match(idempotency_key):
        raise ValidationFailedError("Idempotency-Key must be 1-128 characters of [A-Za-z0-9._:-]")
    data = await _read_limited(file, container.upload_policy.max_bytes)
    filename = validate_upload(data, file.filename, container.upload_policy)
    job, created = await container.jobs.enqueue(
        kind="ingest",
        payload=data,
        filename=filename,
        content_hash=hashlib.sha256(data).hexdigest(),
        params={"collection": collection},
        idempotency_key=idempotency_key,
        api_key_id=None if principal.anonymous else principal.key_id,
        max_attempts=container.settings.job_max_retries,
    )
    if container.settings.ingest_mode == "inline" and job["status"] in {"queued", "running"}:
        job = await container.process_inline(str(job["id"]))
        response.status_code = status.HTTP_200_OK
    elif not created:
        response.status_code = status.HTTP_200_OK
    return UploadAccepted(job=JobOut(**job), created=created, job_url=f"/v1/jobs/{job['id']}")


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Ingestion job status")
async def get_job(job_id: str, container: ContainerDep, principal: PrincipalDep) -> JobOut:
    """Return a job's status, stage, progress and error details."""
    return JobOut(**await container.jobs.get(job_id))


@router.get("/documents", response_model=DocumentPage, summary="List documents")
async def list_documents(
    container: ContainerDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    collection: Annotated[str | None, Query(max_length=128)] = None,
) -> DocumentPage:
    """Newest first, paginated."""
    items, total = await container.documents.list(limit=limit, offset=offset, collection=collection)
    next_offset = offset + limit if offset + limit < total else None
    return DocumentPage(
        items=[DocumentOut(**i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentOut, summary="Get a document")
async def get_document(document_id: str, container: ContainerDep, principal: PrincipalDep) -> DocumentOut:
    """Metadata and chunk count of one document."""
    return DocumentOut(**await container.documents.get(document_id))


@router.patch("/documents/{document_id}", response_model=DocumentOut, summary="Correct document metadata")
async def patch_document(
    document_id: str, patch: DocumentPatch, container: ContainerDep, principal: PrincipalDep
) -> DocumentOut:
    """Edit title, authors, year and other extracted metadata."""
    fields = patch.model_dump(exclude_unset=True)
    if not fields:
        raise ValidationFailedError("No fields to update")
    return DocumentOut(**await container.documents.update_metadata(document_id, fields))


@router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document"
)
async def delete_document(document_id: str, container: ContainerDep, principal: PrincipalDep) -> Response:
    """Delete a document with its pages, chunks and embeddings."""
    await container.documents.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
