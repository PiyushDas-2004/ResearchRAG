"""Request and response models of the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchrag.retrieval.types import SearchFilters


class FiltersIn(BaseModel):
    """Metadata filters (applied inside the SQL search)."""

    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list, max_length=200)
    collection: str | None = Field(None, max_length=128)
    year_min: int | None = Field(None, ge=1900, le=2100)
    year_max: int | None = Field(None, ge=1900, le=2100)
    authors: list[str] = Field(default_factory=list, max_length=20)
    include_references: bool = False

    def to_filters(self) -> SearchFilters:
        """Domain filter object."""
        return SearchFilters(
            document_ids=tuple(self.document_ids),
            collection=self.collection,
            year_min=self.year_min,
            year_max=self.year_max,
            authors=tuple(self.authors),
            include_references=self.include_references,
        )


class TurnIn(BaseModel):
    """A previous conversation turn supplied by the client."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class QueryIn(BaseModel):
    """A question to answer from the indexed papers."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "question": "What afternoon cooling did the CoolCrown roofs deliver in Harbourside?",
                    "filters": {"year_min": 2020},
                    "conversation_id": "demo-1",
                }
            ]
        },
    )

    question: str = Field(..., min_length=3, max_length=2000)
    filters: FiltersIn = Field(default_factory=FiltersIn)
    history: list[TurnIn] = Field(default_factory=list, max_length=20)
    conversation_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$")

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("question is too short")
        return stripped


class FeedbackIn(BaseModel):
    """Thumbs up/down on an answer."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"trace_id": "0b6f...", "rating": "down", "comment": "Wrong page cited"}]
        },
    )

    trace_id: str = Field(..., min_length=8, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(None, max_length=2000)


class FeedbackOut(BaseModel):
    """Stored feedback reference."""

    id: int
    trace_id: str
    rating: Literal["up", "down"]


class DocumentPatch(BaseModel):
    """User-correctable metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, max_length=500)
    authors: list[str] | None = Field(None, max_length=50)
    year: int | None = Field(None, ge=1900, le=2100)
    abstract: str | None = Field(None, max_length=10000)
    doi: str | None = Field(None, max_length=255)
    arxiv_id: str | None = Field(None, max_length=64)
    venue: str | None = Field(None, max_length=500)
    collection: str | None = Field(None, max_length=128)


class DocumentOut(BaseModel):
    """A document summary."""

    id: str
    filename: str
    title: str | None
    authors: list[str]
    year: int | None
    abstract: str | None
    doi: str | None
    arxiv_id: str | None
    venue: str | None
    collection: str | None
    page_count: int
    size_bytes: int
    content_hash: str
    created_at: str
    chunk_count: int | None = None


class DocumentPage(BaseModel):
    """A page of documents."""

    items: list[DocumentOut]
    total: int
    limit: int
    offset: int
    next_offset: int | None


class JobOut(BaseModel):
    """An ingestion job."""

    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str | None
    progress: float
    filename: str | None
    document_id: str | None
    attempts: int
    max_attempts: int
    error: dict[str, Any] | None
    created_at: str | None
    finished_at: str | None


class UploadAccepted(BaseModel):
    """Response to an upload: poll the job URL."""

    job: JobOut
    created: bool
    job_url: str


class Problem(BaseModel):
    """RFC 7807 problem details."""

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None
