"""Answer data models: the raw model payload and the verified result returned by the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Verdict = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class RawClaim(BaseModel):
    """A claim as produced by the model."""

    model_config = ConfigDict(extra="ignore")

    text: str
    citations: list[int] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, int | str):
            value = [value]
        out = []
        for item in value or []:
            try:
                out.append(int(str(item).strip("[] ")))
            except ValueError:
                continue
        return out


class RawQuote(BaseModel):
    """A verbatim supporting quote claimed by the model."""

    model_config = ConfigDict(extra="ignore")

    passage: int
    quote: str


class RawAnswer(BaseModel):
    """The JSON object the model is asked to return (validated leniently)."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    claims: list[RawClaim] = Field(default_factory=list)
    quotes: list[RawQuote] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0, le=1)
    abstained: bool = False
    abstain_reason: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.5


class CitationOut(BaseModel):
    """A citation resolved to a retrieved passage, with quote verification."""

    marker: int
    chunk_id: str
    document_id: str
    title: str
    page: int
    section: str
    quote: str | None = None
    quote_verified: bool = False
    quote_score: float | None = None
    passage: str = Field("", description="The passage text the citation points to")


class ClaimOut(BaseModel):
    """An atomic claim with its verification verdict."""

    text: str
    citations: list[int] = Field(default_factory=list)
    verdict: Verdict = "unsupported"
    rationale: str = ""
    flags: list[str] = Field(default_factory=list)
    removed: bool = False


class SearchedOut(BaseModel):
    """What was searched (shown with abstentions)."""

    queries: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    candidates: int = 0
    passages: int = 0
    documents: list[str] = Field(default_factory=list)
    evidence_score: float = 0.0


class UsageOut(BaseModel):
    """Token usage and estimated cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_models: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """The verified, citation-grounded answer to one query."""

    trace_id: str
    question: str
    standalone_question: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    confidence: float = 0.0
    faithfulness: float | None = None
    claims: list[ClaimOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    searched: SearchedOut = Field(default_factory=SearchedOut)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageOut = Field(default_factory=UsageOut)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    config_name: str = "default"
