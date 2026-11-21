"""Pipeline configuration: every swappable stage is selected and tuned here.

A :class:`PipelineConfig` is loaded from YAML. The evaluation harness sweeps configurations
by applying dotted overrides (``{"retrieval.mode": "vector"}``) to a base config, so every
field that matters for quality must live here rather than in code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from researchrag.errors import ValidationFailedError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(_Strict):
    """How documents are split into retrievable chunks."""

    strategy: Literal["fixed", "sentence", "section", "parent_child"] = "parent_child"
    chunk_tokens: int = Field(160, ge=16, le=4096)
    overlap_tokens: int = Field(32, ge=0, le=1024)
    parent_tokens: int = Field(480, ge=32, le=8192)
    tokenizer: str = "simple"

    @model_validator(mode="after")
    def _check(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        if self.strategy == "parent_child" and self.parent_tokens < self.chunk_tokens:
            raise ValueError("parent_tokens must be >= chunk_tokens for parent_child chunking")
        return self

    def config_hash(self) -> str:
        """Stable hash identifying the chunk set this config produces."""
        return _hash(self.model_dump())


class EmbeddingConfig(_Strict):
    """Which embedding model is used; together with the dimension this is an embedding version."""

    provider: Literal["fake", "openai", "gemini", "local"] = "fake"
    model: str = "fake-hash-v1"
    dimension: int = Field(384, ge=8, le=4096)
    batch_size: int = Field(64, ge=1, le=2048)
    requests_per_minute: int = Field(600, ge=1)
    max_retries: int = Field(4, ge=0, le=10)
    cache_size: int = Field(2048, ge=0)

    @property
    def model_id(self) -> str:
        """Identifier stored with every embedding (provider/model)."""
        return f"{self.provider}/{self.model}"


class RetrievalConfig(_Strict):
    """Candidate generation, fusion and diversification."""

    mode: Literal["hybrid", "vector", "lexical"] = "hybrid"
    candidate_k: int = Field(40, ge=1, le=500)
    top_k: int = Field(8, ge=1, le=100)
    rrf_k: int = Field(60, ge=1)
    vector_weight: float = Field(1.0, ge=0)
    lexical_weight: float = Field(1.0, ge=0)
    mmr_enabled: bool = True
    mmr_lambda: float = Field(0.7, ge=0, le=1)
    ef_search: int = Field(64, ge=1, le=1000)
    include_references: bool = False
    embedding_version: int | None = None


class RewriteConfig(_Strict):
    """Query rewriting, multi-query expansion and conversational condensation."""

    enabled: bool = False
    multi_query: int = Field(0, ge=0, le=5)
    condense_followups: bool = True
    max_history_turns: int = Field(6, ge=0, le=50)


class RerankConfig(_Strict):
    """Second-stage reranking of fused candidates."""

    provider: Literal["none", "lexical", "cross_encoder", "llm"] = "lexical"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(24, ge=1, le=200)
    cache_size: int = Field(4096, ge=0)


class MockLLMConfig(_Strict):
    """Behaviour of the offline mock LLM (only used when ``generation.provider == 'mock'``)."""

    seed: int = 7
    hallucination_rate: float = Field(0.0, ge=0, le=1)
    failure_mode: Literal[
        "none",
        "timeout",
        "rate_limit",
        "malformed_json",
        "fabricated_quote",
        "uncited_claim",
        "follow_injection",
    ] = "none"
    failure_count: int = Field(-1, description="How many calls fail (-1 = all calls)")
    min_term_coverage: float = Field(0.5, ge=0, le=1)
    max_claims: int = Field(3, ge=1, le=10)


class GenerationConfig(_Strict):
    """Answer generation."""

    provider: Literal["mock", "openai", "gemini", "openai_compatible"] = "mock"
    model: str = "mock-extractive-v1"
    base_url: str | None = None
    temperature: float | None = Field(0.0, ge=0, le=2, description="None = provider default")
    max_output_tokens: int = Field(900, ge=64, le=16384)
    max_context_tokens: int = Field(1800, ge=128, le=200_000)
    max_retries: int = Field(2, ge=0, le=5)
    timeout_s: float = Field(60.0, gt=0)
    min_evidence_score: float = Field(0.34, ge=0, le=1)
    mock: MockLLMConfig = Field(default_factory=MockLLMConfig)


class VerificationConfig(_Strict):
    """Claim-level verification of generated answers."""

    enabled: bool = True
    judge: Literal["lexical", "llm", "nli"] = "lexical"
    judge_model: str | None = None
    policy: Literal["flag", "remove"] = "remove"
    regenerate_on_failure: bool = True
    quote_match_threshold: float = Field(90.0, ge=0, le=100)
    supported_threshold: float = Field(0.8, ge=0, le=1)
    partial_threshold: float = Field(0.55, ge=0, le=1)
    min_faithfulness: float = Field(0.5, ge=0, le=1)


class SecurityConfig(_Strict):
    """Handling of untrusted document text."""

    injection_action: Literal["flag", "sanitize", "drop"] = "sanitize"
    output_guard: bool = True


class PipelineConfig(_Strict):
    """Full query + ingestion pipeline configuration."""

    name: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    def config_hash(self) -> str:
        """Stable hash of the whole configuration (excluding its display name)."""
        data = self.model_dump()
        data.pop("name", None)
        return _hash(data)

    def with_overrides(self, overrides: dict[str, Any], *, name: str | None = None) -> PipelineConfig:
        """Return a copy with dotted-path overrides applied, e.g. ``{"retrieval.top_k": 5}``."""
        data = copy.deepcopy(self.model_dump())
        for dotted, value in overrides.items():
            target = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise ValidationFailedError(f"Unknown config section in override: {dotted}")
                target = target[part]
            if parts[-1] not in target:
                raise ValidationFailedError(f"Unknown config key in override: {dotted}")
            target[parts[-1]] = value
        if name is not None:
            data["name"] = name
        try:
            return PipelineConfig.model_validate(data)
        except ValueError as exc:
            raise ValidationFailedError(f"Invalid override: {exc}") from exc


def _hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_pipeline_config(path: str | Path | None) -> PipelineConfig:
    """Load a pipeline config from YAML; ``None`` returns the defaults."""
    if path is None:
        return PipelineConfig()
    file = Path(path)
    if not file.exists():
        raise ValidationFailedError(f"Pipeline config not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValidationFailedError(f"Pipeline config must be a mapping: {file}")
    raw.setdefault("name", file.stem)
    try:
        return PipelineConfig.model_validate(raw)
    except ValueError as exc:
        raise ValidationFailedError(f"Invalid pipeline config {file}: {exc}") from exc
