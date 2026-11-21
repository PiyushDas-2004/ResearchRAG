"""Evaluation runner: index the corpus for a configuration, answer every question, score it."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from researchrag.config.pipeline import PipelineConfig
from researchrag.config.pricing import PricingTable
from researchrag.errors import ResearchRAGError
from researchrag.evaluation.dataset import Dataset, EvalQuestion
from researchrag.evaluation.metrics import (
    RetrievedText,
    abstention_metrics,
    answer_relevance,
    citation_metrics,
    context_metrics,
    retrieval_metrics,
)
from researchrag.evaluation.stats import bootstrap_ci, percentile
from researchrag.evaluation.synthetic import load_papers, render_paper
from researchrag.factory import Stack, build_stack
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.prompts import judge_messages
from researchrag.generation.schema import QueryResult
from researchrag.ingestion.service import ParseOptions
from researchrag.llm.base import LLMClient, LLMRequest
from researchrag.pipeline import QueryRequest
from researchrag.retrieval.memory import MemoryStore
from researchrag.retrieval.types import SearchBackend
from researchrag.storage.repositories import MemoryConversationStore, MemoryTraceSink
from researchrag.storage.store import DocumentStore

# Metrics where higher is better (used by reports, sweeps and the regression gate).
HIGHER_IS_BETTER = {
    "hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "precision@1", "precision@3",
    "precision@5", "ndcg@1", "ndcg@3", "ndcg@5", "mrr", "context_precision", "context_recall",
    "citation_precision", "citation_recall", "faithfulness", "answer_correctness", "answer_relevance",
    "abstention_accuracy",
}  # fmt: skip
LOWER_IS_BETTER = {"hallucination_rate", "over_abstention", "under_abstention", "removed_claim_rate"}
HEADLINE = [
    "recall@5",
    "mrr",
    "ndcg@5",
    "context_recall",
    "citation_precision",
    "faithfulness",
    "hallucination_rate",
    "answer_correctness",
    "abstention_accuracy",
    "over_abstention",
]


class QuestionResult(BaseModel):
    """Per-question outcome."""

    id: str
    type: str
    answerable: bool
    question: str
    gold_answer: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    metrics: dict[str, float | None]
    latency_ms: float
    cost_usd: float
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)


class RunResult(BaseModel):
    """Outcome of evaluating one configuration on one dataset."""

    name: str
    config: dict[str, Any]
    config_hash: str
    dataset: str
    dataset_hash: str
    components: dict[str, str]
    created_at: str
    backend: str
    summary: dict[str, dict[str, float | int]]
    by_type: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    cost: dict[str, float]
    questions: list[QuestionResult]

    def values(self, metric: str) -> list[float | None]:
        """Per-question values of ``metric`` in question order."""
        return [q.metrics.get(metric) for q in self.questions]


@dataclass
class EvalEnvironment:
    """Where the corpus is indexed for evaluation (in-memory or PostgreSQL)."""

    store: DocumentStore
    backend: SearchBackend
    name: str
    papers_dir: Path
    _pdfs: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def memory(cls, papers_dir: str | Path) -> EvalEnvironment:
        """A fresh in-memory environment."""
        store = MemoryStore()
        return cls(store, store, "memory", Path(papers_dir))

    def pdfs(self) -> dict[str, bytes]:
        """Rendered corpus PDFs keyed by paper key (rendered once, deterministic)."""
        if not self._pdfs:
            self._pdfs = {p.key: render_paper(p) for p in load_papers(self.papers_dir)}
        return self._pdfs


async def judge_correctness(llm: LLMClient, question: EvalQuestion, result: QueryResult) -> float:
    """Rubric-based correctness in [0, 1]; abstention is correct exactly for unanswerable questions."""
    if not question.answerable:
        return 1.0 if result.abstained else 0.0
    if result.abstained or not result.answer.strip():
        return 0.0
    try:
        response = await llm.complete(
            LLMRequest(
                judge_messages(question.question, question.gold_answer, result.answer),
                task="judge",
                max_tokens=200,
            )
        )
        return max(0.0, min(1.0, float(parse_json_object(response.text).get("score", 0.0))))
    except (ResearchRAGError, ValueError, TypeError):
        return 0.0


class EvalRunner:
    """Evaluates pipeline configurations against a dataset in one environment."""

    def __init__(
        self, env: EvalEnvironment, dataset: Dataset, *, pricing: PricingTable, judge: LLMClient | None = None
    ):
        self.env = env
        self.dataset = dataset
        self.pricing = pricing
        self.judge = judge

    async def index(self, stack: Stack) -> dict[str, str]:
        """Ensure every corpus paper is indexed for this config; return ``document_id -> paper key``."""
        keys: dict[str, str] = {}
        for key, data in self.env.pdfs().items():
            result = await stack.ingestion.ingest_bytes(data, f"{key}.pdf", collection="synthetic-eval")
            keys[result.document_id] = key
        return keys

    async def run(self, config: PipelineConfig) -> RunResult:
        """Index, answer and score every question for ``config``."""
        traces = MemoryTraceSink()
        stack = await build_stack(
            config,
            store=self.env.store,
            backend=self.env.backend,
            traces=traces,
            conversations=MemoryConversationStore(),
            pricing=self.pricing,
            parse=ParseOptions(isolation="inline"),
        )
        doc_keys = await self.index(stack)
        judge = self.judge or stack.llm
        outcomes: list[QuestionResult] = []
        for question in self.dataset.questions:
            started = time.perf_counter()
            result = await stack.pipeline.answer(QueryRequest(question.question))
            latency = (time.perf_counter() - started) * 1000
            stages = (await traces.get(result.trace_id) or {}).get("stages", {})
            outcomes.append(await self._score(question, result, stages, doc_keys, judge, latency))
        return self._summarise(config, stack, outcomes)

    async def _score(
        self,
        question: EvalQuestion,
        result: QueryResult,
        stages: dict[str, Any],
        doc_keys: dict[str, str],
        judge: LLMClient,
        latency: float,
    ) -> QuestionResult:
        ranked = [
            RetrievedText(doc_keys.get(c["document_id"]), c["text"]) for c in stages.get("candidates", [])
        ]
        context = [
            RetrievedText(doc_keys.get(p["document_id"]), p["text"]) for p in stages.get("passages", [])
        ]
        cited = [RetrievedText(doc_keys.get(c.document_id), c.passage) for c in result.citations]
        metrics: dict[str, float | None] = {}
        metrics.update(retrieval_metrics(question, ranked))
        metrics.update(context_metrics(question, context))
        metrics.update(abstention_metrics(question, result.abstained))
        all_claims = stages.get("generate_claims", [])
        if not result.abstained:
            metrics.update(citation_metrics(question, cited))
            metrics["faithfulness"] = result.faithfulness
            metrics["answer_relevance"] = answer_relevance(question.question, result.answer)
        if all_claims:
            failing = [c for c in all_claims if c["verdict"] in {"unsupported", "contradicted"}]
            metrics["hallucination_rate"] = 1.0 if failing else 0.0
            metrics["removed_claim_rate"] = sum(c["removed"] for c in result.model_dump()["claims"]) / max(
                1, len(result.claims)
            )
        metrics["answer_correctness"] = await judge_correctness(judge, question, result)
        return QuestionResult(
            id=question.id,
            type=question.type,
            answerable=question.answerable,
            question=question.question,
            gold_answer=question.gold_answer,
            answer=result.answer,
            abstained=result.abstained,
            abstain_reason=result.abstain_reason,
            metrics=metrics,
            latency_ms=round(latency, 2),
            cost_usd=result.usage.cost_usd,
            retrieved=[
                {"doc_key": doc_keys.get(c["document_id"]), "page": c["page"], "text": c["text"][:400]}
                for c in stages.get("candidates", [])[:5]
            ],
            claims=[c.model_dump() for c in result.claims],
        )

    def _summarise(self, config: PipelineConfig, stack: Stack, outcomes: list[QuestionResult]) -> RunResult:
        names = sorted({k for q in outcomes for k in q.metrics})
        summary = {
            name: bootstrap_ci([v for q in outcomes if (v := q.metrics.get(name)) is not None]).as_dict()
            for name in names
        }
        by_type: dict[str, dict[str, float]] = defaultdict(dict)
        for qtype in sorted({q.type for q in outcomes}):
            group = [q for q in outcomes if q.type == qtype]
            for name in ("answer_correctness", "abstention_accuracy", "recall@5", "faithfulness"):
                vals = [v for q in group if (v := q.metrics.get(name)) is not None]
                if vals:
                    by_type[qtype][name] = round(sum(vals) / len(vals), 4)
            by_type[qtype]["n"] = len(group)
        latencies = [q.latency_ms for q in outcomes]
        total_cost = sum(q.cost_usd for q in outcomes)
        components = {
            "embedder": stack.embeddings.model_id,
            "llm": f"{stack.llm.provider}/{stack.llm.model}",
            "judge": f"{(self.judge or stack.llm).provider}/{(self.judge or stack.llm).model}",
            "verifier": config.verification.judge,
            "reranker": config.rerank.provider,
        }
        return RunResult(
            name=config.name,
            config=config.model_dump(),
            config_hash=config.config_hash(),
            dataset=self.dataset.name,
            dataset_hash=self.dataset.content_hash,
            components=components,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            backend=self.env.name,
            summary=summary,
            by_type=dict(by_type),
            latency_ms={
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "mean": round(sum(latencies) / max(1, len(latencies)), 2),
            },
            cost={
                "total_usd": round(total_cost, 6),
                "per_query_usd": round(total_cost / max(1, len(outcomes)), 8),
            },
            questions=outcomes,
        )


def is_offline(components: dict[str, str]) -> bool:
    """True when only the offline fake/mock components were used."""
    return components.get("embedder", "").startswith("fake/") and components.get("llm", "").startswith(
        "mock/"
    )


def run_label(run: RunResult) -> str:
    """Human-readable provenance label for reports."""
    if is_offline(run.components):
        return "offline mock components, illustrative (fake embedder + mock extractive LLM)"
    return f"live components: embedder {run.components['embedder']}, LLM {run.components['llm']}"


def short_hash(text: str) -> str:
    """Short stable hash for file names."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]
