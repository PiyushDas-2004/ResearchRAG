"""Retrieval orchestration: (multi-)query -> vector + lexical -> RRF -> rerank -> MMR."""

from __future__ import annotations

from dataclasses import dataclass, field

from researchrag.config.pipeline import PipelineConfig
from researchrag.embeddings.service import EmbeddingService
from researchrag.observability.tracing import StageRecorder
from researchrag.rerank.rerankers import Reranker
from researchrag.retrieval.fusion import rrf
from researchrag.retrieval.mmr import mmr
from researchrag.retrieval.rewrite import QueryRewriter
from researchrag.retrieval.types import Candidate, EmbeddingVersionRef, SearchBackend, SearchFilters


@dataclass
class RetrievalResult:
    """Final candidates plus everything needed for the trace and for ablations."""

    queries: list[str]
    candidates: list[Candidate]
    fused: list[Candidate] = field(default_factory=list)
    per_method: dict[str, list[Candidate]] = field(default_factory=dict)


def _summary(cands: list[Candidate], method: str, n: int = 10) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": c.chunk_id,
            "page": c.page_start,
            "title": c.title[:80],
            "score": round(c.scores.get(method, c.score), 5),
        }
        for c in cands[:n]
    ]


class Retriever:
    """Runs the candidate-generation part of the pipeline for one configuration."""

    def __init__(
        self,
        backend: SearchBackend,
        embeddings: EmbeddingService,
        version: EmbeddingVersionRef,
        config: PipelineConfig,
        *,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.backend = backend
        self.embeddings = embeddings
        self.version = version
        self.config = config
        self.reranker = reranker
        self.rewriter = rewriter
        self.chunk_config_hash = config.chunking.config_hash()

    async def retrieve(self, query: str, filters: SearchFilters, recorder: StageRecorder) -> RetrievalResult:
        """Return the top-k candidates for ``query`` under ``filters``."""
        rc = self.config.retrieval
        filters = SearchFilters(
            **{**filters.__dict__, "include_references": filters.include_references or rc.include_references}
        )
        queries = [query]
        if self.rewriter is not None and self.config.rewrite.enabled:
            async with recorder.stage("rewrite") as st:
                queries = await self.rewriter.expand(query)
                st["queries"] = queries
        lists: list[tuple[str, list[Candidate], float]] = []
        per_method: dict[str, list[Candidate]] = {}
        for i, q in enumerate(queries):
            if rc.mode in {"hybrid", "vector"}:
                async with recorder.stage("retrieve.vector") as st:
                    qvec = await self.embeddings.embed_query(q)
                    found = await self.backend.vector_search(
                        qvec,
                        version=self.version,
                        chunk_config_hash=self.chunk_config_hash,
                        k=rc.candidate_k,
                        filters=filters,
                        ef_search=rc.ef_search,
                    )
                    st["count"] = st.get("count", 0) + len(found)
                    st["ef_search"] = rc.ef_search
                    st.setdefault("top", []).extend(_summary(found, "vector", 5))
                lists.append((f"vector:{i}", found, rc.vector_weight))
                per_method.setdefault("vector", []).extend(found)
            if rc.mode in {"hybrid", "lexical"}:
                async with recorder.stage("retrieve.lexical") as st:
                    found = await self.backend.lexical_search(
                        q, chunk_config_hash=self.chunk_config_hash, k=rc.candidate_k, filters=filters
                    )
                    st["count"] = st.get("count", 0) + len(found)
                    st.setdefault("top", []).extend(_summary(found, "lexical", 5))
                lists.append((f"lexical:{i}", found, rc.lexical_weight))
                per_method.setdefault("lexical", []).extend(found)
        async with recorder.stage("fuse") as st:
            fused = rrf(lists, k=rc.rrf_k)
            for cand_list in per_method.values():
                cand_list.sort(key=lambda c: -c.score)
            st.update(lists=len(lists), count=len(fused), top=_summary(fused, "rrf", 8))
        ranked = fused
        if self.reranker is not None and fused:
            async with recorder.stage("rerank", reranker=self.reranker.name) as st:
                head = fused[: self.config.rerank.top_n]
                scores = await self.reranker.scores(query, head)
                rescored = [c.with_score("rerank", s) for c, s in zip(head, scores, strict=True)]
                rescored.sort(key=lambda c: (-c.score, c.ranks.get("rrf", 0), c.chunk_id))
                ranked = rescored + fused[self.config.rerank.top_n :]
                st.update(count=len(head), top=_summary(rescored, "rerank", 8))
        async with recorder.stage("diversify", enabled=rc.mmr_enabled) as st:
            final = mmr(ranked, k=rc.top_k, lambda_=rc.mmr_lambda) if rc.mmr_enabled else ranked[: rc.top_k]
            st["count"] = len(final)
        return RetrievalResult(queries, final, fused, per_method)
