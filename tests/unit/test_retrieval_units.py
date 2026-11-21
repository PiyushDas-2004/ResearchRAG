"""Fusion, MMR, rerankers, embeddings service and the in-memory backend."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchrag.config.pipeline import EmbeddingConfig, PipelineConfig
from researchrag.embeddings.fake import HashingEmbedder
from researchrag.embeddings.gemini_embedder import GeminiEmbedder
from researchrag.embeddings.local_embedder import LocalEmbedder
from researchrag.embeddings.openai_embedder import OpenAIEmbedder
from researchrag.embeddings.service import EmbeddingService, build_embedder
from researchrag.errors import ProviderError, ProviderTimeoutError, RateLimitedError, RetryableProviderError
from researchrag.ingestion.service import IngestionService, ParseOptions, embedding_input
from researchrag.llm.base import RetryingLLM
from researchrag.llm.mock import MockLLM
from researchrag.observability.tracing import StageRecorder
from researchrag.rerank.rerankers import (
    CachedReranker,
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    build_reranker,
)
from researchrag.resilience import AsyncRateLimiter, retry_async
from researchrag.retrieval.fusion import rrf
from researchrag.retrieval.memory import MemoryStore
from researchrag.retrieval.mmr import mmr
from researchrag.retrieval.retriever import Retriever
from researchrag.retrieval.rewrite import QueryRewriter
from researchrag.retrieval.types import Candidate, SearchFilters
from tests.conftest import paper_pdf


def cand(cid: str, score: float = 0.0, text: str = "", emb: tuple[float, ...] | None = None) -> Candidate:
    return Candidate(cid, "d", text or f"text {cid}", 1, 1, "results", "T", score=score, embedding=emb)


def test_rrf_basic_and_weights() -> None:
    a = [cand("x", 0.9), cand("y", 0.8)]
    b = [cand("y", 3.0), cand("z", 2.0)]
    fused = rrf([("vector", a, 1.0), ("lexical", b, 1.0)], k=60)
    assert [c.chunk_id for c in fused] == ["y", "x", "z"]
    assert fused[0].scores["vector"] == 0.8
    assert fused[0].scores["lexical"] == 3.0
    assert fused[0].ranks == {"vector": 2, "lexical": 1, "rrf": None} or fused[0].ranks["vector"] == 2
    assert math.isclose(fused[0].score, 1 / 62 + 1 / 61)
    weighted = rrf([("vector", a, 0.0), ("lexical", b, 1.0)], k=60, limit=1)
    assert [c.chunk_id for c in weighted] == ["y"]


@given(st.lists(st.lists(st.sampled_from("abcdefgh"), unique=True, max_size=8), min_size=1, max_size=4))
def test_rrf_is_deterministic_and_order_independent(lists: list[list[str]]) -> None:
    named = [(f"m{i}", [cand(c) for c in lst], 1.0) for i, lst in enumerate(lists)]
    first = [c.chunk_id for c in rrf(named)]
    second = [c.chunk_id for c in rrf(list(reversed(named)))]
    assert first == second
    assert len(first) == len({c for lst in lists for c in lst})
    scores = [c.score for c in rrf(named)]
    assert scores == sorted(scores, reverse=True)


def test_rrf_ignores_duplicates_within_a_list() -> None:
    fused = rrf([("m", [cand("a"), cand("a"), cand("b")], 1.0)])
    assert [c.chunk_id for c in fused] == ["a", "b"]


def test_mmr_prefers_diverse_results() -> None:
    dup = (1.0, 0.0)
    items = [cand("a", 1.0, emb=dup), cand("b", 0.99, emb=dup), cand("c", 0.5, emb=(0.0, 1.0))]
    assert [c.chunk_id for c in mmr(items, k=2, lambda_=0.5)] == ["a", "c"]
    assert [c.chunk_id for c in mmr(items, k=2, lambda_=1.0)] == ["a", "b"]
    lexical = [
        cand("a", 1.0, "glacier melt river"),
        cand("b", 0.9, "glacier melt river"),
        cand("c", 0.2, "bees decline farms"),
    ]
    assert [c.chunk_id for c in mmr(lexical, k=2, lambda_=0.5)] == ["a", "c"]
    assert mmr([], k=3, lambda_=0.5) == []


async def test_rerankers() -> None:
    items = [
        cand("a", text="Bees declined on farmland transects."),
        cand("b", text="Glacier meltwater cools rivers by 2.1 degrees."),
    ]
    lexical = await LexicalReranker().scores("glacier meltwater 2.1", items)
    assert lexical[1] > lexical[0]
    llm = await LLMReranker(RetryingLLM(MockLLM(), max_retries=0)).scores("glacier meltwater", items)
    assert llm[1] > llm[0]

    class BadLLM:
        provider, model = "x", "y"

        async def complete(self, request: Any) -> Any:
            return SimpleNamespace(text="not json")

    fallback = await LLMReranker(BadLLM()).scores("q", items)  # type: ignore[arg-type]
    assert fallback == [1.0, 0.5]
    encoder = SimpleNamespace(predict=lambda pairs: [0.1, 0.9])
    assert await CrossEncoderReranker("m", encoder=encoder).scores("q", items) == [0.1, 0.9]
    counting = LexicalReranker()
    cached = CachedReranker(counting, size=1)
    await cached.scores("q", items)
    await cached.scores("q", items[1:])
    assert cached.hits == 1
    assert build_reranker("none", "m", None, 0) is None
    assert isinstance(build_reranker("lexical", "m", None, 0), LexicalReranker)
    with pytest.raises(Exception, match="LLM"):
        build_reranker("llm", "m", None, 10)


def test_hashing_embedder_properties() -> None:
    emb = HashingEmbedder(64)
    v = emb.vector("Glacier meltwater cools the river")
    assert len(v) == 64
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9)
    assert v == emb.vector("Glacier meltwater cools the river")
    near = emb.vector("glacier meltwater cooled rivers")
    far = emb.vector("bees on farmland transects")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(v, near) > dot(v, far)
    assert emb.vector("the of and") == [0.0] * 64


class FlakyEmbedder:
    provider, model, dimension = "fake", "flaky", 8

    def __init__(self, failures: int, error: Exception, out_dim: int = 8) -> None:
        self.out_dim = out_dim
        self.failures = failures
        self.error = error
        self.calls = 0

    async def embed(self, texts: list[str], kind: str) -> Any:
        from researchrag.embeddings.base import EmbeddingBatch

        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return EmbeddingBatch([[0.1] * self.out_dim for _ in texts], len(texts))


async def _nosleep(_: float) -> None:
    return None


async def test_embedding_service_batches_caches_and_retries() -> None:
    inner = FlakyEmbedder(2, RateLimitedError("429", retry_after=0.0))
    service = EmbeddingService(
        inner, EmbeddingConfig(batch_size=2, dimension=8, max_retries=3), sleep=_nosleep
    )
    out = await service.embed(["a", "b", "c"])
    assert len(out) == 3
    assert inner.calls == 4  # 2 failures + 2 batches
    await service.embed(["a", "c"])
    assert service.cache_hits == 2
    assert service.model_id == "fake/flaky"
    assert service.tokens_used == 3
    hard = EmbeddingService(
        FlakyEmbedder(1, ProviderError("bad")), EmbeddingConfig(dimension=8), sleep=_nosleep
    )
    with pytest.raises(ProviderError):
        await hard.embed(["x"])
    exhausted = EmbeddingService(
        FlakyEmbedder(9, ProviderTimeoutError("t")),
        EmbeddingConfig(dimension=8, max_retries=1),
        sleep=_nosleep,
    )
    with pytest.raises(ProviderTimeoutError):
        await exhausted.embed(["x"])
    wrong = EmbeddingService(
        FlakyEmbedder(0, ProviderError("x"), out_dim=5), EmbeddingConfig(dimension=8), sleep=_nosleep
    )
    with pytest.raises(ProviderError, match="wrong number or size"):
        await wrong.embed(["x"])
    nocache = EmbeddingService(
        FlakyEmbedder(0, ProviderError("x")), EmbeddingConfig(dimension=8, cache_size=0)
    )
    await nocache.embed(["x"])
    await nocache.embed(["x"])
    assert nocache.cache_hits == 0


async def test_retry_async_respects_retry_after_and_callbacks() -> None:
    delays: list[float] = []
    attempts: list[int] = []

    async def sleep(d: float) -> None:
        delays.append(d)

    state = {"n": 0}

    async def fn() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise RetryableProviderError("x", retry_after=1.5 if state["n"] == 1 else None)
        return "ok"

    assert await retry_async(fn, max_retries=3, sleep=sleep, on_retry=lambda a, e: attempts.append(a)) == "ok"
    assert delays[0] == 1.5
    assert 0 < delays[1] <= 1.0
    assert attempts == [1, 2]


async def test_rate_limiter_waits_when_bucket_empty() -> None:
    now = [0.0]
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)
        now[0] += d

    limiter = AsyncRateLimiter(60, clock=lambda: now[0], sleep=sleep)
    await limiter.acquire()
    await limiter.acquire()
    assert slept
    assert math.isclose(sum(slept), 1.0, rel_tol=1e-6)


async def test_provider_embedders_with_fake_clients() -> None:
    class OpenAIClient:
        def __init__(self, fail: Exception | None = None) -> None:
            self.fail = fail
            self.embeddings = self

        async def create(self, **kwargs: Any) -> Any:
            if self.fail:
                raise self.fail
            data = [
                SimpleNamespace(index=i, embedding=[float(i)] * kwargs["dimensions"])
                for i in range(len(kwargs["input"]))
            ]
            return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=7))

    batch = await OpenAIEmbedder("text-embedding-3-small", 4, client=OpenAIClient()).embed(
        ["a", "b"], "document"
    )
    assert batch.vectors == [[0.0] * 4, [1.0] * 4]
    assert batch.input_tokens == 7

    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "2"})

    with pytest.raises(RateLimitedError) as exc:
        await OpenAIEmbedder("m", 4, client=OpenAIClient(RateLimitError())).embed(["a"], "query")
    assert exc.value.retry_after == 2.0

    calls: dict[str, Any] = {}

    class GeminiModels:
        async def embed_content(self, **kwargs: Any) -> Any:
            calls.update(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0])], usage_metadata=None)

    client = SimpleNamespace(aio=SimpleNamespace(models=GeminiModels()))
    gem = await GeminiEmbedder("gemini-embedding-001", 2, client=client).embed(["q"], "query")
    assert gem.vectors == [[0.6, 0.8]]
    assert calls["config"]["task_type"] == "RETRIEVAL_QUERY"

    class GeminiFail:
        async def embed_content(self, **kwargs: Any) -> Any:
            err = Exception("quota")
            err.code = 429  # type: ignore[attr-defined]
            raise err

    with pytest.raises(RateLimitedError):
        await GeminiEmbedder("m", 2, client=SimpleNamespace(aio=SimpleNamespace(models=GeminiFail()))).embed(
            ["q"], "document"
        )
    local = LocalEmbedder(
        "m",
        2,
        encoder=SimpleNamespace(encode=lambda texts, normalize_embeddings: [[1.0, 0.0, 9.0]] * len(texts)),
    )
    assert (await local.embed(["a b"], "document")).vectors == [[1.0, 0.0]]
    assert isinstance(build_embedder(EmbeddingConfig()), HashingEmbedder)


def test_embedding_input_header() -> None:
    assert embedding_input("text", "Title", "related_work", True) == "Title. related work: text"
    assert embedding_input("text", None, "results", True) == "results: text"
    assert embedding_input("text", "T", "results", False) == "text"


async def _memory_corpus(
    config: PipelineConfig,
) -> tuple[MemoryStore, EmbeddingService, IngestionService, dict[str, str]]:
    store = MemoryStore()
    embeddings = EmbeddingService(build_embedder(config.embedding), config.embedding)
    ingestion = IngestionService(store, embeddings, config, ParseOptions(isolation="inline"))
    ids = {}
    for key in ("ir-01", "ir-02", "clim-02"):
        ids[key] = (await ingestion.ingest_bytes(paper_pdf(key), f"{key}.pdf", collection="c1")).document_id
    return store, embeddings, ingestion, ids


async def test_memory_backend_retrieval_filters_and_rewrite() -> None:
    config = PipelineConfig().with_overrides({"rewrite.enabled": True, "rewrite.multi_query": 2})
    store, embeddings, ingestion, ids = await _memory_corpus(config)
    again = await ingestion.ingest_bytes(paper_pdf("ir-01"), "dup.pdf")
    assert again.created is False
    version = await ingestion.version()
    rewriter = QueryRewriter(RetryingLLM(MockLLM(), max_retries=0), n=2)
    retriever = Retriever(store, embeddings, version, config, reranker=LexicalReranker(), rewriter=rewriter)
    recorder = StageRecorder()
    result = await retriever.retrieve(
        "What is BraidRank and how does it fuse scores?", SearchFilters(), recorder
    )
    assert len(result.queries) >= 2
    assert result.candidates
    assert ids["ir-01"] in {c.document_id for c in result.candidates[:3]}
    assert {"rewrite", "retrieve.vector", "retrieve.lexical", "fuse", "rerank", "diversify"} <= set(
        recorder.stages
    )
    only = SearchFilters(document_ids=(ids["clim-02"],))
    filtered = await retriever.retrieve("BraidRank fusion", only, StageRecorder())
    assert {c.document_id for c in filtered.candidates} <= {ids["clim-02"]}
    for flt, expected in [
        (SearchFilters(year_min=2023, year_max=2023), {ids["clim-02"], ids["ir-02"]}),
        (SearchFilters(collection="c2"), set()),
        (SearchFilters(authors=("nobody",)), set()),
    ]:
        res = await store.lexical_search(
            "roof cooling reranking", chunk_config_hash=retriever.chunk_config_hash, k=50, filters=flt
        )
        assert {c.document_id for c in res} <= expected
    assert (
        await store.lexical_search(
            "the of", chunk_config_hash=retriever.chunk_config_hash, k=5, filters=SearchFilters()
        )
        == []
    )
    missing = await store.chunks_missing_embeddings(version, None, 5)
    assert missing == []
    assert await store.get_parents(["nope"]) == {}
    assert len(await store.list_document_ids()) == 3


def test_search_filters_helpers() -> None:
    f = SearchFilters(year_min=2020)
    assert f.active
    assert not SearchFilters().active
    assert f.as_dict()["year_min"] == 2020


async def test_query_rewriter_degrades_gracefully() -> None:
    class Broken:
        provider, model = "x", "y"

        async def complete(self, request: Any) -> Any:
            raise ProviderError("down")

    assert await QueryRewriter(Broken(), 2).expand("q") == ["q"]  # type: ignore[arg-type]
    assert await QueryRewriter(Broken(), 0).expand("q") == ["q"]  # type: ignore[arg-type]
    out = await QueryRewriter(RetryingLLM(MockLLM(), max_retries=0), 1).expand(
        "What is the recall of BraidRank?"
    )
    assert out[0] == "What is the recall of BraidRank?"
    assert len(out) == 2
    random.seed(0)
