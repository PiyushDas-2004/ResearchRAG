"""Assembles pipelines from configuration so the API, CLI and evaluation share one wiring."""

from __future__ import annotations

from dataclasses import dataclass

from researchrag.config.pipeline import PipelineConfig
from researchrag.config.pricing import PricingTable
from researchrag.conversation.condense import Condenser
from researchrag.embeddings.service import build_embedder
from researchrag.generation.generator import AnswerGenerator
from researchrag.ingestion.service import IngestionService, ParseOptions
from researchrag.llm.base import LLMClient
from researchrag.llm.factory import build_llm
from researchrag.observability.metering import MeteredEmbeddings, MeteredLLM
from researchrag.pipeline import QueryPipeline
from researchrag.rerank.rerankers import build_reranker
from researchrag.retrieval.retriever import Retriever
from researchrag.retrieval.rewrite import QueryRewriter
from researchrag.retrieval.types import SearchBackend
from researchrag.storage.repositories import ConversationStore, TraceSink
from researchrag.storage.store import DocumentStore
from researchrag.verification.judges import build_judge
from researchrag.verification.verifier import ClaimVerifier


@dataclass
class Stack:
    """A configured ingestion service and query pipeline over one store/backend."""

    config: PipelineConfig
    ingestion: IngestionService
    pipeline: QueryPipeline
    llm: LLMClient
    embeddings: MeteredEmbeddings


async def build_stack(
    config: PipelineConfig,
    *,
    store: DocumentStore,
    backend: SearchBackend,
    traces: TraceSink,
    conversations: ConversationStore,
    pricing: PricingTable,
    parse: ParseOptions | None = None,
    llm: LLMClient | None = None,
    store_prompts: bool = True,
    log_prompts: bool = False,
) -> Stack:
    """Build every component selected by ``config``."""
    embeddings = MeteredEmbeddings(build_embedder(config.embedding), config.embedding)
    model = MeteredLLM(llm or build_llm(config.generation))
    judge_llm: LLMClient = model
    if config.verification.judge == "llm" and config.verification.judge_model:
        judge_llm = MeteredLLM(build_llm(config.generation, model=config.verification.judge_model))
    ingestion = IngestionService(store, embeddings, config, parse)
    version = await ingestion.version()
    reranker = build_reranker(config.rerank.provider, config.rerank.model, model, config.rerank.cache_size)
    rewriter = QueryRewriter(model, config.rewrite.multi_query or 2) if config.rewrite.enabled else None
    retriever = Retriever(backend, embeddings, version, config, reranker=reranker, rewriter=rewriter)
    pipeline = QueryPipeline(
        config=config,
        retriever=retriever,
        backend=backend,
        generator=AnswerGenerator(model, config.generation),
        verifier=ClaimVerifier(build_judge(config.verification, judge_llm), config.verification),
        condenser=Condenser(model, config.rewrite.max_history_turns),
        traces=traces,
        conversations=conversations,
        pricing=pricing,
        store_prompts=store_prompts,
        log_prompts=log_prompts,
    )
    return Stack(config, ingestion, pipeline, model, embeddings)
