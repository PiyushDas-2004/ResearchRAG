"""The end-to-end query pipeline.

condense -> retrieve (rewrite, vector, lexical, RRF, rerank, MMR) -> small-to-big parents ->
injection scan -> token-budget packing -> evidence gate (abstain without calling the LLM) ->
structured generation -> citation resolution + quote checks -> claim verification -> policy
(remove/flag, regenerate once with feedback) -> output guard -> persisted trace.

Every stage is timed into an OpenTelemetry span, a Prometheus histogram and the trace.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from researchrag.config.pipeline import PipelineConfig
from researchrag.config.pricing import CostLedger, PricingTable
from researchrag.conversation.condense import Condenser
from researchrag.generation.generator import AnswerGenerator, GenerationOutcome, TokenSink
from researchrag.generation.packing import Passage, expand_to_parents, pack_context
from researchrag.generation.schema import (
    CitationOut,
    ClaimOut,
    QueryResult,
    SearchedOut,
    UsageOut,
)
from researchrag.ingestion.tokenizer import get_tokenizer
from researchrag.observability.metering import metered
from researchrag.observability.metrics import CLAIM_VERDICTS, INJECTIONS, QUERIES, VERIFICATION_FAILURES
from researchrag.observability.tracing import StageRecorder, current_trace_id
from researchrag.retrieval.retriever import Retriever
from researchrag.retrieval.types import SearchBackend, SearchFilters
from researchrag.security.injection import (
    InjectionFinding,
    new_canary,
    output_guard,
    sanitize_text,
    scan_text,
)
from researchrag.storage.repositories import ConversationStore, TraceSink
from researchrag.textutils import content_terms, coverage
from researchrag.verification.citations import resolve_citations
from researchrag.verification.verifier import ClaimVerifier, VerificationOutcome

log = structlog.get_logger("researchrag.pipeline")

NOT_ENOUGH_EVIDENCE = "Not enough evidence in the indexed papers to answer this question."
WITHHELD = (
    "The answer was withheld because it appeared to follow instructions embedded in a document "
    "or to leak the system prompt."
)
EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class QueryRequest:
    """One question to answer."""

    question: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    history: list[tuple[str, str]] = field(default_factory=list)
    conversation_id: str | None = None
    api_key_id: str | None = None


@dataclass
class _Verified:
    generation: GenerationOutcome
    citations: list[CitationOut]
    verification: VerificationOutcome
    warnings: list[str]
    fabricated: list[int]


def evidence_score(question: str, passages: list[Passage]) -> float:
    """Best single-passage coverage of the question's content terms (0..1)."""
    terms = set(content_terms(question))
    if not terms or not passages:
        return 0.0
    return round(max(coverage(terms, set(content_terms(p.text))) for p in passages), 4)


async def _noop_event(name: str, data: dict[str, Any]) -> None:
    return None


class QueryPipeline:
    """Answers questions with verified citations for one pipeline configuration."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        retriever: Retriever,
        backend: SearchBackend,
        generator: AnswerGenerator,
        verifier: ClaimVerifier,
        condenser: Condenser | None,
        traces: TraceSink,
        conversations: ConversationStore,
        pricing: PricingTable,
        store_prompts: bool = True,
        log_prompts: bool = False,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.backend = backend
        self.generator = generator
        self.verifier = verifier
        self.condenser = condenser
        self.traces = traces
        self.conversations = conversations
        self.pricing = pricing
        self.store_prompts = store_prompts
        self.log_prompts = log_prompts
        self.tokenizer = get_tokenizer(config.chunking.tokenizer)

    async def answer(
        self, request: QueryRequest, *, on_token: TokenSink | None = None, on_event: EventSink | None = None
    ) -> QueryResult:
        """Run the full pipeline and persist the trace."""
        ledger = CostLedger(self.pricing)
        with metered(ledger):
            return await self._answer(request, ledger, on_token, on_event or _noop_event)

    async def _answer(
        self, request: QueryRequest, ledger: CostLedger, on_token: TokenSink | None, emit: EventSink
    ) -> QueryResult:
        recorder = StageRecorder()
        trace_id = str(uuid.uuid4())
        canary = new_canary()
        details: dict[str, Any] = {
            "config_hash": self.config.config_hash(),
            "conversation_id": request.conversation_id,
        }
        warnings: list[str] = []

        history = list(request.history)
        if request.conversation_id:
            history = (
                await self.conversations.history(
                    request.conversation_id, self.config.rewrite.max_history_turns * 2
                )
            ) + history
        standalone = request.question
        if history and self.condenser is not None and self.config.rewrite.condense_followups:
            async with recorder.stage("condense") as st:
                standalone = await self.condenser.condense(history, request.question)
                st.update(history_turns=len(history), standalone=standalone)
        await emit("status", {"stage": "retrieve", "standalone_question": standalone})

        retrieval = await self.retriever.retrieve(standalone, request.filters, recorder)
        async with recorder.stage("expand") as st:
            parent_ids = [c.parent_id for c in retrieval.candidates if c.parent_id]
            parents = await self.backend.get_parents(parent_ids)
            passages = expand_to_parents(retrieval.candidates, parents)
            st.update(children=len(retrieval.candidates), passages=len(passages))
        findings: list[InjectionFinding] = []
        async with recorder.stage("guard.input", action=self.config.security.injection_action) as st:
            passages, findings, flagged = self._scan_passages(passages)
            st.update(flagged=flagged)
            if flagged:
                action = self.config.security.injection_action
                warnings.append(f"{len(flagged)} passage(s) contained instruction-like text ({action})")
        async with recorder.stage("pack", budget=self.config.generation.max_context_tokens) as st:
            packed = pack_context(passages, self.tokenizer, self.config.generation.max_context_tokens)
            st["count"] = len(packed)
        details["passages"] = [
            {
                "number": p.number,
                "chunk_id": p.chunk_id,
                "child_ids": p.child_ids,
                "document_id": p.document_id,
                "title": p.title,
                "pages": [p.page_start, p.page_end],
                "score": round(p.score, 5),
                "suspicious": p.suspicious,
            }
            for p in packed
        ]
        score = evidence_score(standalone, packed)
        searched = SearchedOut(
            queries=retrieval.queries,
            filters=request.filters.as_dict(),
            candidates=len(retrieval.fused),
            passages=len(packed),
            documents=sorted({p.title for p in packed}),
            evidence_score=score,
        )
        await emit("status", {"stage": "generate", "passages": len(packed)})

        async with recorder.stage("gate", threshold=self.config.generation.min_evidence_score) as st:
            st["evidence_score"] = score
            gate_open = bool(packed) and score >= self.config.generation.min_evidence_score
            st["passed"] = gate_open
        if not gate_open:
            reason = NOT_ENOUGH_EVIDENCE + (
                " No passages matched the search."
                if not packed
                else f" The best passage covers {score:.0%} of the question's terms."
            )
            result = self._result(
                trace_id, request, standalone, reason, None, [], [], searched, warnings, ledger, recorder
            )
            return await self._finish(result, details, request)

        verified = await self._generate_and_verify(standalone, packed, canary, recorder, details, on_token)
        warnings.extend(verified.warnings)
        if (
            self.config.verification.enabled
            and self.config.verification.regenerate_on_failure
            and verified.verification.failed_claims
            and not verified.generation.answer.abstained
        ):
            feedback = "\n".join(
                f"- {c.text} ({c.verdict}: {c.rationale})" for c in verified.verification.failed_claims
            )
            await emit(
                "status", {"stage": "regenerate", "failed_claims": len(verified.verification.failed_claims)}
            )
            retry = await self._generate_and_verify(
                standalone, packed, canary, recorder, details, None, feedback=feedback, stage="regenerate"
            )
            better = (retry.verification.faithfulness or 0) >= (
                verified.verification.faithfulness or 0
            ) and retry.verification.supported_count > 0
            details["regenerated"] = {"accepted": better, "faithfulness": retry.verification.faithfulness}
            if better:
                verified = _Verified(
                    retry.generation,
                    retry.citations,
                    retry.verification,
                    warnings + retry.warnings,
                    retry.fabricated,
                )

        gen = verified.generation
        abstain_reason: str | None = None
        if gen.failed:
            abstain_reason = gen.answer.abstain_reason
        elif gen.answer.abstained:
            abstain_reason = f"{NOT_ENOUGH_EVIDENCE} {gen.answer.abstain_reason or ''}".strip()
        elif self.config.verification.enabled and verified.verification.supported_count == 0:
            abstain_reason = (
                f"{NOT_ENOUGH_EVIDENCE} Verification found no claim supported by the retrieved passages."
            )

        answer_text = verified.verification.answer_text
        async with recorder.stage("guard.output", enabled=self.config.security.output_guard) as st:
            guard = output_guard(
                answer_text + " " + " ".join(c.text for c in verified.verification.claims),
                canary=canary,
                findings=findings,
            )
            st["violations"] = guard.violations
        if self.config.security.output_guard and not guard.ok:
            for violation in guard.violations:
                INJECTIONS.labels(kind="output").inc()
                warnings.append(f"output guard: {violation}")
            abstain_reason = WITHHELD

        claims = verified.verification.claims
        citations = verified.citations
        if abstain_reason is not None:
            answer_text = ""
            citations = []
        outcome = verified.verification
        result = self._result(
            trace_id,
            request,
            standalone,
            abstain_reason,
            outcome.faithfulness,
            claims,
            citations,
            searched,
            warnings,
            ledger,
            recorder,
            answer_text=answer_text,
            confidence=min(gen.answer.confidence, outcome.faithfulness or 0.0),
        )
        return await self._finish(result, details, request)

    def _scan_passages(
        self, passages: list[Passage]
    ) -> tuple[list[Passage], list[InjectionFinding], list[dict[str, Any]]]:
        action = self.config.security.injection_action
        kept: list[Passage] = []
        findings: list[InjectionFinding] = []
        flagged: list[dict[str, Any]] = []
        for passage in passages:
            finding = scan_text(passage.text)
            if not finding.found:
                kept.append(passage)
                continue
            INJECTIONS.labels(kind="input").inc()
            findings.append(finding)
            flagged.append(
                {"chunk_id": passage.chunk_id, "title": passage.title, "sentences": list(finding.sentences)}
            )
            passage.suspicious = True
            passage.injection_sentences = finding.sentences
            if action == "drop":
                continue
            if action == "sanitize":
                passage.text = sanitize_text(passage.text, finding)
            kept.append(passage)
        return kept, findings, flagged

    async def _generate_and_verify(
        self,
        question: str,
        packed: list[Passage],
        canary: str,
        recorder: StageRecorder,
        details: dict[str, Any],
        on_token: TokenSink | None,
        *,
        feedback: str | None = None,
        stage: str = "generate",
    ) -> _Verified:
        async with recorder.stage(stage, model=self.generator.config.model) as st:
            gen = await self.generator.generate(
                question, packed, canary, feedback=feedback, on_token=on_token
            )
            st.update(attempts=gen.attempts, repaired=gen.repaired, failed=gen.failed, errors=gen.errors)
        record: dict[str, Any] = {
            "raw_answer": gen.raw_text[:8000],
            "attempts": gen.attempts,
            "errors": gen.errors,
        }
        if self.store_prompts:
            record["prompt"] = [{"role": m.role, "content": m.content} for m in gen.prompt]
        if self.log_prompts:
            log.info("llm.prompt", stage=stage, messages=record.get("prompt"))
        details[stage] = record
        async with recorder.stage("cite" if stage == "generate" else f"{stage}.cite") as st:
            report = resolve_citations(gen.answer, packed, self.config.verification.quote_match_threshold)
            st.update(citations=len(report.citations), dangling=report.dangling, fabricated=report.fabricated)
        async with recorder.stage(
            "verify" if stage == "generate" else f"{stage}.verify", judge=self.verifier.judge.name
        ) as st:
            if self.config.verification.enabled:
                outcome = await self.verifier.verify(gen.answer, packed, report)
            else:
                outcome = VerificationOutcome(
                    [
                        ClaimOut(
                            text=c.text,
                            citations=c.citations,
                            verdict="supported",
                            rationale="verification disabled",
                        )
                        for c in gen.answer.claims
                    ],
                    gen.answer.answer,
                    None,
                )
            st.update(
                verdicts=[c.verdict for c in outcome.claims],
                faithfulness=outcome.faithfulness,
                removed=sum(c.removed for c in outcome.claims),
            )
        details[f"{stage}_claims"] = [c.model_dump() for c in outcome.claims]
        kept_markers = {n for c in outcome.claims if not c.removed for n in c.citations}
        citations = [c for c in report.citations if c.marker in kept_markers]
        return _Verified(gen, citations, outcome, report.warnings + outcome.warnings, report.fabricated)

    def _result(
        self,
        trace_id: str,
        request: QueryRequest,
        standalone: str,
        abstain_reason: str | None,
        faithfulness: float | None,
        claims: list[ClaimOut],
        citations: list[CitationOut],
        searched: SearchedOut,
        warnings: list[str],
        ledger: CostLedger,
        recorder: StageRecorder,
        *,
        answer_text: str = "",
        confidence: float = 0.0,
    ) -> QueryResult:
        return QueryResult(
            trace_id=trace_id,
            question=request.question,
            standalone_question=standalone,
            answer=answer_text if abstain_reason is None else "",
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
            confidence=round(confidence, 4) if abstain_reason is None else 0.0,
            faithfulness=faithfulness,
            claims=claims,
            citations=citations,
            searched=searched,
            warnings=warnings,
            usage=UsageOut(
                input_tokens=ledger.input_tokens,
                output_tokens=ledger.output_tokens,
                embedding_tokens=ledger.embedding_tokens,
                cost_usd=round(ledger.cost_usd, 8),
                unpriced_models=sorted(ledger.unpriced_models),
            ),
            latency_ms=recorder.total_ms,
            stage_latency_ms=dict(recorder.latency_ms),
            config_name=self.config.name,
        )

    async def _finish(
        self, result: QueryResult, details: dict[str, Any], request: QueryRequest
    ) -> QueryResult:
        QUERIES.labels(outcome="abstained" if result.abstained else "answered").inc()
        for claim in result.claims:
            CLAIM_VERDICTS.labels(verdict=claim.verdict).inc()
        if any(c.verdict in {"unsupported", "contradicted"} for c in result.claims):
            VERIFICATION_FAILURES.inc()
        details["otel_trace_id"] = current_trace_id()
        details["stage_latency_ms"] = result.stage_latency_ms
        await self.traces.save(result, details, api_key_id=request.api_key_id)
        if request.conversation_id:
            await self.conversations.append(
                request.conversation_id, "user", request.question, request.api_key_id
            )
            reply = result.answer if not result.abstained else (result.abstain_reason or "")
            await self.conversations.append(request.conversation_id, "assistant", reply, request.api_key_id)
        log.info(
            "query.completed",
            trace_id=result.trace_id,
            abstained=result.abstained,
            faithfulness=result.faithfulness,
            latency_ms=result.latency_ms,
            passages=result.searched.passages,
            cost_usd=result.usage.cost_usd,
        )
        return result
