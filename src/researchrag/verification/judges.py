"""Claim judges: lexical (offline), LLM-based and local NLI cross-encoder."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from researchrag.config.pipeline import VerificationConfig
from researchrag.errors import ConfigurationError, ResearchRAGError
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.prompts import verify_messages
from researchrag.llm.base import LLMClient, LLMRequest
from researchrag.verification.lexical import JudgeResult, Verdict, lexical_judge

VERDICTS: set[str] = {"supported", "partially_supported", "unsupported", "contradicted"}


class ClaimJudge(Protocol):
    """Decides whether evidence supports a claim."""

    name: str

    async def judge(self, claim: str, evidence: str) -> JudgeResult:
        """Return a verdict for ``claim`` given ``evidence``."""
        ...


class LexicalJudge:
    """Offline term-coverage / number / polarity judge."""

    name = "lexical"

    def __init__(self, supported: float = 0.8, partial: float = 0.55) -> None:
        self.supported = supported
        self.partial = partial

    async def judge(self, claim: str, evidence: str) -> JudgeResult:
        """Judge deterministically."""
        return lexical_judge(claim, evidence, supported=self.supported, partial=self.partial)


class LLMJudge:
    """NLI-style judging by an LLM; falls back to the lexical judge on unusable output."""

    name = "llm"

    def __init__(self, llm: LLMClient, fallback: LexicalJudge | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or LexicalJudge()

    async def judge(self, claim: str, evidence: str) -> JudgeResult:
        """Ask the LLM for a verdict."""
        try:
            response = await self.llm.complete(
                LLMRequest(verify_messages(claim, evidence), task="verify", max_tokens=200)
            )
            data = parse_json_object(response.text)
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict not in VERDICTS:
                raise ValueError(f"bad verdict {verdict!r}")
        except (ResearchRAGError, ValueError):
            result = await self.fallback.judge(claim, evidence)
            return JudgeResult(
                result.verdict, f"llm judge failed; lexical fallback: {result.rationale}", result.score
            )
        typed: Verdict = verdict  # type: ignore[assignment]
        return JudgeResult(typed, str(data.get("rationale", ""))[:300], 1.0 if typed == "supported" else 0.0)


class NLIJudge:
    """Local NLI cross-encoder (e.g. a DeBERTa MNLI model via sentence-transformers)."""

    name = "nli"

    def __init__(
        self,
        model: str,
        *,
        encoder: Any | None = None,
        labels: tuple[str, ...] = ("contradiction", "entailment", "neutral"),
    ) -> None:
        if encoder is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415 - optional
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError("pip install 'researchrag[local]' for the NLI judge") from exc
            encoder = CrossEncoder(model)
        self._encoder: Any = encoder
        self.labels = labels
        self._lexical = LexicalJudge()

    async def judge(self, claim: str, evidence: str) -> JudgeResult:
        """Map NLI probabilities onto verdicts (neutral splits by lexical coverage)."""
        scores = await asyncio.to_thread(self._encoder.predict, [(evidence, claim)], apply_softmax=True)
        probs = dict(zip(self.labels, [float(x) for x in scores[0]], strict=True))
        top = max(probs, key=lambda k: probs[k])
        if top == "entailment":
            return JudgeResult("supported", f"entailment p={probs['entailment']:.2f}", probs["entailment"])
        if top == "contradiction":
            return JudgeResult(
                "contradicted", f"contradiction p={probs['contradiction']:.2f}", probs["entailment"]
            )
        lexical = await self._lexical.judge(claim, evidence)
        verdict: Verdict = (
            "partially_supported"
            if lexical.verdict in {"supported", "partially_supported"}
            else "unsupported"
        )
        return JudgeResult(
            verdict, f"neutral p={probs['neutral']:.2f}; {lexical.rationale}", probs["entailment"]
        )


def build_judge(config: VerificationConfig, llm: LLMClient | None) -> ClaimJudge:
    """Create the configured judge."""
    lexical = LexicalJudge(config.supported_threshold, config.partial_threshold)
    if config.judge == "lexical":
        return lexical
    if config.judge == "llm":
        if llm is None:
            raise ConfigurationError("The LLM judge needs an LLM client")
        return LLMJudge(llm, lexical)
    return NLIJudge(config.judge_model or "cross-encoder/nli-deberta-v3-small")
