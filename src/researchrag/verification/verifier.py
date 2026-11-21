"""Claim-level verification and the remove/flag policy.

Each claim is judged against the text of the passages it cites. Claims without citations
(or whose citations all dangle) are flagged ``uncited`` and judged unsupported. Under the
``remove`` policy unsupported and contradicted claims are dropped and the answer text is
rebuilt from the surviving claims; under ``flag`` they stay but are marked. The answer-level
faithfulness score is ``(supported + 0.5 * partially_supported) / claims``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researchrag.config.pipeline import VerificationConfig
from researchrag.generation.packing import Passage
from researchrag.generation.schema import ClaimOut, RawAnswer, RawClaim
from researchrag.verification.citations import MARKER_RE, CitationReport
from researchrag.verification.judges import ClaimJudge

FAILING = {"unsupported", "contradicted"}


@dataclass
class VerificationOutcome:
    """Per-claim verdicts, the policy-applied answer text and summary scores."""

    claims: list[ClaimOut]
    answer_text: str
    faithfulness: float | None
    failed_claims: list[ClaimOut] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def supported_count(self) -> int:
        """Claims kept and at least partially supported."""
        return sum(
            1 for c in self.claims if not c.removed and c.verdict in {"supported", "partially_supported"}
        )


def faithfulness(claims: list[ClaimOut]) -> float | None:
    """Supported-claim ratio counting partial support as half."""
    if not claims:
        return None
    score = sum(
        1.0 if c.verdict == "supported" else 0.5 if c.verdict == "partially_supported" else 0.0
        for c in claims
    )
    return round(score / len(claims), 4)


class ClaimVerifier:
    """Applies a judge to every claim and enforces the configured policy."""

    def __init__(self, judge: ClaimJudge, config: VerificationConfig) -> None:
        self.judge = judge
        self.config = config

    async def verify(
        self, answer: RawAnswer, passages: list[Passage], report: CitationReport
    ) -> VerificationOutcome:
        """Judge claims and apply the policy."""
        by_number = {p.number: p for p in passages}
        fabricated = set(report.fabricated)
        claims: list[ClaimOut] = []
        raw_claims = list(answer.claims)
        if not raw_claims and answer.answer.strip() and not answer.abstained:
            # The model returned prose without claims: treat the whole answer as one claim.
            raw_claims = [
                RawClaim(
                    text=MARKER_RE.sub("", answer.answer).strip(),
                    citations=[int(m) for m in MARKER_RE.findall(answer.answer)],
                )
            ]
        for raw in raw_claims:
            cited = [n for n in raw.citations if n in by_number]
            flags: list[str] = []
            if raw.citations and len(cited) < len(raw.citations):
                flags.append("dangling_citation")
            if any(n in fabricated for n in cited):
                flags.append("fabricated_quote")
            if not cited:
                flags.append("uncited")
                claims.append(
                    ClaimOut(
                        text=raw.text,
                        citations=[],
                        verdict="unsupported",
                        rationale="claim cites no retrieved passage",
                        flags=flags,
                    )
                )
                continue
            evidence = "\n".join(by_number[n].text for n in cited)
            result = await self.judge.judge(raw.text, evidence)
            claims.append(
                ClaimOut(
                    text=raw.text,
                    citations=cited,
                    verdict=result.verdict,
                    rationale=result.rationale,
                    flags=flags,
                )
            )
        failed = [c for c in claims if c.verdict in FAILING]
        if self.config.policy == "remove":
            for claim in failed:
                claim.removed = True
        kept = [c for c in claims if not c.removed]
        if failed and self.config.policy == "remove":
            text = " ".join(c.text.rstrip() + "".join(f" [{n}]" for n in c.citations) for c in kept)
        else:
            text = answer.answer
        warnings = [f"{len(failed)} claim(s) failed verification ({self.config.policy})"] if failed else []
        return VerificationOutcome(claims, text, faithfulness(claims), failed, warnings)
