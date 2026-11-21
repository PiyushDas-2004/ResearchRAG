"""A transparent, offline claim/evidence judge based on lexical and numeric agreement.

This is the default judge for the zero-key stack and the baseline that the LLM and NLI judges
are calibrated against. Decision procedure:

1. Numbers: when the claim otherwise overlaps the evidence well, a number in the claim that is
   absent from the evidence (while the evidence states other numbers) is a contradiction
   (a wrong value); a missing number caps the verdict at ``unsupported``/partial otherwise.
2. Direction/negation: if overlap is high but the claim flips a direction word (``higher`` vs
   ``lower``) or a negation relative to the evidence, the verdict is ``contradicted``.
3. Coverage: the fraction of the claim's content terms found in the evidence decides between
   ``supported``, ``partially_supported`` and ``unsupported``.

It cannot detect paraphrased contradictions or entailment that needs world knowledge; see
docs/EVALUATION.md for its measured agreement with the hand-labelled calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from researchrag.textutils import ANTONYMS, NEGATIONS, content_terms, numbers, split_sentences, words

Verdict = Literal["supported", "partially_supported", "unsupported", "contradicted"]


@dataclass(frozen=True)
class JudgeResult:
    """A verdict with a short explanation and the coverage score that produced it."""

    verdict: Verdict
    rationale: str
    score: float


def best_window(claim_terms: set[str], evidence: str, max_sentences: int = 3) -> tuple[str, float]:
    """The span of 1..``max_sentences`` consecutive evidence sentences that best covers the claim.

    Checks are applied to this local window so that, e.g., a negation elsewhere in a long
    passage does not flip the verdict. Ties prefer the shorter window.
    """
    sentences = [s.text for s in split_sentences(evidence)] or [evidence]
    best_text, best_cov = sentences[0], -1.0
    for size in range(1, max_sentences + 1):
        for i in range(len(sentences) - size + 1):
            window = " ".join(sentences[i : i + size])
            cov = len(claim_terms & set(content_terms(window))) / len(claim_terms)
            if cov > best_cov + 1e-9:
                best_text, best_cov = window, cov
    return best_text, best_cov


def lexical_judge(claim: str, evidence: str, *, supported: float = 0.8, partial: float = 0.55) -> JudgeResult:
    """Judge whether ``evidence`` supports ``claim`` using term coverage, numbers and polarity."""
    claim_terms = set(content_terms(claim))
    if not claim_terms:
        return JudgeResult("unsupported", "claim has no content terms", 0.0)
    window, cov = best_window(claim_terms, evidence)

    claim_nums = numbers(claim)
    window_nums = numbers(window)
    missing_nums = claim_nums - window_nums
    if missing_nums and window_nums and cov >= partial:
        return JudgeResult(
            "contradicted",
            f"numbers {sorted(missing_nums)} not in evidence, which states {sorted(window_nums)[:5]}",
            cov,
        )

    claim_words = set(words(claim))
    window_words = set(words(window))
    if cov >= partial:
        for word in claim_words:
            opposite = ANTONYMS.get(word)
            if opposite and opposite in window_words and word not in window_words:
                return JudgeResult("contradicted", f"claim says '{word}' but evidence says '{opposite}'", cov)
        if bool(claim_words & NEGATIONS) != bool(window_words & NEGATIONS) and cov >= supported:
            return JudgeResult("contradicted", "negation mismatch between claim and evidence", cov)

    if missing_nums and cov >= partial:
        cov = partial - 0.01
    if cov >= supported:
        return JudgeResult("supported", f"{cov:.0%} of claim terms found in evidence", cov)
    if cov >= partial:
        return JudgeResult("partially_supported", f"only {cov:.0%} of claim terms found in evidence", cov)
    return JudgeResult("unsupported", f"evidence covers {cov:.0%} of claim terms", cov)
