"""Retrieval and generation metrics.

Relevance is defined against *gold evidence passages*, independently of how the corpus was
chunked: a retrieved text is relevant to a gold passage when it comes from the gold document
and contains at least ``RELEVANCE_THRESHOLD`` of the passage's content terms. This keeps the
metrics comparable across chunk sizes in ablations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from researchrag.evaluation.dataset import EvalQuestion, GoldEvidence
from researchrag.textutils import content_terms, coverage

RELEVANCE_THRESHOLD = 0.6
RETRIEVAL_KS = (1, 3, 5)


@dataclass(frozen=True)
class RetrievedText:
    """A retrieved chunk or packed passage, with the paper key it came from."""

    doc_key: str | None
    text: str


def matches(item: RetrievedText, gold: GoldEvidence) -> bool:
    """True when ``item`` covers enough of the gold passage and comes from the gold paper."""
    if item.doc_key != gold.doc_key:
        return False
    terms = set(content_terms(gold.passage))
    return bool(terms) and len(terms & set(content_terms(item.text))) / len(terms) >= RELEVANCE_THRESHOLD


def retrieval_metrics(question: EvalQuestion, ranked: Sequence[RetrievedText]) -> dict[str, float | None]:
    """hit@k, recall@k, precision@k, MRR and nDCG@k (binary relevance) for one answerable question."""
    if not question.answerable or not question.evidence:
        return {}
    relevant = [any(matches(item, g) for g in question.evidence) for item in ranked]
    # For nDCG each gold passage earns gain once (at its first match), so DCG <= ideal DCG.
    credited: set[int] = set()
    gains: list[bool] = []
    for item in ranked:
        new = [i for i, g in enumerate(question.evidence) if i not in credited and matches(item, g)]
        if new:
            credited.add(new[0])
        gains.append(bool(new))
    out: dict[str, float | None] = {}
    for k in RETRIEVAL_KS:
        top = ranked[:k]
        found = [g for g in question.evidence if any(matches(item, g) for item in top)]
        out[f"hit@{k}"] = 1.0 if any(relevant[:k]) else 0.0
        out[f"recall@{k}"] = len(found) / len(question.evidence)
        out[f"precision@{k}"] = sum(relevant[:k]) / k
        dcg = sum(1 / math.log2(i + 2) for i, gain in enumerate(gains[:k]) if gain)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(question.evidence))))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    first = next((i for i, rel in enumerate(relevant) if rel), None)
    out["mrr"] = 1 / (first + 1) if first is not None else 0.0
    return out


def context_metrics(question: EvalQuestion, context: Sequence[RetrievedText]) -> dict[str, float | None]:
    """Context precision (share of packed passages that are relevant) and context recall (gold covered)."""
    if not question.answerable or not question.evidence:
        return {}
    if not context:
        return {"context_precision": 0.0, "context_recall": 0.0}
    relevant = sum(any(matches(c, g) for g in question.evidence) for c in context)
    covered = sum(any(matches(c, g) for c in context) for g in question.evidence)
    return {"context_precision": relevant / len(context), "context_recall": covered / len(question.evidence)}


def citation_metrics(question: EvalQuestion, cited: Sequence[RetrievedText]) -> dict[str, float | None]:
    """Citation precision (cited passages that support gold) and recall (gold passages cited)."""
    if not question.answerable or not question.evidence or not cited:
        return {}
    precise = sum(any(matches(c, g) for g in question.evidence) for c in cited)
    covered = sum(any(matches(c, g) for c in cited) for g in question.evidence)
    return {"citation_precision": precise / len(cited), "citation_recall": covered / len(question.evidence)}


def answer_relevance(question: str, answer: str) -> float:
    """Offline proxy: share of the question's content terms addressed in the answer."""
    return coverage(set(content_terms(question)), set(content_terms(answer)))


def abstention_metrics(question: EvalQuestion, abstained: bool) -> dict[str, float | None]:
    """Abstention accuracy plus over-/under-abstention indicators."""
    return {
        "abstention_accuracy": 1.0 if abstained == (not question.answerable) else 0.0,
        "over_abstention": (1.0 if abstained else 0.0) if question.answerable else None,
        "under_abstention": (0.0 if abstained else 1.0) if not question.answerable else None,
    }
