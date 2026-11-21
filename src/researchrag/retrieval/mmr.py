"""Maximal Marginal Relevance diversification."""

from __future__ import annotations

import math
from collections.abc import Sequence

from researchrag.retrieval.types import Candidate
from researchrag.textutils import jaccard, term_set


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr(candidates: Sequence[Candidate], *, k: int, lambda_: float) -> list[Candidate]:
    """Select ``k`` candidates balancing relevance and novelty.

    Relevance is the candidate's current score min-max normalised to [0, 1]. Similarity is
    the cosine of embeddings when both candidates have one, otherwise Jaccard similarity of
    their content terms (lexical-only mode). ``lambda_ = 1`` reduces to plain ranking.
    """
    pool = list(candidates)
    if not pool:
        return []
    scores = [c.score for c in pool]
    lo, hi = min(scores), max(scores)
    rel = {c.chunk_id: (c.score - lo) / (hi - lo) if hi > lo else 1.0 for c in pool}
    terms = {c.chunk_id: term_set(c.text) for c in pool}

    def sim(a: Candidate, b: Candidate) -> float:
        if a.embedding is not None and b.embedding is not None:
            return _cosine(a.embedding, b.embedding)
        return jaccard(terms[a.chunk_id], terms[b.chunk_id])

    selected: list[Candidate] = []
    while pool and len(selected) < k:
        best = max(
            pool,
            key=lambda c: (
                lambda_ * rel[c.chunk_id] - (1 - lambda_) * max((sim(c, s) for s in selected), default=0.0),
                rel[c.chunk_id],
                c.chunk_id,
            ),
        )
        selected.append(best)
        pool.remove(best)
    return selected
