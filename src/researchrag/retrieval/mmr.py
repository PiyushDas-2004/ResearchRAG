"""Maximal Marginal Relevance diversification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from researchrag.retrieval.types import Candidate
from researchrag.textutils import jaccard, term_set


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
    terms: dict[str, set[str]] = {}

    def terms_of(c: Candidate) -> set[str]:
        if c.chunk_id not in terms:
            terms[c.chunk_id] = term_set(c.text)
        return terms[c.chunk_id]

    embedded = [c for c in pool if c.embedding is not None]
    index = {c.chunk_id: i for i, c in enumerate(embedded)}
    matrix: np.ndarray | None = None
    if embedded:
        emb = np.asarray([c.embedding for c in embedded], dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        unit = emb / np.where(norms == 0, 1.0, norms)
        matrix = unit @ unit.T

    def sim(a: Candidate, b: Candidate) -> float:
        if matrix is not None and a.chunk_id in index and b.chunk_id in index:
            return float(matrix[index[a.chunk_id], index[b.chunk_id]])
        return jaccard(terms_of(a), terms_of(b))

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
