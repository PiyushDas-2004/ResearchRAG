"""Reciprocal Rank Fusion (Cormack et al., 2009) with weights and deterministic tie-breaking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from researchrag.retrieval.types import Candidate


def rrf(
    ranked_lists: Sequence[tuple[str, Sequence[Candidate], float]], *, k: int = 60, limit: int | None = None
) -> list[Candidate]:
    """Fuse ranked lists ``[(name, candidates, weight), ...]``.

    ``score(d) = sum_i weight_i / (k + rank_i(d))`` with 1-based ranks. Per-method scores and
    ranks are preserved on the fused candidate. Ties break by best (lowest) rank, then by
    chunk id, so the output is stable for identical inputs regardless of list order.
    """
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    merged: dict[str, Candidate] = {}
    for name, candidates, weight in ranked_lists:
        seen: set[str] = set()
        for rank, cand in enumerate(candidates, start=1):
            if cand.chunk_id in seen:
                continue
            seen.add(cand.chunk_id)
            fused[cand.chunk_id] = fused.get(cand.chunk_id, 0.0) + weight / (k + rank)
            best_rank[cand.chunk_id] = min(best_rank.get(cand.chunk_id, rank), rank)
            previous = merged.get(cand.chunk_id)
            scores = {**(previous.scores if previous else {}), name: cand.score}
            ranks = {**(previous.ranks if previous else {}), name: rank}
            embedding = cand.embedding or (previous.embedding if previous else None)
            base = previous or cand
            merged[cand.chunk_id] = replace(base, scores=scores, ranks=ranks, embedding=embedding)
    order = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    if limit is not None:
        order = order[:limit]
    return [merged[cid].with_score("rrf", fused[cid]) for cid in order]
