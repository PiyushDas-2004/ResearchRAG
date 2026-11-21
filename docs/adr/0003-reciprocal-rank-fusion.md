# ADR 0003: Reciprocal Rank Fusion for hybrid retrieval

- Status: accepted
- Date: 2026-11-21

## Context
Dense retrieval misses exact identifiers, numbers and rare terms; lexical retrieval misses
paraphrases. Their scores (cosine similarity vs `ts_rank_cd`) are on incomparable scales.

## Decision
Fuse ranked lists with weighted Reciprocal Rank Fusion: `score(d) = sum_i w_i / (k + rank_i(d))`,
`k = 60` (Cormack et al., 2009). Weights and `k` are config fields. Multi-query expansion simply
adds more lists. Ties break by best rank, then chunk id, so fusion is deterministic.

## Consequences
- No score calibration or learned weights needed; robust across embedders.
- Absolute relevance is lost after fusion, so abstention uses a separate evidence score and the
  reranker score rather than the fused score.
- Vector-only and lexical-only modes remain available for ablations.
