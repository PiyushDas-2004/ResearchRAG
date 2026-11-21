# ADR 0002: HNSW index parameters and filtered search

- Status: accepted
- Date: 2026-11-21

## Context
pgvector offers IVFFlat and HNSW. HNSW gives better recall/latency trade-offs without a training
step and supports incremental inserts, which suits continuous ingestion.

## Decision
- HNSW with `vector_cosine_ops`, `m = 16`, `ef_construction = 64` (pgvector defaults; good recall
  for corpora up to a few million chunks with modest build time and memory).
- `hnsw.ef_search` is a pipeline-config field (`retrieval.ef_search`, default 64, never below `k`),
  set per query with `SET LOCAL`, so evaluation can sweep it (`configs/sweep-hnsw.yaml`).
- One partial expression index per embedding version (ADR 0006).
- Filtered search: on pgvector >= 0.8 the query enables `hnsw.iterative_scan = relaxed_order` and
  re-sorts by distance; on older versions a short filtered result triggers an exact scan fallback.

## Consequences
- Higher `ef_search` raises recall at the cost of latency; the sweep reports both.
- `m`/`ef_construction` are fixed per index; changing them means building a new index
  (recorded on the `embedding_versions` row).
- On tiny tables (hundreds of vectors) an exact scan + sort can be cheaper and the planner may
  choose it, depending on statistics. Integration tests therefore assert with EXPLAIN that the
  vector query *matches* the per-version partial HNSW index (it is chosen once sequential scans
  are discouraged), on both pgvector 0.6 and 0.8; at realistic sizes the planner picks it on its own.
