# ADR 0006: Versioned embeddings with per-version partial HNSW indexes

- Status: accepted
- Date: 2026-11-21

## Context
Embedding models change (new models, different dimensions). Re-embedding must not corrupt
existing data, and it must be possible to query a specific embedding version while a new one is
being built.

## Decision
- `embedding_versions(id, model_id, dimension, ...)` is unique on `(model_id, dimension)`.
- `chunk_embeddings(chunk_id, version_id, embedding vector)` uses a dimension-less `vector`
  column; each version gets a partial expression index
  `USING hnsw ((embedding::vector(D)) vector_cosine_ops) WHERE version_id = N`, created when the
  version is registered (pgvector's documented pattern for mixed dimensions).
- Queries inline the version id and dimension as validated integers so the planner can match the
  partial index even with generic prepared-statement plans.
- Re-embedding is a background job (`kind = 'reembed'`) that embeds only chunks missing a vector
  for the active version.

## Consequences
- Old and new versions coexist; switching is a config change; rollback is free.
- Storage grows with each version until old versions are deleted.
- pgvector's HNSW limit for `vector` is 2,000 dimensions; larger models must be configured with a
  reduced output dimension (OpenAI `dimensions`, Gemini `output_dimensionality`).
