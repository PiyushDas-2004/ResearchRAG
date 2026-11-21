# ADR 0001: PostgreSQL + pgvector instead of a dedicated vector database

- Status: accepted
- Date: 2026-11-21

## Context
ResearchRAG needs dense vector search, lexical (BM25-style) search, relational metadata
(documents, pages, chunks, traces, jobs, feedback) and transactional ingestion. Dedicated
vector databases (Qdrant, Weaviate, Pinecone, Milvus) are strong at ANN search but would add a
second stateful system to operate, a second consistency domain, and a second security surface.

## Decision
Use one PostgreSQL 16 database with the pgvector extension (HNSW indexes) and Postgres full-text
search (tsvector + GIN). Metadata filters are expressed in the same SQL statement as the vector
or lexical search.

## Consequences
- One transaction writes a document, its chunks and embeddings: no half-ingested documents and no
  cross-system reconciliation.
- Hybrid retrieval, filters, traces and the job queue (`FOR UPDATE SKIP LOCKED`) share one system,
  one backup policy and one set of credentials. Managed options exist (Cloud SQL, AlloyDB, RDS,
  Neon, Supabase).
- At very large scale (hundreds of millions of vectors, heavy write rates) a dedicated engine may
  win on recall/latency per dollar. The retrieval layer is behind the `SearchBackend` protocol, so
  a different backend can be added without touching generation or verification.
- HNSW filtered search has known caveats (see ADR 0002).
