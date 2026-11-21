# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-21

### Added
- Layout-aware PDF ingestion (two-column ordering, header/footer and page-number removal,
  hyphenation repair, tables with captions, section detection, heuristic metadata, optional OCR)
  running in a memory- and time-limited subprocess; magic-byte, size and page-limit validation.
- Fixed, sentence, section-aware and parent-child chunkers with exact character offsets and page
  maps; property-based tests.
- PostgreSQL 16 + pgvector schema with Alembic migrations, chunk sets per chunking config,
  versioned embeddings with per-version partial HNSW indexes, GIN full-text index, transactional
  idempotent ingestion and a resumable `SKIP LOCKED` job queue.
- Fake, OpenAI, Gemini and local (sentence-transformers) embedders behind a batching, caching,
  rate-limited, retrying service.
- Hybrid retrieval (vector + full text, Reciprocal Rank Fusion), metadata filters in SQL, MMR,
  lexical / cross-encoder / LLM rerankers, query rewriting and conversational condensation.
- Structured, citation-grounded generation (OpenAI, Gemini, OpenAI-compatible, deterministic
  mock) with JSON repair, bounded retries, a safe fallback, an evidence gate and SSE streaming.
- Claim-level verification (lexical, LLM and NLI judges), quote verification, remove/flag policy,
  feedback-driven regeneration and faithfulness scoring.
- Prompt-injection defences (input scanning with sanitize/flag/drop, instruction hierarchy,
  canary-based output guard) with a poisoned-PDF test.
- Evaluation harness: retrieval, context, citation, faithfulness, correctness and abstention
  metrics with bootstrap CIs; ablation sweeps with paired comparisons; regression gate; judge
  calibration; Markdown/HTML reports; synthetic 16-paper corpus with 88 questions.
- FastAPI service (API keys, rate limiting, RFC 7807 errors, request ids, idempotent uploads,
  SSE, traces, feedback), demo UI, structlog JSON logs with PII redaction, OpenTelemetry spans,
  Prometheus metrics.
- Docker image, Compose stack with a Jaeger profile, Terraform for Cloud Run + Cloud SQL,
  GitHub Actions for CI, live evaluation, deployment (Workload Identity Federation) and releases.
