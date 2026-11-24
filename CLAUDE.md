# CLAUDE.md — ResearchRAG

Citation-grounded RAG over academic papers. Python 3.11+, `src/` layout, package `researchrag`.

## Dev commands

```bash
uv venv -p 3.11 .venv && . .venv/bin/activate && uv pip install -e ".[dev]"   # or: make install
make lint        # ruff check + format --check
make type        # mypy --strict src/researchrag
make test-unit   # unit tests, no database
make test        # all tests + coverage gate (85%); db tests need RR_TEST_DATABASE_URL or Docker
researchrag corpus build      # render synthetic PDFs into data/corpus/pdfs (deterministic)
researchrag db migrate        # alembic upgrade head
researchrag eval check --baseline main   # offline regression gate
python scripts/validate_corpus.py data   # validate corpus YAML + gold evidence
```

Local Postgres for tests: `docker compose up -d db` (pgvector/pgvector:pg16, pgvector 0.8: exercises
iterative HNSW scans; distro pgvector 0.6 exercises the exact-scan fallback), then
`export RR_TEST_DATABASE_URL=postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag_test`.

## Architecture (one paragraph)

Ingestion: upload -> magic-byte validation -> PDF parse in a resource-limited subprocess
(pymupdf, two-column ordering, header/footer removal, tables, sections, metadata) -> chunker
(fixed / sentence / section / parent_child) -> embed (batched, cached, rate-limited) -> one
transaction writes document, pages, chunk set, chunks, embeddings. Query: condense follow-up ->
optional rewrite / multi-query -> vector (pgvector HNSW, per-embedding-version partial index) +
lexical (tsvector GIN, ts_rank_cd) -> RRF -> rerank -> MMR -> injection scan -> small-to-big
parent swap -> token-budget packing -> evidence gate (abstain) -> LLM structured answer ->
citation + quote checks -> claim verification -> policy (remove/flag/regenerate) -> output guard
-> persisted trace + OTel spans + Prometheus metrics.

## Conventions

* Core pipeline code never imports provider SDKs; adapters live in `llm/` and `embeddings/`
  (ruff `TID251` bans `openai` / `google.genai` imports elsewhere).
* Every tunable lives in `PipelineConfig` (YAML in `configs/`); sweeps use dotted overrides.
* Offline by default: `fake` embedder + `mock` LLM + `lexical` judge. Live tests: `-m live`.
* Never commit secrets, `.env`, third-party PDFs or generated PDFs (`data/corpus/pdfs/`).
* Conventional Commits. mypy --strict clean, ruff clean, coverage >= 85%.
* Numbers in docs only from real runs, labelled with the components used.
* Model names live in config, prices in `configs/pricing.yaml` (with `as_of`).

## Build checklist

- [x] Phase 1 scaffold: tooling, config, CLAUDE.md
- [x] Phase 2 synthetic corpus + PDF ingestion + chunkers + metadata (+ property tests)
- [x] Phase 3 Postgres schema/migrations, embeddings, hybrid retrieval, MMR, rerank, rewrite
- [x] Phase 4 generation, structured citations, abstention, streaming, condensation
- [x] Phase 5 verification layer, security hardening, poisoned-PDF test
- [x] Phase 6 evaluation harness, sweeps + CIs, regression gate, judge calibration
- [x] Phase 7 FastAPI service, auth, rate limit, problem+json, SSE, feedback, UI
- [x] Phase 8 observability: logs, OTel traces, Prometheus, Jaeger profile (real trace in docs)
- [x] Phase 9 Docker, compose, Terraform (zero-cost profile), CI workflows, docs, ADRs, README
- [x] Phase 10 hardening pass (see final report in the session)
- [x] Phase 11 publish: pushed to https://github.com/PiyushDas-2004/ResearchRAG (main); CI and v0.1.0 release still to confirm
- [ ] Phase 12 cloud deploy (owner requires zero cost; nothing is created without explicit go-ahead)

## Cost constraint

The owner requires **zero cost**. Defaults are offline (no paid APIs). Terraform defaults to an
external free-tier Postgres (`database_mode = "external"`) and request-based Cloud Run billing with
inline ingestion; Cloud SQL and always-on CPU are opt-in and documented as paid. Never run live
evaluations or apply infrastructure without the owner's explicit approval.

## Environment notes

Canonical repository: https://github.com/PiyushDas-2004/ResearchRAG (branch `main`). The project
was first developed in the `ResearchRAG/` subfolder of the `unvibecode` repo and exported with
`git subtree split`, which preserved its commit history.
