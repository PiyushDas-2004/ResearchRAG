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

Local Postgres for tests: `docker compose up -d db` (pgvector/pgvector:pg16), then
`export RR_TEST_DATABASE_URL=postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag_test`.

## Architecture (one paragraph)

Ingestion: upload -> magic-byte validation -> PDF parse in a resource-limited subprocess
(pymupdf, two-column ordering, header/footer removal, tables, sections, metadata) -> chunker
(fixed / sentence / section / parent_child) -> embed (batched, cached, rate-limited) -> one
transaction writes document, pages, chunk set, chunks, embeddings. Query: condense follow-up ->
optional rewrite / multi-query -> vector (pgvector HNSW, per-embedding-version partial index) +
lexical (tsvector GIN, ts_rank_cd) -> RRF -> MMR -> rerank -> injection scan -> small-to-big
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
- [ ] Phase 3 Postgres schema/migrations, embeddings, hybrid retrieval, MMR, rerank, rewrite
- [ ] Phase 4 generation, structured citations, abstention, streaming, condensation
- [ ] Phase 5 verification layer, security hardening, poisoned-PDF test
- [ ] Phase 6 evaluation harness, sweeps + CIs, regression gate, judge calibration
- [ ] Phase 7 FastAPI service, auth, rate limit, problem+json, SSE, feedback, UI
- [ ] Phase 8 observability: logs, OTel traces, Prometheus, Jaeger profile
- [ ] Phase 9 Docker, compose, Terraform, CI workflows, docs, ADRs, README
- [ ] Phase 10 hardening pass
- [ ] Phase 11 publish (needs user: session can only push to the unvibecode repo)
- [ ] Phase 12 cloud deploy (needs explicit cost confirmation)

## Environment notes

This project currently lives in the `ResearchRAG/` subfolder of the `unvibecode` repo (branch
`ccr-a05333e2-ynyq69`). Extract with `git subtree split --prefix=ResearchRAG -b researchrag-main`.
