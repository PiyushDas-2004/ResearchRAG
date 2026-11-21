# ResearchRAG

**Citation-grounded question answering over academic papers, with every claim checked against the passage it cites.**

![CI](https://github.com/PiyushDas-2004/ResearchRAG/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A585%25-brightgreen)
![Cost](https://img.shields.io/badge/runs%20at-%240-success)

LLMs answering questions about papers hallucinate, cite things that were never said, and rarely
admit when the evidence is missing. ResearchRAG makes every answer:

- **traceable**: each claim points to a paper, a page and a verbatim passage;
- **verified**: claims are checked against their cited passages, and unsupported claims and fabricated quotes are removed or flagged;
- **abstaining**: it says "not enough evidence in the indexed papers" (and what it searched) instead of guessing;
- **measurable**: retrieval, faithfulness, correctness and abstention are evaluated with confidence intervals, and a regression gate runs in CI.

It runs fully offline with **zero API keys and zero cost** (deterministic fake embedder + mock LLM),
and switches to OpenAI, Gemini, an OpenAI-compatible local server (Ollama) or local
sentence-transformers through configuration.

![Answer with verified citations](docs/images/ui-answer.png)

## Features

- **Ingestion**: magic-byte, size, page and text limits; parsing in a memory- and time-limited subprocess (decompression-bomb defence); two-column layout ordering; header/footer/page-number removal; hyphenation and ligature repair; tables with captions; section detection; heuristic title/authors/year/DOI/arXiv metadata (editable); optional OCR; idempotent, resumable, transactional jobs.
- **Chunking**: fixed-token, sentence, section-aware and parent-child (small-to-big), with exact character offsets and page maps; property-based tests.
- **Storage**: PostgreSQL 16 + pgvector; HNSW per embedding version (partial expression indexes), GIN full text; versioned embeddings and re-embedding jobs; metadata filters inside SQL; Alembic migrations with an advisory-lock guard.
- **Retrieval**: hybrid vector + full-text with Reciprocal Rank Fusion, query rewriting/multi-query, conversational follow-up condensation, MMR diversification, lexical / cross-encoder / LLM reranking, token-budget context packing.
- **Generation**: structured JSON answers (claims + quotes + markers), JSON repair, bounded retries, safe fallback, evidence-gated abstention, SSE streaming.
- **Verification**: quote matching (exact or fuzzy), dangling and uncited claim detection, claim verdicts from a lexical, LLM or NLI judge, remove/flag policy, one feedback-driven regeneration, faithfulness score.
- **Security**: prompt-injection scanning (sanitize/flag/drop), instruction hierarchy, canary output guard, poisoned-PDF tests, API keys, per-key rate limits, RFC 7807 errors, PII redaction in logs.
- **Evaluation**: 16 fictional papers, 88 questions (single-hop, multi-hop, numeric/table, unanswerable, adversarial), retrieval/context/citation/faithfulness/correctness/abstention metrics, bootstrap CIs, ablation sweeps with paired comparisons, a CI regression gate, judge calibration, Markdown/HTML reports.
- **Operations**: FastAPI, demo UI, structlog JSON logs, OpenTelemetry spans per stage (Jaeger profile), Prometheus metrics, Docker, Compose, Terraform for Cloud Run (zero-cost profile), GitHub Actions.

## Architecture

```mermaid
flowchart LR
  subgraph Ingestion
    U[PDF upload] --> V[validate] --> P[isolated parse<br/>layout, tables, sections] --> C[chunk<br/>parent-child] --> E[embed] --> DB[(Postgres + pgvector<br/>HNSW + GIN)]
  end
  subgraph Query
    Q[question] --> CO[condense / rewrite] --> R1[vector search] & R2[full-text search]
    R1 & R2 --> F[RRF] --> RR[rerank + MMR] --> X[parents + injection scan + pack]
    X --> G{evidence gate}
    G -- weak --> AB[abstain]
    G -- ok --> GEN[LLM: claims + quotes] --> CK[citation + quote checks] --> VER[claim verification<br/>remove / flag / regenerate]
    VER --> OG[output guard] --> A[answer + page citations + verdicts]
  end
  DB --- R1
  DB --- R2
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart (5 minutes, offline, no keys, $0)

Requires Docker with Compose.

```bash
git clone https://github.com/PiyushDas-2004/ResearchRAG.git && cd ResearchRAG
docker compose up -d --build
docker compose exec api sh -c "researchrag corpus build && researchrag ingest data/corpus/pdfs"
open http://localhost:8000        # UI; API key: dev-local-key
```

Ask from the command line:

```bash
curl -s http://localhost:8000/v1/query -H 'X-API-Key: dev-local-key' -H 'Content-Type: application/json' \
  -d '{"question": "How many questions and passages does the QuarryQA benchmark contain?"}'
```

Real response (offline components; passage text truncated):

```json
{
  "answer": "We also release QuarryQA, a benchmark of 41,200 questions paired with 1.9 million passages drawn from 63,000 open-access scientific articles. [1]",
  "abstained": false,
  "faithfulness": 1.0,
  "claims": [{"text": "We also release QuarryQA, a benchmark of 41,200 questions paired with 1.9 million passages drawn from 63,000 open-access scientific articles.", "citations": [1], "verdict": "supported", "removed": false}],
  "citations": [{"marker": 1, "title": "BraidRank: Query-Adaptive Fusion of Sparse Term Weights and Dense Embeddings for Scientific Passage Retrieval", "page": 1, "section": "abstract", "quote_verified": true, "quote_score": 100.0}],
  "searched": {"candidates": 61, "passages": 6, "evidence_score": 0.6667},
  "usage": {"input_tokens": 2208, "output_tokens": 147, "cost_usd": 0.0},
  "latency_ms": 63.493
}
```

An unanswerable question gets an abstention with what was searched:

![Abstention](docs/images/ui-abstain.png)

Tracing: `RR_OTEL_EXPORTER_ENDPOINT=http://jaeger:4318/v1/traces docker compose --profile tracing up -d`, then open http://localhost:16686 ([docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)).

### Without Docker

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
researchrag eval run            # full offline evaluation in-memory (no database needed), ~15 s
researchrag eval check          # the CI regression gate
# with a local Postgres + pgvector:
export RR_DATABASE_URL=postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag
researchrag db migrate && researchrag corpus build && researchrag ingest data/corpus/pdfs
researchrag query "What afternoon cooling did the CoolCrown roofs deliver in Harbourside?"
researchrag serve
```

## Configuration

Deployment settings are environment variables with the `RR_` prefix ([.env.example](.env.example)):
`RR_DATABASE_URL`, `RR_API_KEYS`, `RR_PIPELINE_CONFIG`, `RR_RATE_LIMIT_PER_MINUTE`, `RR_CORS_ORIGINS`,
`RR_MAX_UPLOAD_BYTES`, `RR_MAX_PAGES`, `RR_PARSE_ISOLATION`, `RR_PARSE_MEMORY_LIMIT_MB`,
`RR_INGEST_MODE` (`background`/`inline`), `RR_RUN_MIGRATIONS_ON_STARTUP`, `RR_OTEL_EXPORTER_ENDPOINT`,
`RR_LOG_PROMPTS`, `RR_REDACT_PII`, `RR_STORE_PROMPTS_IN_TRACES`. Provider SDKs read
`OPENAI_API_KEY` / `GEMINI_API_KEY` from the environment (never from files).

Pipeline behaviour lives in YAML ([configs/default.yaml](configs/default.yaml)); every field can be
swept:

| Section | Key fields |
|---|---|
| `chunking` | `strategy` (fixed/sentence/section/parent_child), `chunk_tokens`, `overlap_tokens`, `parent_tokens`, `tokenizer` |
| `embedding` | `provider` (fake/openai/gemini/local), `model`, `dimension`, `batch_size`, `requests_per_minute`, `contextual_header` |
| `retrieval` | `mode` (hybrid/vector/lexical), `candidate_k`, `top_k`, `rrf_k`, weights, `mmr_enabled`, `mmr_lambda`, `ef_search`, `include_references`, `embedding_version` |
| `rewrite` | `enabled`, `multi_query`, `condense_followups`, `max_history_turns` |
| `rerank` | `provider` (none/lexical/cross_encoder/llm), `model`, `top_n` |
| `generation` | `provider` (mock/openai/gemini/openai_compatible), `model`, `base_url`, `max_context_tokens`, `max_retries`, `min_evidence_score`, `mock.*` failure injection |
| `verification` | `enabled`, `judge` (lexical/llm/nli), `policy` (remove/flag), `regenerate_on_failure`, `quote_match_threshold` |
| `security` | `injection_action` (sanitize/flag/drop), `output_guard` |

Ready-made configs: `default.yaml` and `eval-offline.yaml` (offline), `openai.yaml`, `gemini.yaml`
(paid APIs), `ollama.yaml` (local, free). Model names and prices live in config and
[configs/pricing.yaml](configs/pricing.yaml) (with an `as_of` date); verify them before relying on them.

## Evaluation

All numbers below come from runs in this repository on 2026-11-21 and are labelled with the
components used. Full reports: [docs/example-report/](docs/example-report/). Metric definitions and
caveats: [docs/EVALUATION.md](docs/EVALUATION.md).

**Offline mock components, illustrative** (fake hashing embedder, mock extractive LLM with 20%
injected numeric hallucinations, lexical judge; 88 questions; in-memory exact search; 95% bootstrap CIs):

| Metric | Value | 95% CI |
|---|---|---|
| Retrieval recall@5 (n=71 answerable) | 0.803 | [0.711, 0.880] |
| MRR | 0.618 | [0.535, 0.708] |
| Context recall | 0.922 | [0.866, 0.965] |
| Citation precision | 0.527 | [0.403, 0.643] |
| Faithfulness of returned answers | 0.918 | [0.867, 0.964] |
| Hallucination rate before verification (any failing claim) | 0.286 | [0.175, 0.397] |
| Answer correctness (lexical rubric judge) | 0.476 | [0.393, 0.560] |
| Abstention accuracy (n=88) | 0.705 | [0.602, 0.796] |
| Over-abstention (answerable but abstained) | 0.296 | [0.197, 0.408] |
| Latency per query, in-process (p50 / p95) | 86 ms / 122 ms | |
| Cost | $0 | |

Ablations (same components, paired bootstrap vs base): 96-token chunks cut recall@5 by 0.141,
256-token chunks cut context recall and citation precision, `top_k = 4` cuts context recall by
0.077, and disabling MMR cuts recall@5 by 0.042 (all significant); vector vs hybrid vs lexical and
query rewriting made no significant difference with the lexical-ish offline embedder; HNSW
`ef_search` had no effect at this corpus size (results equal exact search). Full tables:
[docs/EVALUATION.md](docs/EVALUATION.md#ablation-sweep-offline-components-illustrative).

What these numbers do and do not say: they measure the *pipeline* (retrieval, fusion, packing,
abstention gate, verification) on a synthetic corpus written for this project. The mock LLM copies
sentences, so faithfulness is high by construction, and its abstention heuristic was designed while
looking at this dataset. They are not a measure of language-model quality and overstate real-world
performance. No live-provider evaluation has been run (no API keys were used; the live-eval workflow
exists for when keys are available). The offline lexical judge agrees with the hand-labelled
calibration set only weakly (exact agreement 0.45, Cohen's kappa 0.24 on 40 items), which is why
the LLM and NLI judges exist.

## Load test

**Offline components, illustrative.** Locust, 10 users for 60 s against the Compose stack (one
uvicorn worker, Postgres + pgvector, 16 papers, load generator on the same 4-vCPU host):
`POST /v1/query` 437 requests, **0 failures, p50 100 ms, p95 230 ms**, ~7.3 queries/s.
Details: [docs/EVALUATION.md](docs/EVALUATION.md#latency-and-load).

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [API](docs/API.md) · [Evaluation](docs/EVALUATION.md) ·
[Observability](docs/OBSERVABILITY.md) · [Security](docs/SECURITY.md) ·
[Deployment (zero-cost Cloud Run)](docs/DEPLOYMENT.md) · [ADRs](docs/adr/) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Limitations

- The bundled corpus is synthetic and small (16 papers); results on real literature will be lower,
  and the PDF heuristics were tuned on reportlab-generated layouts (two-column, simple tables).
- The offline judge is lexical: it misses paraphrased support and can accept an added causal
  clause with high word overlap. Use the LLM or NLI judge for real verification.
- Prompt-injection scanning is pattern-based; novel or obfuscated injections can get through the
  input scan (the output guard and prompt structure are the remaining layers).
- Rate limiting is per process; multi-instance deployments need a shared limiter.
- Metadata extraction is heuristic (title by font size, authors by name-like lines); it is editable
  through the API.
- No authentication beyond API keys (no users, tenants or document-level ACLs).

## Roadmap

- Live evaluation with OpenAI/Gemini and a local NLI judge; publish the reports next to the
  offline ones.
- A real-paper benchmark built from openly licensed arXiv papers (`scripts/download_open_access.py`).
- Learned (or calibrated) fusion weights and evidence thresholds per embedder.
- Shared rate limiting and a Redis-backed cache for multi-instance deployments.
- Per-tenant collections and document ACLs enforced inside the SQL filters.
- Table-aware answers (cell-level citations) and figure captions.

## License

MIT, Copyright (c) 2026 Piyush Pradeep Das. The synthetic corpus is original, fictional content
created for this project.
