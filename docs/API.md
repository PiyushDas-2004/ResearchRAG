# API reference

Base URL: `http://localhost:8000` (Compose) or your Cloud Run URL. Interactive OpenAPI docs are
served at `/docs` (Swagger UI) and `/openapi.json`.

## Authentication

Send `X-API-Key: <key>` (or `Authorization: Bearer <key>`) on every `/v1/*` request. Keys are
configured with `RR_API_KEYS` (comma-separated). `/health`, `/ready` and `/metrics` are open.
Each key is rate-limited (`RR_RATE_LIMIT_PER_MINUTE`, default 60); excess requests get
`429` with a `Retry-After` header.

Every response carries `X-Request-ID` (an incoming valid `X-Request-ID` is propagated); the same id
appears in logs and in error bodies.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/documents` | Upload a PDF (multipart `file`, optional `collection`; optional `Idempotency-Key` header). Returns `202` with a job (`200` for an idempotent replay or in `inline` ingest mode). |
| `GET` | `/v1/jobs/{id}` | Job status: `queued` / `running` / `succeeded` / `failed`, stage, progress, error. |
| `GET` | `/v1/documents?limit=&offset=&collection=` | Paginated list (newest first) with `total` and `next_offset`. |
| `GET` | `/v1/documents/{id}` | One document with metadata and chunk count. |
| `PATCH` | `/v1/documents/{id}` | Correct metadata (`title`, `authors`, `year`, `abstract`, `doi`, `arxiv_id`, `venue`, `collection`). |
| `DELETE` | `/v1/documents/{id}` | Delete a document with its pages, chunks and embeddings (`204`). |
| `POST` | `/v1/query` | Answer a question; returns the verified result below. |
| `POST` | `/v1/query/stream` | Same, as Server-Sent Events. |
| `GET` | `/v1/traces/{id}` | Full trace: candidates, packed context, prompt, raw answer, verdicts, latency, cost. |
| `POST` | `/v1/feedback` | `{"trace_id", "rating": "up"|"down", "comment"}` (`201`). |
| `GET` | `/health` | Liveness. |
| `GET` | `/ready` | Readiness: database reachable, migrations at head, pgvector installed. |
| `GET` | `/metrics` | Prometheus exposition. |

### Query request

```json
{
  "question": "How many questions and passages does the QuarryQA benchmark contain?",
  "filters": {"year_min": 2020, "year_max": 2025, "authors": ["Varandov"], "document_ids": [], "collection": null, "include_references": false},
  "conversation_id": "demo-1",
  "history": []
}
```

`filters` are applied inside the SQL search. `conversation_id` keeps short-term memory (follow-ups
such as "what about its limitations?" are condensed into standalone questions); alternatively pass
`history` turns explicitly.

### Query response (real output)

Captured from the Compose stack with the offline components (fake embedder, mock extractive LLM,
lexical judge); passage text truncated here for brevity:

```json
{
  "trace_id": "c90b66cc-b580-4e7d-924e-4f04285bf3ed",
  "question": "How many questions and passages does the QuarryQA benchmark contain?",
  "standalone_question": "How many questions and passages does the QuarryQA benchmark contain?",
  "answer": "We also release QuarryQA, a benchmark of 41,200 questions paired with 1.9 million passages drawn from 63,000 open-access scientific articles. [1]",
  "abstained": false,
  "abstain_reason": null,
  "confidence": 0.717,
  "faithfulness": 1.0,
  "claims": [
    {
      "text": "We also release QuarryQA, a benchmark of 41,200 questions paired with 1.9 million passages drawn from 63,000 open-access scientific articles.",
      "citations": [
        1
      ],
      "verdict": "supported",
      "rationale": "100% of claim terms found in evidence",
      "flags": [],
      "removed": false
    }
  ],
  "citations": [
    {
      "marker": 1,
      "chunk_id": "98513e9e-ab39-5d90-999e-6e3b4dee985c",
      "document_id": "01c0483f-b500-4dda-8fd3-b3226e096402",
      "title": "BraidRank: Query-Adaptive Fusion of Sparse Term Weights and Dense Embeddings for Scientific Passage Retrieval",
      "page": 1,
      "section": "abstract",
      "quote": "We also release QuarryQA, a benchmark of 41,200 questions paired with 1.9 million passages drawn from 63,000 open-access scientific articles.",
      "quote_verified": true,
      "quote_score": 100.0,
      "passage": "Abstract\n\nScientific question answering systems depend on a first-stage retriever that must find relevant passages among\u2026"
    }
  ],
  "searched": {
    "queries": [
      "How many questions and passages does the QuarryQA benchmark contain?"
    ],
    "filters": {
      "document_ids": [],
      "collection": null,
      "year_min": null,
      "year_max": null,
      "authors": [],
      "include_references": false
    },
    "candidates": 61,
    "passages": 6,
    "documents": [
      "BraidRank: Query-Adaptive Fusion of Sparse Term Weights and Dense Embeddings for Scientific Passage Retrieval",
      "Chunk Size and Overlap as Hidden Hyperparameters of Long-Document Retrieval",
      "How Deep Should a Cross-Encoder Rerank? Latency and Accuracy Trade-offs over Hybrid Retrieval Candidates"
    ],
    "evidence_score": 0.6667
  },
  "warnings": [],
  "usage": {
    "input_tokens": 2208,
    "output_tokens": 147,
    "embedding_tokens": 0,
    "cost_usd": 0.0,
    "unpriced_models": []
  },
  "latency_ms": 63.493,
  "stage_latency_ms": {
    "retrieve.vector": 11.387,
    "retrieve.lexical": 3.651,
    "fuse": 4.297,
    "rerank": 0.314,
    "diversify": 18.518,
    "expand": 2.531,
    "guard.input": 2.198,
    "pack": 0.679,
    "gate": 0.047,
    "generate": 12.877,
    "cite": 0.197,
    "verify": 2.548,
    "guard.output": 0.053
  },
  "config_name": "default"
}
```

When the evidence is insufficient, `abstained` is `true`, `answer` is empty, `abstain_reason`
starts with `Not enough evidence in the indexed papers` and `searched` lists what was searched.

### Streaming (SSE)

`POST /v1/query/stream` with the same body. Events, in order:

1. `status` (`accepted`, then `retrieve`, `generate`, possibly `regenerate`),
2. zero or more `token` events (`{"text": "..."}`) with the answer text as it is generated,
3. exactly one `final` event with the full verified result (same schema as `/v1/query`), or an
   `error` event (`{"status", "code", "detail"}`).

Verification can remove claims after streaming, so clients should replace the streamed text with
`final.answer`. Idle connections receive `: keepalive` comments every 15 s.

Real stream excerpt (an unanswerable question):

```text
event: status
data: {"stage":"accepted","request_id":"b679bdcf9163449885f908b08258c22d"}

event: status
data: {"stage":"retrieve","standalone_question":"What is the boiling point of liquid nitrogen?"}

event: status
data: {"stage":"generate","passages":6}

event: final
data: {"trace_id":"...","abstained":true,"abstain_reason":"Not enough evidence in the indexed papers to answer this question. ...", ...}
```

## Errors (RFC 7807)

All errors are `application/problem+json`:

```json
{"type": "https://github.com/PiyushDas-2004/ResearchRAG/blob/main/docs/API.md#upload_rejected",
 "title": "Unsupported media type", "status": 415,
 "detail": "Only PDF files are accepted (checked by content, not extension).",
 "instance": "/v1/documents", "request_id": "3aeff82b170a475bb1f6f737f9b06550",
 "context": {"reason": "not_pdf"}}
```

| Problem `type` suffix | Status | Meaning |
|---|---|---|
| `unauthorized` | 401 | missing or invalid API key |
| `rate_limited` | 429 | per-key limit exceeded (`Retry-After`) |
| `validation_failed` | 422 | request body/params invalid (`errors` lists fields) |
| `upload_rejected` | 413 / 415 / 422 | too large / not a PDF / malformed, encrypted, too many pages, no text, parse timeout |
| `payload_too_large` | 413 | request body over `RR_MAX_REQUEST_BYTES` |
| `not_found` | 404 | unknown document, job, trace or route |
| `conflict` | 409 | idempotency key reused with a different file |
| `provider_error` | 502 | the model provider rejected the request |
| `provider_unavailable`, `provider_rate_limited` | 503 | transient provider failure |
| `provider_timeout` | 504 | provider timed out |
| `internal_error` | 500 | unexpected error (details only in logs) |

### Upload flow

```bash
curl -H "X-API-Key: dev-local-key" -H "Idempotency-Key: paper-42" \
     -F "file=@paper.pdf" http://localhost:8000/v1/documents      # -> 202 {"job": {"id": ...}}
curl -H "X-API-Key: dev-local-key" http://localhost:8000/v1/jobs/<job-id>   # poll until succeeded
```

Re-uploading identical bytes is a no-op (documents are keyed by content hash); repeating a request
with the same `Idempotency-Key` returns the original job.
