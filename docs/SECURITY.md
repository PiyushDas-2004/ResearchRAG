# Security

## Threat model

| Asset / boundary | Threat | Defence |
|---|---|---|
| Uploaded PDFs (untrusted bytes) | malformed or encrypted files, decompression bombs, parser exploits, huge page counts | magic-byte check (not extension), size limit enforced while streaming the upload, page and extracted-text limits, parsing in a child process with an address-space limit (`RLIMIT_AS`) and a wall-clock timeout; clear `problem+json` rejections |
| Document text (untrusted content) | indirect prompt injection ("ignore previous instructions...") | input scan + `sanitize` / `flag` / `drop`, delimiters with tag neutralisation, instruction hierarchy in the system prompt, per-request canary + payload output guard |
| Model output | fabricated quotes, uncited claims, unsupported claims | quote verification, citation resolution, claim verification, remove/flag policy, abstention |
| API | unauthenticated use, abuse, oversized requests | API keys (constant-time digest compare), per-key rate limit with `Retry-After`, request-body size limit, strict pydantic models (`extra="forbid"`), CORS allow-list, security headers (CSP, `nosniff`, `DENY` framing) |
| Secrets | leakage via code, logs, images, history | environment/Secret Manager only, `.env` git-ignored, keys stored and logged only as SHA-256 prefixes, PII and credential redaction in logs, prompts not logged by default |
| Supply chain | vulnerable dependencies or images | `pip-audit`, `bandit`, Trivy image scan in CI, Dependabot, non-root slim image |

## Prompt-injection defence in depth

1. **Input scan** (`security/injection.py`): each passage is split into sentences and checked
   against patterns for instruction overrides ("ignore/disregard ... instructions"), prompt
   exfiltration ("reveal ... system prompt"), role hijacking ("you are now", "Assistant: you
   must"), fake markup (`<system>`), and output coercion ("reply only with"). Matches are recorded
   in the trace and the passage is handled per `security.injection_action`:
   - `sanitize` (default): offending sentences are replaced with `[instruction-like text removed]`;
   - `flag`: the passage is kept but marked `warning="contains instruction-like text"`;
   - `drop`: the passage is removed from the context.
2. **Prompt structure** (`generation/prompts.py`): passages are wrapped in `<passage>` elements,
   tag-like sequences inside document text are defanged so they cannot close the element, and the
   system prompt states that passage text is data and must never be followed.
3. **Output guard**: every answer is checked for the per-request canary (system-prompt leakage),
   for distinctive payload tokens extracted from flagged passages (e.g. `BANANA-PROTOCOL-7`) and
   for phrases of an answer obeying a document. A violation withholds the answer (abstention)
   and adds a warning.

The poisoned-PDF tests (`tests/unit/test_pipeline.py`, `tests/api/test_api.py`,
`tests/e2e/test_server_e2e.py`) render a synthetic paper with a harmless canary instruction,
ingest it, and assert that the answer neither contains the payload nor leaks the prompt, that the
passage is flagged, and that a model which *does* obey the injection is caught by the output guard.

**Limits.** Pattern lists only catch known phrasings; paraphrased, multilingual or obfuscated
injections can pass the scan. The structural and output defences reduce, but do not remove, that
risk. Treat answers as untrusted when documents come from untrusted sources.

## Authentication and rate limiting

- `X-API-Key` (or `Authorization: Bearer`) is required whenever keys are configured, and always in
  `prod` (unless `RR_ALLOW_ANONYMOUS=true`). Keys come from `RR_API_KEYS` (Secret Manager on Cloud
  Run). Only `sha256(key)[:16]` is stored with jobs, traces and feedback, and logged.
- The rate limiter is a per-process sliding window (`RR_RATE_LIMIT_PER_MINUTE`). With several
  instances the effective limit multiplies; use a gateway or shared store for strict limits.

## Logging and privacy

- JSON logs via structlog; a processor redacts e-mail addresses, phone numbers, API-key-like
  strings and sensitive keys (`authorization`, `token`, `password`...) when `RR_REDACT_PII=true`
  (default).
- Prompts are not logged unless `RR_LOG_PROMPTS=true`. Prompts are stored in the database trace
  by default (`RR_STORE_PROMPTS_IN_TRACES`) for debuggability; disable this if documents are
  sensitive.

## Secrets hygiene

- No secrets in code, tests, fixtures, images or history. The repository was scanned before
  publishing (`git grep -nEi "api[_-]?key|secret|token|password"` over the tree and every commit,
  reviewing each hit).
- Test keys in the test suite are obvious dummies (`test-key-123`, `sk-test-not-real`).
- Terraform stores the generated database password and the API keys in its state; keep state in a
  protected remote backend.

## Scanning in CI

`bandit` (static analysis), `pip-audit` (dependency CVEs), Trivy (image CVEs, report-only so a new
upstream CVE does not block unrelated work) and Dependabot updates.
