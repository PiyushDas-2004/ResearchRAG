# ADR 0008: Offline-first, deterministic test components

- Status: accepted
- Date: 2026-11-21

## Context
The project must run, test and evaluate with zero API keys, and CI must be deterministic.

## Decision
- `fake/fake-hash-v1`: a feature-hashing embedder (stemmed unigrams, bigrams, character 4-grams,
  signed hashing, L2-normalised).
- `mock/mock-extractive-v1`: an LLM stand-in that parses the real prompts, answers extractively
  with verbatim quotes, abstains on weak or unmatched evidence and can inject failures (timeouts,
  429s, malformed JSON, fabricated quotes, uncited claims, numeric hallucinations, obeying
  injections) for fault-injection tests.
- Real providers (OpenAI, Gemini, OpenAI-compatible, sentence-transformers) sit behind the same
  interfaces and are exercised only with keys (`pytest -m live`, manual live-eval workflow).

## Consequences
- Every number produced with these components is labelled "offline mock components,
  illustrative". They measure the pipeline mechanics, not language-model quality.
- The mock's abstention heuristic was designed while looking at this dataset, so offline
  abstention numbers are optimistic.
