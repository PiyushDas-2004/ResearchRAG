# ADR 0007: Abstention policy

- Status: accepted
- Date: 2026-11-21

## Context
An answer built on weak evidence is worse than "not enough evidence" for a research tool. But
over-abstaining makes the tool useless, so abstention must be measured both ways.

## Decision
The pipeline abstains, returning "Not enough evidence in the indexed papers" plus what was
searched (queries, filters, passages, papers), when any of these hold:

1. Evidence gate (before the LLM call): no passage survived retrieval/packing, or the best
   passage covers fewer than `generation.min_evidence_score` (default 0.34) of the question's
   content terms. No tokens are spent.
2. The model abstains (it is instructed to when the passages are insufficient).
3. Generation fails after bounded retries and JSON repair (safe fallback).
4. Verification leaves no supported or partially supported claim.
5. The output guard detects prompt leakage or obedience to instructions found in documents.

## Consequences
- Evaluation reports abstention accuracy, over-abstention (answerable but abstained) and
  under-abstention (unanswerable but answered) with confidence intervals.
- The gate threshold is lexical and conservative; it is a config field and can be swept.
