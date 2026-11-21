# ADR 0005: Claim-level verification after generation

- Status: accepted
- Date: 2026-11-21

## Context
Grounding instructions reduce but do not eliminate unsupported statements and fabricated quotes.
Users need to see which statements are supported, and the system should not return uncited facts.

## Decision
1. The model returns structured JSON: answer text with `[n]` markers, atomic claims with passage
   numbers, and verbatim supporting quotes.
2. Citation checks: every marker must resolve to a packed passage (dangling markers are dropped);
   every quote must occur in its passage (exact after normalisation, else fuzzy >= threshold).
   A quote that matches nowhere is a fabricated quote: flagged in the trace and on the claim.
3. Each claim is judged against the text of its cited passages by a pluggable judge: `lexical`
   (offline, deterministic), `llm` (NLI-style prompt) or `nli` (local cross-encoder). Verdicts:
   supported / partially_supported / unsupported / contradicted.
4. Policy: `remove` (default) drops unsupported/contradicted claims and rebuilds the answer from
   survivors; `flag` keeps them marked. With `regenerate_on_failure`, one retry is made with the
   failed claims as feedback; the better result by faithfulness is kept.
5. If no supported claim survives, the system abstains.

## Consequences
- Faithfulness = (supported + 0.5 x partial) / claims, reported per answer and in evaluation.
- The judge can be wrong. Its agreement with a hand-labelled set is measured
  (`researchrag eval calibrate`) and reported honestly; the offline lexical judge is weak on
  paraphrase (see docs/EVALUATION.md).
- Verification adds latency and, with an LLM judge, cost (one judge call per claim).
