# Example evaluation reports

Produced on 2026-11-21 in this repository with the **offline mock components** (fake hashing
embedder, mock extractive LLM, lexical judge). Illustrative only; see
[../EVALUATION.md](../EVALUATION.md) for what these numbers mean and do not mean.

| File | Command |
|---|---|
| `eval-eval-offline.{md,html,json}`, `run.log` | `researchrag eval run --out docs/example-report` |
| `gate.log` | `researchrag eval check --baseline main` (unchanged config: passes, all diffs 0) |
| `gate-seeded-regression.log`, `seeded-regression-config.yaml` | `researchrag eval check --config docs/example-report/seeded-regression-config.yaml` (fails, exit 1) |
| `sweep.{md,html}`, `sweep.log` | `researchrag eval sweep configs/sweep.yaml --out docs/example-report` |
| `sweep-hnsw.{md,html}`, `sweep-hnsw.log` | `researchrag eval sweep configs/sweep-hnsw.yaml --backend postgres` (Compose database, pgvector 0.8.7) |
| `calibration.md`, `calibration-lexical.json` | `researchrag eval calibrate` |

The sweeps' full per-question JSON (several MB) is not committed; rerun the commands to regenerate it.
No live-provider (paid API) evaluation has been run.
