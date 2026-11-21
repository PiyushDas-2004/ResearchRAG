"""Regression gate: compare a run against a stored baseline, paired by question.

A metric *regresses* when its mean drops by more than ``tolerance`` (absolute) AND the paired
bootstrap 95% CI of the change lies entirely on the worse side of zero. Requiring both keeps
the gate quiet on noise (small or non-significant changes) while catching real drops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from researchrag.evaluation.runner import LOWER_IS_BETTER, RunResult
from researchrag.evaluation.stats import paired_bootstrap

GATE_METRICS = (
    "recall@5",
    "mrr",
    "context_recall",
    "faithfulness",
    "answer_correctness",
    "abstention_accuracy",
)


class Baseline(BaseModel):
    """Per-question metric values of a reference run."""

    name: str
    config_hash: str
    dataset_hash: str
    components: dict[str, str]
    created_at: str
    values: dict[str, dict[str, float | None]]  # metric -> question id -> value


class GateCheck(BaseModel):
    """Outcome for one metric."""

    metric: str
    baseline_mean: float
    current_mean: float
    diff: float
    low: float
    high: float
    n: int
    regressed: bool


class GateReport(BaseModel):
    """Outcome of the regression gate."""

    passed: bool
    tolerance: float
    checks: list[GateCheck]
    warnings: list[str]


def baseline_from_run(run: RunResult, metrics: tuple[str, ...] = GATE_METRICS) -> Baseline:
    """Extract a baseline from a run."""
    return Baseline(
        name=run.name,
        config_hash=run.config_hash,
        dataset_hash=run.dataset_hash,
        components=run.components,
        created_at=run.created_at,
        values={m: {q.id: q.metrics.get(m) for q in run.questions} for m in metrics},
    )


def save_baseline(baseline: Baseline, path: str | Path) -> None:
    """Write a baseline JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> Baseline:
    """Read a baseline JSON file."""
    return Baseline.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def compare(
    baseline: dict[str, dict[str, float | None]],
    current: dict[str, dict[str, float | None]],
    *,
    tolerance: float = 0.03,
    metrics: tuple[str, ...] = GATE_METRICS,
) -> GateReport:
    """Paired comparison of per-question values; see module docstring for the rule."""
    checks: list[GateCheck] = []
    warnings: list[str] = []
    for metric in metrics:
        base_vals = baseline.get(metric, {})
        cur_vals = current.get(metric, {})
        ids = sorted(set(base_vals) & set(cur_vals))
        if not ids:
            warnings.append(f"{metric}: no overlapping questions")
            continue
        a = [base_vals[i] for i in ids]
        b = [cur_vals[i] for i in ids]
        cmp = paired_bootstrap(a, b)
        if cmp.n == 0:
            continue
        sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
        worse_by = -sign * cmp.diff
        ci_worse = (cmp.high < 0) if sign > 0 else (cmp.low > 0)
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
        checks.append(
            GateCheck(
                metric=metric,
                baseline_mean=round(sum(x for x, _ in pairs) / len(pairs), 4),
                current_mean=round(sum(y for _, y in pairs) / len(pairs), 4),
                diff=round(cmp.diff, 4),
                low=round(cmp.low, 4),
                high=round(cmp.high, 4),
                n=cmp.n,
                regressed=worse_by > tolerance and ci_worse,
            )
        )
    return GateReport(
        passed=not any(c.regressed for c in checks), tolerance=tolerance, checks=checks, warnings=warnings
    )


def check_run(run: RunResult, baseline: Baseline, *, tolerance: float = 0.03) -> GateReport:
    """Gate a run against a baseline (warns when the dataset changed)."""
    current = {m: {q.id: q.metrics.get(m) for q in run.questions} for m in baseline.values}
    report = compare(baseline.values, current, tolerance=tolerance, metrics=tuple(baseline.values))
    if run.dataset_hash != baseline.dataset_hash:
        report.warnings.append("dataset changed since the baseline; only overlapping questions were compared")
    return report


def report_dict(report: GateReport) -> dict[str, Any]:
    """JSON-friendly gate report."""
    return report.model_dump()
