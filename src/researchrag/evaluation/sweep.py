"""Ablation sweeps over pipeline configurations with paired bootstrap comparisons."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from researchrag.config.pipeline import PipelineConfig, load_pipeline_config
from researchrag.errors import ValidationFailedError
from researchrag.evaluation.runner import HEADLINE, LOWER_IS_BETTER, EvalRunner, RunResult
from researchrag.evaluation.stats import paired_bootstrap


class SweepSpec(BaseModel):
    """A sweep definition loaded from YAML."""

    base: str | None = None
    mode: Literal["ofat", "grid"] = "ofat"
    quality_metric: str = "answer_correctness"
    factors: dict[str, list[Any]] = Field(default_factory=dict)
    max_configs: int = 40


def load_sweep(path: str | Path) -> tuple[SweepSpec, PipelineConfig]:
    """Load a sweep spec and its base pipeline config (relative paths resolve from the spec file)."""
    file = Path(path)
    spec = SweepSpec.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")) or {})
    base_path = (file.parent / spec.base) if spec.base and not Path(spec.base).is_absolute() else spec.base
    if base_path is not None and not Path(base_path).exists():
        base_path = Path(spec.base or "")
    base = load_pipeline_config(base_path).with_overrides({}, name="base")
    return spec, base


def _slug(value: Any) -> str:
    return str(value).lower().replace(" ", "")


def expand(spec: SweepSpec, base: PipelineConfig) -> list[tuple[str, str | None, PipelineConfig]]:
    """``[(name, factor, config), ...]``; the base config comes first with factor ``None``."""
    configs: list[tuple[str, str | None, PipelineConfig]] = [("base", None, base)]
    flat = base.model_dump()
    if spec.mode == "ofat":
        for factor, values in spec.factors.items():
            current = _get(flat, factor)
            for value in values:
                if value == current:
                    continue
                name = f"{factor}={_slug(value)}"
                configs.append((name, factor, base.with_overrides({factor: value}, name=name)))
    else:
        keys = list(spec.factors)
        for combo in itertools.product(*(spec.factors[k] for k in keys)):
            overrides = dict(zip(keys, combo, strict=True))
            if all(_get(flat, k) == v for k, v in overrides.items()):
                continue
            name = ",".join(f"{k}={_slug(v)}" for k, v in overrides.items())
            configs.append((name, "grid", base.with_overrides(overrides, name=name)))
    if len(configs) > spec.max_configs:
        raise ValidationFailedError(
            f"Sweep expands to {len(configs)} configs (max_configs={spec.max_configs})"
        )
    return configs


def _get(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValidationFailedError(f"Unknown sweep factor: {dotted}")
        node = node[part]
    return node


class SweepResult(BaseModel):
    """All runs of a sweep plus paired comparisons against the base."""

    runs: list[RunResult]
    factors: dict[str, str | None]
    comparisons: dict[str, dict[str, dict[str, float | int | bool]]]
    quality_metric: str
    best_quality_per_dollar: str
    best_note: str


async def run_sweep(runner: EvalRunner, spec: SweepSpec, base: PipelineConfig) -> SweepResult:
    """Run every configuration and compare each with the base (paired over questions)."""
    configs = expand(spec, base)
    runs: list[RunResult] = []
    factors: dict[str, str | None] = {}
    for name, factor, config in configs:
        runs.append(await runner.run(config))
        factors[name] = factor
    base_run = runs[0]
    comparisons: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    for run in runs[1:]:
        comparisons[run.name] = {
            metric: paired_bootstrap(base_run.values(metric), run.values(metric)).as_dict()
            for metric in HEADLINE
            if metric in run.summary or metric in base_run.summary
        }
    best, note = best_quality_per_dollar(runs, spec.quality_metric)
    return SweepResult(
        runs=runs,
        factors=factors,
        comparisons=comparisons,
        quality_metric=spec.quality_metric,
        best_quality_per_dollar=best,
        best_note=note,
    )


def best_quality_per_dollar(runs: list[RunResult], metric: str) -> tuple[str, str]:
    """Pick the configuration with the best quality per dollar (ties: quality, then p50 latency)."""

    def quality(run: RunResult) -> float:
        value = run.summary.get(metric, {}).get("mean", 0.0)
        return 0.0 if math.isnan(float(value)) else float(value)

    if all(run.cost["per_query_usd"] == 0 for run in runs):
        best = max(runs, key=lambda r: (quality(r), -r.latency_ms["p50"]))
        return best.name, (
            f"All configurations had zero model cost (offline components), so the pick maximises {metric} "
            "and breaks ties by p50 latency."
        )
    best = max(runs, key=lambda r: (quality(r) / max(r.cost["per_query_usd"], 1e-9), quality(r)))
    return best.name, f"Highest {metric} per USD per query."


def factor_effects(result: SweepResult) -> dict[str, dict[str, dict[str, Any]]]:
    """For each factor and headline metric, the largest change versus base and whether it is significant."""
    effects: dict[str, dict[str, dict[str, Any]]] = {}
    for run_name, comparison in result.comparisons.items():
        factor = result.factors.get(run_name) or run_name
        for metric, cmp in comparison.items():
            current = effects.setdefault(factor, {}).get(metric)
            if current is None or abs(float(cmp["diff"])) > abs(float(current["diff"])):
                worse = (float(cmp["diff"]) > 0) == (metric in LOWER_IS_BETTER)
                effects[factor][metric] = {
                    **cmp,
                    "config": run_name,
                    "direction": "worse" if worse else "better",
                }
    return effects
