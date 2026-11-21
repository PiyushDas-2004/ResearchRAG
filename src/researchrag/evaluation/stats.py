"""Bootstrap confidence intervals and paired comparisons (resampling questions)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A mean with a percentile bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        """JSON-friendly representation."""
        return {
            "mean": round(self.mean, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
        }


@dataclass(frozen=True)
class PairedComparison:
    """Mean difference ``b - a`` over paired questions with a bootstrap CI and two-sided p-value."""

    diff: float
    low: float
    high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """True when the 95% CI excludes zero."""
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float | int | bool]:
        """JSON-friendly representation."""
        return {
            "diff": round(self.diff, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "p_value": round(self.p_value, 4),
            "n": self.n,
            "significant": self.significant,
        }


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Percentile bootstrap CI of the mean."""
    data = np.asarray([v for v in values if v is not None], dtype=float)
    if data.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(data.mean()), float(low), float(high), int(data.size))


def paired_bootstrap(
    a: Sequence[float | None],
    b: Sequence[float | None],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap of ``mean(b - a)`` over items where both values exist."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return PairedComparison(0.0, 0.0, 0.0, 1.0, 0)
    diffs = np.asarray([y - x for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, diffs.size, size=(resamples, diffs.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    observed = float(diffs.mean())
    if np.allclose(diffs, 0):
        p_value = 1.0
    else:
        p_value = float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))
    return PairedComparison(observed, float(low), float(high), p_value, int(diffs.size))


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile (0-100) of a non-empty sequence; NaN when empty."""
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")
