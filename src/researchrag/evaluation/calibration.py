"""Judge calibration against the hand-labelled claim/evidence set."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from researchrag.evaluation.dataset import CalibrationItem
from researchrag.verification.judges import ClaimJudge

LABELS = ("supported", "partially_supported", "unsupported", "contradicted")


class CalibrationReport(BaseModel):
    """Agreement between a judge and the labels."""

    judge: str
    n: int
    accuracy: float
    binary_accuracy: float
    cohen_kappa: float
    confusion: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float]]
    disagreements: list[dict[str, str]]


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa for two label sequences."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


async def calibrate(judge: ClaimJudge, items: list[CalibrationItem]) -> CalibrationReport:
    """Run ``judge`` over the calibration items and report agreement."""
    pairs: list[tuple[str, str]] = []
    disagreements: list[dict[str, str]] = []
    for item in items:
        result = await judge.judge(item.claim, item.evidence)
        pairs.append((item.label, result.verdict))
        if item.label != result.verdict:
            disagreements.append(
                {"id": item.id, "label": item.label, "judge": result.verdict, "claim": item.claim}
            )
    confusion = {gold: dict.fromkeys(LABELS, 0) for gold in LABELS}
    for gold, pred in pairs:
        confusion[gold][pred] += 1
    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in LABELS)
        actual = sum(confusion[label].values())
        per_label[label] = {
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "recall": round(tp / actual, 4) if actual else 0.0,
            "support": float(actual),
        }

    def ok(label: str) -> bool:
        return label in {"supported", "partially_supported"}

    n = len(pairs)
    return CalibrationReport(
        judge=judge.name,
        n=n,
        accuracy=round(sum(a == b for a, b in pairs) / n, 4) if n else 0.0,
        binary_accuracy=round(sum(ok(a) == ok(b) for a, b in pairs) / n, 4) if n else 0.0,
        cohen_kappa=round(cohen_kappa(pairs), 4),
        confusion=confusion,
        per_label=per_label,
        disagreements=disagreements,
    )
