"""Evaluation datasets: questions with gold answers and gold evidence, versioned by content hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["single_hop", "multi_hop", "numeric_table", "unanswerable", "adversarial"]


class GoldEvidence(BaseModel):
    """A gold supporting passage inside a corpus paper."""

    doc_key: str
    passage: str


class EvalQuestion(BaseModel):
    """One evaluation question."""

    id: str
    question: str
    type: QuestionType
    answerable: bool
    gold_answer: str
    evidence: list[GoldEvidence] = Field(default_factory=list)


class CalibrationItem(BaseModel):
    """A hand-labelled claim/evidence pair for judge calibration."""

    id: str
    claim: str
    evidence: str
    label: Literal["supported", "partially_supported", "unsupported", "contradicted"]


@dataclass(frozen=True)
class Dataset:
    """Questions plus a content hash identifying the exact dataset version."""

    questions: list[EvalQuestion]
    content_hash: str
    name: str

    def subset(self, limit: int | None = None, types: set[str] | None = None) -> Dataset:
        """A deterministic subset (first ``limit`` questions after type filtering)."""
        items = [q for q in self.questions if types is None or q.type in types]
        if limit is not None:
            items = items[:limit]
        return Dataset(items, self.content_hash, self.name)


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_dataset(directory: str | Path, pattern: str = "questions_*.jsonl") -> Dataset:
    """Load all question files in ``directory`` (sorted for determinism)."""
    paths = sorted(Path(directory).glob(pattern))
    questions = [
        EvalQuestion.model_validate(json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Dataset(questions, _hash_files(paths), Path(directory).name)


def load_calibration(directory: str | Path) -> list[CalibrationItem]:
    """Load the judge-calibration items."""
    return [
        CalibrationItem.model_validate(json.loads(line))
        for path in sorted(Path(directory).glob("calibration_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
