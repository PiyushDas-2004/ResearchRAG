"""Neutral JSONL export of evaluation cases (for LLMBench or any other harness).

ResearchRAG does not depend on LLMBench. ``export_cases`` writes, per question, the exact
grounded-answer prompt a configuration would send (system + user messages with the packed
passages) plus the reference answer and metadata, so an external benchmarking tool can replay
the same configurations against other models. If the ``llmbench`` package is importable, the
CLI mentions it; it never calls into it, because its API is not part of this project.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from researchrag.evaluation.dataset import Dataset


def llmbench_available() -> bool:
    """True when an ``llmbench`` package is importable in this environment."""
    return importlib.util.find_spec("llmbench") is not None


def export_cases(
    dataset: Dataset, traces: dict[str, dict[str, Any]], config_name: str, out: str | Path
) -> Path:
    """Write one JSONL line per question: prompt messages, reference answer and metadata.

    ``traces`` maps question id to the stored trace stages of a run (prompts must have been
    stored, i.e. ``store_prompts_in_traces`` enabled).
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in dataset.questions:
            stages = traces.get(q.id, {})
            record = {
                "id": q.id,
                "config": config_name,
                "type": q.type,
                "answerable": q.answerable,
                "messages": stages.get("generate", {}).get("prompt"),
                "reference": q.gold_answer,
                "evidence": [e.model_dump() for e in q.evidence],
                "dataset_hash": dataset.content_hash,
            }
            fh.write(json.dumps(record) + "\n")
    return path


def feedback_to_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn low-rated traces into candidate eval questions (gold labels left for a human)."""
    return [
        {
            "id": f"fb-{row['trace_id'][:8]}",
            "question": row["question"],
            "type": "single_hop",
            "answerable": None,
            "gold_answer": None,
            "evidence": [],
            "source": {
                "trace_id": row["trace_id"],
                "model_answer": row.get("answer"),
                "abstained": row.get("abstained"),
                "user_comment": row.get("comment"),
                "retrieved_chunk_ids": row.get("retrieved", []),
            },
            "needs_review": True,
        }
        for row in rows
    ]
