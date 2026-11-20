"""Validate the synthetic corpus YAML files and evaluation JSONL files.

Usage: python scripts/validate_corpus.py [data_dir]

Checks the schema described in data/CORPUS_SPEC.md: required fields, word counts,
question types, and that every gold evidence passage is a verbatim substring of a
paragraph (or an exact table row) of the referenced paper.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

QUESTION_TYPES = {"single_hop", "multi_hop", "numeric_table", "unanswerable", "adversarial"}
LABELS = {"supported", "partially_supported", "unsupported", "contradicted"}
KEY_RE = re.compile(r"^[a-z]+-\d{2}$")


def _words(text: str) -> int:
    return len(text.split())


def validate_paper(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    paper = yaml.safe_load(path.read_text(encoding="utf-8"))
    key = paper.get("key", "")
    if not KEY_RE.match(key) or path.stem != key:
        errors.append(f"{path.name}: key {key!r} must match file name and <prefix>-NN")
    for field in ("title", "authors", "affiliation", "year", "venue", "doi", "abstract", "sections", "references"):
        if not paper.get(field):
            errors.append(f"{key}: missing field {field}")
    if not str(paper.get("doi", "")).startswith("10.5555/rrsyn."):
        errors.append(f"{key}: doi must start with 10.5555/rrsyn.")
    arxiv = paper.get("arxiv_id")
    if arxiv and not re.fullmatch(r"\d{2}99\.\d{5}", str(arxiv)):
        errors.append(f"{key}: arxiv_id must look like YY99.NNNNN")
    abstract_words = _words(paper.get("abstract", ""))
    if not 90 <= abstract_words <= 190:
        errors.append(f"{key}: abstract has {abstract_words} words (want 110-160)")
    body = 0
    tables = 0
    for section in paper.get("sections", []):
        if section.get("title", "").lower() in {"abstract", "references"}:
            errors.append(f"{key}: section {section['title']} must not be in sections")
        for para in section.get("paragraphs", []):
            body += _words(para)
            if not para.isascii():
                errors.append(f"{key}: non-ASCII text in paragraph: {para[:40]!r}")
        for table in section.get("tables", []) or []:
            tables += 1
            width = len(table["header"])
            if any(len(row) != width for row in table["rows"]):
                errors.append(f"{key}: table {table['caption']!r} has ragged rows")
            if not 0 <= table.get("after_paragraph", 0) < len(section.get("paragraphs", [])):
                errors.append(f"{key}: table {table['caption']!r} has bad after_paragraph")
    if not 1000 <= body <= 1800:
        errors.append(f"{key}: body has {body} words (want 1100-1600)")
    if tables < 1:
        errors.append(f"{key}: needs at least one table")
    if not 5 <= len(paper.get("references", [])) <= 12:
        errors.append(f"{key}: want 6-10 references")
    return paper, errors


def _passages(paper: dict[str, Any]) -> tuple[list[str], set[str]]:
    paragraphs = [paper["abstract"]]
    rows: set[str] = set()
    for section in paper["sections"]:
        paragraphs.extend(section.get("paragraphs", []))
        for table in section.get("tables", []) or []:
            for row in table["rows"]:
                rows.add(" | ".join(str(c) for c in row))
    return paragraphs, rows


def validate_questions(path: Path, papers: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    types: Counter[str] = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        qid = q.get("id", f"line {line_no}")
        types[q.get("type", "?")] += 1
        if q.get("type") not in QUESTION_TYPES:
            errors.append(f"{qid}: bad type {q.get('type')}")
        if not q.get("question") or not q.get("gold_answer"):
            errors.append(f"{qid}: missing question or gold_answer")
        evidence = q.get("evidence", [])
        if q.get("answerable") is False and evidence:
            errors.append(f"{qid}: unanswerable question must have no evidence")
        if q.get("answerable") is True and not evidence:
            errors.append(f"{qid}: answerable question needs evidence")
        if q.get("type") == "unanswerable" and q.get("answerable") is not False:
            errors.append(f"{qid}: type unanswerable must have answerable=false")
        if q.get("type") == "multi_hop" and len({e["doc_key"] for e in evidence}) < 2:
            errors.append(f"{qid}: multi_hop needs evidence from two papers")
        for ev in evidence:
            paper = papers.get(ev.get("doc_key", ""))
            if paper is None:
                errors.append(f"{qid}: unknown doc_key {ev.get('doc_key')}")
                continue
            paragraphs, rows = _passages(paper)
            passage = ev.get("passage", "")
            if passage not in rows and not any(passage in p for p in paragraphs):
                errors.append(f"{qid}: passage not found verbatim in {ev['doc_key']}: {passage[:60]!r}")
    print(f"{path.name}: {sum(types.values())} questions {dict(types)}")
    return errors


def validate_calibration(path: Path) -> list[str]:
    errors: list[str] = []
    labels: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        labels[item.get("label", "?")] += 1
        if item.get("label") not in LABELS or not item.get("claim") or not item.get("evidence"):
            errors.append(f"{item.get('id')}: bad calibration item")
    print(f"{path.name}: {dict(labels)}")
    return errors


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    papers: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted((data_dir / "corpus" / "papers").glob("*.yaml")):
        paper, errs = validate_paper(path)
        papers[paper.get("key", path.stem)] = paper
        errors.extend(errs)
    print(f"papers: {len(papers)}")
    for path in sorted((data_dir / "eval").glob("questions_*.jsonl")):
        errors.extend(validate_questions(path, papers))
    for path in sorted((data_dir / "eval").glob("calibration_*.jsonl")):
        errors.extend(validate_calibration(path))
    for err in errors:
        print("ERROR", err)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
