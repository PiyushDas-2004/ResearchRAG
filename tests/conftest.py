"""Shared fixtures: synthetic corpus, rendered PDFs and parsed documents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from researchrag.evaluation.synthetic import PaperSpec, load_papers, render_paper
from researchrag.ingestion.pdf_parser import parse_pdf
from researchrag.ingestion.types import ParsedDocument

pytest_plugins = ["tests.db_fixtures"]

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "data" / "corpus" / "papers"
EVAL_DIR = ROOT / "data" / "eval"


@lru_cache(maxsize=1)
def corpus_papers() -> tuple[PaperSpec, ...]:
    """All authored papers (cached for the session)."""
    return tuple(load_papers(PAPERS_DIR))


@lru_cache(maxsize=64)
def paper_pdf(key: str) -> bytes:
    """Rendered PDF bytes for a paper key."""
    paper = next(p for p in corpus_papers() if p.key == key)
    return render_paper(paper)


@lru_cache(maxsize=64)
def parsed(key: str) -> ParsedDocument:
    """Parsed document for a paper key."""
    return parse_pdf(paper_pdf(key))


@pytest.fixture(scope="session")
def papers() -> tuple[PaperSpec, ...]:
    """All papers of the synthetic corpus."""
    return corpus_papers()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    """One rendered two-column paper with a table."""
    return paper_pdf("clim-02")


@pytest.fixture(scope="session")
def sample_doc() -> ParsedDocument:
    """The parsed form of :func:`sample_pdf`."""
    return parsed("clim-02")
