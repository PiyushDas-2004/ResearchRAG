"""Citation resolution and quote verification.

* Every ``[n]`` marker must resolve to a packed passage; dangling markers are dropped and
  reported.
* Every quote must occur in the cited passage: exact match after normalisation, or a fuzzy
  partial match scoring at least the configured threshold (0-100). A quote that matches
  nowhere is a *fabricated quote* - a hard failure recorded in the trace, flagged on every
  claim citing that passage, and shown as unverified in the API (even if another quote for
  the same passage is genuine).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from researchrag.generation.packing import Passage
from researchrag.generation.schema import CitationOut, RawAnswer
from researchrag.textutils import collapse_whitespace, normalize_unicode

MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def _norm(text: str) -> str:
    return collapse_whitespace(normalize_unicode(text)).lower()


@dataclass(frozen=True)
class QuoteMatch:
    """Result of locating a quote in a passage."""

    verified: bool
    score: float
    offset: int


def match_quote(quote: str, passage: str, threshold: float) -> QuoteMatch:
    """Locate ``quote`` in ``passage`` (exact, then fuzzy). ``offset`` is relative to ``passage``."""
    q = _norm(quote).strip(" .\"'")
    if not q:
        return QuoteMatch(False, 0.0, 0)
    p_norm = _norm(passage)
    idx = p_norm.find(q)
    if idx >= 0:
        return QuoteMatch(True, 100.0, _map_offset(passage, idx))
    alignment = fuzz.partial_ratio_alignment(q, p_norm)
    score = float(alignment.score) if alignment else 0.0
    offset = _map_offset(passage, alignment.dest_start) if alignment else 0
    return QuoteMatch(score >= threshold, round(score, 2), offset)


def _map_offset(original: str, normalized_offset: int) -> int:
    """Approximate an offset in collapsed text back to the original text."""
    seen = 0
    prev_space = False
    for i, ch in enumerate(original.strip()):
        is_space = ch.isspace()
        if not (is_space and prev_space):
            if seen >= normalized_offset:
                return i + (len(original) - len(original.lstrip()))
            seen += 1
        prev_space = is_space
    return len(original)


@dataclass
class CitationReport:
    """Resolved citations plus problems found."""

    citations: list[CitationOut] = field(default_factory=list)
    dangling: list[int] = field(default_factory=list)
    fabricated: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_citations(answer: RawAnswer, passages: list[Passage], threshold: float) -> CitationReport:
    """Resolve markers used in claims/answer text to passages and verify their quotes."""
    by_number = {p.number: p for p in passages}
    used: list[int] = []
    for claim in answer.claims:
        used.extend(claim.citations)
    used.extend(int(m) for m in MARKER_RE.findall(answer.answer))
    report = CitationReport()
    quotes: dict[int, list[str]] = {}
    for q in answer.quotes:
        quotes.setdefault(q.passage, []).append(q.quote)
    for marker in dict.fromkeys(used):
        passage = by_number.get(marker)
        if passage is None:
            report.dangling.append(marker)
            continue
        best: tuple[QuoteMatch, str] | None = None
        any_fabricated = False
        for quote in quotes.get(marker, []):
            match = match_quote(quote, passage.text, threshold)
            any_fabricated = any_fabricated or not match.verified
            if best is None or match.score > best[0].score:
                best = (match, quote)
        if any_fabricated:
            report.fabricated.append(marker)
        page = (
            passage.page_of(best[0].offset) if best is not None and best[0].verified else passage.page_start
        )
        report.citations.append(
            CitationOut(
                marker=marker,
                chunk_id=passage.chunk_id,
                document_id=passage.document_id,
                title=passage.title,
                page=page,
                section=passage.section,
                quote=best[1] if best else None,
                quote_verified=bool(best and best[0].verified),
                quote_score=best[0].score if best else None,
                passage=passage.text,
            )
        )
    if report.dangling:
        report.warnings.append(f"dangling citation markers removed: {report.dangling}")
    if report.fabricated:
        report.warnings.append(f"fabricated quotes (not found in cited passage): {report.fabricated}")
    return report
