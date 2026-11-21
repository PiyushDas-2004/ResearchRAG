"""Deterministic synthetic academic PDFs for tests, demos and evaluation.

Papers are described in YAML (see ``data/CORPUS_SPEC.md``) and rendered with reportlab in a
realistic layout: a full-width title block, a two-column body, section headings, tables with
captions, a references list, a running header, a footer and page numbers. Output bytes are
reproducible (``invariant=True``), so content hashes are stable across runs and machines.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape  # escapes markup we generate; never parses XML  # nosec B406

import yaml
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

FOOTER_TEXT = "Synthetic paper - ResearchRAG test corpus (fictional content)"


class TableSpec(BaseModel):
    """A table placed after a paragraph of a section."""

    caption: str
    header: list[str]
    rows: list[list[str]]
    after_paragraph: int = 0


class SectionSpec(BaseModel):
    """A titled section with paragraphs and optional tables."""

    title: str
    paragraphs: list[str]
    tables: list[TableSpec] = Field(default_factory=list)


class PaperSpec(BaseModel):
    """A synthetic paper as authored in YAML."""

    key: str
    title: str
    authors: list[str]
    affiliation: str = ""
    year: int
    venue: str = "Journal of Synthetic Research"
    doi: str | None = None
    arxiv_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    abstract: str
    sections: list[SectionSpec]
    references: list[str] = Field(default_factory=list)


def load_paper(path: str | Path) -> PaperSpec:
    """Load one paper spec from YAML."""
    return PaperSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_papers(directory: str | Path) -> list[PaperSpec]:
    """Load every ``*.yaml`` paper spec in a directory, sorted by key."""
    return sorted((load_paper(p) for p in Path(directory).glob("*.yaml")), key=lambda p: p.key)


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=17, leading=21, alignment=1),
        "authors": ParagraphStyle("authors", fontName="Times-Roman", fontSize=10.5, leading=13, alignment=1),
        "affil": ParagraphStyle("affil", fontName="Times-Italic", fontSize=9, leading=11, alignment=1),
        "meta": ParagraphStyle("meta", fontName="Times-Roman", fontSize=8.5, leading=10.5, alignment=1),
        "abs_head": ParagraphStyle("abs_head", fontName="Times-Bold", fontSize=10, leading=12, spaceBefore=6),
        "abstract": ParagraphStyle(
            "abstract", fontName="Times-Roman", fontSize=9.5, leading=11.5, leftIndent=24, rightIndent=24
        ),
        "h1": ParagraphStyle(
            "h1", fontName="Times-Bold", fontSize=11, leading=13, spaceBefore=8, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "body", fontName="Times-Roman", fontSize=9.5, leading=11.4, alignment=4, spaceAfter=6
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Times-Italic", fontSize=8.5, leading=10, spaceBefore=4, spaceAfter=3
        ),
        "ref": ParagraphStyle(
            "ref",
            fontName="Times-Roman",
            fontSize=8,
            leading=9.6,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Times-Roman", fontSize=7.2, leading=8.6, splitLongWords=False
        ),
    }


def _table(spec: TableSpec, width: float, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    data = [[Paragraph(escape(str(c)), styles["cell"]) for c in spec.header]]
    data += [[Paragraph(escape(str(c)), styles["cell"]) for c in row] for row in spec.rows]
    # Column widths proportional to the longest word in each column, so words never split.
    longest = [
        max(
            len(w) for cell in [spec.header[i], *[r[i] for r in spec.rows]] for w in str(cell).split() or [""]
        )
        for i in range(len(spec.header))
    ]
    needed = [n * 4.1 + 7 for n in longest]  # ~4.1pt per char at 7.2pt Times + padding
    spare = max(0.0, width - sum(needed))
    col_widths = [w * min(1.0, width / sum(needed)) + spare / len(needed) for w in needed]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [Paragraph(escape(spec.caption), styles["caption"]), table, Spacer(1, 6)]


def render_paper(paper: PaperSpec, *, extra_paragraphs: Sequence[tuple[str, str]] = ()) -> bytes:
    """Render a paper spec to PDF bytes.

    ``extra_paragraphs`` appends ``(section_title, text)`` pairs; tests use it to build
    fixtures such as a poisoned document without touching the authored corpus.
    """
    styles = _styles()
    buffer = io.BytesIO()
    page_w, page_h = LETTER
    margin = 0.75 * inch
    gutter = 0.3 * inch
    col_w = (page_w - 2 * margin - gutter) / 2
    body_h = page_h - 2 * margin

    header: list[Flowable] = [
        Paragraph(escape(paper.title), styles["title"]),
        Spacer(1, 6),
        Paragraph(escape(", ".join(paper.authors)), styles["authors"]),
    ]
    if paper.affiliation:
        header.append(Paragraph(escape(paper.affiliation), styles["affil"]))
    ids = [f"{paper.venue}, {paper.year}"]
    if paper.doi:
        ids.append(f"DOI: {paper.doi}")
    if paper.arxiv_id:
        ids.append(f"arXiv:{paper.arxiv_id}")
    header += [Spacer(1, 3), Paragraph(escape("  |  ".join(ids)), styles["meta"])]
    header += [
        Paragraph("Abstract", styles["abs_head"]),
        Paragraph(escape(paper.abstract), styles["abstract"]),
    ]
    if paper.keywords:
        header.append(Paragraph(escape("Keywords: " + "; ".join(paper.keywords)), styles["meta"]))
    header.append(Spacer(1, 8))
    header_h = sum(f.wrap(page_w - 2 * margin, body_h)[1] for f in header) + 24

    def decorate(canvas: Canvas, doc: BaseDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Times-Italic", 8)
        short = paper.title if len(paper.title) < 70 else paper.title[:67] + "..."
        if doc.page > 1:
            canvas.drawString(margin, page_h - 0.5 * inch, short)
        canvas.drawRightString(page_w - margin, page_h - 0.5 * inch, f"{paper.venue} ({paper.year})")
        canvas.drawString(margin, 0.45 * inch, FOOTER_TEXT)
        canvas.drawRightString(page_w - margin, 0.45 * inch, str(doc.page))
        canvas.restoreState()

    first_frames = [
        Frame(margin, page_h - margin - header_h, page_w - 2 * margin, header_h, id="head", showBoundary=0),
        Frame(margin, margin, col_w, body_h - header_h, id="c1"),
        Frame(margin + col_w + gutter, margin, col_w, body_h - header_h, id="c2"),
    ]
    later_frames = [
        Frame(margin, margin, col_w, body_h, id="l1"),
        Frame(margin + col_w + gutter, margin, col_w, body_h, id="l2"),
    ]
    doc = BaseDocTemplate(
        buffer,
        pagesize=LETTER,
        invariant=True,
        title=paper.title,
        author=", ".join(paper.authors),
        subject=f"{paper.venue} {paper.year}",
        creator="ResearchRAG synthetic corpus",
        producer="ResearchRAG",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="first", frames=first_frames, onPage=decorate),
            PageTemplate(id="later", frames=later_frames, onPage=decorate),
        ]
    )
    story: list[Flowable] = [NextPageTemplate("later"), *header, FrameBreak()]
    sections = list(paper.sections)
    extras: dict[str, list[str]] = {}
    for title, text in extra_paragraphs:
        extras.setdefault(title, []).append(text)
    for title, texts in extras.items():
        existing = next((s for s in sections if s.title == title), None)
        if existing is None:
            sections.append(SectionSpec(title=title, paragraphs=texts))
        else:
            sections[sections.index(existing)] = existing.model_copy(
                update={"paragraphs": existing.paragraphs + texts}
            )
    for number, section in enumerate(sections, start=1):
        story.append(Paragraph(escape(f"{number} {section.title}"), styles["h1"]))
        for idx, para in enumerate(section.paragraphs):
            story.append(Paragraph(escape(para), styles["body"]))
            for table in section.tables:
                if table.after_paragraph == idx:
                    story.extend(_table(table, col_w, styles))
    if paper.references:
        story.append(Paragraph("References", styles["h1"]))
        for idx, ref in enumerate(paper.references, start=1):
            story.append(Paragraph(escape(f"[{idx}] {ref}"), styles["ref"]))
    doc.build(story)
    return buffer.getvalue()


def build_corpus(papers_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Render every paper spec to ``<out_dir>/<key>.pdf``; return ``{key: sha256}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for paper in load_papers(papers_dir):
        data = render_paper(paper)
        (out / f"{paper.key}.pdf").write_bytes(data)
        hashes[paper.key] = hashlib.sha256(data).hexdigest()
    return hashes


def paper_passages(paper: PaperSpec) -> dict[str, Any]:
    """Return the paragraphs and table rows of a paper (used to validate gold evidence)."""
    paragraphs = [paper.abstract] + [p for s in paper.sections for p in s.paragraphs]
    rows = [" | ".join(r) for s in paper.sections for t in s.tables for r in t.rows]
    return {"paragraphs": paragraphs, "rows": rows}
