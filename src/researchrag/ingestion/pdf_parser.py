"""Layout-aware PDF text extraction with pymupdf.

Steps per page: collect text blocks (OCR fallback for image-only pages), detect tables and
render their rows as ``cell | cell`` lines with the caption attached, drop running headers,
footers and page numbers, order blocks for one- or two-column layouts, repair hyphenation
and normalise unicode. The result keeps page numbers and character offsets so that every
chunk can be cited back to a page.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import pymupdf

from researchrag.errors import UploadRejectedError
from researchrag.ingestion.cleaning import (
    fingerprint,
    is_page_number,
    join_lines,
    repeating_margin_texts,
)
from researchrag.ingestion.metadata import extract_metadata
from researchrag.ingestion.sections import heading_label, heading_title
from researchrag.ingestion.types import Block, BlockKind, PageInfo, ParsedDocument, Section, SectionLabel

MARGIN_FRACTION = 0.075
_CAPTION_RE = re.compile(r"^(Table|Tab\.)\s*\d+[:.]", re.IGNORECASE)


@dataclass
class ParseLimits:
    """Resource limits enforced while parsing an untrusted PDF."""

    max_pages: int = 200
    max_chars: int = 2_000_000
    ocr_enabled: bool = False
    detect_tables: bool = True


@dataclass
class RawBlock:
    """A block as read from the page, before cleaning and ordering."""

    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[str]
    font_size: float
    kind: BlockKind = "paragraph"
    caption: str | None = None
    bold: bool = False

    @property
    def text(self) -> str:
        """Cleaned block text."""
        if self.kind == "table":
            return "\n".join(self.lines)
        return join_lines(self.lines)


def _read_blocks(page: Any, ocr_enabled: bool) -> tuple[list[RawBlock], bool]:
    textpage = None
    used_ocr = False
    raw = page.get_text("dict")
    has_text = any(b.get("type") == 0 and b.get("lines") for b in raw["blocks"])
    if not has_text and ocr_enabled and page.get_images():
        try:
            textpage = page.get_textpage_ocr(dpi=200, full=True)
            raw = page.get_text("dict", textpage=textpage)
            used_ocr = True
        except RuntimeError:  # pragma: no cover - tesseract missing at runtime
            used_ocr = False
    blocks: list[RawBlock] = []
    for b in raw["blocks"]:
        if b.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        bold_flags: list[bool] = []
        for line in b["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                lines.append(text)
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(span["size"])
                        bold_flags.append(bool(span["flags"] & 16) or "bold" in span["font"].lower())
        if lines:
            x0, y0, x1, y1 = b["bbox"]
            blocks.append(RawBlock(x0, y0, x1, y1, lines, max(sizes) if sizes else 0.0, bold=all(bold_flags)))
    return blocks, used_ocr


def _extract_tables(page: Any, blocks: list[RawBlock]) -> list[RawBlock]:
    """Replace blocks inside detected tables with one table block per table (caption attached).

    Table detection is the most expensive parsing step, so it only runs on pages that have
    both vector drawings (ruling lines) and a ``Table N`` caption.
    """
    if not any(_CAPTION_RE.match(b.text) for b in blocks) or not page.get_drawings():
        return blocks
    try:
        # Line-based detection only: deterministic and free of the ONNX layout model that
        # newer pymupdf releases load by default (too large for the parser's memory cap).
        try:
            found = page.find_tables(use_layout=False).tables
        except TypeError:  # pragma: no cover - pymupdf < 1.27 has no use_layout argument
            found = page.find_tables().tables
    except (RuntimeError, ValueError):  # pragma: no cover - defensive against odd pages
        return blocks
    if not found:
        return blocks
    kept = list(blocks)
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    for table in found:
        tx0, ty0, tx1, ty1 = table.bbox
        rows: list[str] = []
        for row in table.rows:
            cells = [_cell_text(words, c) if c else "" for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue

        def inside(
            b: RawBlock, tx0: float = tx0, ty0: float = ty0, tx1: float = tx1, ty1: float = ty1
        ) -> bool:
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            return tx0 - 1 <= cx <= tx1 + 1 and ty0 - 1 <= cy <= ty1 + 1

        kept = [b for b in kept if not inside(b)]
        caption_block = None
        candidates = [b for b in kept if _CAPTION_RE.match(b.text) and abs(b.x0 - tx0) < (tx1 - tx0)]
        if candidates:
            caption_block = min(candidates, key=lambda b: min(abs(b.y1 - ty0), abs(b.y0 - ty1)))
            if min(abs(caption_block.y1 - ty0), abs(caption_block.y0 - ty1)) > 40:
                caption_block = None
        caption = caption_block.text if caption_block else None
        if caption_block is not None:
            kept.remove(caption_block)
        lines = ([caption] if caption else []) + rows
        y0 = min(ty0, caption_block.y0) if caption_block else ty0
        kept.append(RawBlock(tx0, y0, tx1, ty1, lines, 0.0, kind="table", caption=caption))
    return kept


def _cell_text(words: list[Any], cell: tuple[float, float, float, float]) -> str:
    """Join the words whose centre lies inside ``cell`` in reading order."""
    x0, y0, x1, y1 = cell
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    inside.sort(key=lambda w: (w[5], w[6], w[7]))
    return join_lines([" ".join(str(w[4]) for w in inside)])


def order_blocks(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
    """Order blocks in reading order for single- and two-column layouts.

    Full-width blocks (title, abstract, wide tables) split the page into vertical bands;
    within a band all left-column blocks are read before the right-column blocks.
    """
    mid = page_width / 2
    tol = page_width * 0.03
    ordered: list[RawBlock] = []
    left: list[RawBlock] = []
    right: list[RawBlock] = []

    def flush() -> None:
        ordered.extend(sorted(left, key=lambda b: (b.y0, b.x0)))
        ordered.extend(sorted(right, key=lambda b: (b.y0, b.x0)))
        left.clear()
        right.clear()

    for block in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        if block.x1 <= mid + tol:
            left.append(block)
        elif block.x0 >= mid - tol:
            right.append(block)
        else:
            flush()
            ordered.append(block)
    flush()
    return ordered


def _split_margins(blocks: list[RawBlock], page_height: float) -> tuple[list[RawBlock], list[RawBlock]]:
    top = page_height * MARGIN_FRACTION
    bottom = page_height * (1 - MARGIN_FRACTION)
    body = [b for b in blocks if not (b.y1 <= top or b.y0 >= bottom)]
    margin = [b for b in blocks if b.y1 <= top or b.y0 >= bottom]
    return body, margin


def parse_pdf(data: bytes, limits: ParseLimits | None = None) -> ParsedDocument:
    """Parse PDF bytes into a :class:`ParsedDocument`.

    Raises :class:`UploadRejectedError` for malformed, encrypted, oversized or empty PDFs.
    """
    limits = limits or ParseLimits()
    try:
        doc: Any = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except (RuntimeError, ValueError) as exc:
        raise UploadRejectedError("The file is not a readable PDF.", detail={"reason": "malformed"}) from exc
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise UploadRejectedError("Encrypted PDFs are not supported.", detail={"reason": "encrypted"})
        if doc.page_count == 0:
            raise UploadRejectedError("The PDF has no pages.", detail={"reason": "empty"})
        if doc.page_count > limits.max_pages:
            raise UploadRejectedError(
                f"The PDF has {doc.page_count} pages; the limit is {limits.max_pages}.",
                detail={"reason": "too_many_pages", "pages": doc.page_count},
            )
        pdf_meta = dict(doc.metadata or {})
        per_page: list[tuple[list[RawBlock], list[RawBlock], float, bool]] = []
        first_page_blocks: list[RawBlock] = []
        total_chars = 0
        for index in range(doc.page_count):
            page = doc[index]
            try:
                blocks, used_ocr = _read_blocks(page, limits.ocr_enabled)
            except RuntimeError as exc:
                raise UploadRejectedError(
                    f"Page {index + 1} could not be read.", detail={"reason": "malformed", "page": index + 1}
                ) from exc
            total_chars += sum(len(line) for b in blocks for line in b.lines)
            if total_chars > limits.max_chars:
                raise UploadRejectedError(
                    "The PDF expands to more text than allowed.",
                    detail={"reason": "too_much_text", "limit_chars": limits.max_chars},
                )
            if index == 0:
                first_page_blocks = list(blocks)
            if limits.detect_tables:
                blocks = _extract_tables(page, blocks)
            body, margin = _split_margins(blocks, page.rect.height)
            per_page.append((order_blocks(body, page.rect.width), margin, page.rect.width, used_ocr))

    repeating = repeating_margin_texts(
        [[line for b in margin for line in b.lines] for _, margin, _, _ in per_page]
    )
    body_sizes = sorted(b.font_size for body, _, _, _ in per_page for b in body if b.font_size > 0)
    body_font = body_sizes[len(body_sizes) // 2] if body_sizes else 0.0

    def is_running_text(block: RawBlock) -> bool:
        if all(fingerprint(line) in repeating or is_page_number(line) for line in block.lines):
            return True
        # Small-font, short margin text is a running head/foot even when it does not repeat
        # (e.g. the first-page header of a two-page paper).
        return 0 < block.font_size < body_font - 0.5 and len(block.text) < 200 and len(block.lines) <= 2

    warnings: list[str] = []
    text_parts: list[str] = []
    pages: list[PageInfo] = []
    blocks_out: list[Block] = []
    cursor = 0
    for page_no, (body, margin, _, used_ocr) in enumerate(per_page, start=1):
        # Margin blocks that do not repeat and are not page numbers are real content (e.g. a
        # first-page footnote); keep them at the end of the page.
        extra = [b for b in margin if not is_running_text(b)]
        page_start = cursor
        for raw in [*body, *extra]:
            text = raw.text
            if not text.strip():
                continue
            if blocks_out or text_parts:
                text_parts.append("\n\n")
                cursor += 2
            kind: BlockKind = raw.kind
            label = heading_label(text) if kind == "paragraph" and len(raw.lines) <= 2 else None
            emphasized = raw.bold or raw.font_size > body_font + 0.5
            if label is not None and (label != "other" or emphasized):
                kind = "heading"
            elif kind == "paragraph" and _CAPTION_RE.match(text):
                kind = "caption"
            blocks_out.append(Block(kind, text, page_no, cursor, cursor + len(text), caption=raw.caption))
            text_parts.append(text)
            cursor += len(text)
        pages.append(PageInfo(page_no, page_start, cursor, ocr=used_ocr))
        if used_ocr:
            warnings.append(f"page {page_no} extracted with OCR")
    full_text = "".join(text_parts)
    if not full_text.strip():
        raise UploadRejectedError(
            "No extractable text (scanned PDF without OCR?).", detail={"reason": "no_text"}
        )
    sections = _detect_sections(blocks_out, len(full_text))
    labelled = [_with_section(b, sections) for b in blocks_out]
    metadata = extract_metadata(first_page_blocks, labelled, pdf_meta)
    return ParsedDocument(
        text=full_text,
        pages=pages,
        blocks=labelled,
        sections=sections,
        metadata=metadata,
        content_hash=hashlib.sha256(data).hexdigest(),
        warnings=warnings,
    )


def _detect_sections(blocks: list[Block], text_len: int) -> list[Section]:
    sections: list[Section] = []
    current_label: SectionLabel = "front_matter"
    current_title = "Front matter"
    current_start = 0
    for block in blocks:
        if block.kind != "heading":
            continue
        label = heading_label(block.text)
        if label is None:
            continue
        if current_label == "references" and label != "appendix":
            continue
        if block.start > current_start:
            sections.append(Section(current_label, current_title, current_start, block.start))
        current_label, current_title, current_start = label, heading_title(block.text), block.start
    sections.append(Section(current_label, current_title, current_start, text_len))
    return sections


def _with_section(block: Block, sections: list[Section]) -> Block:
    for section in sections:
        if section.start <= block.start < section.end:
            return Block(
                block.kind, block.text, block.page, block.start, block.end, section.label, block.caption
            )
    return block
