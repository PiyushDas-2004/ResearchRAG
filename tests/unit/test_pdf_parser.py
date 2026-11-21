"""PDF extraction: layout order, cleaning, tables, sections, metadata and rejections."""

from __future__ import annotations

import io
import json

import pymupdf
import pytest

from researchrag.errors import UploadRejectedError
from researchrag.evaluation.synthetic import FOOTER_TEXT, paper_passages, render_paper
from researchrag.ingestion.pdf_parser import ParseLimits, RawBlock, order_blocks, parse_pdf
from researchrag.textutils import collapse_whitespace
from tests.conftest import EVAL_DIR, corpus_papers, parsed


@pytest.mark.parametrize("key", [p.key for p in corpus_papers()])
def test_metadata_and_sections_match_spec(key: str) -> None:
    paper = next(p for p in corpus_papers() if p.key == key)
    doc = parsed(key)
    assert doc.metadata.title == paper.title
    assert doc.metadata.authors == paper.authors
    assert doc.metadata.year == paper.year
    assert doc.metadata.doi == paper.doi
    assert doc.metadata.arxiv_id == paper.arxiv_id
    assert doc.metadata.abstract is not None
    assert collapse_whitespace(paper.abstract) == collapse_whitespace(doc.metadata.abstract)
    titles = [s.title for s in doc.sections if s.label not in {"front_matter", "abstract", "references"}]
    assert titles == [s.title for s in paper.sections]
    assert doc.sections[-1].label == "references"


@pytest.mark.parametrize("key", [p.key for p in corpus_papers()])
def test_every_paragraph_survives_extraction_in_order(key: str) -> None:
    paper = next(p for p in corpus_papers() if p.key == key)
    text = collapse_whitespace(parsed(key).text)
    passages = paper_passages(paper)
    cursor = 0
    for para in passages["paragraphs"]:
        idx = text.find(collapse_whitespace(para))
        assert idx >= 0, f"paragraph missing: {para[:60]}"
        assert idx >= cursor, "two-column reading order is broken"
        cursor = idx
    for row in passages["rows"]:
        assert row in text, f"table row missing: {row}"


def test_gold_evidence_is_extractable() -> None:
    for path in sorted(EVAL_DIR.glob("questions_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            question = json.loads(line)
            for ev in question["evidence"]:
                text = collapse_whitespace(parsed(ev["doc_key"]).text)
                assert collapse_whitespace(ev["passage"]) in text, question["id"]


def test_headers_footers_and_page_numbers_removed(sample_doc) -> None:  # type: ignore[no-untyped-def]
    assert FOOTER_TEXT not in sample_doc.text
    assert "Urban Thermal Environments Review (2023)" not in sample_doc.text
    for page in sample_doc.pages:
        tail = sample_doc.text[page.start : page.end].strip().split("\n")[-1]
        assert not tail.strip().isdigit()


def test_pages_and_offsets_are_consistent(sample_doc) -> None:  # type: ignore[no-untyped-def]
    assert [p.number for p in sample_doc.pages] == list(range(1, len(sample_doc.pages) + 1))
    for block in sample_doc.blocks:
        assert sample_doc.text[block.start : block.end] == block.text
        page = sample_doc.pages[block.page - 1]
        assert page.start <= block.start <= block.end <= page.end
        assert sample_doc.page_at(block.start) == block.page


def test_table_has_caption_and_rows(sample_doc) -> None:  # type: ignore[no-untyped-def]
    tables = [b for b in sample_doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].caption is not None and tables[0].caption.startswith("Table 1")
    assert "Harbourside | 742 | 0.38 | 1.12" in tables[0].text


def test_order_blocks_two_columns_with_full_width_title() -> None:
    def blk(x0: float, y0: float, x1: float, name: str) -> RawBlock:
        return RawBlock(x0, y0, x1, y0 + 10, [name], 10.0)

    blocks = [
        blk(300, 200, 560, "R1"),
        blk(50, 50, 560, "TITLE"),
        blk(50, 300, 290, "L2"),
        blk(50, 200, 290, "L1"),
        blk(300, 300, 560, "R2"),
        blk(50, 700, 560, "WIDE"),
        blk(50, 720, 290, "L3"),
    ]
    assert [b.text for b in order_blocks(blocks, 612)] == ["TITLE", "L1", "L2", "R1", "R2", "WIDE", "L3"]


def _pdf_with_text(lines: list[str], pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        y = 100.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    data: bytes = doc.tobytes()
    return data


def test_hyphenation_and_ligatures_repaired() -> None:
    data = _pdf_with_text(["The trans-", "former uses state-of-the-", "art weights for GPT-", "4 models."])
    doc = parse_pdf(data, ParseLimits(detect_tables=False))
    assert doc.text == "The transformer uses state-of-the-art weights for GPT-4 models."


def test_rejects_malformed_pdf() -> None:
    with pytest.raises(UploadRejectedError) as exc:
        parse_pdf(b"%PDF-1.7\n" + b"\x00garbage" * 100)
    assert exc.value.detail["reason"] == "malformed"


def test_rejects_encrypted_pdf() -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text")
    buf = io.BytesIO()
    doc.save(buf, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    with pytest.raises(UploadRejectedError) as exc:
        parse_pdf(buf.getvalue())
    assert exc.value.detail["reason"] == "encrypted"


def test_rejects_too_many_pages_and_too_much_text() -> None:
    data = _pdf_with_text(["hello world " * 5], pages=3)
    with pytest.raises(UploadRejectedError) as exc:
        parse_pdf(data, ParseLimits(max_pages=2))
    assert exc.value.detail["reason"] == "too_many_pages"
    with pytest.raises(UploadRejectedError) as exc:
        parse_pdf(data, ParseLimits(max_chars=50))
    assert exc.value.detail["reason"] == "too_much_text"


def test_rejects_pdf_without_text() -> None:
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(UploadRejectedError) as exc:
        parse_pdf(doc.tobytes())
    assert exc.value.detail["reason"] == "no_text"


def test_ocr_fallback_on_scanned_page() -> None:
    import shutil

    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    source = pymupdf.open(stream=render_paper(corpus_papers()[0]), filetype="pdf")
    pix = source[0].get_pixmap(dpi=150)
    scanned = pymupdf.open()
    page = scanned.new_page(width=source[0].rect.width, height=source[0].rect.height)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    data = scanned.tobytes()
    with pytest.raises(UploadRejectedError):
        parse_pdf(data, ParseLimits(ocr_enabled=False))
    try:
        doc = parse_pdf(data, ParseLimits(ocr_enabled=True, detect_tables=False))
    except UploadRejectedError:
        pytest.skip("tesseract language data not available")
    assert doc.pages[0].ocr is True
    assert "Abstract" in doc.text
