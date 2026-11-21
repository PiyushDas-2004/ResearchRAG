"""Chunking: offsets, token budgets, coverage and overlap (including property-based tests)."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from researchrag.config.pipeline import ChunkingConfig
from researchrag.ingestion.chunking import Unit, build_chunker, pack
from researchrag.ingestion.tokenizer import SimpleTokenizer, get_tokenizer
from researchrag.ingestion.types import Block, ChunkDraft, DocumentMetadata, PageInfo, ParsedDocument, Section
from tests.conftest import parsed

TOK = SimpleTokenizer()

word = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFG0123456789", min_size=1, max_size=30)
sentence = st.lists(word, min_size=1, max_size=25).map(lambda ws: " ".join(ws) + ".")
paragraph = st.lists(sentence, min_size=1, max_size=6).map(" ".join)


def make_doc(paragraphs: list[str], headings: list[bool] | None = None) -> ParsedDocument:
    """Build a single-page parsed document from paragraphs (alternating sections)."""
    text = ""
    blocks: list[Block] = []
    sections: list[Section] = []
    headings = headings or [False] * len(paragraphs)
    for para, is_heading in zip(paragraphs, headings, strict=True):
        if text:
            text += "\n\n"
        start = len(text)
        text += para
        blocks.append(Block("heading" if is_heading else "paragraph", para, 1, start, len(text)))
    cuts = [b.start for b in blocks if b.kind == "heading"]
    bounds = [0, *[c for c in cuts if c > 0], len(text)]
    for i in range(len(bounds) - 1):
        sections.append(Section("other", f"s{i}", bounds[i], bounds[i + 1]))
    return ParsedDocument(text, [PageInfo(1, 0, len(text))], blocks, sections, DocumentMetadata(), "h")


def covered(doc: ParsedDocument, chunks: list[ChunkDraft]) -> bool:
    mask = [False] * len(doc.text)
    for c in chunks:
        for i in range(c.start, c.end):
            mask[i] = True
    return all(mask[i] or ch.isspace() for i, ch in enumerate(doc.text))


def check_invariants(doc: ParsedDocument, chunks: list[ChunkDraft], budget: int, overlap: int) -> None:
    assert chunks, "non-empty documents always produce chunks"
    for c in chunks:
        assert c.text == doc.text[c.start : c.end]
        assert c.token_count == TOK.count(c.text)
        assert c.token_count <= budget
        assert c.text == c.text.strip()
    assert covered(doc, chunks), "chunking lost text"
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start > prev.start, "chunks must advance"
        if nxt.start < prev.end:
            assert TOK.count(doc.text[nxt.start : prev.end]) <= overlap, "overlap exceeds configuration"


@settings(max_examples=60, deadline=None)
@given(
    paragraphs=st.lists(paragraph, min_size=1, max_size=8),
    budget=st.integers(min_value=16, max_value=120),
    overlap_frac=st.floats(min_value=0.0, max_value=0.5),
    strategy=st.sampled_from(["fixed", "sentence", "section"]),
)
def test_chunking_properties(paragraphs: list[str], budget: int, overlap_frac: float, strategy: str) -> None:
    overlap = min(int(budget * overlap_frac), budget - 1)
    doc = make_doc(paragraphs)
    config = ChunkingConfig(strategy=strategy, chunk_tokens=budget, overlap_tokens=overlap)  # type: ignore[arg-type]
    check_invariants(doc, build_chunker(config).chunk(doc), budget, overlap)


@settings(max_examples=40, deadline=None)
@given(paragraphs=st.lists(paragraph, min_size=1, max_size=8), child=st.integers(16, 60))
def test_parent_child_properties(paragraphs: list[str], child: int) -> None:
    doc = make_doc(paragraphs)
    config = ChunkingConfig(
        strategy="parent_child", chunk_tokens=child, overlap_tokens=child // 4, parent_tokens=child * 3
    )
    chunks = build_chunker(config).chunk(doc)
    parents = [c for c in chunks if c.kind == "parent"]
    children = [c for c in chunks if c.kind == "child"]
    check_invariants(doc, parents, child * 3, 0)
    assert covered(doc, children)
    by_index = {p.index: p for p in parents}
    for c in children:
        assert c.token_count <= child
        parent = by_index[c.parent_index]  # type: ignore[index]
        assert parent.start <= c.start and c.end <= parent.end
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_section_chunker_never_crosses_sections() -> None:
    doc = make_doc(
        ["1 Intro", "alpha beta. " * 30, "2 Methods", "gamma delta. " * 30], [True, False, True, False]
    )
    chunks = build_chunker(ChunkingConfig(strategy="section", chunk_tokens=40, overlap_tokens=8)).chunk(doc)
    for c in chunks:
        sec = next(s for s in doc.sections if s.start <= c.start < s.end)
        assert c.end <= sec.end
    assert chunks[0].text.startswith("1 Intro")


def test_heading_is_not_left_dangling_at_chunk_end() -> None:
    units = [Unit(0, 5, 4), Unit(6, 10, 4, heading=True), Unit(11, 20, 4)]
    windows = pack(units, "x" * 20, TOK, budget=8, overlap=0)
    assert windows[0][1] == 5


def test_real_paper_parent_child_pages(papers) -> None:  # type: ignore[no-untyped-def]
    doc = parsed("ir-03")
    chunks = build_chunker(ChunkingConfig()).chunk(doc)
    children = [c for c in chunks if c.kind == "child"]
    assert len(children) > 10
    assert {c.page_start for c in children} == {p.number for p in doc.pages}
    assert any(c.section == "references" for c in chunks)
    assert all(c.retrievable for c in children)
    assert not any(c.retrievable for c in chunks if c.kind == "parent")


def test_tables_kept_atomic_when_they_fit() -> None:
    doc = parsed("clim-02")
    table = next(b for b in doc.blocks if b.kind == "table")
    chunks = build_chunker(ChunkingConfig(strategy="section", chunk_tokens=400, overlap_tokens=0)).chunk(doc)
    assert any(c.start <= table.start and table.end <= c.end for c in chunks)


def test_config_hash_changes_with_settings() -> None:
    a = ChunkingConfig()
    assert a.config_hash() == ChunkingConfig().config_hash()
    assert a.config_hash() != ChunkingConfig(chunk_tokens=200).config_hash()


def test_invalid_chunking_config() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(chunk_tokens=32, overlap_tokens=32)
    with pytest.raises(ValueError, match="parent_tokens"):
        ChunkingConfig(chunk_tokens=200, parent_tokens=100)


def test_tokenizers() -> None:
    assert TOK.count("Hello, world!") == 4
    assert TOK.count("supercalifragilistic") == 4
    assert get_tokenizer("simple") is get_tokenizer("simple")
    with pytest.raises(Exception, match="Unknown tokenizer"):
        get_tokenizer("nope")
    try:
        tik = get_tokenizer("tiktoken:cl100k_base")
    except Exception:
        pytest.skip("tiktoken encoding unavailable")
    assert tik.count("hello world") == 2


def test_long_word_split_fits_budget() -> None:
    doc = make_doc(["x" * 500])
    chunks = build_chunker(ChunkingConfig(strategy="fixed", chunk_tokens=16, overlap_tokens=0)).chunk(doc)
    assert "".join(c.text for c in chunks) == "x" * 500
    assert all(c.token_count <= 16 for c in chunks)
