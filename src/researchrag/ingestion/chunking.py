"""Config-driven chunking strategies.

All strategies work on *spans* of the parsed document text, never on copies, so for every
chunk ``chunk.text == document.text[chunk.start:chunk.end]`` holds and citations can always
be mapped back to a page. Strategies:

* ``fixed``: windows of whitespace-delimited units with token overlap.
* ``sentence``: sentences packed up to the budget; overlap is whole sentences.
* ``section``: recursive (paragraph -> sentence -> word) packing that never crosses a
  section boundary; tables stay atomic unless they alone exceed the budget.
* ``parent_child``: section-aware parents (sent to the model) with sentence-packed children
  (embedded and retrieved) - "small-to-big" retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from researchrag.config.pipeline import ChunkingConfig
from researchrag.ingestion.tokenizer import Tokenizer, get_tokenizer
from researchrag.ingestion.types import Block, ChunkDraft, ChunkKind, ParsedDocument
from researchrag.textutils import split_sentences


@dataclass(frozen=True)
class Unit:
    """A packable span of the document text."""

    start: int
    end: int
    tokens: int
    heading: bool = False


class Chunker(Protocol):
    """Splits a parsed document into chunk drafts."""

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Return chunk drafts in document order."""
        ...


def _word_units(text: str, start: int, end: int, tok: Tokenizer, budget: int) -> list[Unit]:
    units: list[Unit] = []
    for match in re.finditer(r"\S+", text[start:end]):
        s, e = start + match.start(), start + match.end()
        n = tok.count(text[s:e])
        if n <= budget:
            units.append(Unit(s, e, n))
        else:
            units.extend(_split_long_word(text, s, e, tok, budget))
    return units


def _split_long_word(text: str, start: int, end: int, tok: Tokenizer, budget: int) -> list[Unit]:
    """Split a single over-long token run into character pieces that each fit the budget."""
    units: list[Unit] = []
    pos = start
    while pos < end:
        lo, hi = pos + 1, end
        best = pos + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if tok.count(text[pos:mid]) <= budget:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        units.append(Unit(pos, best, tok.count(text[pos:best])))
        pos = best
    return units


def _sentence_units(text: str, start: int, end: int, tok: Tokenizer, budget: int) -> list[Unit]:
    units: list[Unit] = []
    for span in split_sentences(text[start:end], offset=start):
        n = tok.count(span.text)
        if n <= budget:
            units.append(Unit(span.start, span.end, n))
        else:
            units.extend(_word_units(text, span.start, span.end, tok, budget))
    return units


def _block_units(doc: ParsedDocument, blocks: Sequence[Block], tok: Tokenizer, budget: int) -> list[Unit]:
    """Recursive units: whole blocks when they fit, else table rows / sentences, else words."""
    units: list[Unit] = []
    for block in blocks:
        n = tok.count(doc.text[block.start : block.end])
        if n <= budget:
            units.append(Unit(block.start, block.end, n, heading=block.kind == "heading"))
        elif block.kind == "table":
            pos = block.start
            for line in doc.text[block.start : block.end].split("\n"):
                line_end = pos + len(line)
                if line.strip():
                    units.extend(_sentence_units(doc.text, pos, line_end, tok, budget))
                pos = line_end + 1
        else:
            units.extend(_sentence_units(doc.text, block.start, block.end, tok, budget))
    return units


def pack(
    units: Sequence[Unit], text: str, tok: Tokenizer, budget: int, overlap: int
) -> list[tuple[int, int, int]]:
    """Greedily pack consecutive units into ``(start, end, tokens)`` windows.

    Each window holds at most ``budget`` tokens (verified on the real text). Consecutive
    windows share trailing units totalling at most ``overlap`` tokens, and every window
    advances by at least one unit, so packing always terminates and never loses a unit.
    """
    windows: list[tuple[int, int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        j = i
        total = 0
        while j < n and total + units[j].tokens <= budget:
            total += units[j].tokens
            j += 1
        j = max(j, i + 1)
        # A trailing heading belongs with the content that follows it.
        while j - 1 > i and units[j - 1].heading and j < n:
            j -= 1
        start, end = units[i].start, units[j - 1].end
        actual = tok.count(text[start:end])
        while actual > budget and j - 1 > i:
            j -= 1
            end = units[j - 1].end
            actual = tok.count(text[start:end])
        windows.append((start, end, actual))
        if j >= n:
            break
        back = j
        shared = 0
        while back - 1 > i and shared + units[back - 1].tokens <= overlap:
            shared += units[back - 1].tokens
            back -= 1
        i = back if back > i else j
    return windows


def _drafts(
    doc: ParsedDocument,
    windows: Sequence[tuple[int, int, int]],
    *,
    kind: ChunkKind,
    first_index: int,
    parent_index: int | None = None,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for offset, (start, end, tokens) in enumerate(windows):
        drafts.append(
            ChunkDraft(
                index=first_index + offset,
                text=doc.text[start:end],
                start=start,
                end=end,
                page_start=doc.page_at(start),
                page_end=doc.page_at(max(start, end - 1)),
                section=doc.section_at(start),
                token_count=tokens,
                kind=kind,
                parent_index=parent_index,
                page_map=doc.page_map(start, end),
            )
        )
    return drafts


class FixedTokenChunker:
    """Fixed token windows over the whole document, with overlap."""

    def __init__(self, chunk_tokens: int, overlap_tokens: int, tokenizer: Tokenizer) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tok = tokenizer

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Return fixed-size windows."""
        units = _word_units(doc.text, 0, len(doc.text), self.tok, self.chunk_tokens)
        windows = pack(units, doc.text, self.tok, self.chunk_tokens, self.overlap_tokens)
        return _drafts(doc, windows, kind="chunk", first_index=0)


class SentenceChunker:
    """Sentence-aware packing over each block; tables are split by row, never mid-cell."""

    def __init__(self, chunk_tokens: int, overlap_tokens: int, tokenizer: Tokenizer) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tok = tokenizer

    def units(self, doc: ParsedDocument, start: int, end: int) -> list[Unit]:
        """Sentence units for ``doc.text[start:end]``, respecting block boundaries."""
        units: list[Unit] = []
        for block in doc.blocks:
            if block.end <= start or block.start >= end:
                continue
            s, e = max(block.start, start), min(block.end, end)
            if block.kind in {"table", "heading"}:
                sub = Block(block.kind, doc.text[s:e], block.page, s, e, block.section)
                units.extend(_block_units(doc, [sub], self.tok, self.chunk_tokens))
            else:
                units.extend(_sentence_units(doc.text, s, e, self.tok, self.chunk_tokens))
        return units

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Return sentence-packed chunks."""
        units = self.units(doc, 0, len(doc.text))
        windows = pack(units, doc.text, self.tok, self.chunk_tokens, self.overlap_tokens)
        return _drafts(doc, windows, kind="chunk", first_index=0)


class SectionChunker:
    """Recursive chunking within sections: paragraphs, then sentences, then words."""

    def __init__(self, chunk_tokens: int, overlap_tokens: int, tokenizer: Tokenizer) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tok = tokenizer

    def windows(self, doc: ParsedDocument) -> list[tuple[int, int, int]]:
        """Return packed windows, never spanning two sections."""
        windows: list[tuple[int, int, int]] = []
        for section in doc.sections:
            blocks = [b for b in doc.blocks if section.start <= b.start < section.end]
            units = _block_units(doc, blocks, self.tok, self.chunk_tokens)
            windows.extend(pack(units, doc.text, self.tok, self.chunk_tokens, self.overlap_tokens))
        return windows

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Return section-aware chunks."""
        return _drafts(doc, self.windows(doc), kind="chunk", first_index=0)


class ParentChildChunker:
    """Small-to-big: section-aware parents, sentence-packed children inside each parent."""

    def __init__(
        self, child_tokens: int, overlap_tokens: int, parent_tokens: int, tokenizer: Tokenizer
    ) -> None:
        self.parents = SectionChunker(parent_tokens, 0, tokenizer)
        self.children = SentenceChunker(child_tokens, overlap_tokens, tokenizer)
        self.tok = tokenizer
        self.child_tokens = child_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: ParsedDocument) -> list[ChunkDraft]:
        """Return parents followed by their children (children reference ``parent_index``)."""
        parent_windows = self.parents.windows(doc)
        parents = _drafts(doc, parent_windows, kind="parent", first_index=0)
        drafts: list[ChunkDraft] = list(parents)
        next_index = len(parents)
        for parent in parents:
            units = self.children.units(doc, parent.start, parent.end)
            windows = pack(units, doc.text, self.tok, self.child_tokens, self.overlap_tokens)
            children = _drafts(doc, windows, kind="child", first_index=next_index, parent_index=parent.index)
            drafts.extend(children)
            next_index += len(children)
        return drafts


def build_chunker(config: ChunkingConfig) -> Chunker:
    """Create the chunker selected by ``config.strategy``."""
    tok = get_tokenizer(config.tokenizer)
    if config.strategy == "fixed":
        return FixedTokenChunker(config.chunk_tokens, config.overlap_tokens, tok)
    if config.strategy == "sentence":
        return SentenceChunker(config.chunk_tokens, config.overlap_tokens, tok)
    if config.strategy == "section":
        return SectionChunker(config.chunk_tokens, config.overlap_tokens, tok)
    return ParentChildChunker(config.chunk_tokens, config.overlap_tokens, config.parent_tokens, tok)
