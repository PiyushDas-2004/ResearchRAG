"""Data structures produced by PDF parsing and consumed by chunkers and storage."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

BlockKind = Literal["heading", "paragraph", "table", "caption"]
SectionLabel = Literal[
    "front_matter",
    "abstract",
    "introduction",
    "related_work",
    "background",
    "methods",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "acknowledgements",
    "references",
    "appendix",
    "other",
]
ChunkKind = Literal["chunk", "parent", "child"]


class DocumentMetadata(BaseModel):
    """Bibliographic metadata, extracted heuristically and editable through the API."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None


@dataclass(frozen=True)
class Block:
    """A layout block of cleaned text with document-level character offsets."""

    kind: BlockKind
    text: str
    page: int
    start: int
    end: int
    section: SectionLabel = "other"
    caption: str | None = None


@dataclass(frozen=True)
class Section:
    """A detected section spanning ``[start, end)`` of the document text."""

    label: SectionLabel
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class PageInfo:
    """Location of one page inside the document text."""

    number: int
    start: int
    end: int
    ocr: bool = False


@dataclass
class ParsedDocument:
    """The cleaned, layout-ordered text of a PDF plus structure and metadata."""

    text: str
    pages: list[PageInfo]
    blocks: list[Block]
    sections: list[Section]
    metadata: DocumentMetadata
    content_hash: str
    warnings: list[str] = field(default_factory=list)

    def page_at(self, offset: int) -> int:
        """Return the 1-based page number containing a document offset."""
        starts = [p.start for p in self.pages]
        idx = bisect.bisect_right(starts, offset) - 1
        return self.pages[max(idx, 0)].number

    def section_at(self, offset: int) -> SectionLabel:
        """Return the section label containing a document offset."""
        for section in self.sections:
            if section.start <= offset < section.end:
                return section.label
        return "other"


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it is persisted. ``text == document.text[start:end]`` always holds."""

    index: int
    text: str
    start: int
    end: int
    page_start: int
    page_end: int
    section: SectionLabel
    token_count: int
    kind: ChunkKind = "chunk"
    parent_index: int | None = None

    @property
    def retrievable(self) -> bool:
        """Parents are context-only; references are kept for metadata but not retrieved by default."""
        return self.kind != "parent"
