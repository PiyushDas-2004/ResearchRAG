"""Heuristic bibliographic metadata extraction from the first page.

No external APIs are called. Title comes from the largest font on page one, authors from
the name-like line under it, DOI/arXiv IDs from regexes and the year from the identifiers
or the most frequent plausible year on page one. Users can correct every field via the API.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from researchrag.ingestion.types import Block, DocumentMetadata
from researchrag.textutils import collapse_whitespace

if TYPE_CHECKING:
    from researchrag.ingestion.pdf_parser import RawBlock

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+[A-Za-z0-9])")
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_NAME_PART_RE = re.compile(r"^[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3}$")


def _clean_name(raw: str) -> str:
    return collapse_whitespace(re.sub(r"[\d*†‡§¶,]+$", "", raw.strip()))


def split_authors(text: str) -> list[str]:
    """Split an author line into names; returns ``[]`` if it does not look like names."""
    parts = re.split(r",\s*|\s+and\s+|;\s*|\s+&\s+", text)
    names = [_clean_name(p) for p in parts if p.strip()]
    names = [n for n in names if n]
    if not names or not all(_NAME_PART_RE.match(n) for n in names):
        return []
    return names


def extract_metadata(
    first_page: Sequence[RawBlock], blocks: Sequence[Block], pdf_meta: Mapping[str, object]
) -> DocumentMetadata:
    """Build :class:`DocumentMetadata` from page-one layout blocks and the parsed document."""
    meta = DocumentMetadata()
    page_one_text = "\n".join(b.text for b in first_page)
    candidates = [b for b in first_page if len(b.text.split()) >= 2 and b.font_size > 0]
    title_block = max(candidates, key=lambda b: (b.font_size, -b.y0), default=None)
    if title_block is not None:
        same_font = [b for b in first_page if b.font_size > 0]
        parts = sorted(
            (
                b
                for b in same_font
                if abs(b.font_size - title_block.font_size) < 0.6 and abs(b.y0 - title_block.y0) < 80
            ),
            key=lambda b: b.y0,
        )
        meta.title = collapse_whitespace(" ".join(b.text for b in parts))
        title_block = parts[-1]
    elif isinstance(pdf_meta.get("title"), str) and str(pdf_meta["title"]).strip():
        meta.title = str(pdf_meta["title"]).strip()

    if title_block is not None:
        below = sorted((b for b in first_page if b.y0 > title_block.y0), key=lambda b: b.y0)
        for block in below[:3]:
            names = split_authors(block.text)
            if names:
                meta.authors = names
                break
    if not meta.authors and isinstance(pdf_meta.get("author"), str):
        meta.authors = split_authors(str(pdf_meta["author"]))

    early_text = page_one_text + "\n" + "\n".join(b.text for b in blocks if b.page <= 2)
    if doi := DOI_RE.search(early_text):
        meta.doi = doi.group(1).rstrip(".")
    if arxiv := ARXIV_RE.search(early_text):
        meta.arxiv_id = arxiv.group(1)

    abstract_head = next((b for b in first_page if re.match(r"^abstract\b", b.text.strip(), re.I)), None)
    front = [b for b in first_page if abstract_head is None or b.y0 < abstract_head.y0]
    front_years = [int(y) for y in YEAR_RE.findall("\n".join(b.text for b in front))]
    page_years = [int(y) for y in YEAR_RE.findall(page_one_text)]
    if front_years:
        meta.year = Counter(front_years).most_common(1)[0][0]
    elif meta.arxiv_id:
        meta.year = 2000 + int(meta.arxiv_id[:2])
    elif page_years:
        meta.year = Counter(page_years).most_common(1)[0][0]

    abstract = [
        b.text
        for b in blocks
        if b.section == "abstract"
        and b.kind == "paragraph"
        and not b.text.lower().startswith(("keywords", "index terms"))
    ]
    if abstract:
        meta.abstract = re.sub(r"\s*\b(Keywords|Index Terms)\s*[:.-].*$", "", " ".join(abstract), flags=re.S)
    else:
        inline = next((b.text for b in blocks if re.match(r"^abstract[\s.:-]+\S", b.text, re.I)), None)
        if inline:
            meta.abstract = re.sub(r"^abstract[\s.:-]+", "", inline, flags=re.I)
    return meta
