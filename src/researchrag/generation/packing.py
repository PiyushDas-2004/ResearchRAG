"""Small-to-big expansion and token-budget context packing."""

from __future__ import annotations

from dataclasses import dataclass, field

from researchrag.ingestion.tokenizer import Tokenizer
from researchrag.retrieval.types import Candidate, ParentPassage, page_of


@dataclass
class Passage:
    """A passage sent to the model; ``number`` is its citation marker."""

    number: int
    chunk_id: str
    document_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    section: str
    score: float
    child_ids: list[str] = field(default_factory=list)
    page_map: tuple[tuple[int, int], ...] = ()
    suspicious: bool = False
    injection_sentences: tuple[str, ...] = ()

    def page_of(self, relative_offset: int) -> int:
        """Page number of an offset inside the passage text."""
        return page_of(self.page_start, self.page_map, relative_offset)


def expand_to_parents(candidates: list[Candidate], parents: dict[str, ParentPassage]) -> list[Passage]:
    """Replace retrieved children by their parent passages, de-duplicating shared parents.

    Ranking order is preserved (a parent takes the rank of its best child). Candidates
    without a parent are used as-is.
    """
    out: list[Passage] = []
    index: dict[str, Passage] = {}
    for cand in candidates:
        parent = parents.get(cand.parent_id) if cand.parent_id else None
        key = parent.chunk_id if parent else cand.chunk_id
        if key in index:
            index[key].child_ids.append(cand.chunk_id)
            continue
        if parent is not None:
            passage = Passage(
                0, parent.chunk_id, parent.document_id, cand.title, parent.text, parent.page_start,
                parent.page_end, parent.section, cand.score, [cand.chunk_id], parent.page_map,
            )  # fmt: skip
        else:
            passage = Passage(
                0, cand.chunk_id, cand.document_id, cand.title, cand.text, cand.page_start,
                cand.page_end, cand.section, cand.score, [cand.chunk_id], cand.page_map,
            )  # fmt: skip
        index[key] = passage
        out.append(passage)
    return out


def _overlaps(a: Passage, b: Passage) -> bool:
    if a.document_id != b.document_id:
        return False
    shorter, longer = (a.text, b.text) if len(a.text) <= len(b.text) else (b.text, a.text)
    return shorter in longer


def pack_context(passages: list[Passage], tokenizer: Tokenizer, budget: int) -> list[Passage]:
    """Greedily keep the best passages that fit in ``budget`` tokens; number them 1..n.

    Passages fully contained in an already-kept passage of the same document are dropped.
    A passage that does not fit is skipped (smaller later ones may still fit).
    """
    kept: list[Passage] = []
    used = 0
    for passage in passages:
        if any(_overlaps(passage, k) for k in kept):
            continue
        cost = tokenizer.count(passage.text) + 24  # element tags and attributes
        if used + cost > budget:
            continue
        kept.append(passage)
        used += cost
    for number, passage in enumerate(kept, start=1):
        passage.number = number
    return kept
