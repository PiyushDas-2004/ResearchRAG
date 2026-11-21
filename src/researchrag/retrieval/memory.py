"""In-memory document store and search backend.

Used by unit tests and by ``researchrag eval --backend memory`` for quick, database-free runs.
Vector search is exact cosine similarity and lexical search is Okapi BM25 over the same
content terms the rest of the offline stack uses. Filter semantics mirror the SQL backend.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass

from researchrag.config.pipeline import ChunkingConfig
from researchrag.errors import NotFoundError
from researchrag.ingestion.types import ChunkDraft, ParsedDocument
from researchrag.retrieval.types import Candidate, EmbeddingVersionRef, ParentPassage, SearchFilters
from researchrag.storage.store import IngestBundle, chunk_uuid
from researchrag.textutils import content_terms


@dataclass
class _ChunkRow:
    id: str
    document_id: str
    config_hash: str
    draft: ChunkDraft
    parent_id: str | None
    terms: Counter[str]


class MemoryStore:
    """A :class:`~researchrag.storage.store.DocumentStore` and search backend held in memory."""

    def __init__(self) -> None:
        self.documents: dict[str, ParsedDocument] = {}
        self.collections: dict[str, str | None] = {}
        self.by_hash: dict[str, str] = {}
        self.chunks: dict[str, _ChunkRow] = {}
        self.chunk_sets: set[tuple[str, str]] = set()
        self.versions: dict[tuple[str, int], EmbeddingVersionRef] = {}
        self.embeddings: dict[tuple[str, int], tuple[float, ...]] = {}

    # ------------------------------------------------------------------ store
    async def find_document(self, content_hash: str) -> str | None:
        """Return the id of a document with these bytes, if any."""
        return self.by_hash.get(content_hash)

    async def register_embedding_version(
        self, provider: str, model: str, dimension: int
    ) -> EmbeddingVersionRef:
        """Get or create an embedding version."""
        key = (f"{provider}/{model}", dimension)
        if key not in self.versions:
            self.versions[key] = EmbeddingVersionRef(len(self.versions) + 1, dimension, key[0])
        return self.versions[key]

    async def chunk_set_exists(self, document_id: str, config_hash: str) -> bool:
        """True if the document already has a chunk set for this config."""
        return (document_id, config_hash) in self.chunk_sets

    async def write_document(self, bundle: IngestBundle) -> str:
        """Store a parsed document with its chunks and vectors."""
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = bundle.parsed
        self.collections[doc_id] = bundle.collection
        self.by_hash[bundle.parsed.content_hash] = doc_id
        await self.write_chunk_set(doc_id, bundle.chunking, bundle.drafts, bundle.version, bundle.vectors)
        return doc_id

    async def write_chunk_set(
        self,
        document_id: str,
        chunking: ChunkingConfig,
        drafts: list[ChunkDraft],
        version: EmbeddingVersionRef,
        vectors: dict[int, list[float]],
    ) -> int:
        """Add a chunk set to a stored document."""
        doc = self.documents.get(document_id)
        if doc is None:
            raise NotFoundError(f"Document {document_id} not found")
        config_hash = chunking.config_hash()
        ids = {d.index: str(chunk_uuid(doc.content_hash, config_hash, d.index)) for d in drafts}
        for draft in drafts:
            parent = ids[draft.parent_index] if draft.parent_index is not None else None
            row = _ChunkRow(
                ids[draft.index], document_id, config_hash, draft, parent, Counter(content_terms(draft.text))
            )
            self.chunks[row.id] = row
        for index, vec in vectors.items():
            self.embeddings[(ids[index], version.id)] = tuple(vec)
        self.chunk_sets.add((document_id, config_hash))
        return len(drafts)

    async def load_document(self, document_id: str) -> ParsedDocument:
        """Return the stored parsed document."""
        doc = self.documents.get(document_id)
        if doc is None:
            raise NotFoundError(f"Document {document_id} not found")
        return doc

    async def list_document_ids(self) -> list[str]:
        """All document ids in insertion order."""
        return list(self.documents)

    async def chunks_missing_embeddings(
        self, version: EmbeddingVersionRef, config_hash: str | None, limit: int
    ) -> list[tuple[str, str, str | None, str]]:
        """Retrievable chunks lacking an embedding for ``version``."""
        missing: list[tuple[str, str, str | None, str]] = [
            (row.id, row.draft.text, self.documents[row.document_id].metadata.title, row.draft.section)
            for row in self.chunks.values()
            if row.draft.retrievable
            and (config_hash is None or row.config_hash == config_hash)
            and (row.id, version.id) not in self.embeddings
        ]
        return sorted(missing)[:limit]

    async def write_embeddings(
        self, version: EmbeddingVersionRef, items: list[tuple[str, list[float]]]
    ) -> None:
        """Upsert embeddings."""
        for chunk_id, vec in items:
            self.embeddings[(chunk_id, version.id)] = tuple(vec)

    # ------------------------------------------------------------------ search
    def _eligible(self, row: _ChunkRow, config_hash: str, filters: SearchFilters) -> bool:
        if row.config_hash != config_hash or not row.draft.retrievable:
            return False
        if not filters.include_references and row.draft.section == "references":
            return False
        meta = self.documents[row.document_id].metadata
        if filters.document_ids and row.document_id not in filters.document_ids:
            return False
        if filters.collection and self.collections.get(row.document_id) != filters.collection:
            return False
        if filters.year_min is not None and (meta.year is None or meta.year < filters.year_min):
            return False
        if filters.year_max is not None and (meta.year is None or meta.year > filters.year_max):
            return False
        if filters.authors:
            names = [a.lower() for a in meta.authors]
            if not any(f.lower() in n for f in filters.authors for n in names):
                return False
        return True

    def _candidate(
        self, row: _ChunkRow, score: float, method: str, rank: int, emb: tuple[float, ...] | None
    ) -> Candidate:
        meta = self.documents[row.document_id].metadata
        d = row.draft
        return Candidate(
            chunk_id=row.id,
            document_id=row.document_id,
            text=d.text,
            page_start=d.page_start,
            page_end=d.page_end,
            section=d.section,
            title=meta.title or "Untitled",
            authors=tuple(meta.authors),
            year=meta.year,
            parent_id=row.parent_id,
            page_map=d.page_map,
            score=score,
            scores={method: score},
            ranks={method: rank},
            embedding=emb,
        )

    async def vector_search(
        self,
        query_vector: list[float],
        *,
        version: EmbeddingVersionRef,
        chunk_config_hash: str,
        k: int,
        filters: SearchFilters,
        ef_search: int,
    ) -> list[Candidate]:
        """Exact cosine nearest neighbours."""
        qn = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        scored: list[tuple[float, str, tuple[float, ...]]] = []
        for row in self.chunks.values():
            emb = self.embeddings.get((row.id, version.id))
            if emb is None or not self._eligible(row, chunk_config_hash, filters):
                continue
            en = math.sqrt(sum(v * v for v in emb)) or 1.0
            cos = sum(a * b for a, b in zip(query_vector, emb, strict=False)) / (qn * en)
            scored.append((cos, row.id, emb))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            self._candidate(self.chunks[cid], score, "vector", i, emb)
            for i, (score, cid, emb) in enumerate(scored[:k], start=1)
        ]

    async def lexical_search(
        self, query: str, *, chunk_config_hash: str, k: int, filters: SearchFilters
    ) -> list[Candidate]:
        """Okapi BM25 (k1 = 1.2, b = 0.75) over eligible chunks."""
        q_terms = list(dict.fromkeys(content_terms(query)))
        rows = [r for r in self.chunks.values() if self._eligible(r, chunk_config_hash, filters)]
        if not q_terms or not rows:
            return []
        n = len(rows)
        avg_len = sum(sum(r.terms.values()) for r in rows) / n
        df = {t: sum(1 for r in rows if t in r.terms) for t in q_terms}
        scored: list[tuple[float, str]] = []
        for row in rows:
            length = sum(row.terms.values())
            score = 0.0
            for term in q_terms:
                tf = row.terms.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg_len))
            if score > 0:
                scored.append((score, row.id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            self._candidate(self.chunks[cid], s, "lexical", i, None)
            for i, (s, cid) in enumerate(scored[:k], 1)
        ]

    async def get_parents(self, parent_ids: list[str]) -> dict[str, ParentPassage]:
        """Fetch parent passages by id."""
        out: dict[str, ParentPassage] = {}
        for pid in parent_ids:
            row = self.chunks.get(pid)
            if row is not None:
                d = row.draft
                out[pid] = ParentPassage(
                    pid, row.document_id, d.text, d.page_start, d.page_end, d.section, d.page_map
                )
        return out
