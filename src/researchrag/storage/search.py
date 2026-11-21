"""PostgreSQL search backend: pgvector HNSW and full-text search with filters inside SQL.

Vector search orders by ``embedding::vector(D) <=> query`` restricted to one embedding
version, which matches that version's partial HNSW expression index. The version id and the
dimension are validated integers inlined into the SQL text (not bind parameters) so the
planner can prove the partial-index predicate even for generic prepared-statement plans.

Filtered ANN search: HNSW returns ``ef_search`` neighbours *before* SQL filters apply, so a
selective filter can leave fewer than ``k`` rows. On pgvector >= 0.8 we enable iterative index
scans (``hnsw.iterative_scan = relaxed_order``) and re-sort by distance; on older versions we
fall back to an exact (sequential) scan when the filtered result is short.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchrag.retrieval.types import Candidate, EmbeddingVersionRef, ParentPassage, SearchFilters
from researchrag.storage.db import pgvector_version
from researchrag.textutils import STOPWORDS

_TERM_RE = re.compile(r"[a-z0-9]+")
_SELECT = """
SELECT c.id::text AS id, c.document_id::text AS document_id, c.text, c.page_start, c.page_end,
       c.section, c.parent_id::text AS parent_id, d.title, d.authors, d.year
"""


def vector_literal(vec: list[float]) -> str:
    """Format a vector as a pgvector text literal."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


def parse_vector(raw: str) -> tuple[float, ...]:
    """Parse a pgvector text literal."""
    return tuple(float(x) for x in raw.strip("[]").split(",") if x)


def tsquery_terms(query: str) -> str:
    """Build an OR-ed ``to_tsquery`` expression from safe alphanumeric terms ('' if none)."""
    terms = [t for t in _TERM_RE.findall(query.lower()) if t not in STOPWORDS and len(t) > 1]
    return " | ".join(dict.fromkeys(terms))


def filter_sql(filters: SearchFilters) -> tuple[str, dict[str, Any]]:
    """SQL ``AND ...`` fragment and parameters for metadata filters (applied in the query)."""
    parts: list[str] = []
    params: dict[str, Any] = {}
    if filters.document_ids:
        parts.append("d.id = ANY(CAST(:doc_ids AS uuid[]))")
        params["doc_ids"] = list(filters.document_ids)
    if filters.collection:
        parts.append("d.collection = :collection")
        params["collection"] = filters.collection
    if filters.year_min is not None:
        parts.append("d.year >= :year_min")
        params["year_min"] = filters.year_min
    if filters.year_max is not None:
        parts.append("d.year <= :year_max")
        params["year_max"] = filters.year_max
    if filters.authors:
        parts.append(
            "EXISTS (SELECT 1 FROM unnest(d.authors) AS au(name) WHERE lower(au.name) LIKE ANY(:authors))"
        )
        params["authors"] = [f"%{a.lower().replace('%', '').replace('_', '')}%" for a in filters.authors]
    if not filters.include_references:
        parts.append("NOT c.is_reference")
    return "".join(f" AND {p}" for p in parts), params


def _candidate(
    row: Any, score: float, method: str, rank: int, embedding: tuple[float, ...] | None
) -> Candidate:
    return Candidate(
        chunk_id=row.id,
        document_id=row.document_id,
        text=row.text,
        page_start=row.page_start,
        page_end=row.page_end,
        section=row.section,
        title=row.title or "Untitled",
        authors=tuple(row.authors or ()),
        year=row.year,
        parent_id=row.parent_id,
        score=score,
        scores={method: score},
        ranks={method: rank},
        embedding=embedding,
    )


class PgSearchBackend:
    """:class:`~researchrag.retrieval.types.SearchBackend` over PostgreSQL + pgvector."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._iterative: bool | None = None
        self.last_plan_fallback = False

    async def _supports_iterative(self, session: AsyncSession) -> bool:
        if self._iterative is None:
            self._iterative = await pgvector_version(session) >= (0, 8)
        return self._iterative

    @staticmethod
    def vector_sql(version: EmbeddingVersionRef, filters: SearchFilters) -> tuple[str, dict[str, Any]]:
        """SQL text and parameters for a vector search (exposed for EXPLAIN tests)."""
        dim = int(version.dimension)
        vid = int(version.id)
        where, params = filter_sql(filters)
        sql = (
            _SELECT
            + f""", c.config_hash, e.embedding::text AS embedding,
       e.embedding::vector({dim}) <=> CAST(:q AS vector({dim})) AS distance
FROM chunk_embeddings e
JOIN chunks c ON c.id = e.chunk_id
JOIN documents d ON d.id = c.document_id
WHERE e.version_id = {vid} AND c.config_hash = :ch AND c.retrievable{where}
ORDER BY e.embedding::vector({dim}) <=> CAST(:q AS vector({dim}))
LIMIT :k"""  # only validated integers are interpolated; values are bound
        )
        return sql, params

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
        """Approximate nearest neighbours via HNSW (exact fallback for short filtered results)."""
        sql, params = self.vector_sql(version, filters)
        params |= {"q": vector_literal(query_vector), "ch": chunk_config_hash, "k": k}
        self.last_plan_fallback = False
        async with self._factory() as session, session.begin():
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {max(int(ef_search), int(k))}"))
            iterative = await self._supports_iterative(session)
            if iterative:
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            rows = (await session.execute(text(sql), params)).all()
            if len(rows) < k and not iterative:
                await session.execute(text("SET LOCAL enable_indexscan = off"))
                rows = (await session.execute(text(sql), params)).all()
                self.last_plan_fallback = True
        rows = sorted(rows, key=lambda r: (r.distance, r.id))
        return [
            _candidate(r, 1.0 - float(r.distance), "vector", i, parse_vector(r.embedding))
            for i, r in enumerate(rows, start=1)
        ]

    async def lexical_search(
        self, query: str, *, chunk_config_hash: str, k: int, filters: SearchFilters
    ) -> list[Candidate]:
        """Postgres full-text search (GIN on tsvector), ranked with ``ts_rank_cd``."""
        terms = tsquery_terms(query)
        if not terms:
            return []
        where, params = filter_sql(filters)
        sql = (
            _SELECT
            + f""", ts_rank_cd(c.tsv, q, 32) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN to_tsquery('english', :tsq) AS q
WHERE c.tsv @@ q AND c.config_hash = :ch AND c.retrievable{where}
ORDER BY score DESC, c.id
LIMIT :k"""  # the filter fragment contains only bound placeholders
        )
        params |= {"tsq": terms, "ch": chunk_config_hash, "k": k}
        async with self._factory() as session:
            rows = (await session.execute(text(sql), params)).all()
        return [_candidate(r, float(r.score), "lexical", i, None) for i, r in enumerate(rows, start=1)]

    async def get_parents(self, parent_ids: list[str]) -> dict[str, ParentPassage]:
        """Fetch parent passages by id."""
        if not parent_ids:
            return {}
        sql = text(
            "SELECT id::text AS id, document_id::text AS document_id, text, page_start, page_end, section "
            "FROM chunks WHERE id = ANY(CAST(:ids AS uuid[]))"
        )
        async with self._factory() as session:
            rows = (await session.execute(sql, {"ids": list(set(parent_ids))})).all()
        return {
            r.id: ParentPassage(r.id, r.document_id, r.text, r.page_start, r.page_end, r.section)
            for r in rows
        }
