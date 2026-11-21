"""LLM-based query rewriting and multi-query expansion."""

from __future__ import annotations

from researchrag.errors import ResearchRAGError
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.prompts import rewrite_messages
from researchrag.llm.base import LLMClient, LLMRequest


class QueryRewriter:
    """Produces alternative phrasings of a query; failures degrade to the original query only."""

    def __init__(self, llm: LLMClient, n: int) -> None:
        self.llm = llm
        self.n = n

    async def expand(self, query: str) -> list[str]:
        """Return ``[query, *alternatives]`` (deduplicated, at most ``n`` alternatives)."""
        if self.n <= 0:
            return [query]
        try:
            response = await self.llm.complete(
                LLMRequest(rewrite_messages(query, self.n), task="rewrite", max_tokens=300)
            )
            alternatives = parse_json_object(response.text).get("queries", [])
        except (ResearchRAGError, ValueError):
            return [query]
        out = [query]
        for alt in alternatives:
            if isinstance(alt, str) and alt.strip() and alt.strip().lower() not in {q.lower() for q in out}:
                out.append(alt.strip())
            if len(out) > self.n:
                break
        return out
