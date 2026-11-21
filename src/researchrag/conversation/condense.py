"""Conversational follow-up condensation ("what about its limitations?" -> standalone query)."""

from __future__ import annotations

from researchrag.errors import ResearchRAGError
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.prompts import condense_messages
from researchrag.llm.base import LLMClient, LLMRequest


class Condenser:
    """Rewrites a follow-up question into a standalone one using recent history."""

    def __init__(self, llm: LLMClient, max_turns: int = 6) -> None:
        self.llm = llm
        self.max_turns = max_turns

    async def condense(self, history: list[tuple[str, str]], followup: str) -> str:
        """Return a standalone question (the follow-up itself if there is no history or on failure)."""
        recent = history[-self.max_turns :] if self.max_turns else []
        if not recent:
            return followup
        try:
            response = await self.llm.complete(
                LLMRequest(condense_messages(recent, followup), task="condense", max_tokens=200)
            )
            standalone = parse_json_object(response.text).get("standalone_question")
        except (ResearchRAGError, ValueError):
            return followup
        return standalone.strip() if isinstance(standalone, str) and standalone.strip() else followup
