"""Live provider smoke tests: real API calls, so they cost money.

Skipped by default (``-m 'not live'`` in pytest config) and also skipped when the provider key
is absent. Run explicitly with ``pytest -m live``.
"""

from __future__ import annotations

import os

import pytest

from researchrag.config.pipeline import GenerationConfig
from researchrag.llm.base import LLMRequest, Message
from researchrag.llm.factory import build_llm

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_embeddings_and_chat() -> None:
    from researchrag.embeddings.openai_embedder import OpenAIEmbedder

    batch = await OpenAIEmbedder("text-embedding-3-small", 256).embed(["hello world"], "query")
    assert len(batch.vectors[0]) == 256
    llm = build_llm(GenerationConfig(provider="openai", model="gpt-5-mini", temperature=None))
    out = await llm.complete(
        LLMRequest([Message("user", 'Reply with the JSON object {"ok": true}.')], task="t", max_tokens=200)
    )
    assert "ok" in out.text


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_embeddings_and_chat() -> None:
    from researchrag.embeddings.gemini_embedder import GeminiEmbedder

    batch = await GeminiEmbedder("gemini-embedding-001", 256).embed(["hello world"], "document")
    assert len(batch.vectors[0]) == 256
    llm = build_llm(GenerationConfig(provider="gemini", model="gemini-flash-latest"))
    out = await llm.complete(
        LLMRequest([Message("user", 'Reply with the JSON object {"ok": true}.')], task="t", max_tokens=200)
    )
    assert "ok" in out.text
