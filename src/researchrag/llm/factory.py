"""Build LLM clients from configuration."""

from __future__ import annotations

from researchrag.config.pipeline import GenerationConfig
from researchrag.llm.base import LLMClient, RetryingLLM


def build_llm(config: GenerationConfig, model: str | None = None) -> RetryingLLM:
    """Create the configured provider wrapped with bounded retries.

    ``model`` overrides ``config.model`` (used for judge models).
    """
    name = model or config.model
    inner: LLMClient
    if config.provider == "mock":
        from researchrag.llm.mock import MockLLM  # noqa: PLC0415

        inner = MockLLM(config.mock, model=name)
    elif config.provider in {"openai", "openai_compatible"}:
        from researchrag.llm.openai_llm import OpenAIChatLLM  # noqa: PLC0415

        compatible = config.provider == "openai_compatible"
        inner = OpenAIChatLLM(
            name,
            base_url=config.base_url,
            compatible=compatible,
            provider="openai" if not compatible else "compat",
        )
    else:
        from researchrag.llm.gemini_llm import GeminiLLM  # noqa: PLC0415

        inner = GeminiLLM(name)
    return RetryingLLM(inner, max_retries=config.max_retries)
