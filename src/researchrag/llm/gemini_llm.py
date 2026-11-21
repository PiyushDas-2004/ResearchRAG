"""Gemini adapter using the ``google-genai`` SDK (``client.aio.models``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from researchrag.errors import ConfigurationError
from researchrag.llm.base import LLMRequest, LLMResponse, StreamEvent, estimate_tokens
from researchrag.llm.gemini_errors import map_gemini_error


class GeminiLLM:
    """Gemini generate_content client; system messages become ``system_instruction``."""

    provider = "gemini"

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        self.model = model
        if client is None:
            try:
                from google import genai  # noqa: PLC0415, TID251 - adapter module
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError("pip install 'researchrag[gemini]' to use Gemini models") from exc
            client = genai.Client()
        self._client: Any = client

    def _payload(self, request: LLMRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {"max_output_tokens": request.max_tokens}
        if system:
            config["system_instruction"] = system
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.json_mode:
            config["response_mime_type"] = "application/json"
        return contents, config

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a full completion."""
        contents, config = self._payload(request)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0)
            or estimate_tokens(request.messages),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            model_id=f"gemini/{self.model}",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then usage from the last chunk."""
        contents, config = self._payload(request)
        usage: Any = None
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            async for chunk in stream:
                if chunk.text:
                    yield StreamEvent(delta=chunk.text)
                usage = getattr(chunk, "usage_metadata", None) or usage
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        yield StreamEvent(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0)
            or estimate_tokens(request.messages),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )
