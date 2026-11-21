"""OpenAI chat-completions adapter; also used for OpenAI-compatible servers (Ollama, vLLM)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from researchrag.errors import ConfigurationError
from researchrag.llm.base import LLMRequest, LLMResponse, StreamEvent, estimate_tokens
from researchrag.llm.openai_errors import map_openai_error


class OpenAIChatLLM:
    """Chat Completions API client.

    ``compatible=True`` targets OpenAI-compatible endpoints: it sends ``max_tokens`` instead
    of ``max_completion_tokens`` and does not request usage in stream options.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        compatible: bool = False,
        client: Any | None = None,
        provider: str = "openai",
    ) -> None:
        self.model = model
        self.provider = provider
        self.compatible = compatible
        if client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415, TID251 - adapter module
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ConfigurationError("pip install 'researchrag[openai]' to use OpenAI models") from exc
            client = AsyncOpenAI(base_url=base_url, max_retries=0) if base_url else AsyncOpenAI(max_retries=0)
        self._client: Any = client

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "timeout": request.timeout_s,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        params["max_tokens" if self.compatible else "max_completion_tokens"] = request.max_tokens
        if request.json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a full completion."""
        try:
            response = await self._client.chat.completions.create(**self._params(request))
        except Exception as exc:
            raise map_openai_error(exc) from exc
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=int(usage.prompt_tokens) if usage else estimate_tokens(request.messages),
            output_tokens=int(usage.completion_tokens) if usage else 0,
            model_id=f"{self.provider}/{self.model}",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then a usage event."""
        params = self._params(request) | {"stream": True}
        if not self.compatible:
            params["stream_options"] = {"include_usage": True}
        output_chars = 0
        usage_sent = False
        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    output_chars += len(delta)
                    yield StreamEvent(delta=delta)
                if getattr(chunk, "usage", None):
                    usage_sent = True
                    yield StreamEvent(
                        input_tokens=int(chunk.usage.prompt_tokens),
                        output_tokens=int(chunk.usage.completion_tokens),
                    )
        except Exception as exc:
            raise map_openai_error(exc) from exc
        if not usage_sent:
            yield StreamEvent(input_tokens=estimate_tokens(request.messages), output_tokens=output_chars // 4)
