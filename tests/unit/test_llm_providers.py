"""LLM provider adapters with fake SDK clients, error mapping, retries and the factory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from researchrag.config.pipeline import GenerationConfig
from researchrag.errors import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    RetryableProviderError,
)
from researchrag.llm.base import LLMRequest, Message, RetryingLLM, StreamEvent, estimate_tokens
from researchrag.llm.factory import build_llm
from researchrag.llm.gemini_errors import map_gemini_error
from researchrag.llm.gemini_llm import GeminiLLM
from researchrag.llm.mock import MockLLM
from researchrag.llm.openai_errors import map_openai_error
from researchrag.llm.openai_llm import OpenAIChatLLM

REQ = LLMRequest(
    [Message("system", "sys"), Message("user", "hi"), Message("assistant", "a")], task="answer", max_tokens=50
)


class _AsyncIter:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> Any:
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


class FakeOpenAI:
    def __init__(self, usage: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.usage = usage
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5) if self.usage else None
        if kwargs.get("stream"):
            chunks = [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"a"'))], usage=None),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=": 1}"))], usage=None),
                SimpleNamespace(choices=[], usage=usage),
            ]
            return _AsyncIter(chunks)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))], usage=usage
        )


async def test_openai_chat_adapter_complete_and_stream() -> None:
    client = FakeOpenAI()
    llm = OpenAIChatLLM("gpt-x", client=client)
    out = await llm.complete(REQ)
    assert out.text == '{"a": 1}'
    assert (out.input_tokens, out.output_tokens, out.model_id) == (11, 5, "openai/gpt-x")
    params = client.calls[0]
    assert params["max_completion_tokens"] == 50
    assert params["response_format"] == {"type": "json_object"}
    assert params["temperature"] == 0.0
    events = [e async for e in llm.stream(REQ)]
    assert "".join(e.delta for e in events) == '{"a": 1}'
    assert events[-1].input_tokens == 11
    assert client.calls[1]["stream_options"] == {"include_usage": True}


async def test_openai_compatible_mode_and_missing_usage() -> None:
    client = FakeOpenAI(usage=False)
    llm = OpenAIChatLLM("llama", client=client, compatible=True, provider="compat")
    req = LLMRequest([Message("user", "x")], task="t", temperature=None, json_mode=False)
    out = await llm.complete(req)
    assert out.input_tokens == estimate_tokens(req.messages)
    assert "max_tokens" in client.calls[0]
    assert "temperature" not in client.calls[0]
    assert "response_format" not in client.calls[0]
    events = [e async for e in llm.stream(req)]
    assert events[-1].input_tokens == estimate_tokens(req.messages)
    assert "stream_options" not in client.calls[1]


def test_openai_error_mapping() -> None:
    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "soon"})

    class InternalServerError(Exception):
        status_code = 503

    class BadRequestError(Exception):
        status_code = 400

    assert isinstance(map_openai_error(APITimeoutError()), ProviderTimeoutError)
    rl = map_openai_error(RateLimitError())
    assert isinstance(rl, RateLimitedError)
    assert rl.retry_after is None
    assert isinstance(map_openai_error(InternalServerError()), RetryableProviderError)
    bad = map_openai_error(BadRequestError())
    assert type(bad) is ProviderError


async def test_openai_adapter_maps_errors() -> None:
    class Failing:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs: Any) -> Any:
            raise TimeoutError

    llm = OpenAIChatLLM("m", client=Failing())
    with pytest.raises(ProviderTimeoutError):
        await llm.complete(REQ)
    with pytest.raises(ProviderTimeoutError):
        _ = [e async for e in llm.stream(REQ)]


class FakeGeminiModels:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return SimpleNamespace(
            text='{"ok": true}',
            usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=3),
        )

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return _AsyncIter(
            [
                SimpleNamespace(text='{"ok"', usage_metadata=None),
                SimpleNamespace(
                    text=": true}",
                    usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=3),
                ),
            ]
        )


async def test_gemini_adapter() -> None:
    models = FakeGeminiModels()
    llm = GeminiLLM("gemini-x", client=SimpleNamespace(aio=SimpleNamespace(models=models)))
    out = await llm.complete(REQ)
    assert out.text == '{"ok": true}'
    assert out.input_tokens == 9
    call = models.calls[0]
    assert call["config"]["system_instruction"] == "sys"
    assert call["config"]["response_mime_type"] == "application/json"
    assert [c["role"] for c in call["contents"]] == ["user", "model"]
    events = [e async for e in llm.stream(REQ)]
    assert "".join(e.delta for e in events) == '{"ok": true}'
    assert events[-1].output_tokens == 3
    err = Exception("boom")
    err.code = 503  # type: ignore[attr-defined]
    failing = GeminiLLM("g", client=SimpleNamespace(aio=SimpleNamespace(models=FakeGeminiModels(err))))
    with pytest.raises(RetryableProviderError):
        await failing.complete(REQ)
    with pytest.raises(RetryableProviderError):
        _ = [e async for e in failing.stream(REQ)]


def test_gemini_error_mapping() -> None:
    timeout = type("ReadTimeout", (Exception,), {})()
    assert isinstance(map_gemini_error(timeout), ProviderTimeoutError)
    client = Exception("x")
    client.code = 400  # type: ignore[attr-defined]
    assert type(map_gemini_error(client)) is ProviderError


async def test_retrying_llm_stream_retries_before_first_token() -> None:
    class Flaky:
        provider, model = "x", "y"

        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: Any) -> Any:
            raise NotImplementedError

        async def stream(self, request: Any) -> Any:
            self.n += 1
            if self.n == 1:
                raise RateLimitedError("429", retry_after=0.0)
            yield StreamEvent(delta="ok")
            if self.n == 2:
                raise RetryableProviderError("mid-stream")

    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    llm = RetryingLLM(Flaky(), max_retries=3, sleep=sleep)  # type: ignore[arg-type]
    seen: list[str] = []
    with pytest.raises(RetryableProviderError, match="mid-stream"):
        async for event in llm.stream(REQ):
            seen.append(event.delta)
    assert seen == ["ok"]
    assert slept == [0.0]


async def test_retrying_llm_records_usage() -> None:
    llm = RetryingLLM(MockLLM(), max_retries=0)
    await llm.complete(LLMRequest([Message("user", "<query>q</query>")], task="rewrite"))
    _ = [e async for e in llm.stream(LLMRequest([Message("user", "<query>q</query>")], task="rewrite"))]
    assert llm.usage.calls == 2
    assert llm.usage.by_task["rewrite"] > 0
    assert llm.model_id == "mock/mock-extractive-v1"


def test_factory_builds_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-real")
    assert isinstance(build_llm(GenerationConfig()).inner, MockLLM)
    assert isinstance(build_llm(GenerationConfig(provider="openai", model="gpt-x")).inner, OpenAIChatLLM)
    compat = build_llm(
        GenerationConfig(provider="openai_compatible", model="llama", base_url="http://localhost:11434/v1")
    )
    assert compat.provider == "compat"
    assert isinstance(build_llm(GenerationConfig(provider="gemini", model="g")).inner, GeminiLLM)
