"""Structured, citation-grounded answer generation with repair, bounded retries and fallback."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from researchrag.config.pipeline import GenerationConfig
from researchrag.errors import ResearchRAGError
from researchrag.generation.json_repair import parse_json_object
from researchrag.generation.packing import Passage
from researchrag.generation.prompts import PromptPassage, answer_messages
from researchrag.generation.schema import RawAnswer
from researchrag.llm.base import LLMClient, LLMRequest, Message

TokenSink = Callable[[str], Awaitable[None]]
_REPAIR_PROMPT = (
    "That was not a valid JSON object matching the schema. Reply again with only the JSON object."
)
GENERATION_FAILED = "The answer could not be generated reliably; no claims are returned."


@dataclass
class GenerationOutcome:
    """The parsed answer plus diagnostics for the trace."""

    answer: RawAnswer
    raw_text: str
    attempts: int
    repaired: bool = False
    failed: bool = False
    errors: list[str] = field(default_factory=list)
    prompt: list[Message] = field(default_factory=list)


class AnswerStreamExtractor:
    """Incrementally decodes the ``"answer"`` string field from a streamed JSON object."""

    def __init__(self) -> None:
        self.buffer = ""
        self._pos: int | None = None
        self._done = False

    def feed(self, chunk: str) -> str:
        """Add raw model text; return newly decoded characters of the answer field."""
        self.buffer += chunk
        if self._done:
            return ""
        if self._pos is None:
            key = self.buffer.find('"answer"')
            if key < 0:
                return ""
            colon = self.buffer.find(":", key + 8)
            if colon < 0:
                return ""
            quote = self.buffer.find('"', colon + 1)
            if quote < 0:
                return ""
            self._pos = quote + 1
        out: list[str] = []
        i = self._pos
        while i < len(self.buffer):
            ch = self.buffer[i]
            if ch == "\\":
                if i + 1 >= len(self.buffer):
                    break
                nxt = self.buffer[i + 1]
                if nxt == "u":
                    if i + 6 > len(self.buffer):
                        break
                    try:
                        out.append(chr(int(self.buffer[i + 2 : i + 6], 16)))
                    except ValueError:
                        out.append("?")
                    i += 6
                    continue
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/"}.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                self._done = True
                i += 1
                break
            out.append(ch)
            i += 1
        self._pos = i
        return "".join(out)


def to_prompt_passages(passages: list[Passage]) -> list[PromptPassage]:
    """Convert packed passages to prompt passages."""
    return [PromptPassage(p.number, p.text, p.title, p.page_start, p.section, p.suspicious) for p in passages]


class AnswerGenerator:
    """Calls the LLM for a structured answer and validates it."""

    def __init__(self, llm: LLMClient, config: GenerationConfig) -> None:
        self.llm = llm
        self.config = config

    def _request(self, messages: list[Message]) -> LLMRequest:
        return LLMRequest(
            messages,
            task="answer",
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            timeout_s=self.config.timeout_s,
        )

    @staticmethod
    def parse(text: str) -> tuple[RawAnswer, bool]:
        """Parse model text into :class:`RawAnswer`; returns ``(answer, repaired)``."""
        try:
            data = json.loads(text)
            repaired = False
        except json.JSONDecodeError:
            data = parse_json_object(text)
            repaired = True
        if not isinstance(data, dict):
            raise ValueError("answer is not a JSON object")
        return RawAnswer.model_validate(data), repaired

    async def generate(
        self,
        question: str,
        passages: list[Passage],
        canary: str,
        *,
        feedback: str | None = None,
        on_token: TokenSink | None = None,
    ) -> GenerationOutcome:
        """Generate an answer; malformed output is repaired, then retried, then replaced by a fallback."""
        base = answer_messages(question, to_prompt_passages(passages), canary, feedback)
        messages = list(base)
        errors: list[str] = []
        attempts = 0
        raw_text = ""
        for attempt in range(self.config.max_retries + 1):
            attempts = attempt + 1
            try:
                if on_token is not None and attempt == 0:
                    raw_text = await self._stream(messages, on_token)
                else:
                    raw_text = (await self.llm.complete(self._request(messages))).text
            except ResearchRAGError as exc:
                errors.append(f"{exc.code}: {exc.message}")
                break
            try:
                answer, repaired = self.parse(raw_text)
            except (ValueError, ValidationError) as exc:
                errors.append(f"malformed_output: {str(exc)[:200]}")
                messages = [
                    *base,
                    Message("assistant", raw_text[:4000]),
                    Message(
                        "user",
                        _REPAIR_PROMPT,
                    ),
                ]
                continue
            return GenerationOutcome(answer, raw_text, attempts, repaired, False, errors, base)
        fallback = RawAnswer(answer="", abstained=True, abstain_reason=GENERATION_FAILED, confidence=0.0)
        return GenerationOutcome(fallback, raw_text, attempts, False, True, errors, base)

    async def _stream(self, messages: list[Message], on_token: TokenSink) -> str:
        extractor = AnswerStreamExtractor()
        async for event in self.llm.stream(self._request(messages)):
            delta = getattr(event, "delta", "")
            if delta:
                visible = extractor.feed(delta)
                if visible:
                    await on_token(visible)
        return extractor.buffer
