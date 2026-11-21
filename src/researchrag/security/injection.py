"""Indirect prompt-injection defences for untrusted document text.

Three layers (defence in depth; none is sufficient alone):

1. Input scan: passages are checked for instruction-like patterns before they reach the model.
   Depending on ``security.injection_action`` the passage is flagged, the offending sentences
   are removed (``sanitize``), or the passage is dropped.
2. Prompt structure: passages are delimited and neutralised, and the system prompt states an
   instruction hierarchy (see :mod:`researchrag.generation.prompts`).
3. Output guard: the answer is checked for the per-request canary (system-prompt leakage),
   for distinctive payload tokens taken from flagged passages, and for tell-tale phrases of an
   answer that is obeying a document instead of the user.

Pattern lists catch common and known phrasings only; novel or obfuscated injections can pass.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from researchrag.textutils import split_sentences

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|above|earlier|all|any|system|your)\b"
        r"[^.]{0,20}\b(instructions?|prompts?|rules|directions|guidelines)\b",
        r"\b(reveal|print|show|output|repeat|leak|disclose)\b[^.]{0,30}"
        r"\b(system prompt|hidden prompt|instructions|initial prompt|developer message)\b",
        r"\byou are now\b",
        r"\bnew instructions?\s*:",
        r"\b(respond|reply|answer)\s+only\s+with\b",
        r"\b(assistant|ai model|language model|llm|chatbot)\s*[,:]\s*"
        r"(please\s+)?(you must|ignore|respond|say)\b",
        r"<\s*/?\s*(system|assistant|instructions?)\s*>",
        r"\bbegin (system|hidden) prompt\b",
        r"\bdo not (cite|mention|reveal) (this|these|the sources?)\b",
    )
]
_OBEYING_RE = re.compile(
    r"\b(as instructed by the (document|passage)|my system prompt|i (have been|was) instructed to|"
    r"ignoring (the|my) previous instructions)\b",
    re.I,
)
_PAYLOAD_RE = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*[-\d])[A-Z0-9][A-Z0-9-]{5,}\b")


@dataclass(frozen=True)
class InjectionFinding:
    """Instruction-like sentences found in one passage."""

    sentences: tuple[str, ...]
    payload_tokens: tuple[str, ...]

    @property
    def found(self) -> bool:
        """True when at least one suspicious sentence was found."""
        return bool(self.sentences)


def scan_text(text: str) -> InjectionFinding:
    """Return the sentences of ``text`` that look like instructions to an AI system."""
    flagged: list[str] = []
    for span in split_sentences(text):
        if any(p.search(span.text) for p in INJECTION_PATTERNS):
            flagged.append(span.text)
    payload = sorted({tok for s in flagged for tok in _PAYLOAD_RE.findall(s)})
    return InjectionFinding(tuple(flagged), tuple(payload))


def sanitize_text(text: str, finding: InjectionFinding) -> str:
    """Remove flagged sentences, leaving a visible marker so the model knows text was withheld."""
    out = text
    for sentence in finding.sentences:
        out = out.replace(sentence, "[instruction-like text removed]")
    return out


def new_canary() -> str:
    """A random per-request canary token embedded in the system prompt."""
    return "RRCANARY-" + secrets.token_hex(6).upper()


@dataclass
class GuardResult:
    """Outcome of the output guard."""

    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no violation was detected."""
        return not self.violations


def output_guard(answer: str, *, canary: str, findings: list[InjectionFinding]) -> GuardResult:
    """Check a model answer for prompt leakage or for obeying instructions found in documents."""
    result = GuardResult()
    if canary and canary in answer:
        result.violations.append("system_prompt_leak")
    for finding in findings:
        for token in finding.payload_tokens:
            if token in answer:
                result.violations.append(f"injection_payload:{token}")
    if _OBEYING_RE.search(answer):
        result.violations.append("obeying_document_instructions")
    return result
