"""Tokenizer abstraction used for chunk budgets and context packing.

``simple`` is a deterministic regex tokenizer that approximates BPE token counts without
any download; ``tiktoken:<encoding>`` uses tiktoken when installed. Chunkers only ever ask
for *counts* of whitespace-delimited units, so offsets always map back to the source text
regardless of which tokenizer is configured.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Protocol

from researchrag.errors import ConfigurationError

_PIECE_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer(Protocol):
    """Counts tokens in a string."""

    name: str

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class SimpleTokenizer:
    """Regex tokenizer: words and punctuation; long words count as several tokens (~6 chars each)."""

    name = "simple"

    def count(self, text: str) -> int:
        """Return an approximate BPE-like token count."""
        total = 0
        for piece in _PIECE_RE.findall(text):
            total += max(1, math.ceil(len(piece) / 6)) if len(piece) > 8 else 1
        return total


class TiktokenTokenizer:
    """Adapter around a tiktoken encoding."""

    def __init__(self, encoding: str) -> None:
        try:
            import tiktoken  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ConfigurationError(
                "tiktoken is not installed; pip install 'researchrag[tiktoken]'"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        """Return the exact token count for the encoding."""
        return len(self._enc.encode(text, disallowed_special=()))


@lru_cache(maxsize=8)
def get_tokenizer(spec: str = "simple") -> Tokenizer:
    """Return a tokenizer for a spec such as ``simple`` or ``tiktoken:cl100k_base``."""
    if spec == "simple":
        return SimpleTokenizer()
    if spec.startswith("tiktoken:"):
        return TiktokenTokenizer(spec.split(":", 1)[1])
    raise ConfigurationError(f"Unknown tokenizer spec: {spec}")
