"""Tolerant JSON extraction for model output.

Models sometimes wrap JSON in prose or code fences, use single quotes, leave trailing commas
or stop before closing brackets. :func:`parse_json_object` tries strict parsing first and then
applies a small set of conservative repairs; anything still unparseable raises ``ValueError``
so the caller can retry or fall back.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SINGLE_QUOTED_KEY_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:")


def _candidate_text(text: str) -> str:
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    elif "```" in text:
        text = text.split("```", 1)[1].removeprefix("json")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    return text[start:]


def _close_brackets(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_string:
        text += '"'
    return text + "".join(reversed(stack))


def _balanced_prefix(text: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``, repairing common defects."""
    candidate = _balanced_prefix(_candidate_text(text))
    attempts = [
        candidate,
        _TRAILING_COMMA_RE.sub(r"\1", candidate),
        _close_brackets(
            _TRAILING_COMMA_RE.sub(r"\1", _SINGLE_QUOTED_KEY_RE.sub(r'"\1":', candidate)).rstrip().rstrip(",")
        ),
    ]
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output is not a repairable JSON object")
