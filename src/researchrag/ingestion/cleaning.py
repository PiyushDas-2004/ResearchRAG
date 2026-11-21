"""Text cleaning for extracted PDF text: hyphenation, unicode, running headers and footers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from researchrag.textutils import collapse_whitespace, normalize_unicode

_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"(\S*[A-Za-z0-9])-[ \t]*\n\s*(?=\S)")


def join_lines(lines: Sequence[str]) -> str:
    """Join the lines of a block into one paragraph, repairing end-of-line hyphenation.

    ``"trans-\\nformer"`` becomes ``"transformer"``. The hyphen is kept (without a space)
    when the continuation is upper-case or numeric (``"GPT-\\n4"``) or when the fragment is
    already a compound (``"state-of-the-\\nart"``), because those are probably real hyphens.
    """
    text = "\n".join(normalize_unicode(line).rstrip() for line in lines)
    text = _HYPHEN_BREAK_RE.sub(_rejoin, text)
    return collapse_whitespace(text)


def _rejoin(match: re.Match[str]) -> str:
    fragment = match.group(1)
    following = match.string[match.end() : match.end() + 1]
    if following.islower() and "-" not in fragment and fragment[-1].isalpha():
        return fragment
    return fragment + "-"


def fingerprint(text: str) -> str:
    """Normalise a margin line so repeating headers match across pages (digits ignored)."""
    return re.sub(r"\d+", "#", collapse_whitespace(text).lower())


def is_page_number(text: str) -> bool:
    """True for strings like ``7``, ``Page 7`` or ``7 of 12``."""
    return bool(_PAGE_NUMBER_RE.match(collapse_whitespace(text)))


def repeating_margin_texts(
    margin_texts_per_page: Sequence[Sequence[str]], min_fraction: float = 0.5
) -> set[str]:
    """Return fingerprints of margin texts that repeat on at least ``min_fraction`` of pages."""
    pages = len(margin_texts_per_page)
    if pages < 2:
        return set()
    counts: Counter[str] = Counter()
    for texts in margin_texts_per_page:
        counts.update({fingerprint(t) for t in texts})
    threshold = max(2, int(pages * min_fraction + 0.5))
    return {fp for fp, n in counts.items() if n >= threshold}
