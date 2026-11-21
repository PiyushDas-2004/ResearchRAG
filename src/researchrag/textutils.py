"""Small, dependency-free text utilities shared by offline components and metrics.

These functions are deliberately simple and deterministic: they back the fake embedder,
the mock LLM, the lexical claim judge and several evaluation metrics, so their behaviour
must not depend on network access or model downloads.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

_STOPWORDS_TEXT = """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    paper papers study studies authors author report reported reports describe described
    according use used using does did what which whose show shows shown
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())

NEGATIONS: frozenset[str] = frozenset({"not", "no", "never", "none", "neither", "nor", "without", "cannot"})

ANTONYMS: dict[str, str] = {
    "increase": "decrease",
    "increased": "decreased",
    "increases": "decreases",
    "higher": "lower",
    "more": "less",
    "faster": "slower",
    "larger": "smaller",
    "improved": "worsened",
    "above": "below",
    "gain": "loss",
    "rose": "fell",
    "better": "worse",
    "outperformed": "underperformed",
}
ANTONYMS.update({v: k for k, v in list(ANTONYMS.items())})

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*|\d+(?:[.,]\d+)*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_SENT_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "etc.",
    "Fig.",
    "fig.",
    "Eq.",
    "approx.",
    "vs.",
    "Dr.",
    "No.",
    "cf.",
)


def normalize_unicode(text: str) -> str:
    """NFKC-normalise text (expands ligatures such as ``ﬁ``) and unify dashes and quotes."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": " - ",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        " ": " ",
        "­": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def stem(word: str) -> str:
    """A light suffix-stripping stemmer (good enough for lexical overlap, not linguistics)."""
    w = word.lower()
    if len(w) <= 4 or w.isdigit():
        return w
    for suffix, repl in (
        ("ational", "ate"),
        ("ization", "ize"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("ations", "ate"),
        ("ation", "ate"),
        ("ments", "ment"),
        ("ities", "ity"),
        ("ness", ""),
        ("ings", ""),
        ("ing", ""),
        ("ies", "y"),
        ("ied", "y"),
        ("sses", "ss"),
        ("edly", ""),
        ("ed", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            if suffix == "s" and w.endswith(("ss", "us", "is")):
                return w
            return w[: len(w) - len(suffix)] + repl
    return w


def words(text: str) -> list[str]:
    """Lower-cased word and number tokens.

    Possessives are dropped (``model's`` -> ``model``), hyphenated compounds are split into
    their parts (``molten-salt`` -> ``molten``, ``salt``) and thousands separators are removed
    (``2,048`` -> ``2048``) so that surface variants of the same term match.
    """
    out: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower()
        if token[0].isdigit():
            out.append(token.replace(",", ""))
            continue
        token = re.sub(r"'s?$", "", token).strip("-'")
        out.extend(part for part in token.split("-") if part)
    return out


def content_terms(text: str) -> list[str]:
    """Stemmed, stopword-free terms (order and duplicates preserved)."""
    return [stem(w) for w in words(text) if w and w not in STOPWORDS and len(w) > 1]


def term_set(text: str) -> set[str]:
    """Set of content terms."""
    return set(content_terms(text))


def numbers(text: str) -> set[str]:
    """Normalised numeric literals in text (``1,200`` and ``1200`` compare equal; ``0.50`` == ``0.5``)."""
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0).replace(",", "").lstrip("+")
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        found.add(raw)
    return found


def coverage(query_terms: Iterable[str], text_terms: set[str]) -> float:
    """Fraction of distinct query terms that occur in ``text_terms`` (0 when no terms)."""
    terms = set(query_terms)
    if not terms:
        return 0.0
    return len(terms & text_terms) / len(terms)


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Span:
    """A ``[start, end)`` character span of a source string."""

    start: int
    end: int
    text: str


def split_sentences(text: str, offset: int = 0) -> list[Span]:
    """Split text into sentence spans whose offsets map back to ``text`` (shifted by ``offset``).

    Abbreviations such as ``e.g.`` and ``et al.`` are protected. The concatenation of the
    returned spans plus the whitespace between them reconstructs ``text`` exactly.
    """
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "\x00"))
    spans: list[Span] = []
    start = 0
    for match in _SENT_END_RE.finditer(protected):
        end = match.start()
        _append_span(spans, text, start, end, offset)
        start = match.end()
    _append_span(spans, text, start, len(text), offset)
    return spans


def _append_span(spans: list[Span], text: str, start: int, end: int, offset: int) -> None:
    piece = text[start:end]
    stripped_left = len(piece) - len(piece.lstrip())
    stripped_right = len(piece.rstrip())
    if stripped_right <= stripped_left:
        return
    s, e = start + stripped_left, start + stripped_right
    spans.append(Span(s + offset, e + offset, text[s:e]))


def iter_word_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` of every whitespace-delimited unit in ``text``."""
    for match in re.finditer(r"\S+", text):
        yield match.start(), match.end()


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()
