"""Section heading detection and canonical section labels."""

from __future__ import annotations

import re

from researchrag.ingestion.types import SectionLabel

_CANONICAL: list[tuple[re.Pattern[str], SectionLabel]] = [
    (re.compile(r"^abstract$"), "abstract"),
    (re.compile(r"^(introduction|overview)$"), "introduction"),
    (re.compile(r"^(related work|prior work|literature review)$"), "related_work"),
    (re.compile(r"^(background|preliminaries)$"), "background"),
    (
        re.compile(r"^(methods?|methodology|approach|materials and methods|data and methods|study design)$"),
        "methods",
    ),
    (re.compile(r"^(experiments?|experimental setup|evaluation|setup)$"), "experiments"),
    (re.compile(r"^(results?|findings|results and discussion)$"), "results"),
    (re.compile(r"^discussion$"), "discussion"),
    (re.compile(r"^(limitations?|threats to validity)$"), "limitations"),
    (re.compile(r"^(conclusions?|concluding remarks|summary)$"), "conclusion"),
    (re.compile(r"^(acknowledge?ments?)$"), "acknowledgements"),
    (re.compile(r"^(references|bibliography|works cited)$"), "references"),
    (re.compile(r"^(appendix|supplementary material)( [a-z])?$"), "appendix"),
]
_NUMBERING_RE = re.compile(r"^((\d+(\.\d+)*\.?)|([IVX]+\.)|([A-Z]\.))\s+")


def heading_label(text: str) -> SectionLabel | None:
    """Return the canonical label if ``text`` looks like a section heading, else ``None``.

    Numbered headings that are not one of the canonical names (``"3 Site Description"``)
    are still headings and get the label ``"other"``.
    """
    stripped = text.strip().rstrip(":.")
    if not stripped or len(stripped) > 80 or len(stripped.split()) > 9:
        return None
    numbered = bool(_NUMBERING_RE.match(stripped))
    name = _NUMBERING_RE.sub("", stripped).strip().lower()
    for pattern, label in _CANONICAL:
        if pattern.match(name):
            return label
    if (
        numbered
        and name
        and name[0].isalpha()
        and not name.endswith((",", ";"))
        and not re.search(r"\d", name)
    ):
        words = stripped.split()
        if len(words) <= 8 and sum(w[:1].isupper() for w in words[1:]) >= max(1, (len(words) - 1) // 2):
            return "other"
    return None


def heading_title(text: str) -> str:
    """Strip numbering from a heading: ``"2.1 Methods"`` -> ``"Methods"``."""
    return _NUMBERING_RE.sub("", text.strip()).strip().rstrip(":.")
