"""Sentence and page-title checks for concept-graph evidence.

Wikipedia is the external definition when its page title matches the concept.
An IUCr sentence is kept as a fallback, and marked related when the dictionary
page is about a neighbouring entry rather than this concept.
"""
from __future__ import annotations

import re

_STOP = {"a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "with"}
_ANAPHORA = re.compile(
    r"^(in the second case|in this case|it|this|these)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"\.(?=\s|$)")


def _tokens(text: str) -> set[str]:
    stripped = re.sub(r"\([^)]*\)", " ", text or "")
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", stripped.lower()):
        if token in _STOP or len(token) < 3:
            continue
        out.add(token)
        if token.endswith("s") and len(token) > 4:
            out.add(token[:-1])
    return out


def page_matches(page_title: str, label: str, aliases: list[str] | None = None) -> bool:
    """True when the page title is about this concept's label or an alias."""
    page = _tokens(page_title)
    if not page:
        return False
    for name in [label, *(aliases or [])]:
        name_tokens = _tokens(name)
        if not name_tokens:
            continue
        if page <= name_tokens or name_tokens <= page:
            return True
        if any(len(token) >= 5 and token in page for token in name_tokens):
            return True
    return False


def is_anaphoric(sentence: str) -> bool:
    return bool(_ANAPHORA.match((sentence or "").strip()))


def pick_sentence(candidates: list[str], needles: list[str]) -> str:
    """Prefer a sentence that names the concept, and skip an anaphoric opener
    when any other candidate exists."""
    rows = [row.strip() for row in candidates if row and row.strip()]
    if not rows:
        return ""
    named = []
    for row in rows:
        low = row.lower()
        if any(len(needle) >= 4 and needle.lower() in low for needle in needles):
            named.append(row)
    pool = named or rows
    plain = [row for row in pool if not is_anaphoric(row)]
    return (plain or pool)[0]


def first_sentence(doc: str, limit: int = 200) -> str:
    """First sentence of a docstring, joining wrapped lines with one space.

    Stops at the first blank line or at the first period that ends a sentence,
    whichever comes first. Adds an ellipsis only when the sentence is cut to
    ``limit`` characters.
    """
    if not doc or not doc.strip():
        return ""
    head = re.split(r"\n\s*\n", doc, maxsplit=1)[0]
    flat = re.sub(r"\s+", " ", head).strip()
    match = _SENTENCE_END.search(flat)
    sentence = flat[: match.end()].strip() if match else flat
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 3].rstrip() + "..."


def needles_for(label: str, aliases: list[str] | None = None) -> list[str]:
    return [label, *(aliases or [])]
