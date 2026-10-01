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
# A sentence period sits after a letter, digit, closing bracket, or closing backtick.
_BEFORE_PERIOD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789)]}`")
_PREP = {
    "in", "on", "for", "to", "with", "of", "at", "by", "from", "as", "into",
    "over", "under", "about", "between", "through", "after", "before",
    "without", "within", "among",
}
_DET = {
    "a", "an", "the", "every", "each", "some", "any", "no",
    "this", "that", "these", "those",
}


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
    """Prefer a sentence that names the concept.

    An anaphoric opener ("In the second case", "This", …) is never the
    definition. On a joint dictionary page the sentence that names the
    concept wins over a later sentence that only makes sense with the one
    before it.
    """
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
    return plain[0] if plain else ""


def _sentence_end(flat: str) -> int | None:
    """Index just past the period that ends the sentence, or None.

    A period inside a double-backtick span is not a boundary. Outside one,
    the period must follow a letter, digit, closing bracket, or backtick,
    and must be followed by whitespace or the end of the text. That keeps
    ``c . v`` and other spaced operators intact.
    """
    i = 0
    n = len(flat)
    in_code = False
    while i < n:
        if flat.startswith("``", i):
            in_code = not in_code
            i += 2
            continue
        if (
            not in_code
            and flat[i] == "."
            and i > 0
            and flat[i - 1] in _BEFORE_PERIOD
            and (i + 1 == n or flat[i + 1].isspace())
        ):
            return i + 1
        i += 1
    return None


def first_sentence(doc: str, limit: int = 200) -> str:
    """First sentence of a docstring, joining wrapped lines with one space.

    Stops at the first blank line or at the first period that ends a sentence,
    whichever comes first. Periods inside ``double backticks`` and spaced
    operators such as ``c . v`` are not boundaries. Adds an ellipsis only when
    the sentence is cut to ``limit`` characters.
    """
    if not doc or not doc.strip():
        return ""
    head = re.split(r"\n\s*\n", doc, maxsplit=1)[0]
    flat = re.sub(r"\s+", " ", head).strip()
    end = _sentence_end(flat)
    sentence = flat[:end].strip() if end else flat
    if len(sentence) <= limit:
        return sentence
    cut = sentence[: limit - 3].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut + "..."


def _removed_run(raw: str, stripped: str) -> int:
    """Longest run of raw tokens missing from the stripped sentence."""
    raw_tokens = re.findall(r"[A-Za-z0-9]+", raw or "")
    kept = re.findall(r"[A-Za-z0-9]+", stripped or "")
    gap = 0
    longest = 0
    index = 0
    for token in raw_tokens:
        if index < len(kept) and token.lower() == kept[index].lower():
            longest = max(longest, gap)
            gap = 0
            index += 1
        else:
            gap += 1
    return max(longest, gap)


# Grammatical pairs a hole-detector must not treat as deleted markup.
_GRAMMATICAL_PAIRS = {
    ("with", "at"),
    ("that", "of"),
    ("that", "for"),
    ("each", "of"),
    ("each", "for"),
    ("any", "of"),
    ("some", "of"),
    ("this", "of"),
    ("those", "of"),
}
def is_unreadable_markup(sentence: str, raw: str | None = None) -> bool:
    """True when markup stripping left a sentence that cannot be quoted.

    Holes are a preposition or ``every`` sitting against another preposition
    (``every in``), ``in such that``, ``the form for`` where a formula was
    deleted, a preposition against a verb (``of is``), ``that and``, or a
    sentence that ends on ``that``. Ordinary collocations (``with at least``,
    ``each of``, ``echelon form for``) stay quoted. A raw page sentence is
    also unreadable when more than two adjacent tokens were dropped.
    """
    text = (sentence or "").strip()
    if not text:
        return False
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    if re.search(r"\bin such(?!\s+a\b)", low):
        return True
    if " every in " in low or re.search(r"\b(?:the|a|an) form for\b", low):
        return True
    if re.search(
        r"\b(?:in|on|of|for|to|with|by|from)\s+(?:is|are|was|were|can|has|have)\b",
        low,
    ):
        return True
    if re.search(r"\b(?:that|which)\s+and\b", low):
        return True
    if re.search(r"\b(?:in|on|of|for|to|with|by|from)\s+[,)]", low):
        return True
    tokens = re.findall(r"[A-Za-z]+|[,.]", text)
    for index, token in enumerate(tokens):
        # A capital "A" is a symbol, not the determiner.
        if not token.islower():
            continue
        word = token.lower()
        if word not in _PREP and word != "every" and word != "that":
            continue
        if index + 1 >= len(tokens):
            # "it holds that" — the equation was stripped. A cut at "by" is
            # a length cap, not a deleted symbol.
            if word == "that":
                return True
            continue
        nxt = tokens[index + 1]
        nxt_word = nxt.lower()
        if (word, nxt_word) in _GRAMMATICAL_PAIRS:
            continue
        if nxt_word in _PREP and (word in _PREP or word == "every"):
            return True
        if nxt in ",." and word in _PREP:
            return True
    if raw and _removed_run(raw, text) > 2:
        return True
    return False


def needles_for(label: str, aliases: list[str] | None = None) -> list[str]:
    """Label, aliases, and any long token (so a joint page can still match)."""
    names = [label, *(aliases or [])]
    extra: list[str] = []
    for name in names:
        for token in re.findall(r"[a-z0-9]+", (name or "").lower()):
            if len(token) >= 12 and token not in extra:
                extra.append(token)
    return names + extra
