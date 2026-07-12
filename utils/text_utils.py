"""
utils/text_utils.py — Generic, domain-agnostic text processing helpers.

Moved verbatim from rag1.py's "PURE UTILITIES" section. These functions
know nothing about loans specifically; they operate on plain strings.
"""

from __future__ import annotations

from config import (
    _LAUGH_EMOJI,
    _RE_COLLAPSE_WS,
    _RE_MYANMAR,
    _RE_STRIP_PUNCT,
    _RE_STRIP_PUNCT_KEEP_MYANMAR,
    _RE_WORDS,
    _THANK_WORDS_LONG,
    _THANK_WORDS_SHORT,
    _AFFIRMATIVE_LONG,
    _AFFIRMATIVE_SHORT,
    _NEGATIVE_LONG,
    _NEGATIVE_SHORT,
)


import unicodedata


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation (preserving Myanmar U+1000–U+109F),
    and collapse whitespace.  Returns empty string for falsy input.
    """
    if not text:
        return ""
    text = _RE_STRIP_PUNCT.sub(" ", text.lower().strip())
    return _RE_COLLAPSE_WS.sub(" ", text).strip()


def normalize_query(text: str) -> str:
    """
    Unicode-normalize, lowercase, strip punctuation (preserving Myanmar
    script), and collapse whitespace to single spaces.

    FIX #4: previously this deleted ALL whitespace (re.sub(r"\\s+", "",
    text)), which silently broke every multi-word keyword phrase in sets
    like BORROW_INTENT_KEYWORDS (e.g. "can i borrow" -> "caniborrow",
    "want to apply" -> "wanttoapply" — neither could ever match again).
    It now collapses repeated whitespace to a single space instead.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _RE_STRIP_PUNCT_KEEP_MYANMAR.sub(" ", text)
    text = _RE_COLLAPSE_WS.sub(" ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return ``"my"`` when Myanmar codepoints are present, else ``"en"``."""
    return "my" if _RE_MYANMAR.search(text) else "en"


def contains_any(haystack: str, needles: frozenset[str]) -> bool:
    """Return True if any needle is a substring of haystack."""
    return any(needle in haystack for needle in needles)


def is_thanks(q: str) -> bool:
    """
    FIX #2: long thank-you phrases are still matched as substrings (safe,
    since they're long enough not to appear accidentally inside unrelated
    words). Short tokens ("ty", "thx") are matched as WHOLE WORDS ONLY,
    so "ty" no longer falsely matches inside "types", "duty", "safety",
    "quantity", "specialty", etc.
    """
    if contains_any(q, _THANK_WORDS_LONG):
        return True
    words = _RE_WORDS.findall(q)
    return any(w in _THANK_WORDS_SHORT for w in words)


def contains_laugh_emoji(raw_text: str) -> bool:
    """
    FIX #3: checked against the RAW (pre-normalization) query, since
    normalize_query() strips emoji characters (they are neither \\w, \\s,
    nor in the Myanmar Unicode block). Checking post-normalization text
    meant this could never match.
    """
    return any(e in raw_text for e in _LAUGH_EMOJI)

def is_affirmative(q: str) -> bool:
    """Word-boundary check, same pattern as is_thanks — avoids '1' falsely
    matching inside '12' or similar."""
    if contains_any(q, _AFFIRMATIVE_LONG):
        return True
    words = _RE_WORDS.findall(q)
    return any(w in _AFFIRMATIVE_SHORT for w in words)


def is_negative(q: str) -> bool:
    if contains_any(q, _NEGATIVE_LONG):
        return True
    words = _RE_WORDS.findall(q)
    return any(w in _NEGATIVE_SHORT for w in words)