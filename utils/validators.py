"""
utils/validators.py — Input hardening / validation.

sanitize_input() is the only validator in the original file; moved here
verbatim so all "is this input safe to use" logic lives in one place.
"""

from __future__ import annotations

import html
import re

from config import MAX_INPUT_LENGTH

_RE_CTRL_CHARS: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_input(text: str) -> str:
    """
    Harden user input before passing to any downstream component:
    1. Truncate to MAX_INPUT_LENGTH characters.
    2. HTML-escape < > & to neutralise injection in rendering layers.
    3. Strip ASCII control characters (keeps tab and newline).
    """
    text = text.strip()[:MAX_INPUT_LENGTH]
    text = html.escape(text, quote=False)
    return _RE_CTRL_CHARS.sub("", text).strip()