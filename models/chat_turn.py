"""
models/chat_turn.py — ChatTurn dataclass.

Single conversation turn for history injection into the prompt.
Moved verbatim from rag1.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """Single conversation turn for history injection into the prompt."""

    role:    str   # "user" | "assistant"
    content: str