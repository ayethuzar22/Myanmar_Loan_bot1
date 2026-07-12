"""
models/loan_document.py — LoanDocument dataclass.

Immutable, hashable representation of one validated loan.json entry.
Moved verbatim from rag1.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoanDocument:
    """
    Immutable, hashable representation of one validated loan.json entry.
    frozen=True prevents accidental mutation and allows safe cross-thread sharing.
    """

    id:             int
    category:       str
    topic:          str
    language:       str
    question:       str
    aliases:        tuple[str, ...]
    keywords:       tuple[str, ...]
    answer:         str
    related_topics: tuple[str, ...]
    source:         str

    @property
    def semantic_text(self) -> str:
        """
        Single concatenated string for embedding.
        Category + Topic + Question + Aliases + Keywords + Answer yields richer
        retrieval signal than embedding the question field alone.
        """
        parts: list[str] = [
            f"Category: {self.category}",
            f"Topic: {self.topic}",
            f"Question: {self.question}",
        ]
        if self.aliases:
            parts.append(f"Aliases: {' | '.join(self.aliases)}")
        if self.keywords:
            parts.append(f"Keywords: {' '.join(self.keywords)}")
        parts.append(f"Answer: {self.answer}")
        return "\n".join(parts)