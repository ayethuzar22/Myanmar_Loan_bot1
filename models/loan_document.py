

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

    priority: int = 0

    @property
    def search_text(self) -> str:
        return " ".join([
            self.category,
            self.topic,
            self.question,
            " ".join(self.aliases),
            " ".join(self.keywords),
            self.answer
        ]).lower()

    @property
    def semantic_text(self) -> str:

        parts = [
            f"Category: {self.category}",
            f"Topic: {self.topic}",
            f"Question: {self.question}",
        ]

        if self.aliases:
            parts.append(
                f"Aliases: {' | '.join(self.aliases)}"
            )

        if self.keywords:
            parts.append(
                f"Keywords: {' '.join(self.keywords)}"
            )

        parts.append(
            f"Answer: {self.answer}"
        )

        return "\n".join(parts)