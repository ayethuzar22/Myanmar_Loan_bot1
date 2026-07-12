"""
models/rag_response.py — RAGResponse dataclass.

Structured response returned to every caller (Django view or CLI).
Moved verbatim from rag1.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGResponse:
    """Structured response returned to every caller (Django view or CLI)."""

    answer:           str
    source:           str
    matched_topic:    str   = ""
    matched_category: str   = ""
    similarity_score: float = 0.0
    confidence:       float = 1.0