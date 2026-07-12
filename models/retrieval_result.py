"""
models/retrieval_result.py — RetrievalResult dataclass.

Typed retrieval hit returned by FAISSIndex.search(). Moved verbatim
from rag1.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.loan_document import LoanDocument


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Typed retrieval hit returned by FAISSIndex.search()."""

    document: LoanDocument
    score:    float
    rank:     int