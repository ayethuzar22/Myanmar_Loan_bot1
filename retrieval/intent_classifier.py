

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class Intent(str, Enum):
    MAX_LOAN_AMOUNT = "MAX_LOAN_AMOUNT"
    MIN_LOAN_AMOUNT = "MIN_LOAN_AMOUNT"
    INTEREST_RATE = "INTEREST_RATE"
    REQUIRED_DOCUMENTS = "REQUIRED_DOCUMENTS"
    ELIGIBILITY = "ELIGIBILITY"


# Keyword -> intent, checked as a case-insensitive substring match
# against the normalized query. Dict order matters: MAX is checked
# before MIN since a query containing "max" should never also trip a
# "min" match (it can't here, but keeping MAX first is the safer
# convention if more overlapping keywords are added later).
_INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.MAX_LOAN_AMOUNT: (
        "အများဆုံး", "အမြင့်ဆုံး", "maximum", "max",
    ),
    Intent.MIN_LOAN_AMOUNT: (
        "အနည်းဆုံး", "minimum", "min",
    ),
    Intent.INTEREST_RATE: (
        "အတိုးနှုန်း", "အတိုး", "interest", "rate",
    ),
    Intent.REQUIRED_DOCUMENTS: (
        "စာရွက်စာတမ်း", "အထောက်အထား", "document", "documents", "nrc",
    ),
    Intent.ELIGIBILITY: (
        "ချေးလို့ရလား", "လျှောက်နိုင်", "eligib", "qualify", "qualification",
    ),
}

# Topics allowed to answer each intent. A FAISS/reranker candidate whose
# `topic` is NOT in this allow-list is dropped from the pool when this
# intent is detected. These must match your knowledge/loan.json `topic`
# field exactly — update this table if you rename or add topics.
INTENT_TOPIC_ALLOWLIST: dict[Intent, tuple[str, ...]] = {
    Intent.MAX_LOAN_AMOUNT: (
        "Loan Amount Limits Overview",
        "Individual Loan Amount Limit",
        "Group Loan Amount Limit",
        "Agriculture Loan Amount Limit",
    ),
    Intent.MIN_LOAN_AMOUNT: (
        "Minimum Loan Amount",
    ),
    Intent.INTEREST_RATE: (
        "Annual Interest Rate",
        "Declining Balance Method",
        "Interest Variance",
    ),
    Intent.REQUIRED_DOCUMENTS: (
        "Basic Document Requirements",
        "NRC Requirement",
        "Household List Requirement",
    ),
    Intent.ELIGIBILITY: (
        "Eligibility Criteria Overview",
        "Citizenship Requirement",
        "Age Limit",
    ),
}


def classify_intent(q_norm: str) -> Optional[Intent]:
    """
    Return the first matching Intent for a normalized query, or None if
    no keyword matches — meaning no topic filtering should be applied
    and retrieval should proceed exactly as it does today.
    """
    q_lower = q_norm.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in q_lower:
                return intent
    return None


def get_topic(doc: Any) -> str:
    """
    Extract the `topic` string from a document, regardless of whether
    it's a plain dict (with a "topic" key) or a RetrievalResult-like
    object exposing `.document.topic` (this project's actual shape from
    retrieval/retriever.py). Returns "" if topic can't be found, which
    simply won't match any allow-list entry.
    """
    if isinstance(doc, dict):
        return doc.get("topic", "") or ""
    inner = getattr(doc, "document", doc)
    return getattr(inner, "topic", "") or ""


def filter_by_intent(query_norm: str, documents: list) -> tuple[list, Optional[Intent]]:
    """
    Convenience wrapper combining classify_intent() + the allow-list
    filter in one call, with the fallback-to-unfiltered safety rule
    built in.

    Returns (filtered_documents, detected_intent). `detected_intent` is
    None if no keyword matched (in which case `filtered_documents` is
    just `documents`, unchanged). If an intent WAS detected but its
    allow-list matched none of the candidates, `filtered_documents`
    falls back to the original unfiltered `documents` rather than an
    empty list — so a topic-name mismatch degrades gracefully to
    today's behavior instead of returning no context at all.
    """
    intent = classify_intent(query_norm)
    if intent is None:
        return documents, None

    allowed_topics = INTENT_TOPIC_ALLOWLIST.get(intent, ())
    filtered = [doc for doc in documents if get_topic(doc) in allowed_topics]

    if not filtered:
        return documents, intent

    return filtered, intent