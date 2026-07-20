from __future__ import annotations
import logging
import threading
from typing import Any, Optional

log = logging.getLogger("wonderami.rag")


def _extract_qa_text(doc: Any) -> str:
    """Build the text the cross-encoder actually scores against the query."""
    if isinstance(doc, dict):
        category = doc.get("category", "") or ""
        topic = doc.get("topic", "") or ""
        keywords = doc.get("keywords", []) or []
        question = doc.get("question", "") or ""
        answer = doc.get("answer", "") or ""
    else:
        inner = getattr(doc, "document", doc)
        category = getattr(inner, "category", "") or ""
        topic = getattr(inner, "topic", "") or ""
        keywords = getattr(inner, "keywords", []) or []
        question = getattr(inner, "question", "") or ""
        answer = getattr(inner, "answer", "") or ""

    keywords_str = ",".join(keywords) if keywords else ""

    return (
        f"Category: {category}\n"
        f"Topic: {topic}\n"
        f"Keywords: {keywords_str}\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    ).strip()


class Reranker:
    """
    Thread-safe wrapper around a BAAI/bge-reranker-base CrossEncoder,
    loaded lazily on first use and cached as a singleton for the process
    lifetime.

        rerank(query, documents) -> list   (same items, reordered best-first)
    """

    _MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._lock: threading.Lock = threading.Lock()
        self._load_failed: bool = False  # sticky — don't retry a hard failure every call

    # ── Public interface ─────────────────────────────────────────────────

    def rerank(self, query: str, documents: list) -> list:

        if not documents:
            return documents

        if not self._ensure_loaded():
            log.warning("Reranker: unavailable — keeping original FAISS ranking.")
            return documents

        try:
            log.info("Reranking %d documents...", len(documents))

            # IMPORTANT: pairs must be (query_text, doc_text) strings, not
            # (query_text, doc_object) — CrossEncoder.predict() scores text pairs.
            pairs = [(query, _extract_qa_text(doc)) for doc in documents]

            scores = self._model.predict(pairs)

            ranked = sorted(
                zip(documents, scores), key=lambda pair: pair[1], reverse=True
            )
            if ranked:
                log.info("Top reranked score: %.4f", ranked[0][1])

            return [doc for doc, _score in ranked]

        except Exception as exc:
            log.error(
                "Reranker: scoring failed (%s) — keeping original FAISS ranking.", exc
            )
            return documents

    # ── Internal ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                self._load()
                return True
            except Exception as exc:
                log.error(
                    "Reranker: model load failed (%s) — falling back to "
                    "original FAISS ranking for all future requests.", exc,
                )
                self._load_failed = True
                return False

    def _load(self) -> None:
        log.info("Loading reranker...")
        from sentence_transformers import CrossEncoder  # local import: no hard dep at module import time

        self._model = CrossEncoder(self._MODEL_NAME, device="cpu")
        log.info("Reranker ready.")