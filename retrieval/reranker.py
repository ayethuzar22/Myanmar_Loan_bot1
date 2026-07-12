"""
retrieval/reranker.py — CrossEncoder reranking stage (BAAI/bge-reranker-base).

Sits between FAISS retrieval and the LLM: takes FAISS's top-K candidates
and re-scores each (query, candidate) pair with a cross-encoder. A
cross-encoder is slower per-pair than the bi-encoder cosine similarity
FAISS uses, but far more accurate, since it lets the model directly
attend to the query and the candidate document together instead of
comparing two independently-computed embeddings.

Design notes
------------
- The CrossEncoder model is loaded lazily (only on first real use) and
  cached as a singleton for the lifetime of the process — it is never
  reloaded per request. This mirrors the same lazy-singleton pattern
  already used elsewhere in this codebase (EmbeddingEngine, QwenClient),
  for consistency.
- CPU-only: the torch device is pinned to "cpu" explicitly, matching
  this machine's hardware (no CUDA GPU).
- Fails soft: if the reranker model can't be loaded (missing package,
  no network on first download, out of memory, etc.) or scoring throws
  for any reason, rerank() logs the failure and returns the original
  FAISS-ordered documents completely unchanged. A reranker outage can
  never crash or block the chatbot — it just silently falls back to
  pre-reranking behavior.
- Accepts `documents` as either plain dicts (with "question"/"answer"
  keys) or objects exposing a `.document` attribute with `.question` /
  `.answer` (matching this project's RetrievalResult shape from
  retrieval/retriever.py) — so it works with rag_pipeline.py's actual
  retriever output without requiring any change to the data model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

log = logging.getLogger("wonderami.rag")


def _extract_qa_text(doc: Any) -> str:
    """
    Pull "question + answer" text out of a single document, regardless
    of whether it's a plain dict or a RetrievalResult-like object with
    a nested `.document`.
    """
    if isinstance(doc, dict):
        question = doc.get("question", "") or ""
        answer = doc.get("answer", "") or ""
        return f"{question} {answer}".strip()

    # RetrievalResult-like object: doc.document.question / doc.document.answer
    inner = getattr(doc, "document", doc)
    question = getattr(inner, "question", "") or ""
    answer = getattr(inner, "answer", "") or ""
    return f"{question} {answer}".strip()


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
        """
        Re-score `documents` against `query` using the cross-encoder and
        return them sorted best-first (highest relevance score first).

        Falls back to the original input order, completely unchanged,
        if the reranker is unavailable or scoring fails for any reason.
        This method never raises.
        """
        if not documents:
            return documents

        if not self._ensure_loaded():
            log.warning("Reranker: unavailable — keeping original FAISS ranking.")
            return documents

        try:
            log.info("Reranking %d documents...", len(documents))
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