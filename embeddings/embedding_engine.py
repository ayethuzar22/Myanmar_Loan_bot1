"""
embeddings/embedding_engine.py — EmbeddingEngine class.

Thread-safe SentenceTransformer (BGE-M3) wrapper. Moved verbatim from
rag1.py, including the retry-on-transient-network-failure logic added
around the first-time Hugging Face Hub download.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_BATCH_SIZE, EMBED_MODEL_NAME, EMBED_QUERY_PREFIX, log


class EmbeddingEngine:
    """
    Thread-safe SentenceTransformer wrapper with double-checked lazy init.

    The model is loaded once per process and reused across all requests.
    BGE-M3 is used by default; change EMBED_MODEL_NAME to switch globally.
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        self.model_name: str                        = model_name
        self._model: Optional[SentenceTransformer] = None
        self._lock:  threading.Lock                 = threading.Lock()

    def encode(
        self,
        texts: list[str],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Encode texts into L2-normalised float32 embeddings of shape (N, D).

        Args:
            texts:      Non-empty list of strings.
            batch_size: SentenceTransformer encode batch size.

        Raises:
            ValueError: On empty input list.
        """
        if not texts:
            raise ValueError("EmbeddingEngine.encode: texts list is empty.")
        model = self._get_model()
        t0    = time.perf_counter()
        vecs: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        ).astype("float32")
        log.info(
            "EmbeddingEngine: encoded %d text(s) in %.3fs",
            len(texts), time.perf_counter() - t0,
        )
        return vecs

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query with the BGE-M3 retrieval prefix.
        Returns shape (1, D).
        """
        return self.encode([f"{EMBED_QUERY_PREFIX}{query}"])

    def _get_model(self) -> SentenceTransformer:
        """Double-checked locking for thread-safe lazy init, with retry on
        transient network failures during the first-time HF Hub download."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                log.info("EmbeddingEngine: loading '%s' ...", self.model_name)
                last_exc: Optional[Exception] = None
                for attempt in range(1, 4):
                    try:
                        self._model = SentenceTransformer(self.model_name)
                        log.info("EmbeddingEngine: model ready.")
                        break
                    except Exception as exc:
                        last_exc = exc
                        log.warning(
                            "EmbeddingEngine: load attempt %d/3 failed (%s) — "
                            "retrying in 3s...", attempt, exc,
                        )
                        time.sleep(3)
                else:
                    log.error(
                        "EmbeddingEngine: failed to load '%s' after 3 attempts — %s. "
                        "Check network connectivity to huggingface.co. Once the "
                        "model downloads successfully once, it's cached locally "
                        "and future runs won't need network access for it.",
                        self.model_name, last_exc,
                    )
                    raise last_exc
        return self._model  # type: ignore[return-value]