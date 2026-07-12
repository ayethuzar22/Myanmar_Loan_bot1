"""
vectorstore/faiss_index.py — FAISSIndex class.

Thread-safe manager for a FAISS IndexFlatIP (inner-product / cosine).
Moved verbatim from rag1.py. _sha256_file is a private helper used only
by this class (change detection for loan.json), so it lives here rather
than in a generic utils module.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from typing import Optional

import faiss
import numpy as np

from config import (
    ARTIFACTS_DIR,
    CHUNKS_PATH,
    EMBED_CACHE_PATH,
    HASH_CACHE_PATH,
    INDEX_PATH,
    RAW_JSON_PATH,
    log,
)
from embeddings.embedding_engine import EmbeddingEngine
from models.loan_document import LoanDocument
from models.retrieval_result import RetrievalResult


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 of a file.  Used for loan.json change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


class FAISSIndex:
    """
    Thread-safe manager for a FAISS IndexFlatIP (inner-product / cosine).

    Persistence strategy:
      build() — writes index + chunks + SHA-256 of loan.json to disk.
      load()  — reads index + chunks from disk.
      needs_rebuild() — compares stored SHA-256 with current file hash.

    Thread-safety:
      An RLock guards _index and _chunks.  search() acquires a read-consistent
      snapshot of both pointers; build() swaps them atomically under lock.
    """

    def __init__(
        self,
        index_path:  str = INDEX_PATH,
        chunks_path: str = CHUNKS_PATH,
        embed_cache: str = EMBED_CACHE_PATH,
        hash_cache:  str = HASH_CACHE_PATH,
    ) -> None:
        self.index_path:  str = index_path
        self.chunks_path: str = chunks_path
        self.embed_cache: str = embed_cache
        self.hash_cache:  str = hash_cache
        self._index:  Optional[faiss.Index]        = None
        self._chunks: Optional[list[LoanDocument]] = None
        self._lock:   threading.RLock              = threading.RLock()

    def needs_rebuild(self, json_path: str = RAW_JSON_PATH) -> bool:
        """True when artifacts are missing or loan.json has changed."""
        if not all(
            os.path.exists(p)
            for p in (self.index_path, self.chunks_path, self.hash_cache)
        ):
            return True
        try:
            with open(self.hash_cache, "r") as fh:
                return fh.read().strip() != _sha256_file(json_path)
        except OSError:
            return True

    def build(
        self,
        documents: list[LoanDocument],
        engine: EmbeddingEngine,
        json_path: str = RAW_JSON_PATH,
    ) -> None:
        """
        Embed all documents, build IndexFlatIP, persist artifacts, and
        swap in-memory pointers atomically.
        """
        if not documents:
            raise ValueError("FAISSIndex.build: document list is empty.")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        vecs = engine.encode([doc.semantic_text for doc in documents])
        faiss.normalize_L2(vecs)  # safety net — encode() already normalises

        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        faiss.write_index(index, self.index_path)
        with open(self.chunks_path, "wb") as fh:
            pickle.dump(documents, fh, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(self.embed_cache, vecs)

        try:
            with open(self.hash_cache, "w") as fh:
                fh.write(_sha256_file(json_path))
        except OSError as exc:
            log.warning("FAISSIndex.build: hash cache write failed — %s", exc)

        with self._lock:
            self._index  = index
            self._chunks = documents

        log.info(
            "FAISSIndex: built %d vectors dim=%d from %d documents.",
            index.ntotal, dim, len(documents),
        )

    def load(self) -> None:
        """
        Load index and chunks from disk.

        Raises:
            FileNotFoundError: When artifact files are absent.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                "FAISS artifacts not found.  Run: python rag1.py --build"
            )
        t0    = time.perf_counter()
        index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "rb") as fh:
            chunks: list[LoanDocument] = pickle.load(fh)
        with self._lock:
            self._index  = index
            self._chunks = chunks
        log.info(
            "FAISSIndex: loaded %d vectors in %.3fs.",
            index.ntotal, time.perf_counter() - t0,
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Search for top_k nearest neighbours.

        Args:
            query_vec: Shape (1, D) float32, L2-normalised.
            top_k:     Number of candidates to retrieve.

        Returns:
            List of RetrievalResult sorted descending by score.
            Returns [] when index is not yet loaded.
        """
        with self._lock:
            if self._index is None or self._chunks is None:
                log.error("FAISSIndex.search: index not loaded.")
                return []
            t0 = time.perf_counter()
            scores, indices = self._index.search(query_vec, top_k)
            log.debug("FAISSIndex: search %.4fs", time.perf_counter() - t0)
            results: list[RetrievalResult] = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:
                    results.append(
                        RetrievalResult(
                            document=self._chunks[int(idx)],
                            score=float(score),
                            rank=rank + 1,
                        )
                    )
        return results