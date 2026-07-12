"""
retrieval/retriever.py — Retriever class.

Coordinates EmbeddingEngine + FAISSIndex with a similarity threshold gate.
Moved verbatim from rag1.py.
"""
from __future__ import annotations
from config import FAISS_TOP_K, SIMILARITY_THRESHOLD, log
from embeddings.embedding_engine import EmbeddingEngine
from models.retrieval_result import RetrievalResult
from vectorstore.faiss_index import FAISSIndex
class Retriever:
    """
    Coordinates EmbeddingEngine + FAISSIndex with a similarity threshold gate.
    Only results at or above ``threshold`` are returned to the pipeline.
    """
    def __init__(
        self,
        engine:    EmbeddingEngine,
        index:     FAISSIndex,
        top_k:     int   = FAISS_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.engine    = engine
        self.index     = index
        self.top_k     = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Encode query, search FAISS, apply threshold.

        Args:
            query: Sanitised user query string.

        Returns:
            Results with score >= threshold, sorted descending.
        """
        query_vec = self.engine.encode_query(query)
        all_results = self.index.search(query_vec, self.top_k)
        above       = [r for r in all_results if r.score >= self.threshold]
        best        = above[0].score if above else (all_results[0].score if all_results else 0.0)
        log.info(
            "Retriever: %d/%d above threshold=%.2f best=%.3f",
            len(above), len(all_results), self.threshold, best,
        )
        return above