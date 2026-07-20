"""
retrieval/retriever.py — Retriever class.

Coordinates EmbeddingEngine + FAISSIndex with a similarity threshold gate.
Moved verbatim from rag1.py.
"""
from __future__ import annotations
from config import FAISS_TOP_K, SIMILARITY_THRESHOLD, log
import re
CATEGORY_SIGNAL_TERMS = {
    "individual_loan": [
        "တစ်ဦးတည်း",
        "individual",
        "single loan",
    ],

    "group_loan": [
        "အဖွဲ့",
        "group loan",
    ],

    "agriculture": [
        "စိုက်ပျိုး",
        "farmer",
        "လယ်",
        "တောင်သူ",
        "စပါး",
    ],

    "business": [
        "စီးပွားရေး",
        "shop",
        "msme",
        "business",
    ],
}
def calculate_business_score(query: str, result) -> float:

    score = result.score

    doc = result.document

    q = query.lower()

    # -----------------------
    # keyword bonus
    # -----------------------

    for kw in doc.keywords:
        if kw.lower() in q:
            score += 0.25

    for alias in doc.aliases:
        if alias.lower() in q:
            score += 0.30

    # -----------------------
    # category bonus
    # -----------------------

    category = doc.category.lower()

    for cat, words in CATEGORY_SIGNAL_TERMS.items():

        if cat in category:

            for w in words:

                if w.lower() in q:

                    score += 0.50
                    break

    # -----------------------
    # priority bonus
    # -----------------------

    score += getattr(doc, "priority", 0) * 0.05

    return score
from embeddings.embedding_engine import EmbeddingEngine
from models.retrieval_result import RetrievalResult
from vectorstore.faiss_index import FAISSIndex
from utils.text_utils import clean_text

def keyword_score(query:str, doc)->float:

    q = query.lower()

    score = 0


    for word in doc.keywords:
        if word.lower() in q:
            score += 0.15


    for alias in doc.aliases:
        if alias.lower() in q:
            score += 0.20


    return min(score,0.5)



def intent_score(query:str, doc)->float:

    q=query.lower()


    score=0


    individual=[
        "တစ်ဦး",
        "individual",
        "တစ်ယောက်",
        "သိန်း"
    ]


    agriculture=[
        "လယ်",
        "စိုက်",
        "တောင်သူ",
        "farmer",
        "crop"
    ]


    if any(x in q for x in individual):

        if "individual" in doc.category.lower():

            score+=0.4


    if any(x in q for x in agriculture):

        if "agriculture" in doc.category.lower():

            score+=0.4


    return score

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
        candidates = [
            r for r in all_results
            if r.score >= self.threshold
        ]

        ranked = sorted(
            candidates,
            key=lambda r:
            calculate_business_score(query, r),
            reverse=True
        )

        above = ranked
        best        = above[0].score if above else (all_results[0].score if all_results else 0.0)
        log.info(
            "Retriever: %d/%d above threshold=%.2f best=%.3f",
            len(above), len(all_results), self.threshold, best,
        )
        return above