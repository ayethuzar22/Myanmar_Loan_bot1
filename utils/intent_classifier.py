"""
utils/intent_classifier.py
==========================
Production-ready hybrid intent classifier for the Myanmar Loan RAG Chatbot.

Classification pipeline
-----------------------
1. Normalise & sanitise input text.
2. Keyword / pattern fast-path  →  O(1), no model call required.
   Returns immediately when confidence ≥ KEYWORD_CONFIDENCE_THRESHOLD.
3. Embedding similarity fallback  →  cosine similarity against per-intent
   prototype sentences encoded once at construction time via EmbeddingEngine
   (BAAI/bge-m3).
4. Return (IntentLabel, confidence: float).

Supports
--------
* English (full sentences, casual / incomplete phrases)
* Burmese / Myanmar script (Unicode range U+1000–U+109F)
* Mixed-script input

Dependencies
------------
* Only the project-internal EmbeddingEngine — no new pip packages.
* numpy  (already required by any embedding stack).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoids a circular import at runtime; used only for the type annotation.
    from embeddings.embedding_engine import EmbeddingEngine  # noqa: F401

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | intent_classifier | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Intent labels
# ---------------------------------------------------------------------------
# class IntentLabel(str, Enum):
#     """All recognised user-intent categories."""
#
#     GREETING = "GREETING"
#     LOAN_INQUIRY = "LOAN_INQUIRY"
#     LOAN_APPLICATION = "LOAN_APPLICATION"
#     ELIGIBILITY_QUESTION = "ELIGIBILITY_QUESTION"
#     DOCUMENT_QUESTION = "DOCUMENT_QUESTION"
#     LOAN_CATEGORY_SELECTION = "LOAN_CATEGORY_SELECTION"
#     LOAN_AMOUNT_PROVIDED = "LOAN_AMOUNT_PROVIDED"
#     LOAN_CONFIRMATION = "LOAN_CONFIRMATION"
#     COMPLAINT = "COMPLAINT"
#     UNKNOWN = "UNKNOWN"

class IntentLabel(str, Enum):

    GREETING = "GREETING"

    LOAN_INFO = "LOAN_INFO"

    LOAN_START = "LOAN_START"

    LOAN_CATEGORY = "LOAN_CATEGORY"

    LOAN_AMOUNT = "LOAN_AMOUNT"

    LOAN_TERM = "LOAN_TERM"

    ELIGIBILITY = "ELIGIBILITY"

    DOCUMENT = "DOCUMENT"

    APPLICATION_STATUS = "APPLICATION_STATUS"

    CONFIRMATION = "CONFIRMATION"

    COMPLAINT = "COMPLAINT"

    UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Keyword / pattern catalogue
# ---------------------------------------------------------------------------
# Each entry is a list of plain substrings or compiled regex patterns.
# Matching is case-insensitive for Latin text.
# Burmese patterns are matched as-is (Unicode).
#
# Design note: keep patterns specific enough to avoid false positives, but
# broad enough to capture casual / incomplete expressions.

_KW: dict[IntentLabel, list[str | re.Pattern]] = {

    IntentLabel.GREETING: [
        # English
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "what's up", "sup",
        # Burmese
        "မင်္ဂလာပါ", "ဟဲလို", "ဟိုင်း", "နေကောင်းလား",
        "နေကောင်း", "ဟေး",
    ],

    IntentLabel.LOAN_INQUIRY: [
        # English — full & incomplete
        "loan", "borrow", "lend", "credit", "finance", "money",
        "need money", "want money", "get money", "cash",
        "can i get", "how to get", "apply for", "interest rate",
        "loan info", "loan detail", "tell me about loan",
        "what loan", "which loan", "loan type", "loan product",
        # Burmese — ချေးငွေ related
        "ချေးချင်", "ချေး", "ငွေချေး", "ချေးငွေ","ငွေလိုတယ်","ပိုက်ဆံလိုလို့","အရေးပေါ်ငွေလို","သားသမီးကျောင်းစရိတ်","ဆေးဖိုးလို","လုပ်ငန်းတိုးချဲ့မယ်"
        "ငွေလို", "ငွေလိုချင်", "ငွေရချင်",
        "ကြွေးဆပ်", "အတိုး", "ချေးငွေ အကြောင်း",
        "ချေးငွေ အချက်အလက်", "ဘယ်ချေးငွေ",
        # Casual Burmese triggers (medical, business intent that implies borrowing)
        "ဆေးကုမယ်", "ဆေးရုံ", "ကုသ",          # medical → implies loan need
        "လုပ်ငန်းလုပ်", "စီးပွားရေး", "ကုန်သွယ်",  # business → implies loan need
        "တိုးချဲ့", "ရင်းနှီး", "မြှုပ်နှံ",
    ],

    IntentLabel.LOAN_APPLICATION: [
        # English
        "apply", "application", "submit", "fill form", "register",
        "sign up", "start loan", "open loan", "request loan",
        "i want to apply", "how to apply",
        # Burmese
        "လျှောက်ထားချင်", "လျှောက်ချင်", "လျှောက်မယ်",
        "ဖောင်ဖြည့်", "တင်သွင်းချင်", "မှတ်ပုံတင်",
        "လျှောက်လွှာ", "လျှောက်ထားမည်",
    ],

    IntentLabel.ELIGIBILITY_QUESTION: [
        # English
        "eligible", "eligibility", "qualify", "qualification",
        "can i", "am i eligible", "do i qualify", "who can",
        "requirement", "criteria", "age limit", "income requirement",
        "minimum salary", "who is eligible",
        # Burmese
        "ရပါသလား", "ရမလား", "ဘယ်သူ ရ", "ခွင့်ပြုပါသလား",
        "အသက်", "ဝင်ငွေ", "လုပ်ခ", "စည်းကမ်းချက်",
        "ရည်မှန်းချက်", "ကျွန်တော် ရပါသလား",
        "ကျွန်မ ရပါသလား", "ထိုက်တန်",
    ],

    IntentLabel.DOCUMENT_QUESTION: [
        # English
        "document", "paperwork", "id card", "national id", "nrc",
        "passport", "proof", "evidence", "certificate",
        "what document", "which document", "required document",
        "form 7", "business license", "salary slip", "bank statement",
        # Burmese
        "စာရွက်စာတမ်း", "မှတ်ပုံတင်", "နိုင်ငံသားစိစစ်ရေးကတ်",
        "ဘယ် စာရွက်", "ဘာ စာရွက်", "လိုအပ်သော",
        "ကုမ္ပဏီမှတ်ပုံတင်", "ဝင်ငွေ အထောက်အထား",
        "ဘဏ်စာရင်း", "ဖောင် ၇",
    ],

    IntentLabel.LOAN_CATEGORY_SELECTION: [
        # English
        "agriculture", "farming", "paddy", "vegetable", "orchard",
        "msme", "small business", "retail", "wholesale", "manufacturing",
        "consumer", "vehicle", "housing", "personal loan",
        "agri loan", "business loan", "car loan", "home loan",
        # Burmese
        "စိုက်ပျိုးရေး", "တောင်သူ", "လယ်", "ဆန်",
        "သားငါး", "ဥယျာဉ်", "ကျေးလက်",
        "စီးပွားရေးငွေချေး", "လုပ်ငန်းငွေချေး",
        "ကား", "အိမ်", "ကုန်ပစ္စည်း", "ကုန်ထုတ်",
        "သေးငယ်သောလုပ်ငန်း",
    ],

    IntentLabel.LOAN_AMOUNT_PROVIDED: [
        # English numeric patterns  e.g. "500000", "5 lakh", "5 million"
        re.compile(r"\b\d[\d,\.]*\s*(kyat|k|mmk|lakh|lac|million|m|သိန်း|သန်း|ကျပ်)?\b", re.I),
        # English amount words
        "hundred thousand", "half million", "one million",
        "five hundred", "one lakh",
        # Burmese amount words
        "သိန်း", "သန်း", "ကျပ်", "ပိုက်ဆံ",
        # Burmese digit patterns (Myanmar digits U+1040–U+1049)
        re.compile(r"[၀-၉]+"),
    ],

    IntentLabel.LOAN_CONFIRMATION: [
        # English
        "yes", "confirm", "agree", "ok", "okay", "sure", "correct",
        "that's right", "proceed", "go ahead", "accept",
        "i agree", "confirmed", "alright", "yep", "yup",
        # Burmese
        "ဟုတ်", "ဟုတ်ကဲ့", "အိုကေ", "သဘောတူ",
        "ကိုက်", "ကိုက်ညီ", "ဆက်လုပ်", "လုပ်မည်",
        "လက်ခံ", "မှန်", "ဆက်သွားမယ်",
    ],

    IntentLabel.COMPLAINT: [
        # English
        "complaint", "complain", "problem", "issue", "error",
        "wrong", "mistake", "not working", "failed", "rejected",
        "unfair", "bad service", "dissatisfied", "unhappy",
        "i have a problem", "help me", "not happy",
        # Burmese
        "တိုင်ကြားချင်", "မကျေနပ်", "ပြဿနာ", "အမှား",
        "မဖြစ်", "ကျဆုံး", "ငြင်းဆို", "ညံ့",
        "မကောင်း", "ဝမ်းနည်း", "ကူညီပါ",
    ],
}

# Confidence awarded per keyword/pattern hit
_KEYWORD_HIT_CONFIDENCE: float = 0.82

# Confidence threshold above which embedding fallback is skipped
_KEYWORD_CONFIDENCE_THRESHOLD: float = 0.75

_INTENT_PRIORITY = {

    IntentLabel.LOAN_START: 10,

    IntentLabel.LOAN_CATEGORY: 9,

    IntentLabel.ELIGIBILITY: 8,

    IntentLabel.DOCUMENT: 7,

    IntentLabel.LOAN_AMOUNT: 6,

    IntentLabel.LOAN_INFO: 5,

    IntentLabel.CONFIRMATION: 4,

}

# Minimum embedding similarity to assign a non-UNKNOWN label
_EMBEDDING_MIN_CONFIDENCE: float = 0.35

# ---------------------------------------------------------------------------
# Prototype sentences for embedding-based fallback
# ---------------------------------------------------------------------------
# Each intent has several diverse example sentences so that the averaged
# prototype vector better covers the semantic neighbourhood of the class.

_PROTOTYPES: dict[IntentLabel, list[str]] = {

    IntentLabel.GREETING: [
        "Hello, how are you?",
        "Hi there!",
        "Good morning.",
        "မင်္ဂလာပါ",
        "ဟဲလို",
    ],

    IntentLabel.LOAN_INQUIRY: [
        "Can I get a loan?",
        "I want to borrow money.",
        "Tell me about your loan products.",
        "What types of loans do you offer?",
        "want borrow money",
        "need some cash",
        "၅ သိန်းချေးချင်တယ်",
        "ဆေးကုမယ် ငွေလိုတယ်",
        "လုပ်ငန်းလုပ်ချင်လို့ ငွေလိုတယ်",
        "ချေးငွေ အကြောင်း သိချင်တယ်",
        "ငွေချေးလို့ ရမလား",
    ],

    IntentLabel.LOAN_APPLICATION: [
        "I want to apply for a loan.",
        "How do I submit a loan application?",
        "Please help me fill out the form.",
        "I'd like to start the loan process.",
        "လျှောက်ထားချင်တယ်",
        "ချေးငွေ လျှောက်မည်",
        "ဖောင်ဖြည့်ချင်တယ်",
    ],

    IntentLabel.ELIGIBILITY_QUESTION: [
        "Am I eligible for a loan?",
        "Who can apply for this loan?",
        "What are the eligibility requirements?",
        "What is the minimum income required?",
        "ကျွန်တော် ရပါသလား",
        "ဘယ်သူ လျှောက်လို့ ရပါသလဲ",
        "ဘာ အရည်အချင်းတွေ လိုသလဲ",
    ],

    IntentLabel.DOCUMENT_QUESTION: [
        "What documents do I need?",
        "Which papers are required for the loan?",
        "Do I need an NRC card?",
        "Is a bank statement required?",
        "ဘာ စာရွက်တွေ လိုသလဲ",
        "မှတ်ပုံတင် လိုသလား",
        "ဘာ အထောက်အထားတွေ ယူလာရမလဲ",
    ],

    IntentLabel.LOAN_CATEGORY_SELECTION: [
        "I want an agriculture loan.",
        "I need a small business loan.",
        "I'm looking for a vehicle loan.",
        "Can I get a consumer loan?",
        "I am a farmer and need money for my crops.",
        "စိုက်ပျိုးရေး ချေးငွေ လိုချင်တယ်",
        "ကားချေးငွေ လျှောက်ချင်တယ်",
        "စီးပွားရေး ချေးငွေ",
    ],

    IntentLabel.LOAN_AMOUNT_PROVIDED: [
        "I need 500,000 kyat.",
        "I want to borrow 5 lakh.",
        "The amount I need is 2 million.",
        "500000",
        "၅ သိန်း လိုချင်တယ်",
        "သိန်း ၁၀ ချေးချင်တယ်",
        "ကျပ် ၃ သန်း",
    ],

    IntentLabel.LOAN_CONFIRMATION: [
        "Yes, I confirm.",
        "That's correct, please proceed.",
        "OK, go ahead.",
        "I agree with the terms.",
        "ဟုတ်ကဲ့ သဘောတူပါတယ်",
        "အိုကေ ဆက်လုပ်ပါ",
        "ဟုတ် မှန်ပါတယ်",
    ],

    IntentLabel.COMPLAINT: [
        "I have a complaint about my loan.",
        "There is a problem with my application.",
        "Your service is very bad.",
        "I am not happy with the result.",
        "My loan was wrongly rejected.",
        "တိုင်ကြားချင်တယ်",
        "ပြဿနာ တစ်ခု ရှိတယ်",
        "မကျေနပ်ဘူး",
    ],

    IntentLabel.UNKNOWN: [
        "What is the weather today?",
        "Tell me a joke.",
        "Random unrelated question.",
    ],
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """
    Lower-case, strip extra whitespace, remove control characters.
    Preserves Burmese / Myanmar Unicode (U+1000–U+109F).
    """
    # Normalise Unicode to NFC so composed characters compare correctly
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _is_burmese(text: str) -> bool:
    """Return True if the text contains Myanmar-script characters."""
    return bool(re.search(r"[\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF]", text))


def _keyword_match(
    text_lower: str,
    patterns: list[str | re.Pattern],
) -> bool:
    """Return True if any keyword / pattern matches the normalised text."""
    for pat in patterns:
        if isinstance(pat, re.Pattern):
            if pat.search(text_lower):
                return True
        else:
            if pat.lower() in text_lower:
                return True
    return False


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Hybrid intent classifier: keyword fast-path → embedding fallback.

    Parameters
    ----------
    embedding_engine : EmbeddingEngine
        Must expose an  ``encode(texts: list[str]) -> np.ndarray``  method
        that returns a 2-D float array of shape ``(len(texts), dim)``.
        The BAAI/bge-m3 model is recommended (already used project-wide).

    keyword_threshold : float
        Minimum confidence to accept a keyword match without consulting
        the embedding model.  Default 0.75.

    embedding_min_confidence : float
        Minimum cosine similarity required to assign a non-UNKNOWN label
        via the embedding path.  Default 0.35.

    Example
    -------
    >>> from embeddings.embedding_engine import EmbeddingEngine
    >>> engine = EmbeddingEngine()          # initialised with BAAI/bge-m3
    >>> clf = IntentClassifier(engine)
    >>> label, conf = clf.classify("ချေးငွေ လျှောက်ချင်တယ်")
    >>> print(label, conf)
    IntentLabel.LOAN_APPLICATION 0.82
    """

    def __init__(
        self,
        embedding_engine: "EmbeddingEngine",
        keyword_threshold: float = _KEYWORD_CONFIDENCE_THRESHOLD,
        embedding_min_confidence: float = _EMBEDDING_MIN_CONFIDENCE,
    ) -> None:
        self._engine = embedding_engine
        self._keyword_threshold = keyword_threshold
        self._embedding_min_confidence = embedding_min_confidence

        logger.info("IntentClassifier initialising — encoding prototype sentences …")
        self._prototype_vectors: dict[IntentLabel, np.ndarray] = (
            self._build_prototype_vectors()
        )
        logger.info(
            "IntentClassifier ready. %d intent prototypes loaded.",
            len(self._prototype_vectors),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self,
        text:str,
        history:list[str]=None) -> tuple[IntentLabel, float]:
        """
        Classify *text* and return ``(IntentLabel, confidence)``.

        ``confidence`` is a float in ``[0.0, 1.0]``:
        * ≈ 0.82  →  keyword hit
        * 0.35–1.0  →  embedding cosine similarity (rescaled)
        * 0.0  →  UNKNOWN / below threshold

        Parameters
        ----------
        text : str
            Raw user input in English, Burmese, or mixed script.

        Returns
        -------
        tuple[IntentLabel, float]
        """
        if not text or not text.strip():
            logger.debug("Empty input → UNKNOWN")
            return IntentLabel.UNKNOWN, 0.0

        normalised = _normalise(text)
        script = "Burmese" if _is_burmese(text) else "Latin"
        logger.debug("classify | script=%s | input=%r", script, text[:120])

        # ── 1. Keyword fast-path ─────────────────────────────────────────────
        kw_label, kw_conf = self._keyword_classify(normalised)
        if kw_conf >= self._keyword_threshold:
            logger.debug(
                "Keyword match → %s (conf=%.2f)", kw_label.value, kw_conf
            )
            return kw_label, kw_conf

        # ── 2. Embedding similarity fallback ─────────────────────────────────
        emb_label, emb_conf = self._embedding_classify(text)
        logger.debug(
            "Embedding match → %s (conf=%.2f)", emb_label.value, emb_conf
        )

        if emb_conf < self._embedding_min_confidence:
            logger.debug("Confidence below threshold → UNKNOWN")
            return IntentLabel.UNKNOWN, emb_conf

        return emb_label, emb_conf

        if history:

            context = " ".join(history[-5:])

            final_text = context + " " + text

        else:

            final_text = text
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _keyword_classify(
        self, text_lower: str
    ) -> tuple[IntentLabel, float]:
        """
        Scan keyword catalogue and return the best-matching intent.

        Strategy:
        * Count distinct hits per intent.
        * Pick the intent with the most hits.
        * Confidence = _KEYWORD_HIT_CONFIDENCE when ≥ 1 hit; else 0.0.
        """
        hit_counts: dict[IntentLabel, int] = {}

        for intent, patterns in _KW.items():
            count = sum(
                1
                for pat in patterns
                if (
                    pat.search(text_lower)          # regex
                    if isinstance(pat, re.Pattern)
                    else pat.lower() in text_lower  # substring
                )
            )
            if count:
                hit_counts[intent] = count

        if not hit_counts:
            return IntentLabel.UNKNOWN, 0.0

        best_intent = max(
            hit_counts,
            key=lambda k: (
                hit_counts[k],
                _INTENT_PRIORITY.get(k, 0)
            )
        )
        # Scale confidence slightly with number of hits (caps at 0.95)
        conf = min(_KEYWORD_HIT_CONFIDENCE + 0.02 * (hit_counts[best_intent] - 1), 0.95)
        return best_intent, conf

    def _embedding_classify(
        self, text: str
    ) -> tuple[IntentLabel, float]:
        """
        Encode *text* with EmbeddingEngine and return the most similar
        intent prototype via cosine similarity.
        """
        try:
            # encode() is expected to accept list[str] → np.ndarray (N, dim)
            query_vec: np.ndarray = self._engine.encode([text])[0]
        except Exception as exc:
            logger.error("EmbeddingEngine.encode failed: %s", exc, exc_info=True)
            return IntentLabel.UNKNOWN, 0.0

        best_label = IntentLabel.UNKNOWN
        best_sim: float = -1.0

        for label, proto_vec in self._prototype_vectors.items():
            if label is IntentLabel.UNKNOWN:
                continue
            sim = _cosine_similarity(query_vec, proto_vec)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        # Clamp to [0, 1] — cosine similarity is in [-1, 1]
        confidence = max(0.0, min(best_sim, 1.0))
        return best_label, confidence

    def _build_prototype_vectors(self) -> dict[IntentLabel, np.ndarray]:
        """
        Encode all prototype sentences and store one averaged vector per
        intent.  Called once during __init__.
        """
        prototype_vectors: dict[IntentLabel, np.ndarray] = {}

        for label, sentences in _PROTOTYPES.items():
            try:
                vecs: np.ndarray = self._engine.encode(sentences)
                # Average pooling over all prototype sentences
                prototype_vectors[label] = vecs.mean(axis=0)
                logger.debug(
                    "Prototype encoded | %s | %d sentences | dim=%d",
                    label.value,
                    len(sentences),
                    vecs.shape[1],
                )
            except Exception as exc:
                logger.error(
                    "Failed to encode prototypes for %s: %s",
                    label.value,
                    exc,
                    exc_info=True,
                )

        return prototype_vectors

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def batch_classify(
        self, texts: list[str]
    ) -> list[tuple[IntentLabel, float]]:
        """
        Classify a list of texts.  Returns a list of (IntentLabel, confidence)
        tuples in the same order as *texts*.
        """
        return [self.classify(t) for t in texts]

    def __repr__(self) -> str:
        return (
            f"IntentClassifier("
            f"keyword_threshold={self._keyword_threshold}, "
            f"embedding_min_confidence={self._embedding_min_confidence}, "
            f"intents={[l.value for l in IntentLabel if l is not IntentLabel.UNKNOWN]})"
        )