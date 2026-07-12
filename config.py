"""
config.py — Centralised configuration for the Wonderami Loan RAG Engine.

Everything here is a constant: paths, thresholds, domain keyword sets, and
the shared logger. Nothing in this file has behaviour of its own; every
value is copied verbatim from the original rag1.py so downstream modules
see identical constants.
"""

from __future__ import annotations

import logging
import os
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# Named logger: Django's LOGGING dict can reconfigure it without conflict.
# propagate=False prevents double-logging under Django's root handler.
# ─────────────────────────────────────────────────────────────────────────────

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s — %(message)s")
)
log = logging.getLogger("wonderami.rag")
if not log.handlers:
    log.addHandler(_log_handler)
log.setLevel(logging.INFO)
log.propagate = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — all tuneable values in one place
# ─────────────────────────────────────────────────────────────────────────────

_HERE: str = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR: str    = os.path.join(_HERE, "artifacts")
INDEX_PATH: str       = os.path.join(ARTIFACTS_DIR, "faiss_index.bin")
CHUNKS_PATH: str      = os.path.join(ARTIFACTS_DIR, "faiss_chunks.pkl")
EMBED_CACHE_PATH: str = os.path.join(ARTIFACTS_DIR, "embeddings_cache.npy")
HASH_CACHE_PATH: str  = os.path.join(ARTIFACTS_DIR, "loan_json.sha256")
RAW_JSON_PATH: str    = os.path.join(_HERE, "loan.json")

EMBED_MODEL_NAME: str   = "BAAI/bge-m3"
EMBED_BATCH_SIZE: int   = 32
EMBED_QUERY_PREFIX: str = "Represent this sentence for retrieval: "

FAISS_TOP_K: int            = 5
SIMILARITY_THRESHOLD: float = 0.45

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL vs GROUP LOAN CAPS
# ⚠️ PLACEHOLDER VALUES modeled on a competitor's (Sathapana) published rates
# as a UX reference only. CONFIRM the real Wondarmi figures with the business
# team before deploying — do not assume these numbers are correct for Wondarmi.
# ─────────────────────────────────────────────────────────────────────────────
INDIVIDUAL_LOAN_MAX_MMK: float = 20_000_000   # 200 lakhs — CONFIRM with Wondarmi
INDIVIDUAL_LOAN_MAX_MONTHS: int = 24
GROUP_LOAN_MAX_MMK: float = 3_000_000         # 30 lakhs — CONFIRM with Wondarmi
GROUP_LOAN_MAX_MONTHS: int = 12
DEFAULT_CALC_TENURE_MONTHS: int = 12          # used when user doesn't state tenure

INDIVIDUAL_LOAN_KEYWORDS: frozenset[str] = frozenset({
    "တစ်ဦးချင်း", "တစ်ဦးတည်း", "ကိုယ်ပိုင်",
    "individual", "personally", "solo", "alone", "myself",
})

GROUP_LOAN_KEYWORDS: frozenset[str] = frozenset({
    "ဝိုင်းကြီးချုပ်", "အဖွဲ့လိုက်", "အုပ်စုလိုက်",
    "group", "joint liability", "joint", "team",
})


MIN_LOAN_AMOUNT_MMK: float = 100_000
_AFFIRMATIVE_LONG: frozenset[str] = frozenset({
    "yes", "okay", "ready", "done", "yeah", "yep",
    "ဟုတ်ကဲ့", "ရပါပြီ", "ပြင်ဆင်ပြီးပါပြီ", "ပြီးပါပြီ", "ရပြီ", "ရှိပါတယ်",
})
_AFFIRMATIVE_SHORT: frozenset[str] = frozenset({"1", "y", "ok"})

_NEGATIVE_LONG: frozenset[str] = frozenset({
    "no", "not yet", "not ready",
    "မရသေးဘူး", "မပြင်ဆင်ရသေးဘူး", "မဟုတ်ဘူး", "မရှိဘူး",
})
_NEGATIVE_SHORT: frozenset[str] = frozenset({"2", "n"})

LOAN_CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "Agriculture":  frozenset({"agriculture", "farming", "farm", "စိုက်ပျိုး"}),
    "MSME":         frozenset({"business", "shop", "msme", "trade", "စီးပွားရေး", "ဆိုင်"}),
    "Consumption":  frozenset({"consumption", "personal", "salary", "medical", "လူသုံးကုန်", "ဆေးကု"}),
}

# Myanmar digit -> ASCII digit translation table, needed because loan
# amounts and tenures in user messages are frequently written in Myanmar
# numerals (e.g. "၅ သိန်း" = "5 lakhs"), which plain \d regex won't match.
_MYANMAR_DIGIT_MAP = str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789")

_RE_LAKH_AMOUNT: re.Pattern[str] = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:သိန်း|lakh|lakhs)", re.IGNORECASE
)
_RE_PLAIN_MMK_AMOUNT: re.Pattern[str] = re.compile(
    r"(\d{4,})\s*(?:ကျပ်|kyat|mmk)?", re.IGNORECASE
)
_RE_MONTHS: re.Pattern[str] = re.compile(
    r"(\d{1,2})\s*(?:လ|month|months)", re.IGNORECASE
)

# Stable substring used to detect "we already asked individual-vs-group"
# in the previous assistant turn, when resuming from chat_history.
_LOAN_MODE_CLARIFY_MARKER: str = "ဘယ်အမျိုးအစားနှင့် လျှောက်ထားလိုပါသလဲ"
_NUMERAL_TO_MODE: dict[str, str] = {
    "1": "individual", "2": "group",
    "၁": "individual", "၂": "group",
}
_ORDINAL_TO_MODE: dict[str, str] = {
    "first": "individual", "second": "group",
    "ပထမ": "individual", "ဒုတိယ": "group",
}
# FIX #1 note: LOCAL_FALLBACK_MIN_SCORE governs the stricter bar used
# when Gemini is unavailable and we must paste a raw KB answer directly.
LOCAL_FALLBACK_MIN_SCORE: float = 0.55

GEMINI_MODEL: str         = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.15
GEMINI_MAX_TOKENS: int    = 1024
GEMINI_MAX_RETRIES: int   = 3
GEMINI_RETRY_DELAY: float = 2.0
GEMINI_TIMEOUT_SECONDS: float = 20.0   # per-request network timeout

# Autonomous-learning FAISS rebuild executes off the request thread by default
LEARNING_REBUILD_ASYNC: bool = True

HISTORY_WINDOW: int   = 4
MAX_INPUT_LENGTH: int = 1000

LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "qwen").strip().lower()

QWEN_MODEL_NAME: str        = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
QWEN_MODEL_NAME_CPU_FALLBACK: str = "Qwen/Qwen2.5-1.5B-Instruct"  # used automatically if no GPU found
QWEN_MAX_NEW_TOKENS: int    = 512
QWEN_TEMPERATURE: float     = 0.15   # matches GEMINI_TEMPERATURE — low, for factual consistency
QWEN_MIN_VRAM_GB_FOR_4BIT: float = 6.0   # below this, fall back to the smaller CPU model
QWEN_MIN_VRAM_GB_FOR_FP16: float = 15.0  # above this, skip quantization entirely
# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS — frozensets for O(1) membership tests
# ─────────────────────────────────────────────────────────────────────────────

GREETINGS: frozenset[str] = frozenset({
    "hello", "hi", "hey",
    "\u1019\u1004\u103a\u1039\u1002\u101c\u102c\u1015\u102b", "\u1040\u1032\u101c\u102d\u102f", "\u1040\u102d\u102f\u1004\u103a\u1038",
})

# FIX #2: THANK_WORDS is split into "long" (safe as substrings, multi-char
# phrases unlikely to appear inside unrelated words) and "short" tokens
# (2-3 characters, which previously caused false positives — e.g. "ty"
# matched inside "types", "duty", "safety", "quantity", "specialty").
# Short tokens are matched as WHOLE WORDS ONLY via is_thanks(), never as
# raw substrings.
_THANK_WORDS_LONG: frozenset[str] = frozenset({
    "thanks",
    "thank you",
    "thank you so much",
    "thanks a lot",
    "many thanks",
    "thanks so much",
    "thanks!",
    "thnx",
    "tnx",
    "tysm",
    "appreciate it",
    "thank u",
    "thank u so much",
    "okay thanks",
    "ok thanks",

    # Myanmar
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1021\u1019\u103b\u102c\u1038\u1000\u103c\u102e\u1038",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1021\u1011\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1021\u1019\u103b\u102c\u1038\u1000\u103c\u102e\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a",
    "\u1021\u1019\u103b\u102c\u1038\u1000\u103c\u102e\u1038\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a",
    "\u1016\u103c\u1031\u1015\u1031\u1038\u1010\u1032\u1037\u1021\u1010\u103d\u1000\u103a\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1016\u103c\u1031\u1015\u1031\u1038\u101c\u102d\u102f\u1037\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1016\u103c\u1031\u1015\u1031\u1038\u1010\u102c\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u101b\u103e\u1004\u103a\u1038\u1015\u103c\u1015\u1031\u1038\u1010\u1032\u1037\u1021\u1010\u103d\u1000\u103a\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1000\u1030\u100a\u102e\u1015\u1031\u1038\u1010\u1032\u1037\u1021\u1010\u103d\u1000\u103a\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1000\u1030\u100a\u102e\u1015\u1031\u1038\u101c\u102d\u102f\u1037\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
    "\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1014\u1032\u102c\u1037",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1001\u1004\u103a\u1017\u103b",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1001\u1004\u103a\u1017\u103b\u102c",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b\u1001\u1004\u103a\u1017\u103b",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b\u1001\u1004\u103a\u1017\u103b\u102c",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a\u1001\u1004\u103a\u1017\u103b",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a\u1001\u1004\u103a\u1017\u103b\u102c",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b\u1014\u102d\u102c\u1037",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b\u1018\u103b",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u102b\u101b\u103e\u1004\u103a",
    "\u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1010\u1004\u103a\u1015\u102b\u1010\u101a\u103a\u101b\u103e\u1004\u103a",
    "\u1021\u102d\u102f\u1000\u1031\u1000\u103b\u1031\u1038\u1007\u1030\u1038",
})

# Short tokens matched as WHOLE WORDS ONLY (never as raw substrings) to
# avoid false positives inside unrelated words.
_THANK_WORDS_SHORT: frozenset[str] = frozenset({
    "thx", "ty",
})

CALC_TRIGGERS: frozenset[str] = frozenset({
    "\u1078\u1000\u103a", "calculate", "calculator", "\u1021\u1078\u102d\u102f\u1038\u1014\u103e\u102f\u1014\u103a\u1078\u1000\u103a",
})

LOAN_TYPE_TRIGGERS: frozenset[str] = frozenset({
    "how many loan", "types of loan", "what loan do you have",
    "\u1001\u103b\u1031\u1038\u1004\u103a\u1040\u1018\u101a\u103a\u1014\u103e\u1005\u103a\u1019\u103b\u102d\u102f\u1038", "\u1001\u103b\u1031\u1038\u1004\u103a\u1021\u1019\u103b\u102d\u102f\u1021\u1005\u102c\u1038",
})

# Borrow-intent root keywords — any query containing ANY of these signals
# that the user wants to borrow money.  Using root substrings (not full
# phrases) means we catch all natural Myanmar variations automatically:
#   ချေးချင်တယ်, ချေးချင်လို့, ငွေချေးချင်, ငွေချေးလိုတယ် etc.
#
# IMPORTANT: This set is checked with `contains_any()` which does substring
# matching, so short root words must be chosen carefully to avoid false
# positives (e.g. do not add bare "ငွေ" alone — too broad).
BORROW_INTENT_KEYWORDS: frozenset[str] = frozenset({
    # Myanmar root substrings
    "ချေးချင်",     # want to borrow (most common form)
    "ချေးလို",      # want/need to borrow
    "ငွေချေး",      # borrow money
    "ချေးငွေရ",     # get a loan
    "ချေးငွေလို",   # need a loan
    "ချေးငွေလျှောက်", # apply for a loan
    "ချေးငွေအကြောင်း",  # about loans
    "ချေးငွေ အကြောင်း", # about loans (with space)
    "ချေးငွေ သိ",   # want to know about loans
    "ချေးနိုင်",    # can borrow
    "ချေးပေး",      # lend (asking)
    "ငွေလို",       # need money
    "ငွေလိုတယ်",    # need money
    "ချေးရမလား",    # can I borrow?
    "ချေးလို့ရ",    # is it possible to borrow?
    "ချေးလို့ရပါ",  # can borrow
    "ပိုက်ဆံချေး",  # borrow money (colloquial)
    "ပိုက်ဆံလို",   # need money (colloquial)
    "loan လျှောက်",  # apply loan (mixed)
    "loan apply",    # mixed Myanmar-English (catches "loan apply လုပ်ချင်တယ်")
    "loan ရ",        # get loan (mixed)
    "loan ချေး",     # borrow loan (mixed)
    "loan လုပ်",     # do loan (mixed)
    # English root phrases
    "can i borrow", "want to borrow", "need a loan", "want a loan",
    "i need money", "need money", "borrow money", "get a loan",
    "apply for a loan", "how to apply", "how do i apply",
    "loan application", "want to apply",
})

TRANSLATE_TRIGGERS: frozenset[str] = frozenset({
    "translate with myanmar", "translate to myanmar",
    "\u1019\u103c\u1014\u103a\u1019\u102c\u101c\u102d\u102f\u1018\u102c\u101e\u102c\u1015\u103c\u1014\u103a", "\u1019\u103c\u1014\u103a\u1019\u102c\u101c\u102d\u102f\u1015\u103c\u1014\u103a\u1015\u1031\u1038",
})

BAD_WORDS: frozenset[str] = frozenset({
    "wtf", "scam", "\u101c\u1030\u101c\u102d\u1019\u103a", "\u101c\u102e\u1038", "\u1005\u1031\u102c\u1000\u103a", "\u100a\u1036\u1037\u101c\u102d\u102f\u1000\u103a\u1078\u102c",
})

OFF_TOPIC_WORDS: frozenset[str] = frozenset({
    "coffee", "tea", "food", "movie", "song", "dating",
    "girl", "boyfriend", "weather", "sport",
})

LOAN_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "loan", "borrow", "money", "rate", "interest", "pay", "credit", "finance",
    "\u1001\u103b\u1031\u1038", "\u1004\u103a\u1040\u1031", "\u1021\u1078\u102d\u102f\u1038", "\u1015\u103c\u1014\u103a\u1006\u1015\u103a", "\u1001\u103b\u1031\u1038\u1004\u103a\u1040", "\u1018\u100f\u103a",
})

_GEMINI_FATAL_TAGS: tuple[str, ...] = (
    "api_key", "quota", "permission", "403", "401", "invalid_argument",
)

_GENERIC_ANSWER_MARKERS: frozenset[str] = frozenset({
    "\u1014\u102c\u1038\u1019\u101c\u100a\u103a\u1015\u102b", "\u1019\u101e\u102d\u1015\u102b", "\u1011\u1015\u103a\u1019\u1036\u1019\u1031\u1038\u1019\u1036\u1014\u102d\u102f\u1004\u103a",
    "don't understand", "not sure", "i don't know",
})

# Emoji set checked against the RAW (pre-normalization) query — see FIX #3.
_LAUGH_EMOJI: tuple[str, ...] = ("\U0001F602", "\U0001F923", "\U0001F604", "\U0001F606")
# i.e. 😂 🤣 😄 😆

# Shared regex utilities (moved from the original "PURE UTILITIES" section;
# kept here since they are cross-cutting and consumed by several modules).
_RE_STRIP_PUNCT: re.Pattern[str] = re.compile(r"[^\w\s\u1000-\u109f]")
_RE_COLLAPSE_WS: re.Pattern[str] = re.compile(r"\s+")
_RE_MYANMAR:     re.Pattern[str] = re.compile(r"[\u1000-\u109f]")
_RE_DIGITS:      re.Pattern[str] = re.compile(r"\d+\.?\d*")
_RE_WORDS:       re.Pattern[str] = re.compile(r"\w+", re.UNICODE)
_RE_STRIP_PUNCT_KEEP_MYANMAR: re.Pattern[str] = re.compile(r"[^\w\s\u1000-\u109F]")