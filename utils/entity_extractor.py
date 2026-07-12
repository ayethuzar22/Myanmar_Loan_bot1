"""
utils/entity_extractor.py
=========================
Rule-based entity extractor for the Myanmar Microfinance Loan AI Chatbot.

Extraction is intentionally LLM-free: regex + keyword lookup + Myanmar-digit
conversion covers all known entity patterns reliably and at zero inference cost.

Extracted entities
------------------
  amount          – loan amount in raw kyat (int)
  loan_category   – "agriculture" | "msme" | "consumer"  (str)
  term_months     – loan duration in months (int)
  income          – monthly income in raw kyat (int)
  documents       – list of canonical document tags (list[str])
  guarantor_count – number of guarantors (int)

Integration path
----------------
  1. Intent classifier (utils/intent_classifier.py)
       → tells us what the user wants
  2. THIS FILE: Entity extractor
       → tells us what data the user provided
  3. models/application_state.py  (future Step 4)
       → ApplicationState.merge(ExtractedEntities) stores fields,
         derives missing_fields, advances current_stage
  4. rag/pipeline.py  (future Step 5)
       → calls classifier + extractor every turn, passes state to prompt builder

No new pip packages required.  Uses only Python stdlib (re, dataclasses,
unicodedata, logging) and a small amount of numpy (already project-present).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | entity_extractor | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_h)


# ===========================================================================
# Myanmar / Burmese script utilities
# ===========================================================================

# Myanmar digits  ၀ ၁ ၂ ၃ ၄ ၅ ၆ ၇ ၈ ၉  →  Arabic digits
_MY_DIGIT_MAP: dict[str, str] = {
    "၀": "0", "၁": "1", "၂": "2", "၃": "3", "၄": "4",
    "၅": "5", "၆": "6", "၇": "7", "၈": "8", "၉": "9",
    # Extended Myanmar Digit block (U+A9F0–A9F9)
    "\uA9F0": "0", "\uA9F1": "1", "\uA9F2": "2", "\uA9F3": "3",
    "\uA9F4": "4", "\uA9F5": "5", "\uA9F6": "6", "\uA9F7": "7",
    "\uA9F8": "8", "\uA9F9": "9",
}

# Burmese spoken-number words → integer value
# Used when a user writes "တစ်" instead of "1"
_MY_NUMBER_WORDS: dict[str, int] = {
    "သုည": 0,
    "တစ်": 1, "တစ": 1,
    "နှစ်": 2, "နှစ": 2,
    "သုံး": 3, "သုံ": 3,
    "လေး": 4,
    "ငါး": 5,
    "ခြောက်": 6, "ခြောက": 6,
    "ခုနစ်": 7, "ခုနစ": 7,
    "ရှစ်": 8, "ရှစ": 8,
    "ကိုး": 9,
    "တစ်ဆယ်": 10, "ဆယ်": 10,
    "တစ်ဆယ့်တစ်": 11,
    "တစ်ဆယ့်နှစ်": 12,
    "တစ်ဆယ့်သုံး": 13,
    "တစ်ဆယ့်လေး": 14,
    "တစ်ဆယ့်ငါး": 15,
    "တစ်ဆယ့်ခြောက်": 16,
    "တစ်ဆယ့်ခုနစ်": 17,
    "တစ်ဆယ့်ရှစ်": 18,
    "တစ်ဆယ့်ကိုး": 19,
    "နှစ်ဆယ်": 20,
    "သုံးဆယ်": 30,
    "လေးဆယ်": 40,
    "ငါးဆယ်": 50,
    "ခြောက်ဆယ်": 60,
    "တစ်ရာ": 100,
}

# Amount-unit multipliers (Myanmar + English)
_UNIT_MULTIPLIERS: dict[str, int] = {
    # Burmese units
    "သိန်း":  100_000,      # 1 thein  = 100,000 kyat
    "သိန်":   100_000,
    "သန်း":  1_000_000,    # 1 than   = 1,000,000 kyat
    "ကျပ်":         1,      # raw kyat
    "ကျပ":          1,
    # English units
    "lakh":   100_000,
    "lac":    100_000,
    "million":1_000_000,
    "m":      1_000_000,
    "k":          1_000,
    "mmk":          1,
    "kyat":         1,
}

# Year-to-month conversion word list (both languages)
_YEAR_WORDS: frozenset[str] = frozenset({"year", "years", "yr", "yrs", "နှစ်"})
_MONTH_WORDS: frozenset[str] = frozenset({"month", "months", "mo", "mos", "လ", "လ​"})


def _mm_to_arabic(text: str) -> str:
    """Replace Myanmar digits with their ASCII equivalents."""
    return "".join(_MY_DIGIT_MAP.get(ch, ch) for ch in text)


def _normalise(text: str) -> str:
    """
    NFC-normalise, convert Myanmar digits, collapse whitespace.
    Does NOT lower-case so that Burmese case (unused) is preserved,
    but lower-cases ASCII for keyword matching.
    """
    text = unicodedata.normalize("NFC", text)
    text = _mm_to_arabic(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_numeric(raw: str) -> Optional[int]:
    """
    Parse a cleaned numeric string (may contain commas) to int.
    Returns None on failure.
    """
    try:
        return int(float(raw.replace(",", "")))
    except (ValueError, TypeError):
        return None


def _resolve_burmese_number_word(text: str) -> Optional[int]:
    """
    Look up Burmese number words in _MY_NUMBER_WORDS.
    Longest match wins.
    """
    best_len = 0
    best_val: Optional[int] = None
    for word, val in _MY_NUMBER_WORDS.items():
        if word in text and len(word) > best_len:
            best_len = len(word)
            best_val = val
    return best_val


# ===========================================================================
# ExtractedEntities — the public output type
# ===========================================================================

@dataclass
class ExtractedEntities:
    """
    All entity slots that can be filled from a single user message.

    All fields default to None / empty so callers can distinguish
    "not mentioned" from "zero".  ApplicationState.merge() will only
    overwrite non-None values.
    """

    amount: Optional[int] = None
    """Loan amount in raw Myanmar kyat."""

    loan_category: Optional[str] = None
    """One of: 'agriculture', 'msme', 'consumer'."""

    term_months: Optional[int] = None
    """Loan duration converted to months."""

    income: Optional[int] = None
    """Monthly income in raw Myanmar kyat."""

    documents: list[str] = field(default_factory=list)
    """Canonical document tags, e.g. ['NRC', 'household_registration']."""

    guarantor_count: Optional[int] = None
    """Number of guarantors the user mentioned."""

    # ── Metadata (not stored in ApplicationState) ──────────────────────────
    raw_text: str = ""
    """The original user message, for debugging."""

    confidence: dict[str, float] = field(default_factory=dict)
    """Per-field extraction confidence (0.0–1.0)."""

    def to_dict(self) -> dict:
        """Serialise to plain dict (for JSON logging / state merge)."""
        return {
            "amount": self.amount,
            "loan_category": self.loan_category,
            "term_months": self.term_months,
            "income": self.income,
            "documents": self.documents,
            "guarantor_count": self.guarantor_count,
        }

    def filled_fields(self) -> list[str]:
        """Return names of fields that were successfully extracted."""
        filled = []
        if self.amount is not None:
            filled.append("amount")
        if self.loan_category is not None:
            filled.append("loan_category")
        if self.term_months is not None:
            filled.append("term_months")
        if self.income is not None:
            filled.append("income")
        if self.documents:
            filled.append("documents")
        if self.guarantor_count is not None:
            filled.append("guarantor_count")
        return filled

    def __repr__(self) -> str:
        filled = self.to_dict()
        return f"ExtractedEntities({filled})"


# ===========================================================================
# Keyword catalogues
# ===========================================================================

# ── Loan category keywords ──────────────────────────────────────────────────
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "agriculture": [
        # English
        "agriculture", "agricultural", "agri", "farming", "farm",
        "paddy", "vegetable", "orchard", "crop", "harvest",
        "livestock", "fishery", "rural",
        # Burmese
        "စိုက်ပျိုးရေး", "တောင်သူ", "လယ်ယာ", "လယ်", "ဆန်",
        "ဟင်းသီးဟင်းရွက်", "သီးနှံ", "ကျေးလက်",
        "သားငါး", "ဥယျာဉ်", "ခြံ",
    ],
    "msme": [
        # English
        "business", "msme", "sme", "small business", "enterprise",
        "retail", "wholesale", "manufacturing", "shop", "trade",
        "commerce", "entrepreneur", "self-employed", "self employed",
        # Burmese
        "စီးပွားရေး", "လုပ်ငန်း", "ဆိုင်", "ကုန်ထုတ်",
        "ကုန်သည်", "ကုန်သွယ်", "သေးငယ်သောလုပ်ငန်း",
        "ကုမ္ပဏီ", "အိမ်မှုလုပ်ငန်း",
    ],
    "consumer": [
        # English
        "consumer", "personal", "vehicle", "car", "motorcycle",
        "housing", "house", "home", "medical", "education",
        "consumption", "individual",
        # Burmese
        "ကား", "မော်တော်ဆိုင်ကယ်", "ဆိုင်ကယ်", "အိမ်",
        "ကုန်ပစ္စည်း", "ဆေးကုသ", "ဆေးရုံ",
        "ပညာရေး", "ကိုယ်ပိုင်", "ကျွန်ုပ်တစ်ဦးချင်း",
    ],
}

# ── Document keywords → canonical tag ──────────────────────────────────────
_DOCUMENT_KEYWORDS: dict[str, list[str]] = {
    "NRC": [
        "nrc", "national registration card", "national id",
        "id card", "မှတ်ပုံတင်", "နိုင်ငံသားစိစစ်", "မပတ်",
    ],
    "household_registration": [
        "household", "household registration", "family list",
        "နေရပ်လိပ်စာ", "အိမ်ထောင်စုစာရင်း", "အိမ်ထောင်စု",
        "နေရပ်", "လိပ်စာ",
    ],
    "form_7": [
        "form 7", "form7", "f7", "land certificate", "land title",
        "ဖောင် ၇", "ဖောင်၇", "မြေပုံ", "မြေဇာပုံ",
        "မြေစာရင်း", "ဖောင်",
    ],
    "business_license": [
        "business license", "trade license", "company registration",
        "လုပ်ငန်းလိုင်စင်", "ကုမ္ပဏီမှတ်ပုံတင်", "လိုင်စင်",
        "ကုမ္ပဏီ မှတ်ပုံတင်",
    ],
    "bank_statement": [
        "bank statement", "bank book", "passbook",
        "ဘဏ်စာရင်း", "ဘဏ်", "ငွေစာရင်း",
    ],
    "salary_slip": [
        "salary slip", "payslip", "pay slip", "salary certificate",
        "လစာဖြတ်ပိုင်း", "လစာ အထောက်အထား", "လစာ စာရွက်",
    ],
    "income_evidence": [
        "income proof", "income evidence", "income document",
        "ဝင်ငွေ အထောက်အထား", "ဝင်ငွေ", "ဝင်ငွေစာရွက်",
    ],
    "guarantor_id": [
        "guarantor id", "guarantor nrc", "ကတိပေးသူ မှတ်ပုံတင်",
        "အာမခံသူ မှတ်ပုံတင်",
    ],
}

# ── Income context words (guard to prevent confusing amount with income) ────
_INCOME_CONTEXT_WORDS: list[str] = [
    # English
    "income", "salary", "earn", "wage", "pay", "monthly",
    "per month", "a month",
    # Burmese
    "ဝင်ငွေ", "လစာ", "လုပ်ခ", "တစ်လ", "လ​တစ်လ",
    "တစ်လမှာ", "လပတ်", "ရငွေ",
]

# ── Guarantor context words ─────────────────────────────────────────────────
_GUARANTOR_CONTEXT_WORDS: list[str] = [
    "guarantor", "guarantee", "guarantors",
    "ကတိပေး", "အာမခံ", "သူနာပြု", "ကတိပေးသူ", "အာမခံသူ",
]


# ===========================================================================
# EntityExtractor
# ===========================================================================

class EntityExtractor:
    """
    Stateless rule-based entity extractor.

    Usage
    -----
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract("ငါ ၅ သိန်း ချေးချင်တယ်")
    >>> result.amount
    500000

    Parameters
    ----------
    currency_unit : str
        ISO code logged in debug messages.  Default "MMK".
    """

    def __init__(self, currency_unit: str = "MMK") -> None:
        self._currency = currency_unit
        logger.info("EntityExtractor initialised (currency=%s, LLM-free).", currency_unit)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all recognised entities from *text*.

        Parameters
        ----------
        text : str
            Raw user message in English, Burmese, or mixed script.

        Returns
        -------
        ExtractedEntities
            All slots that could be filled; unrecognised slots remain None.
        """
        if not text or not text.strip():
            return ExtractedEntities(raw_text=text)

        # Normalise once; keep original for logging
        normalised = _normalise(text)
        lower = normalised.lower()

        logger.debug("extract | input=%r | normalised=%r", text[:120], normalised[:120])

        entities = ExtractedEntities(raw_text=text)
        confidence: dict[str, float] = {}

        # Run each extractor in order of specificity.
        # Income is extracted before amount so that income-context
        # amounts are not also captured as loan amounts.

        income, c = self._extract_income(normalised, lower)
        if income is not None:
            entities.income = income
            confidence["income"] = c
            logger.debug("income extracted: %d (conf=%.2f)", income, c)

        amount, c = self._extract_amount(normalised, lower, exclude_income=income)
        if amount is not None:
            entities.amount = amount
            confidence["amount"] = c
            logger.debug("amount extracted: %d (conf=%.2f)", amount, c)

        category, c = self._extract_loan_category(lower)
        if category is not None:
            entities.loan_category = category
            confidence["loan_category"] = c
            logger.debug("loan_category extracted: %s (conf=%.2f)", category, c)

        term, c = self._extract_term_months(normalised, lower)
        if term is not None:
            entities.term_months = term
            confidence["term_months"] = c
            logger.debug("term_months extracted: %d (conf=%.2f)", term, c)

        docs, c = self._extract_documents(lower)
        if docs:
            entities.documents = docs
            confidence["documents"] = c
            logger.debug("documents extracted: %s (conf=%.2f)", docs, c)

        gcount, c = self._extract_guarantor_count(normalised, lower)
        if gcount is not None:
            entities.guarantor_count = gcount
            confidence["guarantor_count"] = c
            logger.debug("guarantor_count extracted: %d (conf=%.2f)", gcount, c)

        entities.confidence = confidence
        logger.info(
            "Extraction complete | filled=%s | text=%r",
            entities.filled_fields(),
            text[:80],
        )
        return entities

    def batch_extract(self, texts: list[str]) -> list[ExtractedEntities]:
        """Extract entities from a list of messages (convenience wrapper)."""
        return [self.extract(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Private extractors                                                   #
    # ------------------------------------------------------------------ #

    # ── Amount ─────────────────────────────────────────────────────────────

    def _extract_amount(
        self,
        normalised: str,
        lower: str,
        exclude_income: Optional[int] = None,
    ) -> tuple[Optional[int], float]:
        """
        Extract loan amount.

        Strategy
        --------
        1. Match <number> <unit> pairs (Burmese + English units).
        2. Match bare integers ≥ 10,000 that are NOT preceded by income
           context words (to avoid double-counting income).
        3. Convert to raw kyat.
        """
        candidates: list[int] = []

        # Pattern: optional decimal/comma number + optional whitespace + unit
        unit_pattern = re.compile(
            r"([\d,\.]+)\s*"
            r"(သိန်း|သိန်|သန်း|ကျပ်|ကျပ|"
            r"lakh|lac|million|mmk|kyat|[km])\b",
            re.IGNORECASE,
        )
        for m in unit_pattern.finditer(normalised):
            num = _parse_numeric(m.group(1))
            unit = m.group(2).lower()
            mult = _UNIT_MULTIPLIERS.get(unit) or _UNIT_MULTIPLIERS.get(
                # Burmese unit lookup (not lower-cased)
                m.group(2)
            ) or 1
            if num is not None:
                candidates.append(num * mult)

        # Pattern: bare integer (no unit) ≥ 10,000, NOT preceded by income word
        bare_pattern = re.compile(r"(?<!\d)([\d,]{5,}|[\d]{5,})(?!\d)")
        for m in bare_pattern.finditer(normalised):
            # Check that no income context word appears within 30 chars before
            start = max(0, m.start() - 30)
            preceding = lower[start : m.start()]
            if not any(kw in preceding for kw in _INCOME_CONTEXT_WORDS):
                val = _parse_numeric(m.group(1))
                if val is not None and val >= 10_000:
                    candidates.append(val)

        if not candidates:
            return None, 0.0

        # If we extracted income and a candidate matches it exactly, skip it
        if exclude_income is not None:
            candidates = [c for c in candidates if c != exclude_income]

        if not candidates:
            return None, 0.0

        # Take the largest candidate (most likely the loan amount, not a date/id)
        return max(candidates), 0.90

    # ── Loan category ────────────────────────────────────────────────────────

    def _extract_loan_category(
        self, lower: str
    ) -> tuple[Optional[str], float]:
        """
        Match category keywords; return the category with most hits.
        """
        hit_counts: dict[str, int] = {}
        for category, keywords in _CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw.lower() in lower)
            if count:
                hit_counts[category] = count

        if not hit_counts:
            return None, 0.0

        best = max(hit_counts, key=lambda k: hit_counts[k])
        conf = min(0.75 + 0.05 * hit_counts[best], 0.95)
        return best, conf

    # ── Loan term ────────────────────────────────────────────────────────────

    def _extract_term_months(
        self, normalised: str, lower: str
    ) -> tuple[Optional[int], float]:
        """
        Extract duration and normalise to months.

        Handles
        -------
        * "12 months", "12 လ", "12လ"
        * "1 year", "2 years", "နှစ်နှစ်"
        * Burmese number words: "တစ်နှစ်" → 12, "နှစ်နှစ်" → 24
        """
        # Numeric + unit
        term_pattern = re.compile(
            r"([\d]+)\s*"
            r"(month|months|mo|mos|လ|year|years|yr|yrs|နှစ်)\b",
            re.IGNORECASE,
        )
        m = term_pattern.search(normalised)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit in _YEAR_WORDS or unit == "နှစ်":
                return num * 12, 0.92
            return num, 0.92

        # Burmese number word + year
        for bword, bval in _MY_NUMBER_WORDS.items():
            if bword in normalised:
                # Check if a year/month word follows within 5 chars
                idx = normalised.find(bword)
                following = normalised[idx + len(bword) : idx + len(bword) + 6]
                if any(yw in following for yw in ["နှစ်", "year", "yr"]):
                    return bval * 12, 0.85
                if any(mw in following for mw in ["လ", "month", "mo"]):
                    return bval, 0.85

        return None, 0.0

    # ── Income ───────────────────────────────────────────────────────────────

    def _extract_income(
        self, normalised: str, lower: str
    ) -> tuple[Optional[int], float]:
        """
        Extract monthly income.

        Strategy: look for an income context word within a 40-char window
        before a numeric+unit pattern.
        """
        # Find all positions of income context words in lower
        income_positions: list[int] = []
        for kw in _INCOME_CONTEXT_WORDS:
            pos = 0
            while True:
                idx = lower.find(kw, pos)
                if idx == -1:
                    break
                income_positions.append(idx)
                pos = idx + 1

        if not income_positions:
            return None, 0.0

        # Now look for numeric values within ±50 chars of any income word
        amount_pattern = re.compile(
            r"([\d,\.]+)\s*(သိန်း|သိန်|သန်း|ကျပ်|ကျပ|lakh|lac|million|mmk|kyat|[km])?\b",
            re.IGNORECASE,
        )
        candidates: list[int] = []
        for m in amount_pattern.finditer(normalised):
            m_center = (m.start() + m.end()) // 2
            nearby = any(abs(m_center - ip) <= 50 for ip in income_positions)
            if not nearby:
                continue

            num = _parse_numeric(m.group(1))
            unit_raw = m.group(2) or ""
            unit = unit_raw.lower()
            mult = (
                _UNIT_MULTIPLIERS.get(unit)
                or _UNIT_MULTIPLIERS.get(unit_raw)
                or (1 if num and num >= 10_000 else None)
            )
            if num is not None and mult is not None:
                candidates.append(num * mult)

        if not candidates:
            return None, 0.0

        return max(candidates), 0.88

    # ── Documents ────────────────────────────────────────────────────────────

    def _extract_documents(
        self, lower: str
    ) -> tuple[list[str], float]:
        """
        Match document keywords and return canonical tags.
        Deduplicates; preserves insertion order.
        """
        found: list[str] = []
        seen: set[str] = set()

        for canonical_tag, keywords in _DOCUMENT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in lower and canonical_tag not in seen:
                    found.append(canonical_tag)
                    seen.add(canonical_tag)
                    break  # one match per tag is enough

        if not found:
            return [], 0.0

        return found, min(0.80 + 0.03 * len(found), 0.95)

    # ── Guarantor count ──────────────────────────────────────────────────────

    def _extract_guarantor_count(
        self, normalised: str, lower: str
    ) -> tuple[Optional[int], float]:
        """
        Extract guarantor count.

        Matches patterns like:
          "2 guarantors", "သုံးယောက် ကတိပေး", "guarantor 3"
        """
        # Is there a guarantor context word at all?
        has_context = any(kw in lower for kw in _GUARANTOR_CONTEXT_WORDS)
        if not has_context:
            return None, 0.0

        # Numeric: <digit> near guarantor context word
        num_pattern = re.compile(r"\b(\d)\b")
        for m in num_pattern.finditer(normalised):
            m_center = m.start()
            window = lower[max(0, m_center - 25) : m_center + 25]
            if any(kw in window for kw in _GUARANTOR_CONTEXT_WORDS):
                val = _parse_numeric(m.group(1))
                if val is not None and 1 <= val <= 10:
                    return val, 0.87

        # Burmese number word near guarantor context word
        for bword, bval in _MY_NUMBER_WORDS.items():
            if bword in normalised and 1 <= bval <= 10:
                idx = normalised.find(bword)
                window = lower[max(0, idx - 25) : idx + 25]
                if any(kw in window for kw in _GUARANTOR_CONTEXT_WORDS):
                    return bval, 0.83

        # Context exists but no count found → assume 1
        return 1, 0.60

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"EntityExtractor("
            f"currency={self._currency!r}, "
            f"categories={list(_CATEGORY_KEYWORDS)}, "
            f"document_types={list(_DOCUMENT_KEYWORDS)})"
        )