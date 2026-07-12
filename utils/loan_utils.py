"""
utils/loan_utils.py — Loan-domain-specific text extraction and the pure
loan repayment calculator.

Moved verbatim from rag1.py. Depends on utils.text_utils.contains_any for
keyword-set membership checks, and on config for all keyword sets and
compiled regex patterns.
"""

from __future__ import annotations

import re
from typing import Optional

from config import (
    GROUP_LOAN_KEYWORDS,
    INDIVIDUAL_LOAN_KEYWORDS,
    _MYANMAR_DIGIT_MAP,
    _NUMERAL_TO_MODE,
    _ORDINAL_TO_MODE,
    _RE_LAKH_AMOUNT,
    _RE_MONTHS,
    _RE_PLAIN_MMK_AMOUNT,
    LOAN_CATEGORY_KEYWORDS,
)
from utils.text_utils import contains_any


def _to_ascii_digits(text: str) -> str:
    """Convert Myanmar numerals (၀-၉) to ASCII digits for regex matching."""
    return text.translate(_MYANMAR_DIGIT_MAP)


def extract_amount_mmk(text: str) -> Optional[float]:
    """
    Extract a loan amount in MMK from free text, handling both:
      - "X သိန်း" / "X lakh(s)" (X * 100,000)
      - Plain 4+ digit amounts optionally followed by ကျပ်/kyat/mmk
    Handles Myanmar numerals transparently.
    """
    ascii_text = _to_ascii_digits(text)

    m = _RE_LAKH_AMOUNT.search(ascii_text)
    if m:
        return float(m.group(1)) * 100_000

    m2 = _RE_PLAIN_MMK_AMOUNT.search(ascii_text)
    if m2:
        val = float(m2.group(1))
        if val >= 1000:
            return val

    return None


def extract_months(text: str) -> Optional[int]:
    """Extract a tenure in months from free text (e.g. '12 လ', '18 months')."""
    ascii_text = _to_ascii_digits(text)
    m = _RE_MONTHS.search(ascii_text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 24:
            return val
    return None


def detect_loan_mode(q_norm: str) -> Optional[str]:
    """Return 'individual', 'group', or None based on keywords present."""
    if contains_any(q_norm, GROUP_LOAN_KEYWORDS):
        return "group"
    if contains_any(q_norm, INDIVIDUAL_LOAN_KEYWORDS):
        return "individual"
    return None


def resolve_mode_reply(raw_query: str, q_norm: str) -> Optional[str]:
    """
    Resolve a reply to the individual/group clarifying question, accepting
    the keyword form ("individual"/"group"), a bare menu number ("1"/"2",
    including Myanmar numerals), or an ordinal word ("first"/"ပထမ").
    """
    cleaned = re.sub(r"[^\w]", "", raw_query.strip())
    if cleaned in _NUMERAL_TO_MODE:
        return _NUMERAL_TO_MODE[cleaned]

    mode = detect_loan_mode(q_norm)
    if mode is not None:
        return mode

    for word, m in _ORDINAL_TO_MODE.items():
        if word in q_norm:
            return m
    return None


def calculate_microfinance_loan(principal: float, months: int) -> str:
    """
    Compute a full Declining Balance loan repayment summary at 28% p.a.

    Args:
        principal: Loan amount in MMK.  Must be > 0.
        months:    Repayment period in months.  Must be in [6, 24].

    Returns:
        Formatted multi-line string ready for display.

    Raises:
        ValueError: When arguments are outside valid ranges.
    """
    if principal <= 0:
        raise ValueError(f"principal must be positive, got {principal}")
    if not 6 <= months <= 24:
        raise ValueError(f"months must be in [6, 24], got {months}")

    monthly_rate     = 0.28 / 12
    service_fee      = principal * 0.02
    welfare_fee      = principal * 0.005
    actual_disbursed = principal - service_fee - welfare_fee
    monthly_principal = principal / months
    total_interest   = 0.0
    remaining        = principal

    for _ in range(months):
        total_interest += remaining * monthly_rate
        remaining      -= monthly_principal

    total_payable       = principal + total_interest
    avg_monthly_payment = total_payable / months
    sep                 = "\u2500" * 50

    return (
        f"💵 ချေးငွေအရင်း                              : {principal:,.0f} MMK\n"
        f"📈 နှစ်စဉ်အတိုးနှုန်း (Declining Balance 28%) : 28%\n"
        f"📅 ပြန်ဆပ်ရမည့် သက်တမ်း                      : {months} လ\n"
        f"{sep}\n"
        f"💰 ထုတ်ယူချိန်တွင် နုတ်ယူမည့် စရိတ်များ\n"
        f"   ▸ ဝန်ဆောင်ခ (2%)          : {service_fee:,.0f} MMK\n"
        f"   ▸ ဖူလုံရေးကြေး (0.5%)    : {welfare_fee:,.0f} MMK\n"
        f"💵 လက်ဝယ်ရရှိမည့် ငွေပမာဏ  : {actual_disbursed:,.0f} MMK\n"
        f"{sep}\n"
        f"📈 ပြန်လည်ပေးဆပ်ရမည့် အခြေအနေ\n"
        f"   ▸ စုစုပေါင်း ကျသင့်သည့် အတိုး             : {total_interest:,.0f} MMK\n"
        f"   ▸ စုစုပေါင်း ပြန်ဆပ်ရမည့် ငွေ (အရင်း+အတိုး) : {total_payable:,.0f} MMK\n"
        f"     (ပထမလ အများဆုံး ဆပ်ရပြီး လစဉ် တဖြည်းဖြည်း လျော့ညွှန်းသွားပါမည်)\n"
        f"   ➡️  ပျမ်းမျှ လစဉ်ဆပ်ရမည့် ငွေ               : {avg_monthly_payment:,.0f} MMK / လ"
    )

def parse_loan_category(q_norm: str) -> Optional[str]:
    """Maps free-text category replies ('business', 'စိုက်ပျိုး') to the
    canonical category name used in loan.json / entities."""
    for category, keywords in LOAN_CATEGORY_KEYWORDS.items():
        if contains_any(q_norm, keywords):
            return category
    return None

