"""
models/application_state.py — Conversation stage + collected loan facts.

This is the shared "memory" object that the state machine (rag/state_machine.py)
and entity extractor (utils/entity_extractor.py) both operate on. It replaces
the ad-hoc single-slot tracking (_is_awaiting_loan_mode / _find_pending_loan_amount)
that previously lived directly inside RAGPipeline.

Nothing in the existing pipeline is modified by adding this file — it is
pure addition until step 6 wires it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConversationStage(str, Enum):
    """
    Where the conversation currently stands, from the loan officer's
    point of view. A real officer doesn't jump straight to numbers —
    they establish purpose, gather details, check eligibility, then quote.
    """
    GREETING            = "greeting"
    DISCOVER_NEED       = "discover_need"        # what do you want the loan for?
    COLLECT_LOAN_TYPE    = "collect_loan_type"     # agriculture / MSME / consumption
    COLLECT_MODE        = "collect_mode"          # individual / group
    COLLECT_AMOUNT      = "collect_amount"
    COLLECT_TENURE       = "collect_tenure"
    ELIGIBILITY_CHECK    = "eligibility_check"
    QUOTE_CALCULATION    = "quote_calculation"
    DOCUMENT_GUIDANCE    = "document_guidance"
    GENERAL_QA          = "general_qa"            # free-form KB questions, any stage
    HANDOFF             = "handoff"               # escalate to human officer


@dataclass
class ApplicationState:
    """
    Accumulated facts about the customer's in-progress loan inquiry.
    One instance persists per conversation (see rag/conversation_memory.py).

    Fields are intentionally Optional — a real conversation fills these in
    gradually, one turn at a time; the state machine reads which are still
    None to decide what to ask next.
    """
    stage: ConversationStage = ConversationStage.GREETING

    loan_category: Optional[str] = None      # "Agriculture" | "MSME" | "Consumption"
    loan_mode: Optional[str] = None           # "individual" | "group"
    amount_mmk: Optional[float] = None
    tenure_months: Optional[int] = None
    monthly_income_mmk: Optional[float] = None
    has_guarantor: Optional[bool] = None
    has_nrc: Optional[bool] = None
    purpose_text: Optional[str] = None        # free-text reason, for logging/handoff

    # Bookkeeping — lets the state machine avoid re-asking the same
    # question, and lets logging/analytics see how far a customer got.
    turns_in_current_stage: int = 0
    unresolved_attempts: int = 0              # consecutive no-info / low-confidence turns

    def missing_fields(self) -> list[str]:
        """Return which core fields are still unknown, in the order a
        loan officer would naturally ask for them."""
        missing = []
        if self.loan_category is None:
            missing.append("loan_category")
        if self.loan_mode is None:
            missing.append("loan_mode")
        if self.amount_mmk is None:
            missing.append("amount_mmk")
        if self.tenure_months is None:
            missing.append("tenure_months")
        return missing

    def is_ready_for_eligibility_check(self) -> bool:
        return not self.missing_fields()