"""
rag/dialogue_manager.py — generalized dialogue management layer.

Sits BETWEEN conversation memory and the existing retrieve → rerank →
generate pipeline. Does not touch embeddings, FAISS, the reranker, or
the Qwen client. Its only job is to answer two questions that apply
identically to every topic (rule #11), not to any single hardcoded
question:

    1. Is this message a continuation of the assistant's last turn
       (a short "Yes"/"No"/"ပြီးပြီ" reply), or a new standalone query?
    2. Once we have retrieved+reranked candidates, do they span more
       than one customer group (Farmer / MSME / Salaried Employee) in a
       way that would produce a merged, contradictory answer — and if
       so, can that be resolved silently from memory/query text, or
       does it genuinely require asking the user one question?

Everything here is data-driven off the KB's existing `category` field,
not per-question rules — the same logic applies whether the topic is
eligibility, documents, interest, fees, branches, or complaints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Customer-group taxonomy, derived from your existing KB categories ──
# Add to this mapping if you add new customer-group-specific categories
# later (e.g. a future "Pensioner Loan" category) — everything NOT
# listed here is treated as general/cross-group and is never filtered
# or flagged as ambiguous.
GROUP_CATEGORIES: dict[str, set[str]] = {
    "farmer":          {"Agriculture Loan"},
    "msme":            {"MSME Loan"},
    "salary_employee": {"Consumption Loan"},
}
_CATEGORY_TO_GROUP: dict[str, str] = {
    cat: group for group, cats in GROUP_CATEGORIES.items() for cat in cats
}
_GROUP_SPECIFIC_CATEGORIES = set(_CATEGORY_TO_GROUP.keys())

_GROUP_LABELS_MY = {
    "farmer": "စိုက်ပျိုးရေးချေးငွေ (Agriculture Loan)",
    "msme": "အသေးစားစီးပွားရေးချေးငွေ (MSME Loan)",
    "salary_employee": "လူသုံးကုန်ချေးငွေ (Consumption Loan)",
}

# Query-text hints for resolving group without asking, when the user's
# own wording already names it (e.g. "Do I need NRC for agriculture
# loan?" should never trigger a clarify question).
_GROUP_QUERY_HINTS: dict[str, tuple[str, ...]] = {
    "farmer": ("လယ်", "တောင်သူ", "စိုက်ပျိုး", "farmer", "agriculture", "livestock", "crop"),
    "msme": ("ဆိုင်", "စီးပွားရေး", "business", "msme", "shop", "enterprise"),
    "salary_employee": ("လစာ", "ဝန်ထမ်း", "salary", "employee", "consumption", "civil servant"),
}

# Continuation-reply vocabulary — short acknowledgement/confirmation
# phrases in Myanmar and English. Checked only against SHORT messages
# (<= 4 words), and only meaningful when there's a pending assistant
# question to attach to — see is_continuation_reply().
_CONTINUATION_PHRASES = (
    "yes", "no", "ok", "okay", "done", "finished", "already done",
    "ပြီးပြီ", "ပြီးပါပြီ", "မရှိဘူး", "ရှိတယ်", "ဟုတ်ကဲ့", "မဟုတ်ဘူး", "ရပြီ",
)


@dataclass
class DialogueDecision:
    """
    What the caller (rag_pipeline.py) should do next.

    action:
        "ANSWER"       — proceed with retrieval/rerank/generate as normal,
                          using `effective_query` in place of the raw
                          user message.
        "ACKNOWLEDGE"  — skip retrieval and generation entirely; return
                          `direct_answer` immediately.
        "CLARIFY"      — skip generation; return `direct_answer` (the
                          clarifying question) immediately, and memory
                          has been updated so the NEXT turn can resolve it.
    """
    action: str
    effective_query: str
    direct_answer: Optional[str] = None
    group_filter: Optional[str] = None


def _resolve_group_from_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    for group, hints in _GROUP_QUERY_HINTS.items():
        if any(h.lower() in text_lower for h in hints):
            return group
    return None


def resolve_customer_group(query: str, memory) -> Optional[str]:
    """
    Try to resolve which customer group the current turn is about,
    WITHOUT asking — checked in priority order: memory (already
    established earlier in this conversation), then the current
    query's own wording. Returns None if genuinely unresolved.
    """
    loan_category = (memory.entities.get("loan_category") or "").lower() if memory else ""
    if loan_category:
        for group, cats in GROUP_CATEGORIES.items():
            if any(loan_category in cat.lower() or cat.lower() in loan_category for cat in cats):
                return group
        # loose match against the group key itself, e.g. entities may
        # already store "agriculture"/"msme"/"consumption" directly
        for group in GROUP_CATEGORIES:
            if group.split("_")[0] in loan_category or loan_category in group:
                return group

    return _resolve_group_from_text(query)


def get_result_category(result) -> str:
    inner = getattr(result, "document", result)
    return getattr(inner, "category", "") or (result.get("category", "") if isinstance(result, dict) else "")


def detect_present_groups(results: list) -> set[str]:
    """Which customer groups are actually present among the candidates."""
    groups = set()
    for r in results:
        cat = get_result_category(r)
        if cat in _GROUP_SPECIFIC_CATEGORIES:
            groups.add(_CATEGORY_TO_GROUP[cat])
    return groups


def filter_by_customer_group(results: list, group: Optional[str]) -> list:
    """
    Drop candidates belonging to a DIFFERENT customer group than
    `group`. General (non-group-specific) candidates are always kept.
    If `group` is None, returns `results` unchanged — filtering only
    ever narrows when we actually know which group is relevant.
    """
    if group is None:
        return results
    kept = []
    for r in results:
        cat = get_result_category(r)
        if cat not in _GROUP_SPECIFIC_CATEGORIES or _CATEGORY_TO_GROUP.get(cat) == group:
            kept.append(r)
    return kept


def is_continuation_reply(q_norm: str) -> bool:
    words = q_norm.split()
    if len(words) > 4:
        return False
    q_lower = q_norm.lower()
    return any(phrase in q_lower for phrase in _CONTINUATION_PHRASES)


class DialogueManager:
    """
    Stateless logic, all state lives in the ConversationMemory instance
    passed in. Call `decide()` once per turn, before retrieval.
    """

    def decide(self, query: str, q_norm: str, memory) -> DialogueDecision:
        # ── Case 1: this reply is answering OUR pending clarify question ──
        if memory is not None and getattr(memory, "pending_clarify_query", None):
            if is_continuation_reply(q_norm) or _resolve_group_from_text(query):
                resolved_group = _resolve_group_from_text(query)
                original_query = memory.pending_clarify_query
                memory.pending_clarify_query = None
                memory.pending_clarify_type = None
                if resolved_group:
                    memory.update_entities({"loan_category": resolved_group})
                return DialogueDecision(
                    action="ANSWER",
                    effective_query=original_query,
                    group_filter=resolved_group,
                )
            # User asked something unrelated instead of answering — drop
            # the pending clarify state and treat this as a fresh query.
            memory.pending_clarify_query = None
            memory.pending_clarify_type = None

        # ── Case 2: generic continuation reply with no pending question ──
        # (rule 7/8) — safest generalized behavior: don't guess what a
        # bare "ပြီးပြီ"/"Yes" means out of context; skip retrieval and
        # the LLM call entirely, and invite the next real question.
        if is_continuation_reply(q_norm) and not (memory and getattr(memory, "pending_clarify_query", None)):
            has_prior_turn = memory is not None and len(memory.messages) > 0
            if has_prior_turn:
                return DialogueDecision(
                    action="ACKNOWLEDGE",
                    effective_query=query,
                    direct_answer=(
                        "ကောင်းပါပြီခင်ဗျာ။ အခြားမေးမြန်းလိုသည်များ ရှိပါက "
                        "ဆက်လက်မေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"
                    ),
                )

        # ── Case 3: normal query — proceed to retrieval as usual ──
        return DialogueDecision(action="ANSWER", effective_query=query)

    def check_group_ambiguity(
        self, query: str, memory, results: list
    ) -> Optional[DialogueDecision]:
        """
        Call AFTER retrieval+rerank. Returns a CLARIFY decision only if
        the top results genuinely span multiple customer groups AND
        that cannot be resolved from memory or the query text — i.e.
        this is a last resort, per rules #1-#5.

        Returns None if no clarification is needed (the normal case).
        """
        present_groups = detect_present_groups(results)
        if len(present_groups) < 2:
            return None  # single group or fully general — no ambiguity

        resolved = resolve_customer_group(query, memory)
        if resolved is not None:
            return None  # resolvable silently — caller should filter, not ask

        labels = "၊ ".join(_GROUP_LABELS_MY[g] for g in sorted(present_groups))
        question = (
            f"ဤအချက်အလက်သည် ချေးငွေအမျိုးအစားပေါ် မူတည်၍ ကွာခြားနိုင်ပါတယ်ခင်ဗျာ။ "
            f"{labels} အနက် မည်သည့်အမျိုးအစားအတွက် သိလိုပါသနည်း?"
        )
        if memory is not None:
            memory.pending_clarify_query = query
            memory.pending_clarify_type = "customer_group"
        return DialogueDecision(action="CLARIFY", effective_query=query, direct_answer=question)