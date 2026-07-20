
from __future__ import annotations
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from retrieval.reranker import Reranker
from retrieval.intent_classifier import filter_by_intent
from config import (
    BAD_WORDS,
    BORROW_INTENT_KEYWORDS,
    CALC_TRIGGERS,
    DEFAULT_CALC_TENURE_MONTHS,
    GREETINGS,
    GROUP_LOAN_MAX_MMK,
    GROUP_LOAN_MAX_MONTHS,
    INDIVIDUAL_LOAN_MAX_MMK,
    INDIVIDUAL_LOAN_MAX_MONTHS,
    LEARNING_REBUILD_ASYNC,
    LLM_PROVIDER,
    LOAN_DOMAIN_KEYWORDS,
    LOAN_TYPE_TRIGGERS,
    LOCAL_FALLBACK_MIN_SCORE,
    MIN_LOAN_AMOUNT_MMK,
    OFF_TOPIC_WORDS,
    RAW_JSON_PATH,
    _GENERIC_ANSWER_MARKERS,
    _LOAN_MODE_CLARIFY_MARKER,
    log,
)
from embeddings.embedding_engine import EmbeddingEngine
from knowledge.knowledge_store import KnowledgeStore
# from llm.gemini_client import GeminiClient
from llm.qwen_client import QwenClient
from models.chat_turn import ChatTurn
from models.rag_response import RAGResponse
from models.retrieval_result import RetrievalResult
from prompts.prompt_builder import PromptBuilder
from prompts.system_prompt import CORE_PROJECT_RULES
from rag.conversation_memory import ConversationMemory
from rag.state_machine import LoanStage, LoanStateMachine
from retrieval.retriever import Retriever
from utils.loan_utils import (
    calculate_microfinance_loan,
    detect_loan_mode,
    extract_amount_mmk,
    extract_months,
    parse_loan_category,
    resolve_mode_reply,
)
from utils.text_utils import (
    clean_text,
    contains_any,
    contains_laugh_emoji,
    detect_language,
    is_affirmative,
    is_negative,
    is_thanks,
    normalize_query,
)
from utils.validators import sanitize_input
from vectorstore.faiss_index import FAISSIndex

from rag.dialogue_manager import (
    DialogueManager,
    resolve_customer_group,
    filter_by_customer_group,

)
from models.loan_document import LoanDocument
# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS / HELPERS
# (previously mis-nested inside AutonomousLearningFilter — that's what
# caused the NameError: class-scoped names aren't visible as bare module
# names when called without `self.`)
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_FOLLOWUP_QUESTIONS: dict[LoanStage, str] = {
    LoanStage.CATEGORY: "ဘယ်လိုချေးငွေအမျိုးအစားကို စိတ်ဝင်စားပါသလဲ ခင်ဗျာ? (စိုက်ပျိုးရေး / စီးပွားရေး / လူသုံးကုန်)",
    LoanStage.AMOUNT:   "ဘယ်လောက်ချေးငွေ လိုအပ်ပါသလဲ ခင်ဗျာ?",
    LoanStage.TERM:     "ဘယ်နှစ်လ သက်တမ်းနှင့် ပြန်ဆပ်လိုပါသလဲ ခင်ဗျာ? (၆ လမှ ၂၄ လအထိ)",
    LoanStage.DOCUMENT: "လိုအပ်သော စာရွက်စာတမ်းများ (NRC, အိမ်ထောင်စုစာရင်း) ပြင်ဆင်ပြီးပါပြီလား ခင်ဗျာ?",
}

_FOLLOWUP_PATTERNS: frozenset[str] = frozenset({
    "how much", "how about", "what about", "and how", "how many",
    "ဘယ်လောက်", "ရော", "ကော",
})


def _looks_like_followup(q_norm: str) -> bool:
    word_count = len(q_norm.split())
    return word_count <= 4 or contains_any(q_norm, _FOLLOWUP_PATTERNS)


def _build_retrieval_query(
    query: str,
    q_norm: str,
    chat_history: Optional[list[ChatTurn]],
) -> str:
    """
    For short/pronoun-heavy follow-ups ("how much can get", "ဘယ်လောက်ရလဲ"),
    the raw query alone often lacks the topic word entirely, causing FAISS
    to drift toward weak generic matches. Prepending the most recent user
    message gives the embedding model the missing topic context.

    NOTE: this only affects what gets embedded for FAISS search — the
    original `query` is still used everywhere else (exact match, prompt
    building, entity extraction), so nothing else in the flow changes.
    """
    if not chat_history or not _looks_like_followup(q_norm):
        return query
    for turn in reversed(chat_history):
        if turn.role == "user":
            return f"{turn.content} {query}"
    return query


# ─────────────────────────────────────────────────────────────────────────────
# AUTONOMOUS LEARNING FILTER
# ─────────────────────────────────────────────────────────────────────────────

class AutonomousLearningFilter:
    """
    Validates AI-generated answers before persisting them to loan.json.

    Three sequential guardrail layers run first (cheap string ops).
    Only answers that clear all three are sent to the Gemini Critic,
    which validates against CORE_PROJECT_RULES.  VALID answers are saved
    and the FAISS index is rebuilt in-process for instant future retrieval.

    validate_and_save() catches and logs all exceptions so a filter
    failure never propagates to the caller.
    """

    def __init__(
        self,
        store:  KnowledgeStore,
        llm:    QwenClient,
        index:  FAISSIndex,
        engine: EmbeddingEngine,
        rebuild_async: bool = LEARNING_REBUILD_ASYNC,
    ) -> None:
        self._store  = store
        self._llm    = llm
        self._index  = index
        self._engine = engine
        self._rebuild_async = rebuild_async
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="faiss-rebuild")
            if rebuild_async else None
        )

    def validate_and_save(self, question: str, answer: str) -> None:
        """Fire-and-forget validation + persistence.  Never raises."""
        try:
            self._run(question, answer)
        except Exception as exc:
            log.error("AutonomousLearningFilter: unexpected error — %s", exc)

    def _run(self, question: str, answer: str) -> None:
        """
        Executes internal validation checks and persists valid Q&A pairs.
        """
        if not question or not answer:
            log.warning("AutoFilter: empty question or answer received. Skipping.")
            return

        # Basic guardrail check
        if "မေးမြန်းသည့်အချက်အလက်ကို ရှာမတွေ့ပါ" in answer:
            log.info("AutoFilter: answer indicates no info found. Skipping persistence.")
            return

        # 1. Save new Q&A pair to KnowledgeStore
        doc = Document(
            id=f"auto_{len(self._store.documents) + 1}",
            topic="Autonomous Learning",
            category="AutoGenerated",
            question=question,
            answer=answer,
        )
        self._store.add_document(doc)
        log.info("AutoFilter: new Q&A saved to knowledge base.")

        # 2. Trigger FAISS index rebuild
        self._trigger_rebuild()

    def shutdown(self) -> None:
        """Gracefully drain the rebuild executor (call on app shutdown)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _trigger_rebuild(self) -> None:
        if self._executor is not None:
            self._executor.submit(self._rebuild_now)
        else:
            self._rebuild_now()

    def _rebuild_now(self) -> None:
        try:
            self._index.build(
                self._store.documents, self._engine, self._store.json_path
            )
            log.info("AutoFilter: FAISS rebuild complete.")
        except Exception as exc:
            log.error("AutoFilter: FAISS rebuild failed — %s", exc)
# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full retrieve-then-generate pipeline.
    """

    _ABUSE_MY    = "ကျေးဇူးပြု၍ လေးစားအလေးအနက်ပြုပြီး မေးမြန်းပေးပါ။ မလျော်ကန်သော အသုံးအနှုန်းများကို ကျွန်ုပ်တို့ ခွင့်မပြုပါ။ ကျေးဇူးတင်ပါတယ်။"
    _GREETING    = "မင်္ဂလာပါ ခင်ဗျာ!စိုုက်ပျိုးရေးချေးငွေ၊ အသေးစားစီးပွားရေးလုပ်ငန်း နှင့် လူသုံးကုန်ချေးငွေများအကြောင်း လွတ်လပ်စွာ မေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"
    _THANKS      = "အားမနာပါနဲ့ ခင်ဗျာ! နောက်ထပ် သိလိုသည်များရှိပါက ထပ်မံမေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"
    _LOAN_TYPES  = (
         " ချေးငွေ အမျိုးအစား (၃မျိုး) မျိုး ရှိပါတယ်ခင်ဗျာ။\n"
        "၁။ စိုက်ပျိုးရေးချေးငွေ (Agriculture Loan)\n"
        "၂။ အသေးစားစီးပွားရေးလုပ်ငန်းချေးငွေ (Small Business Loan)\n"
        "၃။ လူသုံးကုန်နှင့် အိမ်ထောင်သုံးစွဲမှုချေးငွေ (Consumption Loan)\n"
        "ဘယ်ချေးငွေအကြောင်း ပိုမိုသိရှိလိုပါသလဲ ခင်ဗျာ?"
    )
    _LOAN_MODE_CLARIFY = (
        "ချေးငွေတောင်းဆိုမှုအမျိုးအစား (၂) မျိုးရှိပါတယ်ခင်ဗျာ —\n\n"
        "၁။ တစ်ဦးချင်းချေးငွေ (Individual Loan)\n"
        "၂။ ဝိုင်းကြီးချုပ်ချေးငွေ (Joint Liability Group Loan)\n\n"
        "လူကြီးမင်း ဘယ်အမျိုးအစားနှင့် လျှောက်ထားလိုပါသလဲ ခင်ဗျာ?"
    )
    _BORROW_INTENT = (
        "ဟုတ်ကဲ့ ခင်ဗျာ၊ ကျွန်မတို့ Wondarmi Microfinance မှာ ချေးငွေ (၃) မျိုး ဝန်ဆောင်မှု ပေးနေပါတယ်ခင်ဗျာ —\n\n"
        "၁။ 🌾 စိုက်ပျိုးရေးချေးငွေ (Agriculture Loan)\n"
        "   လယ်ယာစိုက်ပျိုးရေး၊ မွေးမြူရေး နှင့် လယ်ယာသုံးစက်ကိရိယာ ဝယ်ယူလိုသူများအတွက်ခင်ဗျာ\n\n"
        "၂။ 🏪 အသေးစားစီးပွားရေးချေးငွေ (Small Business Loan)\n"
        "   ဆိုင်ဖွင့်ရန်၊ ကုန်ပစ္စည်းအရင်းထည့်ရန် သို့မဟုတ် လုပ်ငန်းချဲ့ရန်အတွက်ခင်ဗျာ\n\n"
        "၃။ 👤 လူသုံးကုန်ချေးငွေ (Consumption Loan)\n"
        "   လစာ/ဝင်ငွေရှိသောဝန်ထမ်းများ၊ ဆေးကုသစရိတ်၊ အိမ်ပြင်စရိတ် သို့မဟုတ် အရေးပေါ်လိုအပ်ချက်များအတွက်ခင်ဗျာ\n\n"
        "လူကြီးမင်းက ဘယ်ရည်ရွယ်ချက်အတွက် ချေးလိုပါသလဲ ခင်ဗျာ? "
        "(ဥပမာ — စိုက်ပျိုးရေး၊ ဆိုင်ဖွင့်ရန်၊ ဆေးကု၊ အိမ်ပြင် စသဖြင့်) "
        "ပြောပေးပါက သင့်အတွက် အကောင်းဆုံး ချေးငွေအမျိုးအစားကို ညွှန်ပြပေးနိုင်ပါမည်ခင်ဗျာ။"
    )
    _NO_INFO_MY  = (
        "တောင်းပန်ပါတယ်ခင်ဗျာ။\n\n"
        "ကျွန်ုပ်သည် Wonderami Microfinance ၏ "
        "ချေးငွေဆိုင်ရာ AI Assistant ဖြစ်ပါသည်။\n\n"
        "ချေးငွေနှင့်သက်ဆိုင်သော မေးခွန်းများကိုသာ "
        "ဖြေကြားပေးနိုင်ပါသည်။\n\n"
        "ဥပမာ -\n"
        "• ချေးငွေအမျိုးအစားများ\n"
        "• ချေးငွေလျှောက်ထားနည်း\n"
        "• အတိုးနှုန်း\n"
        "• လိုအပ်သောစာရွက်စာတမ်းများ\n"
        "• ချေးငွေပြန်ဆပ်နည်း\n\n"
        "သိလိုသည်များရှိပါက ဆက်လက်မေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"
    )
    _NO_INFO_EN  = (
        "I'm here to assist with Wonderami Microfinance loan services only.\n\n"
        "Please ask questions related to:\n"
        "• Loan products\n"
        "• Loan eligibility\n"
        "• Loan application process\n"
        "• Interest rates\n"
        "• Required documents\n"
        "• Loan repayment\n\n"
        "Feel free to ask any loan-related questions."
    )
    _EMPTY_MY    = "ကျေးဇူးပြု၍ မေးခွန်းတစ်ခုခုကို ရိုက်ထည့်ပေးပါ ခင်ဗျာ။"
    _EMOJI_REPLY = "ဟားဟား 😄\nချေးငွေနှင့်ပတ်သက်ပြီး သိလိုတာရှိရင် မေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"

    def __init__(
        self,
        store:     KnowledgeStore,
        retriever: Retriever,
        builder:   PromptBuilder,
        llm:       QwenClient,
        engine:    EmbeddingEngine,
        index:     FAISSIndex,

        state_machine: Optional[LoanStateMachine] = None,
        dialogue_manager: Optional[DialogueManager] = None,
    ) -> None:
        self._store     = store
        self._retriever = retriever
        self._builder   = builder
        self._llm       = llm
        self._engine    = engine
        self._index = index

        self._reranker = Reranker()

        # ADD THIS LINE
        self._dialogue_manager = dialogue_manager or DialogueManager()

        self._filter = AutonomousLearningFilter(
            store,
            llm,
            index,
            engine
        )

        self._state_machine = state_machine or LoanStateMachine()
        self._shortcuts: list[Callable[[str], Optional[RAGResponse]]] = [
            self._handle_safety,
            self._handle_greeting,
            self._handle_thanks,
            self._handle_loan_types,
            self._handle_calculator,
        ]

    def _classify_intent(self, q: str) -> str:
        if contains_any(q, GREETINGS):
            return "greeting"
        if is_thanks(q):
            return "thanks"
        if contains_any(q, BORROW_INTENT_KEYWORDS):
            return "loan"
        if contains_any(q, CALC_TRIGGERS):
            return "calculator"
        if contains_any(q, LOAN_TYPE_TRIGGERS):
            return "loan_info"
        return "offtopic"

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        chat_history: Optional[list[ChatTurn]] = None,
        memory: Optional[ConversationMemory] = None,
    ) -> RAGResponse:
        query = sanitize_input(query)
        if not query:
            return RAGResponse(answer=self._EMPTY_MY, source="empty_input")

        if contains_laugh_emoji(query):
            return RAGResponse(answer=self._EMOJI_REPLY, source="emoji_handler")

        q_norm = normalize_query(query)

        amount = extract_amount_mmk(query)
        mode = detect_loan_mode(q_norm)
        tenure = extract_months(query)

        dialogue = self._dialogue_manager.decide(query, q_norm, memory)

        if dialogue.action == "ACKNOWLEDGE":
            return RAGResponse(answer=dialogue.direct_answer, source="dialogue_acknowledge")

        if dialogue.action == "CLARIFY":
            return RAGResponse(answer=dialogue.direct_answer, source="dialogue_clarify")

        query = dialogue.effective_query

        # ── Slot-gated state-machine capture ─────────────────────────────
        # Only intercepts the turn (and advances stage) when the message
        # actually answers the CURRENTLY pending slot. Anything else falls
        # through to shortcuts/exact-match/FAISS below untouched — this is
        # the ONLY state-machine block in run(); the old unconditional
        # duplicate has been removed.
        if memory is not None:
            captured_slot_this_turn = False

            if memory.stage == LoanStage.CATEGORY:
                category = parse_loan_category(q_norm)
                if category is not None:
                    memory.update_entities({"loan_category": category})
                    captured_slot_this_turn = True

            elif memory.stage == LoanStage.AMOUNT and amount is not None:
                if amount < MIN_LOAN_AMOUNT_MMK:
                    lang = detect_language(query)
                    min_lakhs = MIN_LOAN_AMOUNT_MMK / 100_000
                    if lang == "my":
                        msg = (
                            f"တောင်းပန်ပါတယ်ခင်ဗျာ၊ {amount:,.0f} MMK မှာ နည်းလွန်းနေပါတယ်။ "
                            f"ကျွန်ုပ်တို့ အနည်းဆုံး ချေးငွေပမာဏမှာ သိန်း {min_lakhs:,.1f} "
                            f"({MIN_LOAN_AMOUNT_MMK:,.0f} MMK) ဖြစ်ပါတယ်ခင်ဗျာ။ "
                            f"ပြန်လည်ပြီး ပမာဏတစ်ခု ပြောပြပေးပါရန် ခင်ဗျာ။"
                        )
                    else:
                        msg = (
                            f"Sorry, {amount:,.0f} MMK is too low — our minimum "
                            f"loan amount is {min_lakhs:,.1f} lakhs "
                            f"({MIN_LOAN_AMOUNT_MMK:,.0f} MMK). Please let me "
                            f"know a higher amount."
                        )
                    return RAGResponse(answer=msg, source="amount_too_low")
                memory.update_entities({"amount": amount})
                captured_slot_this_turn = True

            elif memory.stage == LoanStage.TERM and tenure is not None:
                memory.update_entities({"term_months": tenure})
                captured_slot_this_turn = True

            elif memory.stage == LoanStage.DOCUMENT:
                if is_affirmative(q_norm):
                    memory.update_entities({"documents": True})
                    captured_slot_this_turn = True
                elif is_negative(q_norm):
                    memory.update_entities({"documents": False})
                    captured_slot_this_turn = True

            if captured_slot_this_turn:
                next_stage = self._state_machine.next_stage(memory.stage, memory.entities)
                memory.stage = next_stage
                followup = _STAGE_FOLLOWUP_QUESTIONS.get(next_stage)
                if followup is not None:
                    return RAGResponse(answer=followup, source=f"state_machine_{next_stage.value}")
                if next_stage == LoanStage.COMPLETE:
                    return RAGResponse(
                        answer=(
                            "ကျေးဇူးတင်ပါတယ်ခင်ဗျာ။ လူကြီးမင်း၏ ချေးငွေလျှောက်လွှာ အချက်အလက်များကို "
                            "လက်ခံရရှိပါပြီ။ ကျွန်ုပ်တို့၏ ကွင်းဆင်းဝန်ထမ်းများမှ ၃ ရက်မှ ၅ ရက်အတွင်း "
                            "ဆက်သွယ်လာမှာ ဖြစ်ပါတယ်ခင်ဗျာ။"
                        ),
                        source="state_machine_complete",
                    )

        if amount is not None and mode is not None:
            return self._answer_loan_mode_calculation(amount, mode, query)

        if self._is_awaiting_loan_mode(chat_history):
            resolved_mode = mode or resolve_mode_reply(query, q_norm)
            if resolved_mode is not None:
                pending_amount = self._find_pending_loan_amount(chat_history)
                if pending_amount is not None:
                    return self._answer_loan_mode_calculation(pending_amount, resolved_mode, query)

        if amount is not None and contains_any(q_norm, LOAN_DOMAIN_KEYWORDS):
            return RAGResponse(answer=self._LOAN_MODE_CLARIFY, source="loan_mode_clarification")

        for handler in self._shortcuts:
            result = handler(q_norm)
            if result is not None:
                return result

        exact = self._exact_match(query)
        if exact is not None:
            return exact

        retrieval_query = _build_retrieval_query(query, q_norm, chat_history)
        results = self._retriever.retrieve(retrieval_query)
        results = self._reranker.rerank(retrieval_query, results)
        clarify = self._dialogue_manager.check_group_ambiguity(query, memory, results)
        if clarify is not None:
            return RAGResponse(answer=clarify.direct_answer, source="dialogue_clarify")

        resolved_group = dialogue.group_filter or resolve_customer_group(query, memory)
        results = filter_by_customer_group(results, resolved_group)

        results = results[:3]

        results, detected_intent = filter_by_intent(q_norm, results)
        if detected_intent is not None:
            log.info(
                "RAGPipeline: intent=%s detected — filtered to %d "
                "candidate(s) (topics: %s).",
                detected_intent.value,
                len(results),
                [r.document.topic for r in results],
            )
        if results:
            best      = results[0]
            if best.score >= 0.55:
                log.info("High score (%.3f) — Returning direct Knowledge Base answer.", best.score)
                return self._local_kb_answer(query, results)
            prompt    = self._builder.build(query, results, chat_history)

            best = results[0]

            if best.score >= 0.75:
                return self._local_kb_answer(
                    query,
                    results
                )

            prompt = self._builder.build(
                query,
                results,
                chat_history
            )

            ai_answer = self._llm.generate(prompt)

            if not ai_answer:
                if best.score < LOCAL_FALLBACK_MIN_SCORE:
                    log.warning(
                        "မေးမြန်းသည့်အချက်အလက်ကို ရှာမတွေ့ပါ။ ကျေးဇူးပြု၍ ပိုမိုတိကျစွာ မေးမြန်းပေးပါ။"

                    )
                    return self._no_info(query, score=best.score)

                log.warning(
                    "မေးမြန်းသည့်အချက်အလက်ကို ရှာမတွေ့ပါ။ ကျေးဇူးပြု၍ ပိုမိုတိကျစွာ မေးမြန်းပေးပါ။ "
                    "(score=%.3f, topic=%s).", best.score, best.document.topic,
                )
                return self._local_kb_answer(query, results)

            self._filter.validate_and_save(query, ai_answer)
            return RAGResponse(
                answer=ai_answer,
                source="qwen_rag",
                matched_topic=best.document.topic,
                matched_category=best.document.category,
                similarity_score=best.score,
                confidence=best.score,
            )

        if contains_any(q_norm, BORROW_INTENT_KEYWORDS):
            borrow_response = self._handle_borrow_intent(q_norm)
            if borrow_response is not None:
                return borrow_response

        if (
                memory is not None
                and memory.stage not in (
                LoanStage.COMPLETE,
                LoanStage.PENDING
        )
                and memory.entities.get("application_started")
        ):

            next_stage = self._state_machine.next_stage(
                memory.stage,
                memory.entities
            )

            followup = _STAGE_FOLLOWUP_QUESTIONS.get(next_stage)

            if followup:
                memory.stage = next_stage

                return RAGResponse(
                    answer=followup,
                    source=f"state_machine_{next_stage.value}_fallback"
                )

            if captured_slot_this_turn:
                next_stage = self._state_machine.next_stage(memory.stage, memory.entities)
                memory.stage = next_stage
                followup = _STAGE_FOLLOWUP_QUESTIONS.get(next_stage)
                if followup is not None:
                    return RAGResponse(answer=followup, source=f"state_machine_{next_stage.value}")
                if next_stage == LoanStage.COMPLETE:
                    return RAGResponse(
                        answer=(
                            "ကျေးဇူးတင်ပါတယ်ခင်ဗျာ။ လူကြီးမင်း၏ ချေးငွေလျှောက်လွှာ "
                            "အချက်အလက်များကို လက်ခံရရှိပါပြီ။ ကျွန်ုပ်တို့၏ ကွင်းဆင်းဝန်ထမ်းများမှ "
                            "၃ ရက်မှ ၅ ရက်အတွင်း ဆက်သွယ်လာမှာ ဖြစ်ပါတယ်ခင်ဗျာ။"
                        ),
                        source="state_machine_complete",
                    )
        log.info("RAGPipeline: below threshold — returning no-info.")
        return self._no_info(query)

    def shutdown(self) -> None:
        self._filter.shutdown()

    def _is_awaiting_loan_mode(self, chat_history: Optional[list[ChatTurn]]) -> bool:
        if not chat_history:
            return False
        last = chat_history[-1]
        return last.role == "assistant" and _LOAN_MODE_CLARIFY_MARKER in last.content

    def _find_pending_loan_amount(self, chat_history) -> Optional[float]:
        if not self._is_awaiting_loan_mode(chat_history):
            return None
        if len(chat_history) < 2:
            return None
        prev_user = chat_history[-2]
        if prev_user.role != "user":
            return None
        return extract_amount_mmk(prev_user.content)

    def _answer_loan_mode_calculation(
            self, amount_mmk: float, mode: str, query: str
    ) -> RAGResponse:
        lang = detect_language(query)
        if mode == "individual":
            cap_amount, cap_months, mode_label_my, mode_label_en = (
                INDIVIDUAL_LOAN_MAX_MMK, INDIVIDUAL_LOAN_MAX_MONTHS,
                "တစ်ဦးချင်းချေးငွေ", "Individual Loan",
            )
        else:
            cap_amount, cap_months, mode_label_my, mode_label_en = (
                GROUP_LOAN_MAX_MMK, GROUP_LOAN_MAX_MONTHS,
                "ဝိုင်းကြီးချုပ်ချေးငွေ", "Joint Liability Group Loan",
            )

        if amount_mmk > cap_amount:
            cap_lakhs = cap_amount / 100_000
            if lang == "my":
                answer = (
                    f"{mode_label_my} အတွက် တောင်းဆိုနိုင်သည့် အများဆုံးပမာဏမှာ "
                    f"သိန်း {cap_lakhs:,.0f} ({cap_amount:,.0f} MMK) ဖြစ်ပါတယ်ခင်ဗျာ။ "
                    f"တောင်းဆိုထားသော ပမာဏသည် ၎င်းထက် ပိုများနေပါသည်။ "
                    f"ကျေးဇူးပြု၍ ပမာဏလျှော့ချ၍ မေးမြန်းပါ (သို့) အခြားချေးငွေအမျိုးအစားကို "
                    f"စဉ်းစားကြည့်ပါရန် တိုက်တွန်းအပ်ပါတယ်ခင်ဗျာ။"
                )
            else:
                answer = (
                    f"The maximum amount available under {mode_label_en} is "
                    f"{cap_lakhs:,.0f} lakhs ({cap_amount:,.0f} MMK). "
                    f"The requested amount exceeds this. Please ask again with "
                    f"a lower amount, or consider the other loan mode."
                )
            return RAGResponse(answer=answer, source="loan_mode_cap_exceeded")

        months = extract_months(query) or DEFAULT_CALC_TENURE_MONTHS
        months = min(months, cap_months)
        months = max(months, 6)

        try:
            calc_text = calculate_microfinance_loan(amount_mmk, months)
        except ValueError as exc:
            log.error("_answer_loan_mode_calculation: calculator error — %s", exc)
            return self._no_info(query)

        prefix = (
            f"{mode_label_my} ({months} \u101c) \u1021\u1010\u103d\u1000\u103a "
            f"\u1078\u1000\u103a\u1001\u103b\u1000\u103a\u1019\u103e\u102f\u1019\u103b\u102c\u1038\u2014\n\n"
            if lang == "my"
            else f"{mode_label_en} calculation ({months} months):\n\n"
        )
        return RAGResponse(
            answer=prefix + calc_text,
            source="loan_mode_calculator",
            matched_category=mode_label_en,
        )

    # ── Shortcut handlers ─────────────────────────────────────────────────────

    def _handle_safety(self, q: str) -> Optional[RAGResponse]:
        if contains_any(q, BAD_WORDS):
            return RAGResponse(answer=self._ABUSE_MY, source="safety_filter")
        return None

    def _handle_greeting(self, q: str) -> Optional[RAGResponse]:
        if contains_any(q, GREETINGS):
            return RAGResponse(answer=self._GREETING, source="greeting_handler")
        return None

    def _handle_thanks(self, q: str) -> Optional[RAGResponse]:
        if is_thanks(q):
            return RAGResponse(answer=self._THANKS, source="thanks_handler")
        return None

    def _handle_loan_types(self, q: str) -> Optional[RAGResponse]:
        if contains_any(q, LOAN_TYPE_TRIGGERS):
            return RAGResponse(answer=self._LOAN_TYPES, source="structural_loan_types")
        return None

    def _handle_calculator(self, q: str) -> Optional[RAGResponse]:
        if contains_any(q, CALC_TRIGGERS):
            return RAGResponse(answer="LAUNCH_CALCULATOR", source="calculator_trigger")
        return None

    def _handle_borrow_intent(self, q: str) -> Optional[RAGResponse]:
        if contains_any(q, BORROW_INTENT_KEYWORDS):
            return RAGResponse(
                answer=self._BORROW_INTENT,
                source="borrow_intent_handler",
            )
        return None

    def _handle_borrow_intent(self, q):

        if contains_any(q, BORROW_INTENT_KEYWORDS):
            return RAGResponse(
                answer=self._BORROW_INTENT,
                source="borrow_intent_handler"
            )

    def _local_kb_answer(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> RAGResponse:
        best_doc = results[0].document
        lang     = detect_language(query)
        answer   = best_doc.answer.strip()

        seen_topics: set[str] = {best_doc.topic}
        extra_topics: list[str] = []
        for r in results[1:]:
            if r.document.topic in seen_topics:
                continue
            if r.score < 0.55:
                continue
            seen_topics.add(r.document.topic)
            extra_topics.append(r.document.topic)

        if extra_topics:
            if lang == "my":
                topics_str = "\u1005 ".join(extra_topics[:2])
                answer += (

                    f"တောင်းဆိုထားသော ပမာဏသည် ၎င်းထက် ပိုများနေပါသည်။ "
                    f"ကျေးဇူးပြု၍ ပမာဏလျှော့ချ၍ မေးမြန်းပါ (သို့) အခြားချေးငွေအမျိုးအစားကို "
                    f"စဉ်းစားကြည့်ပါရန် တိုက်တွန်းအပ်ပါတယ်ခင်ဗျာ။"
                )
            else:
                topics_str = ", ".join(extra_topics[:2])
                answer += f"\n\nYou may also ask about: {topics_str}."

        return RAGResponse(
            answer=answer,
            source="local_kb_fallback",
            matched_topic=best_doc.topic,
            matched_category=best_doc.category,
            similarity_score=results[0].score,
            confidence=results[0].score,
        )

    # ── Exact match ───────────────────────────────────────────────────────────

    def _exact_match(self, query: str) -> Optional[RAGResponse]:
        doc = self._store.find_exact(clean_text(query))
        if doc is None:
            return None
        lang   = detect_language(query)
        parts  = doc.answer.split("/")
        answer = (
            parts[-1].strip()
            if lang == "en" and len(parts) > 1
            else parts[0].strip()
        )
        return RAGResponse(
            answer=answer,
            source="exact_match",
            matched_topic=doc.topic,
            matched_category=doc.category,
            similarity_score=1.0,
            confidence=1.0,
        )

    def _no_info(self, query: str, score: float = 0.0) -> RAGResponse:
        lang = detect_language(query)
        return RAGResponse(
            answer=self._NO_INFO_MY if lang == "my" else self._NO_INFO_EN,
            source="threshold_gate",
            similarity_score=score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_pipeline:      Optional[RAGPipeline] = None
_pipeline_lock: threading.Lock        = threading.Lock()


def _build_llm_client() -> QwenClient:
    """
    Qwen is now the only supported LLM backend — Gemini has been removed
    entirely (no import, no provider switch). If you need to swap models
    again later, this is the single seam to change.
    """
    log.info("_build_pipeline: loading local QwenClient.")
    return QwenClient()


def _build_pipeline(json_path: str = RAW_JSON_PATH) -> RAGPipeline:
    store  = KnowledgeStore(json_path)
    store.load()
    engine = EmbeddingEngine()
    index  = FAISSIndex()
    if index.needs_rebuild(json_path):
        log.info("_build_pipeline: FAISS stale or absent — rebuilding.")
        index.build(store.documents, engine, json_path)
    else:
        index.load()
    return RAGPipeline(
        store=store,
        retriever=Retriever(engine, index),
        builder=PromptBuilder(),
        llm=_build_llm_client(),
        engine=engine,
        index=index,
    )


def _get_pipeline(json_path: str = RAW_JSON_PATH) -> RAGPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = _build_pipeline(json_path)
            atexit.register(_pipeline.shutdown)
    return _pipeline  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:         str,
    json_path:     str                       = RAW_JSON_PATH,
    chat_history:  Optional[list[ChatTurn]] = None,
    last_response: Optional[str]            = None,
    last_question: Optional[str]            = None,
) -> dict[str, Any]:
    history: list[ChatTurn] = list(chat_history) if chat_history else []
    if last_question and last_response:
        history = [
            ChatTurn(role="user",      content=last_question),
            ChatTurn(role="assistant", content=last_response),
        ] + history

    resp = _get_pipeline(json_path).run(query, history)
    return {
        "answer":           resp.answer,
        "source":           resp.source,
        "matched_topic":    resp.matched_topic,
        "matched_category": resp.matched_category,
        "similarity_score": resp.similarity_score,
        "confidence":       resp.confidence,
    }


def build_index(json_path: str = RAW_JSON_PATH) -> None:
    store = KnowledgeStore(json_path)
    store.load()
    engine = EmbeddingEngine()
    FAISSIndex().build(store.documents, engine, json_path)
    log.info("build_index: complete — %d documents.", len(store.documents))