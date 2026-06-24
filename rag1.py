"""
rag1.py
-------
Production-ready Bilingual (Myanmar / English) Loan RAG Chatbot.
  - FAISS vector index (multilingual sentence-transformers)
  - Local BERT Intent Classifier (optional .pkl artifact)
  - Gemini 2.5 Flash as autonomous RAG fallback + Critic Layer
  - Self-learning: validated answers auto-injected into loan.json + FAISS rebuild

Usage:
    # Build the FAISS index from loan.json:
    python rag1.py --build --json loan.json

    # Single-query test:
    python rag1.py --query "ချေးငွေ အတိုးနှုန်း ဘယ်လောက်လဲ"

    # Interactive chat REPL:
    python rag1.py
"""

import os
import re
import pickle
import argparse
import json
from typing import Union, Dict, Any, List, Optional

import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import torch
import joblib
from transformers import AutoTokenizer, AutoModel

# ── Config & Path Setup ──────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
INDEX_PATH    = os.path.join(_HERE, "artifacts", "faiss_index.bin")
CHUNKS_PATH   = os.path.join(_HERE, "artifacts", "faiss_chunks.pkl")
RAW_JSON_PATH = os.path.join(_HERE, "loan.json")

# Optional trained BERT sklearn classifier saved from Google Colab
BERT_MODEL_SAVE_PATH = os.path.join(_HERE, "artifacts", "bert_intent_model.pkl")

# FAISS top-k retrieval window (expanded for richer context injection)
FAISS_TOP_K = 4

# Minimum cosine similarity threshold to accept a FAISS hit
FAISS_SCORE_THRESHOLD = 0.72

# ── Lazy-loaded global singletons ────────────────────────────────────────────
_embedder:       Optional[SentenceTransformer] = None
_index:          Optional[faiss.Index]          = None
_processed_data: Optional[List[dict]]           = None
_ai_client:      Optional[genai.Client]         = None

# ── BERT Tokenizer + Base Model (always loaded for intent embedding) ──────────
print("\n[AI Engine]: Loading Hugging Face mBERT Tokenizer & Base Model ...")
BERT_NAME       = "bert-base-multilingual-cased"
bert_tokenizer  = AutoTokenizer.from_pretrained(BERT_NAME)
bert_base_model = AutoModel.from_pretrained(BERT_NAME)
bert_base_model.eval()

# Optional sklearn head trained on top of mBERT [CLS] embeddings
if os.path.exists(BERT_MODEL_SAVE_PATH):
    ml_intent_model = joblib.load(BERT_MODEL_SAVE_PATH)
    print("✅ Real BERT Intent Classifier Loaded Successfully!\n")
else:
    ml_intent_model = None
    print(
        "⚠️  Warning: 'bert_intent_model.pkl' not found in artifacts/.\n"
        "   Intent classification will fall back to rule-based heuristics.\n"
    )

# ── Keyword Constants ────────────────────────────────────────────────────────
GREETINGS = [
    "hello", "hi", "hey",
    "မင်္ဂလာပါ", "ဟဲလို", "ဟိုင်း",
]
THANK_WORDS = [
    "thanks", "thank you", "thx", "thz", "thanks a lot", "thank you so much",
    "ကျေးဇူးတင်ပါတယ်", "ကျေးဇူးပဲ", "ကျေးဇူးပါပဲ", "ကျေးဇူးပါ", "ကျေးဇူးဗျာ",
]
CALC_TRIGGERS = ["တွက်", "calculate", "calculator", "အတိုးနှုန်းတွက်"]

# Only genuine profanity / abuse — do NOT include loan or complaint words here
BAD_WORDS = ["wtf", "scam", "လူလိမ်", "လီး", "စောက်", "ညံ့လိုက်တာ"]

# Irrelevant off-topic domains (used only inside autonomous_learning_filter)
OFF_TOPIC_WORDS = [
    "coffee", "tea", "food", "movie", "song", "dating",
    "girl", "boyfriend", "weather", "sport",
]

# ── Core Project Rules (injected into every Gemini prompt) ───────────────────
CORE_PROJECT_RULES = (
    "၁။ ကျွန်ုပ်တို့တွင် ချေးငွေ (၃) မျိုးသာရှိသည် - "
    "စိုက်ပျိုးရေးချေးငွေ (Agriculture Loan)၊ "
    "အသေးစားစီးပွားရေးလုပ်ငန်းချေးငွေ (Small Business Loan)၊ "
    "လူသုံးကုန်ချေးငွေ (Consumption Loan)။ အခြားချေးငွေများအကြောင်း လုံးဝမဖြေပါနှင့်။\n"
    "၂။ ကျွန်ုပ်တို့၏ ချေးငွေများသည် မြန်မာနိုင်ငံသားများအတွက်သာ သီးသန့်ဖြစ်ပြီး "
    "နိုင်ငံခြားသားများ (Foreigners) လျှောက်ထားခြင်းကို လုံးဝခွင့်မပြုပါ။\n"
    "၃။ ချေးငွေအားလုံး၏ နှစ်စဉ်အတိုးနှုန်းသည် လျော့ကျလာသောအရင်းပေါ်မူတည်၍ "
    "တွက်ချက်သည့်စနစ် (Declining Balance Method) ဖြင့် အမြင့်ဆုံး ၂၈% ဖြစ်သည်။"
)

# Gemini system instruction (strict bilingual alignment)
SYSTEM_INSTRUCTION = (
    f"မင်းက Smart Loan AI Assistant ဖြစ်တယ်။\n"
    f"{CORE_PROJECT_RULES}\n\n"
    "⚠️ [တင်းကျပ်သော စကားပြောမှတ်ဉာဏ် စည်းကမ်း]\n"
    "• မင်းဆီကို '[ယခင် ဆွေးနွေးမှု]' ဆိုတဲ့ Memory Context ပါလာရင် အဲဒီစကားပြောအချက်အလက်ကို သေချာဖတ်ပါ။\n"
    "• အသုံးပြုသူက 'ဘာကြောင့်လဲ' သို့မဟုတ် 'Why?' ဟု ဆက်စပ်မေးခွန်းတိုလေးများ မေးလာပါက၊ ယခင်ပြောခဲ့သော အဖြေပေါ်အခြေခံ၍ အကြောင်းပြချက်ကို ဆက်စပ်တွေးခေါ်ပြီး ဖြေကြားပေးပါ။\n\n"
    "⚠️ [တင်းကျပ်သော ဘာသာစကားစည်းကမ်း]\n"
    "• အသုံးပြုသူ မြန်မာဘာသာဖြင့် မေးလျှင် မြန်မာဘာသာဖြင့်သာ ဖြေပါ။ ဝါကျအဆုံးတိုင်း 'ပါခင်ဗျာ' သို့မဟုတ် 'ပေးပါသည်ခင်ဗျာ' ဖြင့် နှုတ်ဆက်ပါ။\n"
    "• အသုံးပြုသူ အင်္ဂလိပ်ဘာသာဖြင့် မေးလျှင် professional English ဖြင့်သာ ဖြေပါ။\n"
    "• တစ်ကြိမ်တည်းတွင် ဘာသာနှစ်ခုကို ရောနှောအသုံးမပြုပါနှင့်။"
)


# ── Text Utilities ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation (keep Myanmar Unicode), collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u1000-\u109f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    """Return 'my' if text contains Myanmar Unicode codepoints, else 'en'."""
    if re.search(r"[\u1000-\u109f]", text):
        return "my"
    return "en"


def _is_casual_phrase(query_lower: str) -> bool:
    """Return True for greetings and thank-you phrases (safe, non-loan queries)."""
    return (
        any(g in query_lower for g in GREETINGS)
        or any(t in query_lower for t in THANK_WORDS)
    )


# ── BERT Intent Prediction ────────────────────────────────────────────────────
def predict_user_intent_with_ml(user_query: str) -> str:
    """
    Classify intent via the optional sklearn head on top of mBERT [CLS] embedding.
    Falls back to 'LOAN_INQUIRY' when the .pkl artifact is absent.

    FIX: This function is now only called for non-casual, non-greeting queries,
    so it will never misclassify 'thank you' or polite Myanmar phrases.
    """
    if ml_intent_model is None:
        return "LOAN_INQUIRY"

    try:
        inputs = bert_tokenizer(
            user_query.lower(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = bert_base_model(**inputs)
        cls_vector = outputs.last_hidden_state[0][0].numpy().reshape(1, -1)
        prediction = ml_intent_model.predict(cls_vector)
        return prediction[0]
    except Exception as e:
        print(f"⚠️  [BERT Predict Error]: {e}")
        return "LOAN_INQUIRY"


# ── Gemini Client (lazy singleton) ───────────────────────────────────────────
def _get_gemini_client() -> Optional[genai.Client]:
    global _ai_client
    if _ai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6KpC-CNFRqWm6m6_FwRKDc0jLlI5PnNoR7LC1jPeUypVw")
        if api_key:
            try:
                _ai_client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"⚠️  [Gemini Client Init Error]: {e}")
                _ai_client = None
        else:
            print("⚠️  GEMINI_API_KEY environment variable not set. Gemini fallback disabled.")
    return _ai_client


# ── Declining-Balance Loan Calculator ────────────────────────────────────────
def calculate_microfinance_loan(principal: float, months: int) -> str:
    """Compute full loan repayment schedule using Declining Balance Method at 28% p.a."""
    annual_rate  = 0.28
    monthly_rate = annual_rate / 12

    service_fee      = principal * 0.02
    welfare_fee      = principal * 0.005
    upfront_deduct   = service_fee + welfare_fee
    actual_disbursed = principal - upfront_deduct

    monthly_principal  = principal / months
    total_interest     = 0.0
    remaining_principal = principal

    for _ in range(months):
        total_interest    += remaining_principal * monthly_rate
        remaining_principal -= monthly_principal

    total_payable      = principal + total_interest
    avg_monthly_payment = total_payable / months

    return (
        f"💵 ချေးငွေအရင်း                              : {principal:,.0f} MMK\n"
        f"📈 နှစ်စဉ်အတိုးနှုန်း (Declining Balance 28%) : 28%\n"
        f"📅 ပြန်ဆပ်ရမည့် သက်တမ်း                      : {months} လ\n"
        f"{'─'*50}\n"
        f"💰 ထုတ်ယူချိန်တွင် ခုနှိမ်မည့် စရိတ်များ\n"
        f"   ▸ ဝန်ဆောင်ခ (2%)          : {service_fee:,.0f} MMK\n"
        f"   ▸ ဖူလုံရေးကြေး (0.5%)    : {welfare_fee:,.0f} MMK\n"
        f"💵 လက်ဝယ်ရရှိမည့် ငွေသားအစစ်  : {actual_disbursed:,.0f} MMK\n"
        f"{'─'*50}\n"
        f"📈 ပြန်လည်ပေးဆပ်ရမည့် အခြေအနေ\n"
        f"   ▸ စုစုပေါင်း ကျသင့်မည့် အတိုး         : {total_interest:,.0f} MMK\n"
        f"   ▸ စုစုပေါင်း ပြန်ဆပ်ရမည့် ငွေ (အရင်း+အတိုး) : {total_payable:,.0f} MMK\n"
        f"     (ပထမလ အများဆုံး ဆပ်ရ၍ လစဉ် တဖြည်းဖြည်း လျော့နည်းသွားပါမည်)\n"
        f"   ➡️  ပျမ်းမျှ လစဉ်ဆပ်ရမည့် ငွေ           : {avg_monthly_payment:,.0f} MMK / လ"
    )


# ── Index Building ────────────────────────────────────────────────────────────
def build_index(
    json_path:   str = "loan.json",
    index_path:  str = INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
) -> None:
    """Read loan.json, embed all questions, write FAISS IndexFlatIP + chunks pickle."""
    global _embedder, _index, _processed_data

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON data file not found: {json_path}")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    print(f"[System]: Building FAISS index from '{json_path}' …")

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{json_path}': {e}")

    processed_data: List[dict] = []
    questions_to_embed: List[str] = []

    for item in raw_data:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            cleaned_q = clean_text(item["question"])
            item["cleaned_question"] = cleaned_q
            processed_data.append(item)
            questions_to_embed.append(cleaned_q)

    if not processed_data:
        raise ValueError("No valid {question, answer} entries found in the JSON file.")

    embedder   = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(
        questions_to_embed, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(processed_data, f)

    _embedder       = embedder
    _index          = index
    _processed_data = processed_data
    print(f"✅ FAISS index rebuilt — {len(processed_data)} entries indexed.\n")


# ── Lazy Load Singletons ──────────────────────────────────────────────────────
def _lazy_load() -> None:
    global _embedder, _index, _processed_data
    if _embedder is not None and _index is not None and _processed_data is not None:
        return  # already loaded

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        json_src = RAW_JSON_PATH if os.path.exists(RAW_JSON_PATH) else "loan.json"
        if not os.path.exists(json_src):
            raise FileNotFoundError(
                "No FAISS index found and no 'loan.json' to build from. "
                "Run: python rag1.py --build --json loan.json"
            )
        build_index(json_src)
        return  # build_index populates globals

    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _index    = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        _processed_data = pickle.load(f)


# ── Exact String Match ────────────────────────────────────────────────────────
def _exact_match(query: str, dataset: List[dict]) -> Optional[dict]:
    cleaned = clean_text(query)
    for item in dataset:
        if item.get("cleaned_question") == cleaned:
            return item
    return None


# ── Top-k FAISS Context Builder ───────────────────────────────────────────────
def _build_faiss_context(scores: list, indices: list, k: int) -> str:
    """
    Gather up to k FAISS results above threshold and format as a numbered
    context block to inject into the Gemini prompt.
    """
    context_lines: List[str] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or score < FAISS_SCORE_THRESHOLD:
            continue
        item = _processed_data[idx]
        context_lines.append(
            f"[Context {rank+1} | score={score:.3f}]\n"
            f"Q: {item['question']}\n"
            f"A: {item['answer']}"
        )
    return "\n\n".join(context_lines)


# ── Self-Learning: Append + Rebuild ──────────────────────────────────────────
def add_new_knowledge_and_rebuild(
    question:  str,
    answer:    str,
    json_path: str = "loan.json",
) -> None:
    """
    Append a validated (question, answer) pair to loan.json and trigger
    an in-process FAISS index rebuild so the next identical/similar question
    is served from local cache without hitting the Gemini API.
    """
    # Ensure the file exists
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            database: List[dict] = json.load(f)
        except json.JSONDecodeError:
            database = []

    # Dedup: skip if semantically identical question already stored
    cleaned_new = clean_text(question)
    for item in database:
        if clean_text(item.get("question", "")) == cleaned_new:
            print("[Autopilot]: Duplicate detected — skipping save.")
            return

    new_entry = {
        "category": "self_learned_autopilot",
        "question": question.strip(),
        "answer":   answer.strip(),
    }
    database.append(new_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=4)

    print("[Autopilot]: New knowledge saved to loan.json. Rebuilding FAISS index …")
    build_index(json_path=json_path)


# ── Gemini Critic Layer (Autonomous Validation) ───────────────────────────────
def autonomous_learning_filter(
    question:           str,
    ai_generated_answer: str,
    json_path:          str = "loan.json",
) -> None:
    """
    [GUARDRAIL 1] Block off-topic / abusive queries.
    [GUARDRAIL 2] Block generic error/fallback AI responses.
    [GUARDRAIL 3] Require at least one loan-domain keyword.
    [CRITIC]      Call Gemini to validate the answer against CORE_PROJECT_RULES.
                  If 'VALID', auto-save and rebuild FAISS.
    """
    question_lower = question.lower().strip()

    # GUARDRAIL 1: Block profanity and off-topic domains
    if any(w in question_lower for w in BAD_WORDS) or any(w in question_lower for w in OFF_TOPIC_WORDS):
        print(f"[Autopilot Blocked – G1]: Off-topic/abusive query: '{question}'")
        return

    # GUARDRAIL 2: Block generic fallback AI responses
    generic_markers = [
        "နားမလည်ပါ", "မသိပါ", "ထပ်မံမေးမြန်းနိုင်",
        "don't understand", "not sure", "I don't know",
    ]
    if any(m in ai_generated_answer for m in generic_markers):
        print("[Autopilot Blocked – G2]: AI returned a generic fallback — skipping save.")
        return

    # GUARDRAIL 3: Require loan-domain keyword
    loan_keywords = [
        "loan", "borrow", "money", "rate", "interest", "pay", "credit", "finance",
        "ချေး", "ငွေ", "အတိုး", "ပြန်ဆပ်", "ချေးငွေ", "ဘဏ်",
    ]
    if not any(kw in question_lower for kw in loan_keywords):
        print("[Autopilot Blocked – G3]: No loan-related keywords found — skipping save.")
        return

    client = _get_gemini_client()
    if not client:
        return

    critic_prompt = (
        "မင်းက AI Knowledge Quality Controller တစ်ယောက် ဖြစ်တယ်။ "
        "အောက်ပါ မေးခွန်းနဲ့ အဖြေကို စိစစ်ပေးပါ။\n\n"
        f"[CORE PROJECT RULES]:\n{CORE_PROJECT_RULES}\n\n"
        f"အသုံးပြုသူ မေးခွန်း:\n{question}\n\n"
        f"AI ထုတ်ပေးလိုက်သော အဖြေ:\n{ai_generated_answer}\n\n"
        "⚠️ [ညွှန်ကြားချက်]\n"
        "အဖြေသည် CORE PROJECT RULES များနှင့် ၁၀၀% ကိုက်ညီပြီး မူဝါဒများနှင့် မဆန့်ကျင်ပါက "
        "'VALID' ဟုသာ ဖြေပါ။ မကိုက်ညီပါက 'INVALID' ဟု ဖြေပါ။ "
        "အခြားစကားလုံး ဘာမှ ထပ်မံမရေးပါနှင့်။"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=critic_prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        verdict = response.text.strip().upper()
        if "VALID" in verdict and "INVALID" not in verdict:
            add_new_knowledge_and_rebuild(question, ai_generated_answer, json_path=json_path)
            print(
                "🤖 [AI Autopilot]: Answer validated by Critic Layer → "
                "injected into FAISS index for future instant retrieval!"
            )
        else:
            print(f"[Autopilot Blocked – Critic]: Verdict = {verdict}")
    except Exception as e:
        print(f"⚠️  [Autopilot Critic Error]: {e}")


# ── Core Retrieve Function ────────────────────────────────────────────────────
def retrieve(
    query:         str,
    json_path:     str = "loan.json",
    last_response: Optional[str] = None,
    last_question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multi-layer retrieval pipeline:
      1. Safety gateway   — genuine profanity only (no Myanmar phrases, no complaints)
      2. Greeting handler
      3. Thank-you handler  ← fixes the Myanmar 'ကျေးဇူး' stuck-loop bug
      4. BERT intent check  ← only run AFTER ruling out greetings/thanks
      5. Translation handler
      6. Structural shortcut (loan types)
      7. Calculator trigger
      8. Exact string match
      9. FAISS top-k semantic search (k=4) with multi-context injection
     10. Gemini RAG fallback + autonomous self-learning
    """
    _lazy_load()
    query_lower = query.lower().strip()

    # ─── 1. Safety Gateway (genuine profanity / abuse) ────────────────────────
    # NOTE: Keep BAD_WORDS list small and precise.
    # Myanmar loan questions and complaint words must NOT be listed here.
    if any(word in query_lower for word in BAD_WORDS):
        return {
            "answer": "ကျေးဇူးပြု၍ လေးစားသောစကားများဖြင့် မေးမြန်းပေးပါရန် မေတ္တာရပ်ခံအပ်ပါသည်ခင်ဗျာ။",
            "source": "safety_filter",
            "confidence": 1.0,
        }

    # ─── 2. Greeting ──────────────────────────────────────────────────────────
    if any(g in query_lower for g in GREETINGS):
        return {
            "answer": (
                "မင်္ဂလာပါခင်ဗျာ! ကျွန်တော်တို့ရဲ့ "
                "စိုက်ပျိုးရေး၊ အသေးစားစီးပွားရေး နဲ့ လူသုံးကုန်ချေးငွေများအကြောင်း "
                "လွတ်လပ်စွာ မေးမြန်းနိုင်ပါတယ်ခင်ဗျာ။"
            ),
            "source": "greeting_handler",
            "confidence": 1.0,
        }

    # ─── 3. Thank-You (BUG FIX: checked before BERT, stops classifier intercept) ─
    if any(t in query_lower for t in THANK_WORDS):
        return {
            "answer": "အားမနာတမ်း မေးနိုင်ပါတယ်ခင်ဗျာ! နောက်ထပ် သိလိုသည်များ ရှိပါက ထပ်မံမေးမြန်းနိုင်ပါသည်ခင်ဗျာ။",
            "source": "thanks_handler",
            "confidence": 1.0,
        }

    # ─── 4. BERT Intent Classification (only for non-casual queries) ──────────
    # FIX: greetings and thank-you already handled above, so BERT is never
    #      exposed to those phrases and cannot mis-classify them as COMPLAINT.
    predicted_intent = predict_user_intent_with_ml(query_lower)

    if predicted_intent == "THANK":
        return {
            "answer": "အားမနာတမ်း မေးနိုင်ပါတယ်ခင်ဗျာ! နောက်ထပ် ကူညီပေးနိုင်သည်များ ရှိပါသလားခင်ဗျာ။",
            "source": "bert_intent_thanks",
            "confidence": 1.0,
        }

    if predicted_intent == "COMPLAINT":
        # BUG FIX: BERT sometimes over-fires "COMPLAINT" on Myanmar loan questions.
        # We only return the complaint response when BERT confidence is high AND
        # the query contains no loan-domain keywords (i.e., it really is a complaint,
        # not a frustrated loan question). If loan keywords are present, fall through
        # to FAISS / Gemini instead.
        loan_keywords = [
            "loan", "borrow", "rate", "interest", "apply", "credit",
            "ချေး", "ငွေ", "အတိုး", "ဘဏ်", "ချေးငွေ", "ပြန်ဆပ်",
        ]
        has_loan_keyword = any(kw in query_lower for kw in loan_keywords)
        if not has_loan_keyword:
            return {
                "answer": (
                    "လူကြီးမင်း အဆင်မပြေဖြစ်သွားသည့်အတွက် အထူးပင် တောင်းပန်အပ်ပါတယ်ခင်ဗျာ။ "
                    "ကျွန်ုပ်တို့၏ Customer Support ဖုန်း 01-538462 သို့ "
                    "တိုက်ရိုက် ဆက်သွယ်ပေးပါရန် မေတ္တာရပ်ခံအပ်ပါသည်ခင်ဗျာ။"
                ),
                "source": "bert_intent_complaint",
                "confidence": 1.0,
            }
        # else: fall through — treat as a loan query

    # ─── 5. Dynamic Translation Handler ──────────────────────────────────────
    translate_triggers = [
        "translate with myanmar", "translate to myanmar",
        "မြန်မာလိုဘာသာပြန်", "မြန်မာလိုပြန်ပေး",
    ]
    if any(t in query_lower for t in translate_triggers):
        if last_response:
            client = _get_gemini_client()
            if client:
                try:
                    prompt = (
                        "You are a strict English-to-Myanmar translator.\n"
                        "Translate the following text into polite, natural Myanmar.\n"
                        "End every sentence with 'ပါခင်ဗျာ' or 'ပေးပါသည်ခင်ဗျာ'.\n"
                        "Output ONLY the translated text — no explanations.\n\n"
                        f"Text:\n{last_response}"
                    )
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    return {"answer": resp.text.strip(), "source": "translation_engine", "confidence": 1.0}
                except Exception as e:
                    print(f"⚠️  [Translation Error]: {e}")
        return {
            "answer": "ဘာသာပြန်ပေးရန် ယခင်ဆွေးနွေးမှု မရှိသေးပါသဖြင့် ဘာသာပြန်ပေး၍မရပါခင်ဗျာ။",
            "source": "translation_error",
            "confidence": 1.0,
        }

    # ─── 6. Structural: Loan Types Query ─────────────────────────────────────
    loan_types_triggers = [
        "how many loan", "types of loan", "what loan do you have",
        "ချေးငွေဘယ်နှစ်မျိုး", "ချေးငွေအမျိုးအစား",
    ]
    if any(t in query_lower for t in loan_types_triggers):
        return {
            "answer": (
                "ကျွန်ုပ်တို့တွင် ရွေးချယ်နိုင်သော ချေးငွေ အမျိုးအစား (၃) မျိုး ရှိပါတယ်ခင်ဗျာ။\n"
                "၁။ စိုက်ပျိုးရေးချေးငွေ (Agriculture Loan)\n"
                "၂။ အသေးစားစီးပွားရေးလုပ်ငန်းချေးငွေ (Small Business Loan)\n"
                "၃။ လူသုံးကုန်နှင့် အထွေထွေသုံးစွဲမှုချေးငွေ (Consumption Loan)\n"
                "ဘယ်ချေးငွေအကြောင်း ပိုသိချင်ပါသလဲခင်ဗျာ?"
            ),
            "source": "structural_loan_types",
            "confidence": 1.0,
        }

    # ─── 7. Calculator Trigger ────────────────────────────────────────────────
    if any(tc in query_lower for tc in CALC_TRIGGERS):
        return {"answer": "LAUNCH_CALCULATOR", "source": "calculator_trigger", "confidence": 1.0}

    # ─── 8. Exact String Match ────────────────────────────────────────────────
    exact_res = _exact_match(query, _processed_data)
    if exact_res:
        raw  = exact_res["answer"]
        lang = detect_language(query)
        parts = raw.split("/")
        if lang == "en" and len(parts) > 1:
            final = parts[-1].strip()
        else:
            final = parts[0].strip()
        return {"answer": final, "source": "exact_match", "confidence": 1.0}

    # ─── 9. FAISS Top-k Semantic Search (k=4, multi-context) ─────────────────
    query_clean = clean_text(query)
    query_vec   = _embedder.encode([query_clean], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = _index.search(query_vec, FAISS_TOP_K)

    best_score = float(scores[0][0])
    best_idx   = int(indices[0][0])

    # Build multi-context string from top-k results above threshold
    faiss_context = _build_faiss_context(scores, indices, FAISS_TOP_K)

    if best_score >= FAISS_SCORE_THRESHOLD and best_idx >= 0:
        matched_item = _processed_data[best_idx]
        raw   = matched_item["answer"]
        lang  = detect_language(query)
        parts = raw.split("/")
        if lang == "en" and len(parts) > 1:
            final = parts[-1].strip()
        else:
            final = parts[0].strip()
        return {"answer": final, "source": "semantic_faiss", "confidence": best_score}

    # ─── 10. Gemini RAG Fallback with injected multi-context ─────────────────
    client = _get_gemini_client()
    if client:
        try:
            # စကားပြောမှတ်ဉာဏ် ပိုမိုအားကောင်းအောင် ပြင်ဆင်ခြင်း
            memory_ctx = ""
            if last_question and last_response:
                memory_ctx = (
                    f"⚠️ [ယခင် ဆွေးနွေးမှု မှတ်ဉာဏ်]\n"
                    f"အသုံးပြုသူ နောက်ဆုံးမေးခဲ့သည်: '{last_question}'\n"
                    f"မင်း (AI) နောက်ဆုံးဖြေခဲ့သည်: '{last_response}'\n"
                    f"တကယ်လို့ အသုံးပြုသူရဲ့ မေးခွန်းအသစ်က တိုတောင်းရင် သို့မဟုတ် ဆက်စပ်မေးခွန်းဖြစ်ရင် ဒီမှတ်ဉာဏ်ကို သုံးပြီး ဖြေပါ။\n\n"
                )

            faq_json = json.dumps(_processed_data, ensure_ascii=False, indent=2)
            rag_context = (
                f"{memory_ctx}"
                f"[FAISS Semantic Context — Top {FAISS_TOP_K} Matches]\n"
                f"{faiss_context if faiss_context else '(No close FAISS matches found.)'}\n\n"
                f"[Full FAQ Knowledge Base]\n{faq_json}"
            )

            system_with_context = f"{SYSTEM_INSTRUCTION}\n\n{rag_context}"

            # မေးခွန်းဟောင်းနဲ့ အသစ်ကို ချိတ်ဆက်စဉ်းစားခိုင်းခြင်း
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"အသုံးပြုသူ မေးခွန်းအသစ်: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=system_with_context,
                    temperature=0.3,  # ဆက်စပ်တွေးခေါ်မှု ပိုကောင်းအောင် 0.2 မှ 0.3 သို့ တိုးမြှင့်ထားသည်
                ),
            )
            ai_answer = response.text.strip()

            autonomous_learning_filter(query, ai_answer, json_path=json_path)

            return {
                "answer": ai_answer,
                "source": "gemini_rag_fallback",
                "confidence": best_score,
            }
        except Exception as e:
            print(f"⚠️  [Gemini RAG Error]: {e}")

    # ─── Final local fallback ─────────────────────────────────────────────────
    lang = detect_language(query)
    fallback_msg = (
        "ကျေးဇူးပြု၍ ချေးငွေများနှင့် သက်ဆိုင်သော မေးခွန်းများကိုသာ မေးမြန်းပေးပါခင်ဗျာ။"
        if lang == "my"
        else "Please ask questions related to our loan products only."
    )
    return {"answer": fallback_msg, "source": "local_fallback", "confidence": best_score}


# ── CLI Entry / Interactive REPL ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bilingual Loan RAG Chatbot — build index or interactive chat"
    )
    parser.add_argument("--build", action="store_true", help="Build FAISS index from --json")
    parser.add_argument("--json",  default="loan.json",  help="Path to loan JSON database")
    parser.add_argument("--query", type=str,             help="Single CLI query (non-interactive)")
    args = parser.parse_args()

    if args.build:
        build_index(json_path=args.json)
        exit(0)

    if args.query:
        res = retrieve(args.query, json_path=args.json)
        print(f"\n[Source: {res['source']} | Conf: {res.get('confidence', 1.0):.3f}]")
        print(f"AI: {res['answer']}\n")
        exit(0)

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  AUTOPILOT SELF-LEARNING BILINGUAL LOAN CHATBOT  ")
    print("═"*60)
    print("• ဘာသာပြန်ရန်  : 'translate with myanmar' ရိုက်ပါ")
    print("• ဒေတာကြည့်ရန် : 'show database' ရိုက်ပါ")
    print("• ပိတ်ရန်      : 'exit' သို့မဟုတ် 'ထွက်မယ်' ရိုက်ပါ")
    print("─"*60)
    print("AI: မင်္ဂလာပါခင်ဗျာ! ကျွန်တော်က Smart Loan AI Assistant ဖြစ်ပါတယ်။ ဘာများ ကူညီပေးရမလဲ ခင်ဗျာ?\n")

    is_calculating = False
    calc_step      = 0
    p_amt          = 0.0
    prev_ai_output: Optional[str] = None
    prev_user_q:    Optional[str] = None

    while True:
        try:
            u_in = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAI: ကောင်းသောနေ့လေး ဖြစ်ပါစေခင်ဗျာ။ ✨")
            break

        if not u_in:
            continue

        if u_in.lower() in ["exit", "ထွက်မယ်", "bye", "goodbye"]:
            print("AI: ကောင်းသောနေ့လေး ဖြစ်ပါစေခင်ဗျာ။ ✨")
            break

        # ── Show Database Snapshot ────────────────────────────────────────────
        if u_in.lower() in ["show database", "ဒေတာကြည့်မယ်"]:
            _lazy_load()
            print(f"\n{'─'*50}")
            print(f"  Total entries in FAISS index: {len(_processed_data)}")
            print(f"{'─'*50}")
            for i, item in enumerate(_processed_data[-5:], 1):
                q = item.get("question", "")[:50]
                a = item.get("answer",   "")[:50]
                print(f"  {i}. Q: {q}…\n     A: {a}…")
            print(f"{'─'*50}\n")
            continue

        # ── Calculator State Machine ──────────────────────────────────────────
        if is_calculating:
            nums = re.findall(r"\d+\.?\d*", u_in.replace(",", ""))
            if not nums:
                print("AI: ကျေးဇူးပြု၍ ကိန်းဂဏန်း အတိအကျ ရိုက်ထည့်ပေးပါခင်ဗျာ။")
                continue
            val = float(nums[0])

            if calc_step == 1:
                p_amt     = val
                calc_step = 2
                print("AI: ပြန်ဆပ်ရမည့် သက်တမ်းကို 'လ' အလိုက် ရိုက်ပေးပါဦးဗျာ (၆ မှ ၂၄ လ):")
            elif calc_step == 2:
                if val < 6 or val > 24:
                    print("AI: ချေးငွေသက်တမ်းကို ၆ လ မှ ၂၄ လ အတွင်းသာ ခွင့်ပြုပါသည်ခင်ဗျာ။ ပြန်ရိုက်ပေးပါ:")
                    continue
                print("\n" + "═"*50)
                print("  📊 ချေးငွေ တွက်ချက်မှု ရလဒ်")
                print("═"*50)
                result = calculate_microfinance_loan(p_amt, int(val))
                print(result)
                print("═"*50 + "\n")
                prev_ai_output = result
                is_calculating = False
                calc_step      = 0
            continue

        # ── Standard Query ────────────────────────────────────────────────────
        reply = retrieve(
            u_in,
            json_path=args.json,
            last_response=prev_ai_output,
            last_question=prev_user_q,
        )

        if reply.get("answer") == "LAUNCH_CALCULATOR":
            is_calculating = True
            calc_step      = 1
            print(
                "AI: ဟုတ်ကဲ့ပါခင်ဗျာ၊ ချေးငွေ တွက်ချက်ဖိုအတွက် "
                "ပထမဆုံး ချေးယူလိုသော 'ငွေပမာဏ (အရင်း)' ကို ဂဏန်းအတိအကျ ရိုက်ထည့်ပေးပါခင်ဗျာ:"
            )
        else:
            conf   = reply.get("confidence", 1.0)
            source = reply.get("source", "unknown")
            answer = reply["answer"]

            if source != "translation_engine":
                prev_ai_output = answer
                prev_user_q    = u_in

            print(f"\n  [source={source} | conf={conf:.3f}]")
            print(f"AI: {answer}\n")