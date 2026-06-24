"""
main.py
-------
FastAPI backend for the Loan Chatbot.

Endpoints:
    POST /predict  -> loan approval, interest rate, term prediction
    POST /chat     -> FAISS RAG-only conversational Q&A (no external LLM)

Run locally:
    uvicorn main:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from rag import preload_index, retrieve

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = "artifacts"

MODEL_PATHS = {
    "approval":      os.path.join(ARTIFACTS_DIR, "loan_approval_model.pkl"),
    "interest_rate": os.path.join(ARTIFACTS_DIR, "interest_rate_model.pkl"),
    "term":          os.path.join(ARTIFACTS_DIR, "term_model.pkl"),
}
SCALER_PATH       = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENCODERS_PATH     = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
TERM_MAP_PATH     = os.path.join(ARTIFACTS_DIR, "term_label_map.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")

# ── Global model store (loaded at startup) ────────────────────────────────────

models: dict[str, Any] = {}


# ── Lifespan handler (startup / shutdown) ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models and the RAG index at startup."""
    log.info("Loading ML models…")
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise RuntimeError(
                f"Model file not found: {path}\n"
                "Run train_models.py first to generate model artifacts."
            )
        models[name] = joblib.load(path)
        log.info(f"  ✓ {name} loaded from {path}")

    models["scaler"]         = joblib.load(SCALER_PATH)
    models["label_encoders"] = joblib.load(ENCODERS_PATH)
    models["term_label_map"] = joblib.load(TERM_MAP_PATH)   # {0: 36, 1: 60, …}
    models["feature_names"]  = joblib.load(FEATURE_NAMES_PATH)

    log.info("Loading RAG index…")
    preload_index()

    log.info("✅  All systems ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Chatbot API",
    version="1.0.0",
    description="ML-powered loan prediction + FAISS RAG chat",
    lifespan=lifespan,
)

# ── CORS (allow Flutter / any origin in development) ─────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your Flutter app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoanPredictRequest(BaseModel):
    annual_income:    float = Field(..., gt=0,     description="Annual income in USD")
    loan_amount:      float = Field(..., gt=0,     description="Requested loan amount in USD")
    credit_score:     float = Field(..., ge=300, le=850, description="FICO credit score")
    dti:              float = Field(..., ge=0,     description="Debt-to-income ratio (%)")
    employment_years: float = Field(..., ge=0,     description="Years of employment")
    home_ownership:   str   = Field(...,            description="rent / own / mortgage / other")
    purpose:          str   = Field(...,            description="Loan purpose (e.g. debt_consolidation)")

    @field_validator("home_ownership", "purpose", mode="before")
    @classmethod
    def lowercase_strip(cls, v: str) -> str:
        return v.lower().strip()


class LoanPredictResponse(BaseModel):
    approved:      bool
    interest_rate: float  = Field(description="Predicted APR (%); 0 if rejected")
    term_months:   int    = Field(description="Predicted repayment term in months; 0 if rejected")
    confidence:    float  = Field(description="Approval probability (0–1)")
    explanation:   str    = Field(description="Plain-language summary of the decision")


class ChatMessage(BaseModel):
    role:    str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message:  str                   = Field(..., min_length=1)
    history:  list[ChatMessage]     = Field(default_factory=list)
    loan_profile: dict | None       = Field(None, description="Optional loan context from /predict")


class ChatResponse(BaseModel):
    reply: str


# ── Feature engineering helper ────────────────────────────────────────────────

def _build_feature_vector(req: LoanPredictRequest) -> np.ndarray:
    """
    Transform a LoanPredictRequest into a scaled numpy feature vector
    matching the layout produced by preprocessing.py.
    """
    scaler: Any          = models["scaler"]
    encoders: dict       = models["label_encoders"]
    feature_names: list  = models["feature_names"]

    # Numeric columns (must match NUMERIC_COLS order in preprocessing.py)
    numeric_raw = np.array([[
        req.loan_amount,
        req.annual_income,
        req.dti,
        req.credit_score,
        req.employment_years,
    ]], dtype=float)
    numeric_scaled = scaler.transform(numeric_raw)

    # Categorical columns (must match CATEGORICAL_COLS order)
    categorical_encoded = []
    for col, value in [("home_ownership", req.home_ownership), ("purpose", req.purpose)]:
        le = encoders.get(col)
        if le is None:
            raise HTTPException(500, f"Label encoder for '{col}' not found.")
        if value not in le.classes_:
            # Fall back to the most common class (index 0 after fit)
            log.warning(f"Unknown value '{value}' for '{col}'; defaulting to '{le.classes_[0]}'")
            encoded = 0
        else:
            encoded = int(le.transform([value])[0])
        categorical_encoded.append(encoded)

    feature_vector = np.concatenate(
        [numeric_scaled[0], categorical_encoded]
    ).reshape(1, -1)

    return feature_vector


# ── /predict endpoint ─────────────────────────────────────────────────────────

@app.post("/predict", response_model=LoanPredictResponse, tags=["Prediction"])
async def predict_loan(req: LoanPredictRequest) -> LoanPredictResponse:
    """
    Predict loan approval, interest rate, and repayment term.
    Returns a confidence score and a plain-language explanation.
    """
    try:
        X = _build_feature_vector(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature encoding error: {e}")

    # ── Approval prediction ───────────────────────────────────────────────────
    approval_model = models["approval"]
    proba = approval_model.predict_proba(X)[0]   # [P(rejected), P(approved)]
    confidence = float(proba[1])
    approved = bool(approval_model.predict(X)[0] == 1)

    interest_rate = 0.0
    term_months = 0

    if approved:
        # ── Interest rate prediction ──────────────────────────────────────────
        ir_model = models["interest_rate"]
        interest_rate = round(float(ir_model.predict(X)[0]), 2)
        interest_rate = max(1.0, min(interest_rate, 36.0))  # reasonable clamp

        # ── Term prediction ───────────────────────────────────────────────────
        term_model    = models["term"]
        term_label    = int(term_model.predict(X)[0])
        term_label_map: dict = models["term_label_map"]
        term_months   = int(term_label_map.get(term_label, 36))

    # ── Plain-language explanation ────────────────────────────────────────────
    if approved:
        explanation = (
            f"Based on your profile (credit score {req.credit_score:.0f}, "
            f"annual income ${req.annual_income:,.0f}, DTI {req.dti:.1f}%), "
            f"your loan of ${req.loan_amount:,.0f} is likely to be approved "
            f"with an estimated interest rate of {interest_rate:.2f}% APR "
            f"over {term_months} months."
        )
    else:
        explanation = (
            f"Based on your current profile, this loan application is likely to be "
            f"declined (confidence: {confidence * 100:.1f}%). Common reasons include "
            "a high debt-to-income ratio, lower credit score, or insufficient income "
            "relative to the loan amount."
        )

    return LoanPredictResponse(
        approved=approved,
        interest_rate=interest_rate,
        term_months=term_months,
        confidence=round(confidence, 4),
        explanation=explanation,
    )


# ── /chat endpoint ────────────────────────────────────────────────────────────

def _build_rag_reply(message: str, rag_context: str, loan_profile: dict | None) -> str:
    """
    Compose a plain-text reply from FAISS-retrieved context chunks
    and the optional loan profile — no external LLM required.

    Strategy:
      1. Always open with the retrieved FAQ context (the most relevant chunks).
      2. If a loan profile is present, append a personalised summary.
      3. If retrieval returned nothing useful, fall back to a polite default.
    """
    sections: list[str] = []

    # ── Retrieved knowledge ───────────────────────────────────────────────────
    if rag_context and rag_context.strip():
        # Strip the "[Context N | relevance=X.XXX]" header lines added by retrieve()
        # so the reply reads as clean prose for the Flutter client.
        clean_lines = []
        for line in rag_context.splitlines():
            if line.startswith("[Context") and "relevance=" in line:
                continue
            clean_lines.append(line)
        context_text = "\n".join(clean_lines).strip()
        if context_text:
            sections.append(context_text)

    # ── Personalised loan profile summary ────────────────────────────────────
    if loan_profile:
        approved    = loan_profile.get("approved")
        confidence  = loan_profile.get("confidence", 0)
        int_rate    = loan_profile.get("interest_rate", 0)
        term_months = loan_profile.get("term_months", 0)

        if approved:
            profile_text = (
                f"Based on your submitted profile, your loan application looks likely "
                f"to be approved (confidence: {confidence * 100:.1f}%). "
                f"The estimated interest rate is {int_rate:.2f}% APR over "
                f"{term_months} months. Please contact a loan officer to confirm "
                f"the final offer and documentation requirements."
            )
        else:
            profile_text = (
                f"Based on your submitted profile, your loan application may not be "
                f"approved at this time (confidence of rejection: "
                f"{(1 - confidence) * 100:.1f}%). Common reasons include a high "
                f"debt-to-income ratio, lower credit score, or income that is "
                f"insufficient relative to the requested amount. You may wish to "
                f"speak with a loan officer to explore alternative options."
            )
        sections.append(profile_text)

    # ── Fallback when no context and no profile ───────────────────────────────
    if not sections:
        sections.append(
            "I'm sorry, I couldn't find specific information about that in our "
            "knowledge base. Please contact our loan support team directly or "
            "rephrase your question and try again."
        )

    return "\n\n".join(sections)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    """
    FAISS RAG-only Q&A endpoint.
    Retrieves the most relevant loan FAQ chunks for the user's message
    and returns a composed plain-text reply — no external LLM call.
    """
    # ── Retrieve relevant context from FAISS ──────────────────────────────────
    try:
        rag_context = retrieve(req.message, k=3)
    except Exception as e:
        log.warning(f"RAG retrieval failed: {e}. Returning fallback response.")
        rag_context = ""

    # ── Build and return reply ────────────────────────────────────────────────
    reply = _build_rag_reply(req.message, rag_context, req.loan_profile)
    return ChatResponse(reply=reply)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }