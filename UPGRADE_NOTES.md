# Loan RAG Chatbot — Upgrade Package

## Files delivered

| File | Purpose |
|---|---|
| `intent_training_data.json` | 50-sample bilingual training dataset (4 intents) |
| `train_intent.py` | Full mBERT + sklearn training pipeline |
| `rag1_patches.py` | Three drop-in code patches for `rag1.py` |

---

## 1. Training Dataset (`intent_training_data.json`)

### Distribution
| Intent | Count | Languages |
|---|---|---|
| LOAN_INQUIRY | 25 | my + en |
| CALCULATE | 10 | my + en |
| COMPLAINT | 7 | my + en |
| THANK | 8 | my + en |

### Design decisions
- Myanmar samples use polite particles (`ပါခင်ဗျာ`, `ရှင့်`) to match real user register.
- LOAN_INQUIRY covers all 3 product types + interest method + eligibility edge cases (foreigners).
- CALCULATE includes both keyword-only samples *and* samples with embedded amounts — this teaches BERT to fire the intent even when the user buries the request in a full sentence.
- COMPLAINT is intentionally smaller than LOAN_INQUIRY to avoid the over-firing bug that was present in the original code; the confidence floor patch (Section A) handles ambiguous cases.

### How to train
```bash
python train_intent.py --data intent_training_data.json --folds 5
# Outputs: artifacts/bert_intent_model.pkl + artifacts/label_encoder.pkl
```

---

## 2. CALCULATE Fast-Path (Section B patch)

### Routing architecture
```
User query
    │
    ▼
Safety / Greeting / Thank handlers (unchanged)
    │
    ▼
BERT intent classification
    ├── THANK      → thanks_handler (instant)
    ├── COMPLAINT  → complaint_handler (instant, with loan-keyword guard)
    ├── CALCULATE  → ┐
    │                 ├─ Both amounts detected? → calculate_microfinance_loan() [DETERMINISTIC]
    │                 └─ Missing amounts?       → LAUNCH_CALCULATOR wizard
    └── LOAN_INQUIRY → FAISS → Gemini RAG
```

### Why zero LLM latency matters for CALCULATE
- The LLM could produce rounded or approximate figures that differ from the
  Declining Balance formula — a compliance risk in microfinance.
- `calculate_microfinance_loan()` is already deterministic and auditable.
- BERT fires the intent in ~50 ms on CPU; the calc itself is microseconds.

---

## 3. Grounded RAG Prompt (Section C patch)

### Key changes vs original
| Dimension | Before | After |
|---|---|---|
| Temperature | 0.3 | **0.15** (tighter factual grounding) |
| Financial facts | Embedded in `CORE_PROJECT_RULES` only | **Separate `FINANCIAL_FACTS` block** that the model cannot ignore |
| Hallucination guard | Implicit | **Explicit "NEVER fabricate" instruction + fallback phrase** |
| Numeric accuracy | LLM may calculate inline | LLM redirects user to built-in calculator |
| Prompt structure | Flat string | **`build_rag_prompt()` function** — testable + reusable |

### Hallucination failure modes patched
1. **Rate fabrication** — LLM previously could output e.g. "25% flat rate". Now the FINANCIAL_FACTS block anchors every response to 28% declining balance.
2. **External bank reference** — Added explicit SCOPE GUARD prohibiting comparison with other banks.
3. **Tenure out-of-range** — FINANCIAL_FACTS states 6–24 month bounds explicitly.
4. **Fee omission** — Service fee (2%) and welfare fee (0.5%) are now in the hard-coded facts, not just in loan.json.

---

## Apply patches to rag1.py — checklist

- [ ] Run `python train_intent.py` → confirm `artifacts/bert_intent_model.pkl` and `artifacts/label_encoder.pkl` exist.
- [ ] In `rag1.py`, replace `predict_user_intent_with_ml()` with **Section A**.
- [ ] In `rag1.py` `retrieve()`, add **Section B** immediately after the COMPLAINT block.
- [ ] In `rag1.py`, replace `SYSTEM_INSTRUCTION` and the Gemini call block with **Section C**.
- [ ] Re-run `python rag1.py --build --json loan.json` to refresh the FAISS index.
- [ ] Test: `python rag1.py --query "ချေးငွေ ၅ သိန်း ၁၂ လ ဆပ်မယ်ဆို တွက်ပေးပါ"` → should return inline calc result, NOT Gemini.
