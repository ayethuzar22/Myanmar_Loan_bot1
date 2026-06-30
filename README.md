[//]: # (# Loan Chatbot — Backend)

[//]: # ()
[//]: # (End-to-end ML + RAG chatbot for loan eligibility prediction and Q&A.)

[//]: # (Exposes a FastAPI REST API consumed by a Flutter mobile app.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Project structure)

[//]: # ()
[//]: # (```)

[//]: # (loan_chatbot/)

[//]: # (├── preprocessing.py      # Data cleaning, encoding, scaling, train/test split)

[//]: # (├── train_models.py       # Trains 3 models + SHAP, saves to artifacts/)

[//]: # (├── rag.py                # FAISS index builder + retrieve&#40;&#41; function)

[//]: # (├── main.py               # FastAPI app &#40;/predict + /chat&#41;)

[//]: # (├── requirements.txt)

[//]: # (├── loan_data.csv         # ← your Kaggle dataset &#40;add this&#41;)

[//]: # (├── loan_faq.txt          # ← your FAQ / policy document &#40;add this&#41;)

[//]: # (└── artifacts/            # ← auto-created by training scripts)

[//]: # (    ├── loan_approval_model.pkl)

[//]: # (    ├── interest_rate_model.pkl)

[//]: # (    ├── term_model.pkl)

[//]: # (    ├── scaler.pkl)

[//]: # (    ├── label_encoders.pkl)

[//]: # (    ├── term_label_map.pkl)

[//]: # (    ├── feature_names.pkl)

[//]: # (    ├── shap_explainer.pkl)

[//]: # (    ├── faiss_index.bin)

[//]: # (    └── faiss_chunks.pkl)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Setup)

[//]: # ()
[//]: # (```bash)

[//]: # (# 1. Create and activate a virtual environment)

[//]: # (python -m venv .venv)

[//]: # (source .venv/bin/activate          # Windows: .venv\Scripts\activate)

[//]: # ()
[//]: # (# 2. Install dependencies)

[//]: # (pip install -r requirements.txt)

[//]: # ()
[//]: # (# 3. Add your Kaggle dataset)

[//]: # (#    Download from: https://www.kaggle.com/datasets/)

[//]: # (#    Rename or configure the path in preprocessing.py)

[//]: # (cp ~/Downloads/loan_data.csv .)

[//]: # ()
[//]: # (# 4. Add your FAQ / policy document)

[//]: # (#    Plain text file with loan policies, interest rate tables, eligibility rules)

[//]: # (echo "Your loan FAQ content here..." > loan_faq.txt)

[//]: # ()
[//]: # (# 5. Set your Anthropic API key)

[//]: # (export ANTHROPIC_API_KEY="sk-ant-..."     # or add to a .env file)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Step 1 — Preprocess & Train)

[//]: # ()
[//]: # (```bash)

[//]: # (# Train all three models &#40;runs preprocessing internally&#41;)

[//]: # ()
[//]: # ()
[//]: # (# Outputs:)

[//]: # (#   artifacts/loan_approval_model.pkl   &#40;XGBoost classifier&#41;)

[//]: # (#   artifacts/interest_rate_model.pkl   &#40;RandomForest regressor&#41;)

[//]: # (#   artifacts/term_model.pkl            &#40;XGBoost multi-class&#41;)

[//]: # (#   plots/shap_approval_bar.png)

[//]: # (#   plots/shap_approval_beeswarm.png)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Step 2 — Build the RAG index)

[//]: # ()
[//]: # (```bash)

[//]: # (python rag.py --build --faq loan_faq.txt)

[//]: # ()
[//]: # (# Test retrieval:)

[//]: # (python rag.py --query "What is the maximum loan amount I can apply for?")

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Step 3 — Run the API)

[//]: # ()
[//]: # (```bash)

[//]: # (uvicorn main:app --reload --port 8000)

[//]: # ()
[//]: # (# API docs: http://localhost:8000/docs)

[//]: # (# Health:   http://localhost:8000/health)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## API reference)

[//]: # ()
[//]: # (### POST /predict)

[//]: # ()
[//]: # (**Request:**)

[//]: # (```json)

[//]: # ({)

[//]: # (  "annual_income": 75000,)

[//]: # (  "loan_amount": 15000,)

[//]: # (  "credit_score": 720,)

[//]: # (  "dti": 18.5,)

[//]: # (  "employment_years": 4,)

[//]: # (  "home_ownership": "rent",)

[//]: # (  "purpose": "debt_consolidation")

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (**Response:**)

[//]: # (```json)

[//]: # ({)

[//]: # (  "approved": true,)

[//]: # (  "interest_rate": 11.45,)

[//]: # (  "term_months": 36,)

[//]: # (  "confidence": 0.8712,)

[//]: # (  "explanation": "Based on your profile &#40;credit score 720, annual income $75,000, DTI 18.5%&#41;, your loan of $15,000 is likely to be approved with an estimated interest rate of 11.45% APR over 36 months.")

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (### POST /chat)

[//]: # ()
[//]: # (**Request:**)

[//]: # (```json)

[//]: # ({)

[//]: # (  "message": "What documents do I need to apply?",)

[//]: # (  "history": [)

[//]: # (    {"role": "user", "content": "Hi, can I get a loan?"},)

[//]: # (    {"role": "assistant", "content": "Hello! I'd be happy to help..."})

[//]: # (  ],)

[//]: # (  "loan_profile": {)

[//]: # (    "approved": true,)

[//]: # (    "confidence": 0.87,)

[//]: # (    "interest_rate": 11.45,)

[//]: # (    "term_months": 36)

[//]: # (  })

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (**Response:**)

[//]: # (```json)

[//]: # ({)

[//]: # (  "reply": "To apply for a personal loan, you'll typically need to provide...")

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Flutter integration)

[//]: # ()
[//]: # (In your Flutter app, add the `http` package to `pubspec.yaml`:)

[//]: # ()
[//]: # (```yaml)

[//]: # (dependencies:)

[//]: # (  http: ^1.2.0)

[//]: # (```)

[//]: # ()
[//]: # (Then use this Dart service class:)

[//]: # ()
[//]: # (```dart)

[//]: # (import 'dart:convert';)

[//]: # (import 'package:http/http.dart' as http;)

[//]: # ()
[//]: # (class LoanApiService {)

[//]: # (  static const String baseUrl = 'http://your-server:8000';)

[//]: # ()
[//]: # (  static Future<Map<String, dynamic>> predictLoan&#40;{)

[//]: # (    required double annualIncome,)

[//]: # (    required double loanAmount,)

[//]: # (    required double creditScore,)

[//]: # (    required double dti,)

[//]: # (    required double employmentYears,)

[//]: # (    required String homeOwnership,)

[//]: # (    required String purpose,)

[//]: # (  }&#41; async {)

[//]: # (    final response = await http.post&#40;)

[//]: # (      Uri.parse&#40;'$baseUrl/predict'&#41;,)

[//]: # (      headers: {'Content-Type': 'application/json'},)

[//]: # (      body: jsonEncode&#40;{)

[//]: # (        'annual_income': annualIncome,)

[//]: # (        'loan_amount': loanAmount,)

[//]: # (        'credit_score': creditScore,)

[//]: # (        'dti': dti,)

[//]: # (        'employment_years': employmentYears,)

[//]: # (        'home_ownership': homeOwnership,)

[//]: # (        'purpose': purpose,)

[//]: # (      }&#41;,)

[//]: # (    &#41;;)

[//]: # (    if &#40;response.statusCode == 200&#41; {)

[//]: # (      return jsonDecode&#40;response.body&#41;;)

[//]: # (    })

[//]: # (    throw Exception&#40;'Prediction failed: ${response.body}'&#41;;)

[//]: # (  })

[//]: # ()
[//]: # (  static Future<String> sendMessage&#40;{)

[//]: # (    required String message,)

[//]: # (    required List<Map<String, String>> history,)

[//]: # (    Map<String, dynamic>? loanProfile,)

[//]: # (  }&#41; async {)

[//]: # (    final response = await http.post&#40;)

[//]: # (      Uri.parse&#40;'$baseUrl/chat'&#41;,)

[//]: # (      headers: {'Content-Type': 'application/json'},)

[//]: # (      body: jsonEncode&#40;{)

[//]: # (        'message': message,)

[//]: # (        'history': history,)

[//]: # (        if &#40;loanProfile != null&#41; 'loan_profile': loanProfile,)

[//]: # (      }&#41;,)

[//]: # (    &#41;;)

[//]: # (    if &#40;response.statusCode == 200&#41; {)

[//]: # (      return jsonDecode&#40;response.body&#41;['reply'] as String;)

[//]: # (    })

[//]: # (    throw Exception&#40;'Chat failed: ${response.body}'&#41;;)

[//]: # (  })

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Deployment &#40;production checklist&#41;)

[//]: # ()
[//]: # (- [ ] Set `ANTHROPIC_API_KEY` as a secret environment variable)

[//]: # (- [ ] Change CORS `allow_origins` from `["*"]` to your Flutter app's domain)

[//]: # (- [ ] Use `gunicorn -k uvicorn.workers.UvicornWorker main:app` for multi-worker production)

[//]: # (- [ ] Mount `artifacts/` as a persistent volume &#40;Docker&#41; so models survive restarts)

[//]: # (- [ ] Add rate limiting &#40;e.g. `slowapi`&#41; to the `/chat` endpoint)

[//]: # (- [ ] Monitor SHAP drift over time — retrain when feature importance shifts significantly)

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