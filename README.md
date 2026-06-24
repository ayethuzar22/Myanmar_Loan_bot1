# Loan Chatbot — Backend

End-to-end ML + RAG chatbot for loan eligibility prediction and Q&A.
Exposes a FastAPI REST API consumed by a Flutter mobile app.

---

## Project structure

```
loan_chatbot/
├── preprocessing.py      # Data cleaning, encoding, scaling, train/test split
├── train_models.py       # Trains 3 models + SHAP, saves to artifacts/
├── rag.py                # FAISS index builder + retrieve() function
├── main.py               # FastAPI app (/predict + /chat)
├── requirements.txt
├── loan_data.csv         # ← your Kaggle dataset (add this)
├── loan_faq.txt          # ← your FAQ / policy document (add this)
└── artifacts/            # ← auto-created by training scripts
    ├── loan_approval_model.pkl
    ├── interest_rate_model.pkl
    ├── term_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── term_label_map.pkl
    ├── feature_names.pkl
    ├── shap_explainer.pkl
    ├── faiss_index.bin
    └── faiss_chunks.pkl
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Kaggle dataset
#    Download from: https://www.kaggle.com/datasets/
#    Rename or configure the path in preprocessing.py
cp ~/Downloads/loan_data.csv .

# 4. Add your FAQ / policy document
#    Plain text file with loan policies, interest rate tables, eligibility rules
echo "Your loan FAQ content here..." > loan_faq.txt

# 5. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."     # or add to a .env file
```

---

## Step 1 — Preprocess & Train

```bash
# Train all three models (runs preprocessing internally)


# Outputs:
#   artifacts/loan_approval_model.pkl   (XGBoost classifier)
#   artifacts/interest_rate_model.pkl   (RandomForest regressor)
#   artifacts/term_model.pkl            (XGBoost multi-class)
#   plots/shap_approval_bar.png
#   plots/shap_approval_beeswarm.png
```

---

## Step 2 — Build the RAG index

```bash
python rag.py --build --faq loan_faq.txt

# Test retrieval:
python rag.py --query "What is the maximum loan amount I can apply for?"
```

---

## Step 3 — Run the API

```bash
uvicorn main:app --reload --port 8000

# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

---

## API reference

### POST /predict

**Request:**
```json
{
  "annual_income": 75000,
  "loan_amount": 15000,
  "credit_score": 720,
  "dti": 18.5,
  "employment_years": 4,
  "home_ownership": "rent",
  "purpose": "debt_consolidation"
}
```

**Response:**
```json
{
  "approved": true,
  "interest_rate": 11.45,
  "term_months": 36,
  "confidence": 0.8712,
  "explanation": "Based on your profile (credit score 720, annual income $75,000, DTI 18.5%), your loan of $15,000 is likely to be approved with an estimated interest rate of 11.45% APR over 36 months."
}
```

### POST /chat

**Request:**
```json
{
  "message": "What documents do I need to apply?",
  "history": [
    {"role": "user", "content": "Hi, can I get a loan?"},
    {"role": "assistant", "content": "Hello! I'd be happy to help..."}
  ],
  "loan_profile": {
    "approved": true,
    "confidence": 0.87,
    "interest_rate": 11.45,
    "term_months": 36
  }
}
```

**Response:**
```json
{
  "reply": "To apply for a personal loan, you'll typically need to provide..."
}
```

---

## Flutter integration

In your Flutter app, add the `http` package to `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.2.0
```

Then use this Dart service class:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class LoanApiService {
  static const String baseUrl = 'http://your-server:8000';

  static Future<Map<String, dynamic>> predictLoan({
    required double annualIncome,
    required double loanAmount,
    required double creditScore,
    required double dti,
    required double employmentYears,
    required String homeOwnership,
    required String purpose,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'annual_income': annualIncome,
        'loan_amount': loanAmount,
        'credit_score': creditScore,
        'dti': dti,
        'employment_years': employmentYears,
        'home_ownership': homeOwnership,
        'purpose': purpose,
      }),
    );
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }
    throw Exception('Prediction failed: ${response.body}');
  }

  static Future<String> sendMessage({
    required String message,
    required List<Map<String, String>> history,
    Map<String, dynamic>? loanProfile,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'history': history,
        if (loanProfile != null) 'loan_profile': loanProfile,
      }),
    );
    if (response.statusCode == 200) {
      return jsonDecode(response.body)['reply'] as String;
    }
    throw Exception('Chat failed: ${response.body}');
  }
}
```

---

## Deployment (production checklist)

- [ ] Set `ANTHROPIC_API_KEY` as a secret environment variable
- [ ] Change CORS `allow_origins` from `["*"]` to your Flutter app's domain
- [ ] Use `gunicorn -k uvicorn.workers.UvicornWorker main:app` for multi-worker production
- [ ] Mount `artifacts/` as a persistent volume (Docker) so models survive restarts
- [ ] Add rate limiting (e.g. `slowapi`) to the `/chat` endpoint
- [ ] Monitor SHAP drift over time — retrain when feature importance shifts significantly
