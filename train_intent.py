"""
train_intent.py
---------------
Train an mBERT-based sklearn intent classifier for the Bilingual Loan Chatbot.
Saves: artifacts/bert_intent_model.pkl

Intents: LOAN_INQUIRY | CALCULATE | COMPLAINT | THANK

Usage:
    python train_intent.py --data intent_training_data.json [--epochs 4]
"""

import os
import json
import argparse
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel

# ── Config ───────────────────────────────────────────────────────────────────
BERT_NAME         = "bert-base-multilingual-cased"
ARTIFACTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_SAVE_PATH   = os.path.join(ARTIFACTS_DIR, "bert_intent_model.pkl")
ENCODER_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

VALID_INTENTS = {"LOAN_INQUIRY", "CALCULATE", "COMPLAINT", "THANK"}


# ── CLS Embedding ─────────────────────────────────────────────────────────────
def get_cls_embedding(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract [CLS] token embedding from mBERT for each text.
    Returns: np.ndarray of shape (n_samples, hidden_size=768)
    """
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            outputs = model(**inputs)
            # [CLS] token is always index 0
            cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_vecs)
    return np.vstack(all_embeddings)


# ── Data Loading + Validation ─────────────────────────────────────────────────
def load_training_data(json_path: str) -> tuple[list[str], list[str]]:
    """
    Load intent_training_data.json.
    Strips JSON-style comments (// ...) so the file is human-editable.
    Returns (texts, labels).
    """
    import re
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip // line comments (not valid JSON but useful for annotation)
    raw_clean = re.sub(r"//[^\n]*", "", raw)

    data = json.loads(raw_clean)

    texts, labels = [], []
    skipped = 0
    for item in data:
        text   = item.get("text", "").strip()
        intent = item.get("intent", "").strip().upper()
        if not text or intent not in VALID_INTENTS:
            skipped += 1
            continue
        texts.append(text)
        labels.append(intent)

    if skipped:
        print(f"⚠️  Skipped {skipped} invalid entries (missing text or unknown intent).")

    print(f"✅ Loaded {len(texts)} training samples across {len(set(labels))} intents.")
    dist = {i: labels.count(i) for i in VALID_INTENTS}
    print(f"   Distribution: {dist}")
    return texts, labels


# ── Training Pipeline ─────────────────────────────────────────────────────────
def train(json_path: str, cv_folds: int = 5) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load data
    texts, labels = load_training_data(json_path)
    assert len(texts) >= 10, "Need at least 10 samples to train."

    # 2. Load mBERT
    print(f"\n[Train]: Loading {BERT_NAME} tokenizer & model …")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model     = AutoModel.from_pretrained(BERT_NAME)
    model.eval()

    # 3. Embed
    print("[Train]: Generating mBERT [CLS] embeddings …")
    X = get_cls_embedding(texts, tokenizer, model)

    # 4. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"[Train]: Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 5. Cross-validation
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        # multi_class="multinomial",
        random_state=42,
    )
    skf    = StratifiedKFold(n_splits=min(cv_folds, len(set(labels))), shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    print(f"\n[CV Results]: Macro F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    # 6. Final fit on full data
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n[Train Set Classification Report]")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # 7. Save classifier + label encoder
    joblib.dump(clf, MODEL_SAVE_PATH)
    joblib.dump(le,  ENCODER_SAVE_PATH)
    print(f"✅ Classifier saved → {MODEL_SAVE_PATH}")
    print(f"✅ LabelEncoder  saved → {ENCODER_SAVE_PATH}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="intent_training_data.json",
                        help="Path to training JSON")
    parser.add_argument("--folds",  type=int, default=5,
                        help="Number of StratifiedKFold CV folds")
    args = parser.parse_args()
    train(args.data, cv_folds=args.folds)
