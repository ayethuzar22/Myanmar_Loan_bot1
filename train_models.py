"""
train_models.py
---------------
Trains three models on the loan dataset:
  1. XGBoost classifier     → loan approval (binary)
  2. RandomForestRegressor  → interest rate (regression)
  3. XGBoost multi-class    → repayment term in months (12/24/36/60)

For each model: 5-fold CV, metrics report, joblib save.
SHAP values are generated for the approval classifier.
"""

import os
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    accuracy_score,
)
from xgboost import XGBClassifier, XGBRegressor
import shap

from preprocessing import preprocess

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = "plots"


# ── Utility ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_model(model, name: str) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  ✓ Saved → {path}")
    return path


# ── Model 1: Loan Approval Classifier (XGBoost binary) ───────────────────────

def train_approval_classifier(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "=" * 60)
    print("MODEL 1: Loan Approval Classifier (XGBoost)")
    print("=" * 60)

    # Class imbalance handling via scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1
    print(f"  Class ratio neg/pos = {scale_pos_weight:.2f} → used as scale_pos_weight")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")
    print("\n  ── 5-Fold CV (Out-of-Fold) Report ──")
    print(classification_report(y_train, oof_preds, target_names=["Rejected", "Approved"]))

    # Final fit on full training set
    model.fit(X_train, y_train)

    # Hold-out test evaluation
    test_preds = model.predict(X_test)
    print("  ── Hold-out Test Set Report ──")
    print(classification_report(y_test, test_preds, target_names=["Rejected", "Approved"]))

    # ── SHAP values ──────────────────────────────────────────────────────────
    print("  Generating SHAP values (this may take a moment)…")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    explainer = shap.TreeExplainer(model)

    # Use a sample of test set for speed (max 500 rows)
    sample_size = min(500, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]

    shap_values = explainer.shap_values(X_sample)

    # Summary bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance — Loan Approval")
    plt.tight_layout()
    bar_path = os.path.join(PLOTS_DIR, "shap_approval_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  ✓ SHAP bar plot saved → {bar_path}")

    # Beeswarm plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm — Loan Approval")
    plt.tight_layout()
    bee_path = os.path.join(PLOTS_DIR, "shap_approval_beeswarm.png")
    plt.savefig(bee_path, dpi=150)
    plt.close()
    print(f"  ✓ SHAP beeswarm plot saved → {bee_path}")

    # Save SHAP explainer alongside model
    joblib.dump(explainer, os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"))

    save_model(model, "loan_approval_model")
    return model


# ── Model 2: Interest Rate Regressor (RandomForest) ───────────────────────────

def train_interest_rate_regressor(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODEL 2: Interest Rate Regressor (Random Forest)")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold CV (regression)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(model, X_train, y_train, cv=cv)
    cv_rmse = rmse(y_train, oof_preds)
    print(f"\n  5-Fold CV RMSE : {cv_rmse:.4f}%")

    # Final fit
    model.fit(X_train, y_train)

    # Hold-out evaluation
    test_preds = model.predict(X_test)
    test_rmse = rmse(y_test, test_preds)
    print(f"  Test set RMSE  : {test_rmse:.4f}%")
    print(f"  Test set R²    : {model.score(X_test, y_test):.4f}")

    # Feature importance plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    importances = model.feature_importances_
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=30, ha="right")
    plt.title("Feature Importance — Interest Rate (Random Forest)")
    plt.tight_layout()
    fi_path = os.path.join(PLOTS_DIR, "rf_interest_rate_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  ✓ Feature importance plot saved → {fi_path}")

    save_model(model, "interest_rate_model")
    return model


# ── Model 3: Repayment Term Classifier (XGBoost multi-class) ─────────────────

def train_term_classifier(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODEL 3: Repayment Term Classifier (XGBoost multi-class)")
    print("=" * 60)

    unique_terms = sorted(np.unique(y_train))
    print(f"  Unique term classes (months): {unique_terms}")

    # Re-encode term values to 0-indexed class labels
    term_to_label = {t: i for i, t in enumerate(unique_terms)}
    label_to_term = {i: t for t, i in term_to_label.items()}
    y_train_enc = np.array([term_to_label[t] for t in y_train])
    y_test_enc = np.array([term_to_label.get(t, 0) for t in y_test])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(unique_terms),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(model, X_train, y_train_enc, cv=cv, method="predict")
    print("\n  ── 5-Fold CV Report ──")
    print(classification_report(
        y_train_enc, oof_preds,
        target_names=[f"{label_to_term[i]}mo" for i in range(len(unique_terms))]
    ))

    # Final fit
    model.fit(X_train, y_train_enc)

    # Hold-out evaluation
    test_preds = model.predict(X_test)
    print("  ── Hold-out Test Set Report ──")
    print(classification_report(
        y_test_enc, test_preds,
        target_names=[f"{label_to_term[i]}mo" for i in range(len(unique_terms))]
    ))

    # Save term mapping alongside model
    joblib.dump(label_to_term, os.path.join(ARTIFACTS_DIR, "term_label_map.pkl"))
    print(f"  ✓ Term label map saved → {ARTIFACTS_DIR}/term_label_map.pkl")

    save_model(model, "term_model")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main(csv_path: str = "loan_data.csv"):
    # Preprocess
    splits = preprocess(csv_path, artifacts_dir=ARTIFACTS_DIR)
    feature_names = splits["feature_names"]

    # ── Model 1: Loan approval classifier ────────────────────────────────────
    c = splits["classification"]
    train_approval_classifier(
        c["X_train"], c["y_train"],
        c["X_test"], c["y_test"],
        feature_names,
    )

    # ── Model 2: Interest rate regressor (trained on approved loans only) ─────
    # Filter training set to approved loans only, so the model learns rates
    # conditioned on approval (matching real-world logic).
    r = splits["int_rate"]
    approved_mask_train = splits["classification"]["y_train"] == 1
    approved_mask_test = splits["classification"]["y_test"] == 1
    train_interest_rate_regressor(
        r["X_train"][approved_mask_train], r["y_train"][approved_mask_train],
        r["X_test"][approved_mask_test], r["y_test"][approved_mask_test],
    )

    # ── Model 3: Repayment term classifier ───────────────────────────────────
    t = splits["term"]
    train_term_classifier(
        t["X_train"], t["y_train"],
        t["X_test"], t["y_test"],
    )

    print("\n✅  All models trained and saved to:", ARTIFACTS_DIR)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "loan_data.csv"
    main(csv_path)
