"""
preprocessing.py
----------------
Preprocesses the Kaggle loan dataset with columns:
    loan_id, no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value, loan_status

Returns train/test splits for:
    - classification : loan_status  (Approved=1 / Rejected=0)
    - regression     : cibil_score  (proxy for interest rate — see NOTE)
    - term           : loan_term    (repayment months)

NOTE: This dataset does not contain an interest rate column.
      cibil_score is used as the regression demonstration target.
      In production, replace it with your actual interest_rate column
      if available, or derive it from a rate table keyed on cibil_score.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Column definitions ─────────────────────────────────────────────────────────

# loan_id is an identifier — dropped before modelling
ID_COL = "loan_id"

# All numeric feature columns (used as model inputs)
NUMERIC_COLS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Categorical feature columns (will be label-encoded)
CATEGORICAL_COLS = [
    "education",      # e.g. "Graduate" / "Not Graduate"
    "self_employed",  # e.g. "Yes" / "No"
]

CLASSIFICATION_TARGET = "loan_status"   # "Approved" or "Rejected"
REGRESSION_TARGET_RATE = "loan_term"    # used as interest-rate proxy (replace if you have int_rate)
REGRESSION_TARGET_TERM = "loan_term"    # repayment period in months

ALL_REQUIRED_COLS = (
    NUMERIC_COLS + CATEGORICAL_COLS + [CLASSIFICATION_TARGET]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_loan_status(series: pd.Series) -> pd.Series:
    """Map loan_status to binary: 1 = Approved, 0 = Rejected."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"approved": 1, "rejected": 0})
    )


def _derive_interest_rate(df: pd.DataFrame) -> pd.Series:
    """
    Derive a synthetic interest rate from cibil_score and loan_amount.
    Replace this function with your actual int_rate column if available.

    Rate logic (illustrative):
        cibil >= 750  →  8%  – 10%
        cibil 700-749 → 10%  – 13%
        cibil 650-699 → 13%  – 18%
        cibil < 650   → 18%  – 24%
    """
    score = df["cibil_score"].clip(300, 900)
    # Linear interpolation: higher score → lower rate
    rate = 24 - ((score - 300) / 600) * 16   # range 8% – 24%
    # Add small noise based on loan_amount (larger loans → slightly higher rate)
    loan_norm = (df["loan_amount"] - df["loan_amount"].min()) / (
        df["loan_amount"].max() - df["loan_amount"].min() + 1e-9
    )
    rate = rate + loan_norm * 1.5
    return rate.round(2)


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Load, clean, encode, scale, and split the loan dataset.

    Returns a dict with keys:
        classification → {X_train, X_test, y_train, y_test}
        int_rate       → {X_train, X_test, y_train, y_test}
        term           → {X_train, X_test, y_train, y_test}
        feature_names  → list[str]
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape : {df.shape}")

    # Strip whitespace from column names (common Kaggle issue)
    df.columns = df.columns.str.strip().str.lower()

    # Drop the ID column — it carries no predictive signal
    if ID_COL in df.columns:
        df.drop(columns=[ID_COL], inplace=True)
        print(f"  Dropped identifier column: '{ID_COL}'")

    # Verify required columns exist
    missing_cols = set(c.lower() for c in ALL_REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}\n"
            f"Columns found: {list(df.columns)}"
        )

    # Keep only the columns we need
    df = df[[c for c in ALL_REQUIRED_COLS if c in df.columns]].copy()
    print(f"  Rows loaded : {len(df):,}")

    # ── 2. Clean the classification target ───────────────────────────────────
    df[CLASSIFICATION_TARGET] = _clean_loan_status(df[CLASSIFICATION_TARGET])
    unmapped = df[CLASSIFICATION_TARGET].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} rows had unrecognised loan_status values → dropped")
    df.dropna(subset=[CLASSIFICATION_TARGET], inplace=True)
    df[CLASSIFICATION_TARGET] = df[CLASSIFICATION_TARGET].astype(int)

    # ── 3. Derive interest rate proxy (synthetic) ─────────────────────────────
    df["int_rate"] = _derive_interest_rate(df)
    print(f"  Interest rate range: {df['int_rate'].min():.2f}% – {df['int_rate'].max():.2f}%")

    # ── 4. Drop rows where loan_term is NaN (regression target) ───────────────
    df.dropna(subset=["loan_term", "int_rate"], inplace=True)
    print(f"  Rows after target NaN drop : {len(df):,}")

    # ── 5. Impute missing values in numeric features ──────────────────────────
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = num_imputer.fit_transform(df[NUMERIC_COLS])

    # ── 6. Encode categorical features ───────────────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  Encoded '{col}' → {list(le.classes_)}")

    # ── 7. Build feature matrix ───────────────────────────────────────────────
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values.astype(float)

    # ── 8. Scale numeric columns only (categoricals stay integer-encoded) ─────
    scaler = StandardScaler()
    X[:, : len(NUMERIC_COLS)] = scaler.fit_transform(X[:, : len(NUMERIC_COLS)])

    # ── 9. Extract targets ────────────────────────────────────────────────────
    y_class    = df[CLASSIFICATION_TARGET].values.astype(int)
    y_int_rate = df["int_rate"].values.astype(float)
    y_term     = df["loan_term"].values.astype(int)

    # ── 10. Train / test split (stratified on approval status) ───────────────
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class,
    )

    splits = {
        "classification": {
            "X_train": X[idx_train],
            "X_test":  X[idx_test],
            "y_train": y_class[idx_train],
            "y_test":  y_class[idx_test],
        },
        "int_rate": {
            "X_train": X[idx_train],
            "X_test":  X[idx_test],
            "y_train": y_int_rate[idx_train],
            "y_test":  y_int_rate[idx_test],
        },
        "term": {
            "X_train": X[idx_train],
            "X_test":  X[idx_test],
            "y_train": y_term[idx_train],
            "y_test":  y_term[idx_test],
        },
        "feature_names": feature_cols,
    }

    # ── 11. Save preprocessing artifacts for inference ────────────────────────
    joblib.dump(scaler,          os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(label_encoders,  os.path.join(artifacts_dir, "label_encoders.pkl"))
    joblib.dump(num_imputer,     os.path.join(artifacts_dir, "num_imputer.pkl"))
    joblib.dump(feature_cols,    os.path.join(artifacts_dir, "feature_names.pkl"))

    # ── 12. Summary ───────────────────────────────────────────────────────────
    approved = int(y_class.sum())
    rejected = int(len(y_class) - approved)
    print(f"\nPreprocessing complete.")
    print(f"  Train samples : {len(idx_train):,}")
    print(f"  Test  samples : {len(idx_test):,}")
    print(f"  Features      : {feature_cols}")
    print(f"  Class balance : Approved={approved:,} / Rejected={rejected:,}")
    print(f"  Artifacts saved to: {artifacts_dir}/")

    return splits


# ── Run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "loan_data.csv"
    splits = preprocess(csv_path)

    print("\nSplit shapes:")
    for task, data in splits.items():
        if isinstance(data, dict):
            print(
                f"  {task:20s}: "
                f"X_train={data['X_train'].shape}  "
                f"y_train={data['y_train'].shape}"
            )