"""
app/utils.py
------------
Helper utilities for loading serialised artefacts and running inference.
"""

import os
import joblib
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "model.pkl")
TRANS_PATH = os.path.join(_BASE_DIR, "transformer.pkl")


def load_artifacts():
    """Load and return (transformer, model) from disk. Raises if files missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run train.py first."
        )
    if not os.path.exists(TRANS_PATH):
        raise FileNotFoundError(
            f"Transformer file not found at {TRANS_PATH}. "
            "Please run train.py first."
        )
    transformer = joblib.load(TRANS_PATH)
    model       = joblib.load(MODEL_PATH)
    return transformer, model


def predict_single(customer: dict, transformer, model) -> dict:
    """
    Run inference on a single customer record.

    Parameters
    ----------
    customer    : dict  – raw customer fields from the "customer" key in the request
    transformer : fitted ColumnTransformer (SimpleImputer + OneHotEncoder)
    model       : fitted RandomForestClassifier

    Returns
    -------
    dict with keys: churn_prediction (str "Yes"/"No"), churn_probability (float)
    """
    # Build a single-row DataFrame; the transformer was fit on the original
    # column order so we just pass whatever columns the customer dict has.
    df = pd.DataFrame([customer])

    # Coerce TotalCharges to numeric in case it arrives as a string
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    X_tf = transformer.transform(df)
    prob = float(model.predict_proba(X_tf)[0, 1])
    pred = "Yes" if model.predict(X_tf)[0] == 1 else "No"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  pred
    }
