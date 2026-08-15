"""
train.py
--------
Training script to build and serialize the churn prediction model.
Run this script once to generate app/model.pkl and app/transformer.pkl.

Usage:
    python train.py
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("test_data", "all_customers.csv")
MODEL_PATH = os.path.join("app", "model.pkl")
TRANS_PATH = os.path.join("app", "transformer.pkl")
TARGET_COL = "Churn"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def main():
    print("Loading data ...")
    df = load_data(DATA_PATH)

    # Creating X and y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Step 1: Drop the 'customerID' column
    X = X.drop(columns=['customerID'])

    # Step 2: Convert 'TotalCharges' to numeric (handles spaces or non-numeric values)
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

    # Step 3: Convert target column 'y' to binary values
    y = y.map({'Yes': 1, 'No': 0})

    # Step 4: Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Numeric features    : {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")

    # Step 5: Define preprocessing pipeline (no model yet)
    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    # Step 6: Apply the transformation to the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Fitting transformer ...")
    X_train_tf = preprocessor.fit_transform(X_train)
    X_test_tf  = preprocessor.transform(X_test)

    print("Training model ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_tf, y_train)

    # -- Evaluation -----------------------------------------------------------
    y_pred = model.predict(X_test_tf)
    y_prob = model.predict_proba(X_test_tf)[:, 1]
    print("\n-- Evaluation --------------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

    # -- Persist artefacts ----------------------------------------------------
    os.makedirs("app", exist_ok=True)
    joblib.dump(preprocessor, TRANS_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved transformer -> {TRANS_PATH}")
    print(f"Saved model       -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
