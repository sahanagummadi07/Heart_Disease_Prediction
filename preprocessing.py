"""Feature engineering and scaling aligned with training artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load_feature_order() -> list[str]:
    path = ARTIFACTS / "feature_order.json"
    if not path.exists():
        return [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def load_imputer() -> SimpleImputer | None:
    p = ARTIFACTS / "imputer.joblib"
    if p.exists():
        return joblib.load(p)
    return None


def load_scaler():
    p = ARTIFACTS / "scaler.joblib"
    if not p.exists():
        raise FileNotFoundError("Run scripts/train_models.py first to create artifacts/scaler.joblib")
    return joblib.load(p)


def patient_to_frame(patient: dict, feature_order: list[str]) -> pd.DataFrame:
    row = {k: patient[k] for k in feature_order}
    return pd.DataFrame([row])


def transform_features(X: pd.DataFrame, feature_order: list[str]) -> np.ndarray:
    X = X[feature_order].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    imputer = load_imputer()
    if imputer is not None:
        Xv = imputer.transform(X.values)
    else:
        Xv = X.fillna(X.median(numeric_only=True)).values
    scaler = load_scaler()
    return scaler.transform(Xv)
