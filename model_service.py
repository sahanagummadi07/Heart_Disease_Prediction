"""Model loading, prediction, and business logic between HTTP and sklearn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from middleware.preprocessing import load_feature_order, patient_to_frame, transform_features
from middleware.schemas import ModelName, PatientFeatures

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"

_MODEL_FILES: dict[ModelName, str] = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "random_forest": "random_forest.joblib",
    "neural_network": "mlp_classifier.joblib",
}


class ModelService:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._feature_order = load_feature_order()
        self._load_all()

    def _load_all(self) -> None:
        for name, fname in _MODEL_FILES.items():
            path = MODEL_DIR / fname
            if path.exists():
                self._models[name] = joblib.load(path)

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def is_ready(self) -> bool:
        return len(self._models) > 0

    def predict(self, patient: PatientFeatures, model: ModelName) -> tuple[int, float]:
        if model not in self._models:
            raise KeyError(f"Model '{model}' is not trained or missing on disk.")
        clf = self._models[model]
        frame = patient_to_frame(patient.model_dump(), self._feature_order)
        Xs = transform_features(frame, self._feature_order)
        proba = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xs)[0, 1]
        pred = int(clf.predict(Xs)[0])
        if proba is None:
            proba = float(pred)
        return pred, float(proba)

    def metrics(self) -> dict:
        p = ARTIFACTS / "metrics.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def feature_importance(self) -> dict:
        p = ARTIFACTS / "feature_importance.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))


_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
