"""Pydantic schemas: API contract and validation (middleware layer)."""

from typing import Literal

from pydantic import BaseModel, Field


ModelName = Literal["logistic_regression", "decision_tree", "random_forest", "neural_network"]


class PatientFeatures(BaseModel):
    age: int = Field(ge=18, le=120, description="Age in years")
    sex: int = Field(ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(ge=0, le=3, description="Chest pain type (0–3)")
    trestbps: int = Field(ge=80, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(ge=60, le=220, description="Max heart rate achieved")
    exang: int = Field(ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(ge=0, le=3, description="Number of major vessels colored by fluoroscopy")
    # Cleveland / UCI encodings often use 1–3 or 3 / 6 / 7; allow a wide range so valid rows are not rejected.
    thal: int = Field(ge=0, le=9, description="Thalassemia / perfusion (dataset-specific encoding)")


class PredictRequest(BaseModel):
    patient: PatientFeatures
    model: ModelName = "random_forest"


class PredictResponse(BaseModel):
    model: str
    prediction: int
    probability_heart_disease: float
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
