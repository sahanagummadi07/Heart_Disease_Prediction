"""FastAPI backend: REST API for heart disease risk scoring."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from middleware.http_middleware import RequestContextMiddleware
from middleware.model_service import get_model_service
from middleware.schemas import HealthResponse, PredictRequest, PredictResponse

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_model_service()
    return HealthResponse(status="ok" if svc.is_ready() else "degraded", models_loaded=svc.list_models())


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    svc = get_model_service()
    if not svc.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run: python scripts/train_models.py",
        )
    try:
        pred, proba = svc.predict(body.patient, body.model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    msg = (
        "Elevated likelihood of heart disease (model output)."
        if pred == 1
        else "Lower likelihood of heart disease (model output)."
    )
    return PredictResponse(
        model=body.model,
        prediction=pred,
        probability_heart_disease=round(proba, 4),
        message=msg,
    )


@app.get("/api/metrics")
def metrics() -> dict:
    return get_model_service().metrics()


@app.get("/api/feature-importance")
def feature_importance() -> dict:
    return get_model_service().feature_importance()


@app.get("/styles.css")
def serve_styles() -> FileResponse:
    path = FRONTEND / "styles.css"
    if not path.exists():
        raise HTTPException(status_code=404, detail="frontend/styles.css missing")
    return FileResponse(path, media_type="text/css")


@app.get("/app.js")
def serve_app_js() -> FileResponse:
    path = FRONTEND / "app.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="frontend/app.js missing")
    return FileResponse(path, media_type="application/javascript")


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html missing")
    return FileResponse(index_path, media_type="text/html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
