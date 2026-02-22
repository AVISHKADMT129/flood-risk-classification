"""
FastAPI backend for the Flood Risk Classification system.
Provides /health, /metadata, and /predict endpoints.
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionInput, PredictionOutput, HealthResponse
from api.model_loader import get_model
from src.inference import predict
from src.utils import get_categorical_metadata, get_district_mappings, get_feature_statistics
from src.config import METRICS_PATH

app = FastAPI(
    title="Flood Risk Classification API",
    description="Predict flood risk for Sri Lankan districts using ML",
    version="1.0.0",
)

# CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metadata")
def get_metadata():
    """Return unique categorical values, district mappings, feature stats, and model metrics."""
    model_metrics = {}
    try:
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)
    except Exception:
        pass

    return {
        "categorical": get_categorical_metadata(),
        "district_mappings": get_district_mappings(),
        "feature_stats": get_feature_statistics(),
        "model_metrics": model_metrics,
    }


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: PredictionInput):
    """Accept feature inputs and return flood prediction."""
    try:
        model = get_model()
        result = predict(model, data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
