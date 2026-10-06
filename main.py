"""FastAPI service that serves the Iris species classifier in model.pkl."""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    load_error = None
except Exception as exc:  # keep the app up so /health can report the problem
    model = None
    load_error = f"{type(exc).__name__}: {exc}"

app = FastAPI(
    title="Iris Species Classifier",
    description="Predicts the species of an Iris flower from its sepal and petal measurements.",
    version="1.0.0",
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    }


class Prediction(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {"message": "Iris Species Classifier API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "error": load_error,
    }


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    row = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    probs = model.predict_proba(row)[0]
    probabilities = {str(cls): round(float(p), 4) for cls, p in zip(model.classes_, probs)}
    species = max(probabilities, key=probabilities.get)

    return Prediction(
        species=species,
        confidence=probabilities[species],
        probabilities=probabilities,
    )
