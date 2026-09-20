"""
FastAPI service for the toxic comment detector.

Local run:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from model import toxicity_model
from tanglish_model import tanglish_model
from schemas import HealthResponse, PredictRequest, PredictResponse, TanglishPredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at startup, not on every request
    toxicity_model.load()
    tanglish_model.load()
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(
    title="Toxic Comment Detector API",
    description="Multi-label toxicity classification for text comments.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    both_loaded = toxicity_model.is_loaded() and tanglish_model.is_loaded()
    return HealthResponse(
        status="ok" if both_loaded else "model not loaded",
        model_loaded=both_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not toxicity_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = toxicity_model.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result


@app.post("/predict-tanglish", response_model=TanglishPredictResponse)
def predict_tanglish(request: PredictRequest):
    """Binary offensive/not-offensive classification for romanized
    Tamil-English (Tanglish) text. See training/train_tanglish.py for
    dataset and methodology notes."""
    if not tanglish_model.is_loaded():
        raise HTTPException(status_code=503, detail="Tanglish model is not loaded yet.")

    try:
        result = tanglish_model.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result
