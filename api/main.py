from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_DIR = Path("models/topic_model/2026-05-12")

app = FastAPI(
    title="News Topic Modeling API",
    description="API for assigning topic IDs to news text using a trained NMF topic model.",
    version="0.1.0",
)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    topic_id: int
    topic_score: float
    topic_keywords: list[str]

def load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "topic_metadata.json"
    return pd.read_json(metadata_path, typ="series").to_dict()

model = joblib.load(MODEL_DIR / "nmf_topic_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
metadata = load_metadata(MODEL_DIR)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/topics")
def topics() -> dict[str, Any]:
    return {
        "algorithm": metadata["algorithm"],
        "n_topics": metadata["n_topics"],
        "topic_keywords": metadata["topic_keywords"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text_vector = vectorizer.transform([request.text])
    topic_matrix = model.transform(text_vector)

    topic_id = int(topic_matrix.argmax(axis=1)[0])
    topic_score = float(topic_matrix.max(axis=1)[0])
    topic_keywords = metadata["topic_keywords"][str(topic_id)]

    return PredictionResponse(
        topic_id=topic_id,
        topic_score=topic_score,
        topic_keywords=topic_keywords,
    )