
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "models" / "iris_model.pkl"

model = None

FLOWER_NAMES = ["setosa", "versicolor", "virginica"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Run python train.py first."
        )

    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="AWS MLOps Prediction API",
    description="Iris flower classification using machine learning",
    version="1.0.0",
    lifespan=lifespan
)


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "AWS MLOps Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(flower: FlowerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "predicted_class": prediction,
        "flower_name": FLOWER_NAMES[prediction],
        "confidence": round(
            float(probabilities[prediction]), 4
        ),
        "probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(
                FLOWER_NAMES, probabilities
            )
        }
    }
