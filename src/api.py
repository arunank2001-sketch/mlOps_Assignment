from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path

# MODEL_URI = "model_artifact"
#MODEL_URI = "runs:/cc37b857bcf04fc8848c2818b23c67f7/model"  # replaced below

MODEL_URI = str(Path(__file__).resolve().parent.parent / "model_artifact")

state = {}


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float
    chol: float
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int = Field(..., ge=1, le=3)
    ca: float
    thal: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientFeatures):
    row = pd.DataFrame([patient.model_dump()])
    prob = float(state["model"].predict_proba(row)[0, 1])
    return {
        "prediction": int(prob >= 0.5),
        "confidence": round(prob, 4),
    }