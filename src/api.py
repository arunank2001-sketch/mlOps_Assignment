import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api_requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api")

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


Instrumentator().instrument(app).expose(app)


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