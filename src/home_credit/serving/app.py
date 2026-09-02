"""FastAPI scoring endpoint.

Start with:
    uvicorn home_credit.serving.app:app --reload

POST /predict with a JSON body matching ApplicationInput to get a default
probability score between 0 and 1.
"""

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from home_credit.config import settings

app = FastAPI(title="Home Credit Default Risk Scorer", version="0.1.0")

_model: mlflow.pyfunc.PyFuncModel | None = None


def _get_model() -> mlflow.pyfunc.PyFuncModel:
    global _model
    if _model is None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.model_registry_name}/Production"
        try:
            _model = mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model from {model_uri}. "
                "Train and promote a model first: python scripts/train.py"
            ) from exc
    return _model


class ApplicationInput(BaseModel):
    SK_ID_CURR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float | None = None
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None
    NAME_CONTRACT_TYPE: str | None = None
    CODE_GENDER: str | None = None
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    default_probability: float = Field(..., ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ApplicationInput) -> PredictionResponse:
    try:
        model = _get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([payload.model_dump()])
    try:
        prob = float(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(SK_ID_CURR=payload.SK_ID_CURR, default_probability=prob)
