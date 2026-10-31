"""Churn prediction API."""
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.train.train import FEATURES

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="AI381 churn API")


class Customer(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    return {"churn_probability": round(p, 3)}