"""API tests: each test protects one thing that broke (or could break) in Lab 1."""
import joblib
import pytest
from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)

VALID = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


# 1) API contract: /health and /predict answer with the agreed shape
def test_health_contract():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_contract():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0


# 2) Model loads: the saved file is a real model with predict_proba
def test_model_loads():
    model = joblib.load("model.joblib")
    assert hasattr(model, "predict_proba")


# 3) Input schema: bad or missing fields are rejected with 422
def test_rejects_unknown_contract():
    bad = {**VALID, "contract": "weekly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_missing_field():
    bad = {k: v for k, v in VALID.items() if k != "monthly_charges"}
    assert client.post("/predict", json=bad).status_code == 422


# 4) Edge case: brand-new customer, and a negative value
def test_edge_new_customer_zero_tenure():
    edge = {**VALID, "tenure_months": 0, "support_calls": 0}
    r = client.post("/predict", json=edge)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_rejects_negative_tenure():
    bad = {**VALID, "tenure_months": -1}
    assert client.post("/predict", json=bad).status_code == 422


# 5) Deterministic prediction: same input, same known answer
def test_deterministic_prediction():
    first = client.post("/predict", json=VALID).json()["churn_probability"]
    second = client.post("/predict", json=VALID).json()["churn_probability"]
    assert first == second
    assert first == pytest.approx(0.895, abs=0.01)