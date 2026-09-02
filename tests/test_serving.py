"""Tests for the FastAPI scoring endpoint — no model file required."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from home_credit.serving.app import app

client = TestClient(app)

PAYLOAD = {
    "SK_ID_CURR": 100001,
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "EXT_SOURCE_1": 0.083,
    "EXT_SOURCE_2": 0.263,
    "EXT_SOURCE_3": None,
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.15])

    with patch("home_credit.serving.app._get_model", return_value=mock_model):
        response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["SK_ID_CURR"] == 100001
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["default_probability"] == pytest.approx(0.15, abs=1e-6)


def test_predict_passes_sk_id_through():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.42])

    with patch("home_credit.serving.app._get_model", return_value=mock_model):
        response = client.post("/predict", json={**PAYLOAD, "SK_ID_CURR": 999})

    assert response.json()["SK_ID_CURR"] == 999


def test_predict_returns_503_when_model_not_loaded():
    with patch("home_credit.serving.app._get_model", side_effect=RuntimeError("no model")):
        response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 503
