"""
API tests (need trained artefacts: run `python scripts/train_models.py` first).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAVED = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SAVED, "linear_model.pkl")),
                                reason="models not trained")


@pytest.fixture(scope="module")
def client():
    from backend.app import app
    app.testing = True
    return app.test_client()


def test_health(client):
    j = client.get("/api/health").get_json()
    assert j["status"] == "ok" and j["models"]["linear_loaded"]


def test_predict_uses_model_not_random(client):
    a = client.post("/api/predict", json={"model": "linear", "n_steps": 24}).get_json()
    b = client.post("/api/predict", json={"model": "linear", "n_steps": 24}).get_json()
    assert a["predicted"] == b["predicted"]              # deterministic
    assert len(a["predicted"]) == 24
    assert all(lo <= p <= hi for lo, p, hi in
               zip(a["confidence_lower"], a["predicted"], a["confidence_upper"]))
    assert max(a["predicted"]) - min(a["predicted"]) > 0.1


def test_predict_responds_to_past_usage(client):
    base = client.post("/api/predict", json={"n_steps": 6}).get_json()["predicted"]
    hi = client.post("/api/predict", json={"n_steps": 6, "past_usage": [4.0] * 24}).get_json()["predicted"]
    assert sum(hi) > sum(base)


def test_predict_validation(client):
    assert client.post("/api/predict", json={"model": "xgboost"}).status_code == 400
    assert client.post("/api/predict", json={"past_usage": ["a"]}).status_code == 400
    assert client.post("/api/predict", json={"past_usage": [-1, 2]}).status_code == 400
    r = client.post("/api/predict", json={"n_steps": 9999}).get_json()
    assert len(r["predicted"]) == 72                     # clamped


def test_history_is_real_and_deterministic(client):
    a = client.get("/api/history?days=3").get_json()
    b = client.get("/api/history?days=3").get_json()
    assert a["values"] == b["values"] and len(a["values"]) == 72


@pytest.mark.parametrize("method", ["zscore", "iqr", "rolling_zscore", "isolation_forest"])
def test_anomalies_all_methods(client, method):
    s = [1.0, 1.1, 0.9, 1.0] * 30
    s[60] = 9.0
    r = client.post("/api/anomalies", json={"series": s, "method": method}).get_json()
    assert 60 in r["indices"]


def test_anomalies_bad_method(client):
    assert client.post("/api/anomalies", json={"method": "x"}).status_code == 400


def test_suggestions(client):
    j = client.post("/api/suggestions", json={}).get_json()
    assert j["suggestions"] and "basis" in j["suggestions"][0]


def test_metrics_come_from_file(client):
    m = client.get("/api/model-metrics").get_json()
    assert "LinearRegression" in m and "Persistence" in m
    assert set(m["LinearRegression"]) == {"mae", "rmse", "r2", "mape"}


def test_simulate_returns_prediction_and_actual(client):
    j = client.post("/api/simulate", json={"step": 5, "model": "linear"}).get_json()
    assert {"next_value", "actual", "timestamp"} <= set(j)
    assert 0 <= j["next_value"] < 10
