"""
Unit tests for the data pipeline, anomaly detectors and forecaster.
Run from the project root:  pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import data_loader as dl                                    # noqa: E402
from models.anomaly_detector import detect_anomalies, generate_suggestions  # noqa: E402
from models.forecaster import (recursive_forecast, persistence_forecast,   # noqa: E402
                               seasonal_naive_forecast)


@pytest.fixture(scope="module")
def data():
    return dl.get_processed_data(look_back=24, use_synthetic=True, save=False)


# ── Data ─────────────────────────────────────────────────────────────────────

def test_target_is_not_constant(data):
    """Regression test: an earlier generator saturated the target at 8.0 kW."""
    y = data["y_hourly"]
    assert y.std() > 0.3
    assert (y >= 7.99).mean() < 0.01


def test_hourly_series_is_gap_free(data):
    idx = data["y_hourly"].index
    assert (np.diff(idx.values) == np.timedelta64(1, "h")).all()
    assert not data["y_hourly"].isna().any()


def test_split_is_chronological_and_leak_free(data):
    split = data["split_ts"]
    assert data["test_index"].min() >= split
    # every TRAIN target must be strictly before the split
    n_train = len(data["y_train_seq"])
    feats_idx = data["features"].index[data["look_back"]:]
    assert (feats_idx[:n_train] < split).all()


def test_scaler_fitted_on_train_only(data):
    feats, split, scaler = data["features"], data["split_ts"], data["scaler"]
    train_max = feats.loc[feats.index < split, "Global_active_power"].max()
    assert scaler.data_max_[0] == pytest.approx(train_max)


def test_window_does_not_contain_its_own_target(data):
    """Window k covers rows [k, k+24); its target is row k+24."""
    feats, lb = data["features"], data["look_back"]
    scaled = data["scaler"].transform(feats.values)
    X, y = dl.build_sequences(scaled, lb)
    assert np.allclose(X[10], scaled[10:10 + lb])
    assert y[10] == pytest.approx(scaled[10 + lb, 0])


def test_inverse_target_roundtrip(data):
    kw = data["y_test_kw"]
    assert kw.min() >= 0 and kw.max() < 10
    assert np.allclose(kw, data["features"].loc[data["test_index"], "Global_active_power"].values)


def test_features_depend_only_on_past():
    """Changing a FUTURE value must not change features of earlier rows."""
    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    y = pd.Series(np.random.default_rng(0).uniform(0.5, 3, 400), index=idx)
    a = dl.build_feature_frame(y)
    y2 = y.copy(); y2.iloc[-1] += 5
    b = dl.build_feature_frame(y2)
    assert a.iloc[:-1].equals(b.iloc[:-1])


# ── Anomaly detection ────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["zscore", "iqr", "rolling_zscore", "isolation_forest"])
def test_every_method_runs_and_finds_spikes(method):
    s = np.random.default_rng(0).normal(1.5, 0.1, 200)
    s[[50, 120]] *= 3
    r = detect_anomalies(s, method=method, threshold=3.0)   # API always passes threshold
    assert {50, 120} <= set(r["indices"])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        detect_anomalies(np.ones(10), method="nope")


def test_peak_suggestion_needs_timestamps_and_uses_correct_maths():
    idx = pd.date_range("2024-01-01", periods=24 * 7, freq="h")
    s = np.where((idx.hour >= 17) & (idx.hour <= 21), 3.0, 1.0)
    tips = generate_suggestions(s, timestamps=idx)
    peak = [t for t in tips if t["category"] == "Peak Hours"][0]
    assert "200.0% above" in peak["detail"]          # (3-1)/1 = 200 %
    assert peak["basis"] == "computed"
    assert not any(t["category"] == "Peak Hours" for t in generate_suggestions(s))


# ── Forecaster ───────────────────────────────────────────────────────────────

def test_baselines():
    y = pd.Series(np.arange(48, dtype=float), index=pd.date_range("2024-01-01", periods=48, freq="h"))
    assert (persistence_forecast(y, 3) == 47).all()
    assert seasonal_naive_forecast(y, 3).tolist() == [24, 25, 26]


def test_recursive_forecast_shape_and_non_negative(data):
    from models.linear_model import train_linear
    lr = train_linear(data["X_train_flat"], data["y_train_seq"], alpha=10.0)
    fn = lambda X: lr.predict(X.reshape(len(X), -1)).ravel()
    pred = recursive_forecast(data["y_hourly"].iloc[:-100], 48, fn, data["scaler"])
    assert pred.shape == (48,) and (pred >= 0).all()
    assert pred.std() > 0.05                         # not a flat line


def test_forecast_beats_persistence_over_a_day(data):
    """Sanity check that the model learned something (daily cycle)."""
    from models.linear_model import train_linear
    lr = train_linear(data["X_train_flat"], data["y_train_seq"], alpha=10.0)
    fn = lambda X: lr.predict(X.reshape(len(X), -1)).ravel()
    y, split = data["y_hourly"], data["split_ts"]
    pos = np.searchsorted(y.index, split)
    errs_m, errs_p = [], []
    for p in range(pos + 200, pos + 1000, 80):
        truth = y.iloc[p:p + 24].values
        errs_m.append(np.sqrt(np.mean((recursive_forecast(y.iloc[:p], 24, fn, data["scaler"]) - truth) ** 2)))
        errs_p.append(np.sqrt(np.mean((persistence_forecast(y.iloc[:p], 24) - truth) ** 2)))
    assert np.mean(errs_m) < np.mean(errs_p)
