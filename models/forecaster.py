"""
forecaster.py
─────────────
Recursive multi-step forecasting and honest back-testing.

A model trained for ONE-step-ahead prediction is rolled forward: each
prediction is appended to the history, features are rebuilt with the same
`build_feature_frame` used in training, and the next step is predicted.

Because errors compound, `backtest_horizons` measures the real RMSE at every
horizon (1 … H hours) over many forecast origins in the test period.  Those
numbers are what the API uses for its prediction intervals (instead of a fixed
±8 % band).
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .data_loader import build_feature_frame, MIN_HISTORY, FEATURE_COLS, inverse_target

log = logging.getLogger(__name__)

# predict_fn: scaled windows (n, look_back, n_features) -> scaled predictions (n,)
PredictFn = Callable[[np.ndarray], np.ndarray]


def make_predict_fn(kind: str, model) -> PredictFn:
    """Wrap a fitted model so it maps (n, look_back, F) windows → (n,) predictions."""
    if kind == "linear":
        return lambda X: model.predict(X.reshape(len(X), -1)).ravel()
    if kind == "lstm":
        return lambda X: np.asarray(model(X.astype("float32"), training=False)).ravel()
    raise ValueError(f"Unknown model kind '{kind}'")


def recursive_forecast_batch(
    histories: list[pd.Series],
    n_steps: int,
    predict_fn: PredictFn,
    scaler,
    look_back: int = 24,
) -> np.ndarray:
    """
    Forecast ``n_steps`` hours for several histories at once.

    Parameters
    ----------
    histories : hourly kW Series (DatetimeIndex, contiguous), each with at
                least ``MIN_HISTORY + look_back`` points.
    Returns   : array (len(histories), n_steps) in kW.
    """
    need = MIN_HISTORY + look_back
    ys = []
    for h in histories:
        if len(h) < need:
            raise ValueError(f"Need at least {need} hours of history, got {len(h)}")
        ys.append(h.iloc[-need:].astype(float).copy())

    out = np.zeros((len(ys), n_steps))
    for s in range(n_steps):
        windows = []
        for y in ys:
            f = build_feature_frame(y.iloc[-need:])          # → exactly look_back rows
            windows.append(scaler.transform(f[FEATURE_COLS].values))
        pred_kw = inverse_target(predict_fn(np.stack(windows)), scaler)
        pred_kw = np.clip(pred_kw, 0.0, None)                # power cannot be negative
        out[:, s] = pred_kw
        for k, y in enumerate(ys):
            nxt = y.index[-1] + pd.Timedelta(hours=1)
            ys[k] = pd.concat([y, pd.Series([pred_kw[k]], index=[nxt])])
    return out


def recursive_forecast(history: pd.Series, n_steps: int, predict_fn: PredictFn,
                       scaler, look_back: int = 24) -> np.ndarray:
    """Single-history convenience wrapper → 1-D array of kW forecasts."""
    return recursive_forecast_batch([history], n_steps, predict_fn, scaler, look_back)[0]


# ── Baselines ────────────────────────────────────────────────────────────────

def persistence_forecast(history: pd.Series, n_steps: int) -> np.ndarray:
    """Naive: repeat the last observed value."""
    return np.repeat(history.iloc[-1], n_steps)


def seasonal_naive_forecast(history: pd.Series, n_steps: int, season: int = 24) -> np.ndarray:
    """Seasonal naive: same hour of the previous day (repeated)."""
    last_season = history.iloc[-season:].values
    return np.array([last_season[h % season] for h in range(n_steps)])


# ── Back-test ────────────────────────────────────────────────────────────────

def backtest_horizons(
    y_hourly: pd.Series,
    split_ts: pd.Timestamp,
    predict_fns: dict[str, PredictFn],
    scaler,
    look_back: int = 24,
    horizon: int = 72,
    n_origins: int = 30,
) -> dict:
    """
    Evaluate multi-step forecasts from ``n_origins`` evenly spaced forecast
    origins inside the test period.

    Returns {'rmse': {name: [rmse_h1..rmse_H]}, 'mae': {...}, 'n_origins': int,
             'origins': [timestamps]}.
    Baselines 'Persistence' and 'SeasonalNaive' are always included.
    """
    positions = np.where(y_hourly.index >= split_ts)[0]
    first, last = positions[0], len(y_hourly) - horizon
    origins = np.linspace(first, last, n_origins).astype(int)

    truth = np.stack([y_hourly.iloc[p:p + horizon].values for p in origins])
    histories = [y_hourly.iloc[:p] for p in origins]

    preds = {name: recursive_forecast_batch(histories, horizon, fn, scaler, look_back)
             for name, fn in predict_fns.items()}
    preds["Persistence"] = np.stack([persistence_forecast(h, horizon) for h in histories])
    preds["SeasonalNaive"] = np.stack([seasonal_naive_forecast(h, horizon) for h in histories])

    rmse = {k: np.sqrt(((v - truth) ** 2).mean(axis=0)).round(5).tolist() for k, v in preds.items()}
    mae = {k: np.abs(v - truth).mean(axis=0).round(5).tolist() for k, v in preds.items()}
    log.info("Back-test done: %d origins × %d h", len(origins), horizon)
    return {"rmse": rmse, "mae": mae, "n_origins": int(len(origins)),
            "horizon": horizon,
            "origins": [str(y_hourly.index[p]) for p in origins]}
