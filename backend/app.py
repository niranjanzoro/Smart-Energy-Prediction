# app.py  –  Smart Energy Consumption Predictor  –  Flask REST API
#
# Endpoints:
#   GET  /                  Serve frontend
#   GET  /api/health        Health check + model availability
#   POST /api/predict       Multi-step forecast with the TRAINED model (linear / lstm)
#   POST /api/suggestions   Rule-based optimisation tips
#   POST /api/anomalies     Anomaly detection
#   GET  /api/history       Real hourly history from the processed dataset
#   GET  /api/model-metrics Metrics saved by scripts/train_models.py
#   POST /api/simulate      Replay of held-out data: one-step prediction vs actual
#
# Run:  python backend/app.py        (after:  python scripts/train_models.py)

import json
import logging
import os
import sys
from datetime import datetime, timezone
from functools import wraps

# Silence TensorFlow before anything imports it
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "frontend", "templates"),
    static_folder=os.path.join(ROOT, "frontend", "static"),
)
CORS(app)

SAVED_DIR = os.path.join(ROOT, "models", "saved")
SCALER_PATH = os.path.join(SAVED_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(SAVED_DIR, "metrics.json")
META_PATH = os.path.join(SAVED_DIR, "meta.json")
PROCESSED_CSV = os.path.join(ROOT, "data", "processed_energy.csv")

MODEL_LABEL = {"linear": "LinearRegression", "lstm": "LSTM"}   # key in metrics / meta
MAX_STEPS = 72
MAX_SERIES_LEN = 5000
TS_FMT = "%Y-%m-%dT%H:%M"

# ── Registry ──────────────────────────────────────────────────────────────────
_reg = {"linear": None, "lstm": None, "lstm_tried": False, "scaler": None,
        "meta": {}, "metrics": {}, "history": None, "loaded": False}


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def _load_models():
    """Load Linear model, scaler, metadata and the hourly history. LSTM is lazy."""
    try:
        from models.linear_model import load_linear
        _reg["linear"] = load_linear()
    except Exception as exc:                                        # noqa: BLE001
        log.warning("Linear model not loaded: %s", exc)

    try:
        import joblib
        _reg["scaler"] = joblib.load(SCALER_PATH)
    except Exception as exc:                                        # noqa: BLE001
        log.warning("Scaler not loaded: %s", exc)

    for key, path in (("metrics", METRICS_PATH), ("meta", META_PATH)):
        try:
            with open(path) as fh:
                _reg[key] = json.load(fh)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("%s not loaded: %s", os.path.basename(path), exc)

    try:
        df = pd.read_csv(PROCESSED_CSV, index_col=0, parse_dates=True)
        _reg["history"] = df["Global_active_power"].asfreq("h")
    except Exception as exc:                                        # noqa: BLE001
        log.warning("History not loaded: %s", exc)

    _reg["loaded"] = True
    log.info("Model loading complete (linear=%s scaler=%s history=%s).",
             _reg["linear"] is not None, _reg["scaler"] is not None,
             _reg["history"] is not None)


def _get_lstm():
    """Lazily load the LSTM (imports TensorFlow only on first use)."""
    if _reg["lstm"] is None and not _reg["lstm_tried"]:
        _reg["lstm_tried"] = True
        try:
            from models.lstm_model import load_lstm
            _reg["lstm"] = load_lstm()
        except Exception as exc:                                    # noqa: BLE001
            log.warning("LSTM not available: %s", exc)
    return _reg["lstm"]


def _ensure_loaded(fn):
    @wraps(fn)
    def _inner(*args, **kwargs):
        if not _reg["loaded"]:
            _load_models()
        return fn(*args, **kwargs)
    return _inner


def _require_ready():
    if _reg["linear"] is None or _reg["scaler"] is None or _reg["history"] is None:
        raise ApiError("Models are not trained yet. Run: python scripts/train_models.py", 503)


# ── Validation helpers ────────────────────────────────────────────────────────

def _float_list(raw, name="series", min_len=1):
    if not isinstance(raw, list) or len(raw) > MAX_SERIES_LEN:
        raise ApiError(f"'{name}' must be a list of at most {MAX_SERIES_LEN} numbers.")
    try:
        arr = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise ApiError(f"'{name}' must contain only numbers.")
    if arr.ndim != 1 or len(arr) < min_len or not np.isfinite(arr).all():
        raise ApiError(f"'{name}' must be a 1-D list of ≥{min_len} finite numbers.")
    return arr


def _int_arg(value, name, default, lo, hi):
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ApiError(f"'{name}' must be an integer.")
    return max(lo, min(v, hi))


def _body():
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _fmt(idx):
    return [t.strftime(TS_FMT) for t in idx]


def _predictor(kind):
    """Return (kind_actually_used, predict_fn, note)."""
    from models.forecaster import make_predict_fn
    note = None
    if kind == "lstm":
        model = _get_lstm()
        if model is not None:
            return "lstm", make_predict_fn("lstm", model), note
        note = "LSTM model not available on this server – used the linear model instead."
    return "linear", make_predict_fn("linear", _reg["linear"]), note


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
@_ensure_loaded
def health():
    meta = _reg["meta"]
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {
            "linear_loaded": _reg["linear"] is not None,
            "scaler_loaded": _reg["scaler"] is not None,
            "lstm_file_present": os.path.exists(os.path.join(SAVED_DIR, "lstm_model.keras")),
            "lstm_loaded": _reg["lstm"] is not None,
        },
        "data_source": meta.get("data_source"),
        "trained_at": meta.get("trained_at"),
    })


@app.route("/api/predict", methods=["POST"])
@_ensure_loaded
def predict():
    """
    POST {model: 'linear'|'lstm', n_steps: 1-72, past_usage: [kW, …] (optional)}

    Recursive multi-step forecast from the end of the stored history.  If
    ``past_usage`` is given, those values replace the MOST RECENT hours of the
    history (oldest → newest) before forecasting.  Interval = prediction ±
    1.96 × the back-tested RMSE at that horizon.
    """
    _require_ready()
    from models.forecaster import recursive_forecast

    body = _body()
    requested = str(body.get("model", "linear")).lower()
    if requested not in MODEL_LABEL:
        raise ApiError(f"'model' must be one of {list(MODEL_LABEL)}.")
    n_steps = _int_arg(body.get("n_steps"), "n_steps", 24, 1, MAX_STEPS)

    history = _reg["history"].copy()
    past = body.get("past_usage")
    if past:
        vals = _float_list(past, "past_usage")
        if (vals < 0).any():
            raise ApiError("'past_usage' values must be ≥ 0 kW.")
        k = min(len(vals), len(history))
        history.iloc[-k:] = vals[-k:]

    kind, fn, note = _predictor(requested)
    look_back = _reg["meta"].get("look_back", 24)
    pred = recursive_forecast(history, n_steps, fn, _reg["scaler"], look_back)

    sigma = _reg["meta"].get("horizon_rmse", {}).get(MODEL_LABEL[kind])
    if sigma:
        s = np.array([sigma[min(h, len(sigma) - 1)] for h in range(n_steps)])
        lower, upper = np.clip(pred - 1.96 * s, 0, None), pred + 1.96 * s
    else:
        lower = upper = pred
        note = (note + " " if note else "") + "No back-test statistics found – interval unavailable."

    future_idx = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=n_steps, freq="h")
    return jsonify({
        "timestamps": _fmt(future_idx),
        "predicted": np.round(pred, 4).tolist(),
        "confidence_lower": np.round(lower, 4).tolist(),
        "confidence_upper": np.round(upper, 4).tolist(),
        "interval": "≈95% prediction interval from back-tested RMSE per horizon",
        "model_used": kind,
        "model_requested": requested,
        "note": note,
        "history_ends": history.index[-1].strftime(TS_FMT),
    })


def _series_and_index(body):
    """Series (+ aligned DatetimeIndex) from the request, defaulting to the last 7 days."""
    hist = _reg["history"]
    raw = body.get("series")
    if raw is None or raw == []:
        tail = hist.iloc[-168:]
        return tail.to_numpy(), tail.index
    series = _float_list(raw, "series", min_len=2)
    ts = body.get("timestamps")
    if ts is not None:
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(ts))
        except Exception:                                           # noqa: BLE001
            raise ApiError("'timestamps' must be a list of ISO datetime strings.")
        if len(idx) != len(series):
            raise ApiError("'timestamps' and 'series' must have the same length.")
    else:   # assume the series ends at the end of the stored history
        idx = pd.date_range(end=hist.index[-1], periods=len(series), freq="h")
    return series, idx


@app.route("/api/suggestions", methods=["GET", "POST"])
@_ensure_loaded
def suggestions():
    """POST {series?, timestamps?} → rule-based optimisation tips."""
    _require_ready()
    from models.anomaly_detector import detect_anomalies, generate_suggestions
    series, idx = _series_and_index(_body())
    anom = detect_anomalies(series, method="rolling_zscore")
    tips = generate_suggestions(series, timestamps=idx, anomaly_result=anom)
    return jsonify({
        "suggestions": tips,
        "series_stats": {
            "mean_kw": round(float(np.mean(series)), 3),
            "peak_kw": round(float(np.max(series)), 3),
            "min_kw": round(float(np.min(series)), 3),
            "std_kw": round(float(np.std(series)), 3),
        },
    })


@app.route("/api/anomalies", methods=["POST"])
@_ensure_loaded
def anomalies():
    """POST {series?, method, threshold} → anomaly report."""
    _require_ready()
    from models.anomaly_detector import detect_anomalies
    body = _body()
    series, _ = _series_and_index(body)
    method = body.get("method", "rolling_zscore")
    try:
        threshold = float(body.get("threshold", 3.0))
    except (TypeError, ValueError):
        raise ApiError("'threshold' must be a number.")
    try:
        result = detect_anomalies(series, method=method, threshold=threshold)
    except ValueError as exc:
        raise ApiError(str(exc))
    return jsonify({
        "method": method,
        "indices": result["indices"], "values": result["values"],
        "count": result["count"], "pct": result["pct"], "severity": result["severity"],
        "series_length": len(series),
    })


@app.route("/api/history")
@_ensure_loaded
def history():
    """GET ?days=7 → the most recent hourly kW values from the processed dataset."""
    _require_ready()
    days = _int_arg(request.args.get("days"), "days", 7, 1, 90)
    tail = _reg["history"].iloc[-days * 24:]
    return jsonify({"timestamps": _fmt(tail.index), "values": np.round(tail.values, 4).tolist(),
                    "unit": "kW", "source": _reg["meta"].get("data_source")})


@app.route("/api/model-metrics")
@_ensure_loaded
def model_metrics():
    """Metrics written by scripts/train_models.py (one-step-ahead, test period, kW)."""
    if not _reg["metrics"]:
        raise ApiError("No metrics found. Run: python scripts/train_models.py", 503)
    return jsonify(_reg["metrics"])


@app.route("/api/simulate", methods=["POST"])
@_ensure_loaded
def simulate():
    """
    POST {step: int, model: 'linear'|'lstm'}

    Replays the held-out TEST period: at replay step k the model sees the real
    history up to hour k and predicts hour k; the actual value is returned
    alongside, so the stream shows genuine model error.
    """
    _require_ready()
    from models.forecaster import recursive_forecast
    body = _body()
    step = _int_arg(body.get("step"), "step", 0, 0, 10**9)
    requested = str(body.get("model", "linear")).lower()
    if requested not in MODEL_LABEL:
        raise ApiError(f"'model' must be one of {list(MODEL_LABEL)}.")

    y = _reg["history"]
    split = pd.Timestamp(_reg["meta"]["split_ts"])
    start = int(np.searchsorted(y.index, split))
    p = start + step % (len(y) - start)
    kind, fn, _ = _predictor(requested)
    pred = recursive_forecast(y.iloc[:p], 1, fn, _reg["scaler"],
                              _reg["meta"].get("look_back", 24))[0]
    return jsonify({
        "next_value": round(float(pred), 4),
        "actual": round(float(y.iloc[p]), 4),
        "timestamp": y.index[p].strftime("%Y-%m-%dT%H:%M:%S"),
        "model_used": kind,
    })


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(ApiError)
def api_error(e):
    return jsonify({"error": e.message, "code": e.status}), e.status


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found", "code": 404}), 404


@app.errorhandler(405)
def bad_method(_e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405


@app.errorhandler(Exception)
def server_error(e):
    log.exception("Unhandled error: %s", e)
    return jsonify({"error": "Internal server error", "code": 500}), 500


if __name__ == "__main__":
    log.info("Loading models…")
    _load_models()
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Smart Energy API  →  http://localhost:%d", port)
    # use_reloader=False avoids importing twice (silent TensorFlow crash on Windows)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
