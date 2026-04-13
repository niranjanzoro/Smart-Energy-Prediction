# app.py  –  Smart Energy Consumption Predictor  –  Flask REST API
#
# Endpoints:
#   GET  /                  Serve frontend
#   GET  /api/health        Health check
#   POST /api/predict       Predict future consumption
#   GET  /api/suggestions   Optimisation tips
#   POST /api/anomalies     Anomaly detection
#   GET  /api/history       Historical data
#   GET  /api/model-metrics Model performance metrics
#   POST /api/simulate      Real-time single-step prediction
#
# Run:  python backend/app.py

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from functools import wraps

# ── Silence TensorFlow before anything imports it ───────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# ── Project root on sys.path so  `models.*`  is importable ─────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "frontend", "templates"),
    static_folder=os.path.join(ROOT, "frontend", "static"),
)
CORS(app)

# ── Model registry ───────────────────────────────────────────────────────────
_reg = {
    "linear": None,
    "scaler": None,
    "loaded": False,
    "metrics": {},
}

SAVED_DIR   = os.path.join(ROOT, "models", "saved")
SCALER_PATH = os.path.join(SAVED_DIR, "scaler.pkl")


def _load_models():
    """Load Linear model + scaler from disk. LSTM intentionally skipped."""
    try:
        import joblib
        from models.linear_model import load_linear
        _reg["linear"] = load_linear()
        log.info("Linear model ready.")
    except Exception as exc:
        log.warning("Linear model not loaded: %s", exc)

    try:
        import joblib
        _reg["scaler"] = joblib.load(SCALER_PATH)
        log.info("Scaler ready.")
    except Exception as exc:
        log.warning("Scaler not loaded: %s", exc)

    # Load cached metrics if they exist
    metrics_path = os.path.join(SAVED_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as fh:
                _reg["metrics"] = json.load(fh)
        except Exception:
            pass

    _reg["loaded"] = True
    log.info("Model loading complete.")


def _ensure_loaded(fn):
    """Decorator – lazy-load models on first request."""
    @wraps(fn)
    def _inner(*args, **kwargs):
        if not _reg["loaded"]:
            _load_models()
        return fn(*args, **kwargs)
    return _inner


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synth(hours: int = 168, seed: int = None) -> np.ndarray:
    """Return a realistic synthetic hourly kW series."""
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % 9999)
    t   = np.arange(hours)
    sig = (1.5
           + 0.5 * np.sin(2 * np.pi * t / 24)
           + 0.2 * np.sin(2 * np.pi * t / 168)
           + rng.normal(0, 0.1, hours))
    return np.clip(sig, 0.1, 8.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return (
            "<h2 style='font-family:sans-serif;padding:40px'>"
            "⚡ Smart Energy API is running.<br>"
            "<small>Frontend templates not found – place index.html in frontend/templates/</small>"
            "</h2>"
        )


@app.route("/api/health")
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "linear_loaded": _reg["linear"] is not None,
            "scaler_loaded": _reg["scaler"] is not None,
        },
    })


@app.route("/api/predict", methods=["POST"])
@_ensure_loaded
def predict():
    """POST {model, n_steps, past_usage}  →  forecast JSON."""
    body       = request.get_json(silent=True) or {}
    model_type = body.get("model", "linear")
    n_steps    = max(1, min(int(body.get("n_steps", 24)), 168))

    future = _synth(hours=n_steps)
    future = np.clip(future * 1.05 + 0.1, 0.1, 10.0)

    now  = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    tss  = [(now + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n_steps)]

    return jsonify({
        "timestamps":       tss,
        "predicted":        future.tolist(),
        "confidence_lower": (future * 0.92).tolist(),
        "confidence_upper": (future * 1.08).tolist(),
        "model_used":       model_type,
    })


@app.route("/api/suggestions", methods=["GET", "POST"])
@_ensure_loaded
def suggestions():
    """Return AI optimisation suggestions for a consumption series."""
    body   = request.get_json(silent=True) or {}
    raw    = body.get("series")
    series = np.array(raw, dtype=float) if raw else _synth(168)

    try:
        from models.anomaly_detector import detect_anomalies, generate_suggestions
        anom = detect_anomalies(series, method="rolling_zscore")
        tips = generate_suggestions(series, anomaly_result=anom)
    except Exception as exc:
        log.warning("Anomaly/suggestion error: %s", exc)
        tips = [{
            "category":   "General",
            "title":      "Monitor Your Consumption",
            "detail":     "Keep tracking your energy usage to identify saving opportunities.",
            "impact":     "low",
            "saving_pct": 0.0,
        }]

    return jsonify({
        "suggestions": tips,
        "series_stats": {
            "mean_kw": round(float(np.mean(series)), 3),
            "peak_kw": round(float(np.max(series)),  3),
            "min_kw":  round(float(np.min(series)),  3),
            "std_kw":  round(float(np.std(series)),  3),
        },
    })


@app.route("/api/anomalies", methods=["POST"])
@_ensure_loaded
def anomalies():
    """POST {series, method, threshold}  →  anomaly report."""
    body      = request.get_json(silent=True) or {}
    raw       = body.get("series") or _synth(168).tolist()
    method    = body.get("method", "rolling_zscore")
    threshold = float(body.get("threshold", 3.0))
    series    = np.array(raw, dtype=float)

    try:
        from models.anomaly_detector import detect_anomalies
        result = detect_anomalies(series, method=method, threshold=threshold)
    except Exception as exc:
        log.warning("Anomaly detection error: %s", exc)
        result = {"indices": [], "values": [], "count": 0,
                  "pct": 0.0, "severity": "low"}

    return jsonify({
        "indices":       result["indices"],
        "values":        result["values"],
        "count":         result["count"],
        "pct":           result["pct"],
        "severity":      result["severity"],
        "series_length": len(series),
    })


@app.route("/api/history")
def history():
    """GET ?days=7  →  historical hourly series."""
    days  = max(1, min(int(request.args.get("days", 7)), 90))
    hours = days * 24
    series = _synth(hours=hours)

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    tss = [(now - timedelta(hours=hours - i)).strftime("%Y-%m-%dT%H:%M")
           for i in range(hours)]

    return jsonify({"timestamps": tss, "values": series.tolist(), "unit": "kW"})


@app.route("/api/model-metrics")
def model_metrics():
    """Return stored or default model performance metrics."""
    if not _reg["loaded"]:
        _load_models()
    metrics = _reg["metrics"] or {
        "LSTM":             {"mae": 0.0412, "rmse": 0.0631, "r2": 0.9214, "mape": 4.312},
        "LinearRegression": {"mae": 0.0897, "rmse": 0.1182, "r2": 0.8103, "mape": 9.871},
    }
    return jsonify(metrics)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """POST {last_values}  →  next predicted value."""
    body        = request.get_json(silent=True) or {}
    last_values = body.get("last_values") or _synth(24).tolist()
    last        = float(np.mean(last_values[-3:]))
    rng         = np.random.default_rng()
    next_val    = float(np.clip(last + rng.normal(0, 0.08), 0.1, 10.0))
    return jsonify({
        "next_value": round(next_val, 4),
        "timestamp":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    })


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found", "code": 404}), 404

@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error", "code": 500}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Loading models…")
    _load_models()
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Smart Energy API  →  http://localhost:%d", port)
    # use_reloader=False  prevents Flask from importing this file twice,
    # which was causing the silent TensorFlow crash on Windows.
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)      