"""
train_models.py
───────────────
End-to-end training / evaluation script.

  1. Load + preprocess data (synthetic by default, UCI with --real)
  2. Fit Ridge (alpha tuned with TimeSeriesSplit) and, optionally, the LSTM
  3. Evaluate on the held-out TEST period in kW:
       a) one-step-ahead metrics (MAE / RMSE / R² / MAPE)
       b) multi-step back-test (RMSE at every horizon 1…72 h)
     against two naive baselines (Persistence, Seasonal-naive)
  4. Save models, scaler, metrics.json, meta.json and charts

Run from the project root:
    python scripts/train_models.py                    # synthetic data
    python scripts/train_models.py --real             # UCI dataset (auto-download)
    python scripts/train_models.py --no-lstm          # skip TensorFlow
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAVED_DIR = os.path.join(ROOT, "models", "saved")
CHART_DIR = os.path.join(ROOT, "data", "charts")
SCALER_PATH = os.path.join(SAVED_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(SAVED_DIR, "metrics.json")
META_PATH = os.path.join(SAVED_DIR, "meta.json")

HORIZON = 72


def parse_args():
    p = argparse.ArgumentParser(description="Train energy prediction models")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="use synthetic data (default)")
    src.add_argument("--real", action="store_true", help="use the UCI dataset (downloads if missing)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--look-back", type=int, default=24, help="input window in hours")
    p.add_argument("--no-lstm", action="store_true", help="skip LSTM (no TensorFlow needed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Charts ───────────────────────────────────────────────────────────────────

def save_charts(history, test_index, y_kw, preds_kw, backtest):
    os.makedirs(CHART_DIR, exist_ok=True)

    if history:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["loss"], label="Train loss", lw=2)
        ax.plot(history["val_loss"], label="Validation loss", lw=2, ls="--")
        ax.set(title="LSTM training / validation loss", xlabel="Epoch", ylabel="Huber loss (scaled)")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "training_loss.png"), dpi=120); plt.close(fig)

    n = min(168, len(y_kw))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_index[:n], y_kw[:n], label="Actual", lw=2, color="#2563eb")
    styles = {"LSTM": ("#16a34a", "--"), "LinearRegression": ("#dc2626", ":"),
              "SeasonalNaive": ("#9ca3af", "-.")}
    for name, p in preds_kw.items():
        if name in styles:
            c, ls = styles[name]
            ax.plot(test_index[:n], p[:n], label=name, lw=1.5, ls=ls, color=c)
    ax.set(title="One-step-ahead prediction vs actual (first 7 test days)", ylabel="kW")
    ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "prediction_comparison.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, r in backtest["rmse"].items():
        ax.plot(range(1, len(r) + 1), r, label=name, lw=2)
    ax.set(title=f"Multi-step forecast error vs horizon ({backtest['n_origins']} test origins)",
           xlabel="Forecast horizon (hours)", ylabel="RMSE (kW)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "horizon_rmse.png"), dpi=120); plt.close(fig)
    log.info("Charts saved → %s", CHART_DIR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    use_synthetic = not args.real

    from models import data_loader as dl
    from models.metrics import evaluate_predictions, compare_models
    from models.forecaster import make_predict_fn, backtest_horizons

    if args.real:
        dl.download_dataset()

    log.info("Preparing data (source=%s, look_back=%d) …",
             "synthetic" if use_synthetic else "UCI", args.look_back)
    data = dl.get_processed_data(look_back=args.look_back, use_synthetic=use_synthetic)
    scaler = data["scaler"]
    inv = lambda a: dl.inverse_target(a, scaler)

    X_tr, y_tr = data["X_train_seq"], data["y_train_seq"]
    X_te, y_kw = data["X_test_seq"], data["y_test_kw"]

    # Validation = chronologically last 15 % of the TRAIN windows (test set untouched)
    v = int(len(X_tr) * 0.85)
    X_fit, y_fit, X_val, y_val = X_tr[:v], y_tr[:v], X_tr[v:], y_tr[v:]

    results, preds_kw, predict_fns, history = {}, {}, {}, {}

    # ── Ridge ────────────────────────────────────────────────────────────
    from models.linear_model import train_linear, predict_linear, save_linear
    lr = train_linear(data["X_train_flat"], y_tr)
    preds_kw["LinearRegression"] = inv(predict_linear(lr, data["X_test_flat"]))
    save_linear(lr)
    predict_fns["LinearRegression"] = make_predict_fn("linear", lr)
    best_alpha = float(lr.named_steps["ridge"].alpha)

    # ── LSTM ─────────────────────────────────────────────────────────────
    if not args.no_lstm:
        try:
            from models.lstm_model import train_lstm, predict_lstm, save_lstm
            lstm, history = train_lstm(X_fit, y_fit, X_val, y_val, epochs=args.epochs,
                                       batch_size=args.batch_size, seed=args.seed)
            preds_kw["LSTM"] = inv(predict_lstm(lstm, X_te))
            save_lstm(lstm)
            predict_fns["LSTM"] = make_predict_fn("lstm", lstm)
        except Exception as e:                                   # noqa: BLE001
            log.warning("LSTM training failed (%s) – continuing without it.", e)

    # ── Baselines (one-step) ─────────────────────────────────────────────
    # window row 0 = t-24, row -1 = t-1  (scaled target in column 0)
    preds_kw["Persistence"] = inv(X_te[:, -1, 0])
    preds_kw["SeasonalNaive"] = inv(X_te[:, 0, 0]) if args.look_back == 24 else inv(X_te[:, -1, 0])

    # ── One-step metrics (kW, held-out test period) ──────────────────────
    for name in ["LSTM", "LinearRegression", "Persistence", "SeasonalNaive"]:
        if name in preds_kw:
            results[name] = evaluate_predictions(y_kw, preds_kw[name], name)
    compare_models(results)

    # ── Multi-step back-test ─────────────────────────────────────────────
    log.info("Running %d-hour multi-step back-test …", HORIZON)
    bt = backtest_horizons(data["y_hourly"], data["split_ts"], predict_fns, scaler,
                           look_back=args.look_back, horizon=HORIZON)
    for name in bt["rmse"]:
        r = bt["rmse"][name]
        log.info("  %-17s RMSE @1h=%.3f  @24h=%.3f  @72h=%.3f kW", name, r[0], r[23], r[-1])

    # ── Persist everything the API needs ─────────────────────────────────
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": data["source"],
        "look_back": args.look_back,
        "feature_cols": data["feature_cols"],
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "split_ts": str(data["split_ts"]),
        "ridge_alpha": best_alpha,
        "horizon": HORIZON,
        "horizon_rmse": bt["rmse"], "horizon_mae": bt["mae"],
        "backtest_origins": bt["n_origins"],
        "lstm_epochs_run": len(history.get("loss", [])),
        "seed": args.seed,
        "metrics_note": "One-step-ahead, held-out test period, kW.",
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    save_charts(history, data["test_index"], y_kw, preds_kw, bt)
    log.info("✓ Done. Models, metrics.json, meta.json and charts saved.")


if __name__ == "__main__":
    main()
