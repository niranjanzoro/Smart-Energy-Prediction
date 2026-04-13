"""
train_models.py
───────────────
Entry-point script that:
  1. Loads and preprocesses the dataset (synthetic by default)
  2. Trains both the LSTM and Linear Regression models
  3. Evaluates and compares them on the test set
  4. Saves both trained models + the fitted scaler to disk
  5. Generates and saves comparison charts

Run from the project root:
    python scripts/train_models.py [--synthetic] [--epochs 30]
"""

import argparse
import logging
import os
import sys

# Allow imports from sibling directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAVED_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
SCALER_PATH = os.path.join(SAVED_DIR, "scaler.pkl")
CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "charts")


def parse_args():
    p = argparse.ArgumentParser(description="Train energy prediction models")
    p.add_argument("--synthetic", action="store_true", default=True,
                   help="Use synthetic data (default: True)")
    p.add_argument("--real", action="store_true",
                   help="Use real UCI dataset (requires download)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--look-back", type=int, default=24,
                   help="Number of past hours used as input window")
    p.add_argument("--no-lstm", action="store_true",
                   help="Skip LSTM training (faster, no TF required)")
    return p.parse_args()


def save_training_charts(history: dict, y_test: np.ndarray,
                         lstm_preds: np.ndarray, lr_preds: np.ndarray) -> None:
    """Produce and save training-loss curve + prediction comparison chart."""
    os.makedirs(CHART_DIR, exist_ok=True)

    # ── Loss curve ──────────────────────────────────────────────────────
    if history:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["loss"], label="Train Loss", lw=2)
        ax.plot(history["val_loss"], label="Val Loss", lw=2, linestyle="--")
        ax.set_title("LSTM Training / Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "training_loss.png"), dpi=120)
        plt.close(fig)

    # ── Prediction comparison ────────────────────────────────────────────
    n_plot = min(200, len(y_test))
    x_ax = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_ax, y_test[:n_plot], label="Actual", lw=2, color="#2563eb")
    if lstm_preds is not None:
        ax.plot(x_ax, lstm_preds[:n_plot], label="LSTM", lw=1.5,
                linestyle="--", color="#16a34a")
    ax.plot(x_ax, lr_preds[:n_plot], label="Linear Regression", lw=1.5,
            linestyle=":", color="#dc2626")
    ax.set_title("Prediction vs Actual (first 200 test samples)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalised Energy Consumption")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "prediction_comparison.png"), dpi=120)
    plt.close(fig)

    log.info("Charts saved → %s", CHART_DIR)


def main():
    args = parse_args()
    use_synthetic = not args.real

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Preparing data (synthetic=%s, look_back=%d) …", use_synthetic, args.look_back)
    from models.data_loader import get_processed_data
    data = get_processed_data(
        look_back=args.look_back,
        use_synthetic=use_synthetic,
    )

    X_tr_seq = data["X_train_seq"]
    y_tr_seq = data["y_train_seq"]
    X_te_seq = data["X_test_seq"]
    y_te_seq = data["y_test_seq"]
    X_tr_flat = data["X_train_flat"]
    X_te_flat = data["X_test_flat"]
    scaler = data["scaler"]

    # Validation split from training data (last 15%)
    val_split = int(len(X_tr_seq) * 0.85)
    X_val_seq, y_val_seq = X_tr_seq[val_split:], y_tr_seq[val_split:]
    X_tr_seq_t, y_tr_seq_t = X_tr_seq[:val_split], y_tr_seq[:val_split]

    from models.metrics import evaluate_predictions, compare_models

    results = {}
    lstm_preds = None
    history = {}

    # ── LSTM ──────────────────────────────────────────────────────────────
    if not args.no_lstm:
        try:
            from models.lstm_model import train_lstm, predict_lstm, save_lstm
            model_lstm, history = train_lstm(
                X_tr_seq_t, y_tr_seq_t,
                X_val_seq, y_val_seq,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            lstm_preds = predict_lstm(model_lstm, X_te_seq)
            results["LSTM"] = evaluate_predictions(y_te_seq, lstm_preds, "LSTM")
            save_lstm(model_lstm)
        except Exception as e:
            log.warning("LSTM training failed (%s) – skipping.", e)
    else:
        log.info("LSTM training skipped (--no-lstm flag).")

    # ── Linear Regression ────────────────────────────────────────────────
    from models.linear_model import train_linear, predict_linear, save_linear
    model_lr = train_linear(X_tr_flat, y_tr_seq)
    lr_preds = predict_linear(model_lr, X_te_flat)
    results["LinearRegression"] = evaluate_predictions(y_te_seq, lr_preds, "LinearRegression")
    save_linear(model_lr)

    # Save scaler
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Scaler saved → %s", SCALER_PATH)

    # ── Comparison ───────────────────────────────────────────────────────
    compare_models(results)
    save_training_charts(history, y_te_seq, lstm_preds, lr_preds)
    log.info("✓ All done.  Models and charts saved.")


if __name__ == "__main__":
    main()
