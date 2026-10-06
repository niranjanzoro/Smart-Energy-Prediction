"""
lstm_model.py
─────────────
Defines, trains, evaluates, and persists an LSTM model for time-series
energy consumption forecasting.

Architecture
────────────
Input  → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
       → Dense(32, relu) → Dense(1)

The first LSTM layer returns sequences so the second LSTM can operate
on the full hidden-state trajectory.
"""

import os
import logging

import numpy as np
import joblib

# TensorFlow / Keras imports (gracefully degrade if TF is not installed)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .metrics import evaluate_predictions  # noqa: F401 (re-exported)

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


# ── Model Definition ──────────────────────────────────────────────────────────

def build_lstm(
    look_back: int,
    n_features: int,
    units_1: int = 128,
    units_2: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    """
    Build and compile the stacked LSTM model.

    Parameters
    ----------
    look_back    : number of past time-steps used as input window
    n_features   : number of input features per time-step
    units_1/2    : LSTM cell counts for layer 1 and 2
    dropout      : dropout rate applied after each LSTM layer
    learning_rate: initial Adam learning rate

    Returns
    -------
    Compiled Keras Sequential model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.  Run: pip install tensorflow")

    model = Sequential(
        [
            Input(shape=(look_back, n_features)),
            LSTM(units_1, return_sequences=True),
            Dropout(dropout),
            LSTM(units_2, return_sequences=False),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="EnergyLSTM",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",            # robust to outliers
        metrics=["mae"],
    )

    log.info("LSTM model built – param count: %s", model.count_params())
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 8,
    seed: int | None = 42,
) -> tuple:
    """
    Train the LSTM model with early-stopping and LR reduction on plateau.

    Parameters
    ----------
    X_train / y_train : training sequences and targets
    X_val   / y_val   : validation sequences and targets
    epochs            : max training epochs
    batch_size        : mini-batch size
    patience          : early-stopping patience

    Returns
    -------
    (model, history)  – trained Keras model + training history dict
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    look_back, n_features = X_train.shape[1], X_train.shape[2]
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)        # numpy + python + TF seeds
    model = build_lstm(look_back, n_features)

    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            LSTM_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    log.info(
        "Training LSTM | epochs=%d  batch=%d  X_train=%s",
        epochs,
        batch_size,
        X_train.shape,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    log.info("Training complete – best val_loss=%.5f", min(history.history["val_loss"]))
    return model, history.history


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_lstm(model, X: np.ndarray) -> np.ndarray:
    """Return raw (scaled) predictions from the LSTM model."""
    return model.predict(X, verbose=0).flatten()


# ── Persistence ───────────────────────────────────────────────────────────────

def save_lstm(model) -> None:
    """Save the trained LSTM model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(LSTM_PATH)
    log.info("LSTM saved → %s", LSTM_PATH)


def load_lstm():
    """Load a previously trained LSTM model from disk."""
    if not os.path.exists(LSTM_PATH):
        raise FileNotFoundError(
            f"No saved LSTM found at {LSTM_PATH}.  Run train_models.py first."
        )
    model = load_model(LSTM_PATH)
    log.info("LSTM loaded from %s", LSTM_PATH)
    return model


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test with random data
    look_back, n_features, n_samples = 24, 10, 500
    X_dummy = np.random.rand(n_samples, look_back, n_features).astype("float32")
    y_dummy = np.random.rand(n_samples).astype("float32")

    model, hist = train_lstm(
        X_dummy[:400], y_dummy[:400],
        X_dummy[400:], y_dummy[400:],
        epochs=3,
        batch_size=32,
        patience=2,
    )
    preds = predict_lstm(model, X_dummy[400:])
    print("Predictions shape:", preds.shape)
    print("Sample predictions:", preds[:5])
