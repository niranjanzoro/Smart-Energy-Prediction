"""
linear_model.py
───────────────
Linear Regression baseline model for energy consumption prediction.

Acts as a performance benchmark against the LSTM.  Uses the same
windowed feature representation (flattened look-back window) so that
comparisons are on equal footing.
"""

import os
import logging

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .metrics import evaluate_predictions  # noqa: F401

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
LR_PATH = os.path.join(MODELS_DIR, "linear_model.pkl")


# ── Model Definition ──────────────────────────────────────────────────────────

def build_linear(degree: int = 1, alpha: float = 1.0) -> Pipeline:
    """
    Build a Ridge Regression pipeline.

    Parameters
    ----------
    degree : polynomial degree (1 = plain linear, 2 = quadratic interactions)
    alpha  : L2 regularisation strength

    Returns
    -------
    Scikit-learn Pipeline: PolynomialFeatures → StandardScaler → Ridge
    """
    steps = [
        ("scaler", StandardScaler()),
    ]
    if degree > 1:
        # Insert polynomial expansion BEFORE the scaler
        steps.insert(0, ("poly", PolynomialFeatures(degree=degree, include_bias=False)))

    steps.append(("ridge", Ridge(alpha=alpha)))
    pipe = Pipeline(steps)
    log.info("Linear model built | degree=%d  alpha=%.3f", degree, alpha)
    return pipe


# ── Training ──────────────────────────────────────────────────────────────────

def train_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int = 1,
    alpha: float = 1.0,
) -> Pipeline:
    """
    Fit the linear model.

    Parameters
    ----------
    X_train : 2-D array of shape (samples, look_back * n_features)
    y_train : 1-D target array
    degree  : polynomial feature degree
    alpha   : Ridge regularisation

    Returns
    -------
    Fitted Pipeline
    """
    model = build_linear(degree=degree, alpha=alpha)
    log.info("Training Linear model | X_train=%s", X_train.shape)
    model.fit(X_train, y_train)
    log.info("Training complete")
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_linear(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return predictions from the linear model."""
    return model.predict(X).flatten()


# ── Persistence ───────────────────────────────────────────────────────────────

def save_linear(model: Pipeline) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, LR_PATH)
    log.info("Linear model saved → %s", LR_PATH)


def load_linear() -> Pipeline:
    if not os.path.exists(LR_PATH):
        raise FileNotFoundError(
            f"No saved linear model found at {LR_PATH}.  Run train_models.py first."
        )
    model = joblib.load(LR_PATH)
    log.info("Linear model loaded from %s", LR_PATH)
    return model


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X = np.random.rand(300, 24 * 12).astype("float32")
    y = np.random.rand(300).astype("float32")
    model = train_linear(X[:250], y[:250])
    preds = predict_linear(model, X[250:])
    print("Predictions shape:", preds.shape)
    print("Sample predictions:", preds[:5])
