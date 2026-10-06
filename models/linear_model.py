"""
linear_model.py
───────────────
Ridge-regression baseline for energy forecasting.

Uses the same flattened look-back window as the LSTM input so the comparison
is on equal footing.  The regularisation strength is tuned with
time-series cross-validation (expanding window, no shuffling).
"""

import os
import logging

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
LR_PATH = os.path.join(MODELS_DIR, "linear_model.pkl")
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def build_linear(degree: int = 1, alpha: float = 1.0) -> Pipeline:
    """
    Pipeline: [PolynomialFeatures →] StandardScaler → Ridge.

    degree > 1 expands 240 inputs to tens of thousands of columns – only
    sensible for a much smaller input window.
    """
    steps = []
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps += [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    return Pipeline(steps)


def train_linear(X_train: np.ndarray, y_train: np.ndarray, degree: int = 1,
                 alpha: float | None = None, n_splits: int = 5) -> Pipeline:
    """
    Fit the Ridge pipeline.

    If ``alpha`` is None the best alpha from ALPHA_GRID is chosen by
    TimeSeriesSplit cross-validation (scoring = RMSE) on the training data.
    """
    if alpha is not None:
        model = build_linear(degree, alpha).fit(X_train, y_train)
        log.info("Ridge fitted | alpha=%.3g (fixed)", alpha)
        return model

    search = GridSearchCV(
        build_linear(degree),
        {"ridge__alpha": ALPHA_GRID},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
    )
    search.fit(X_train, y_train)
    log.info("Ridge tuned with TimeSeriesSplit | best alpha=%.3g  CV-RMSE(scaled)=%.5f",
             search.best_params_["ridge__alpha"], -search.best_score_)
    return search.best_estimator_


def predict_linear(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X).ravel()


def save_linear(model: Pipeline) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, LR_PATH)
    log.info("Linear model saved → %s", LR_PATH)


def load_linear() -> Pipeline:
    if not os.path.exists(LR_PATH):
        raise FileNotFoundError(
            f"No saved linear model at {LR_PATH}. Run scripts/train_models.py first."
        )
    return joblib.load(LR_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X = np.random.rand(300, 240).astype("float32")
    y = np.random.rand(300).astype("float32")
    m = train_linear(X[:250], y[:250])
    print("Predictions:", predict_linear(m, X[250:])[:5])
