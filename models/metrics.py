"""
metrics.py
──────────
Shared evaluation utilities: MAE, RMSE, R², MAPE, and a side-by-side
comparison printer.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """
    Compute regression metrics and return them in a dict.

    Metrics
    -------
    mae  : Mean Absolute Error
    rmse : Root Mean Squared Error
    r2   : R² coefficient of determination
    mape : Mean Absolute Percentage Error (%)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE – guard against zero true values
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    metrics = {"mae": round(float(mae), 5),
               "rmse": round(float(rmse), 5),
               "r2": round(float(r2), 5),
               "mape": round(float(mape), 3)}

    if model_name:
        _print_metrics(model_name, metrics)

    return metrics


def compare_models(results: dict) -> None:
    """
    Pretty-print a comparison table for multiple models.

    Parameters
    ----------
    results : {model_name: metrics_dict}
    """
    col_w = 14
    header = f"{'Model':<20} {'MAE':>{col_w}} {'RMSE':>{col_w}} {'R²':>{col_w}} {'MAPE %':>{col_w}}"
    sep = "─" * len(header)
    print("\n" + sep)
    print("  Model Performance Comparison")
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"{name:<20} "
            f"{m['mae']:>{col_w}.5f} "
            f"{m['rmse']:>{col_w}.5f} "
            f"{m['r2']:>{col_w}.5f} "
            f"{m['mape']:>{col_w}.3f}"
        )
    print(sep + "\n")


def _print_metrics(name: str, m: dict) -> None:
    print(
        f"[{name}]  MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}"
        f"  R²={m['r2']:.5f}  MAPE={m['mape']:.3f}%"
    )


def moving_average(series: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply a centred moving average to smooth a 1-D array.

    Edges are handled by a decreasing window (same as `np.convolve` in
    'same' mode with a uniform kernel).
    """
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")
