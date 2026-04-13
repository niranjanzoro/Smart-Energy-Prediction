"""
anomaly_detector.py
───────────────────
Lightweight anomaly detection for energy consumption time-series data.

Methods
───────
1. Z-Score           – flag readings beyond N standard deviations
2. IQR Fence         – flag readings beyond IQR-based fences
3. Rolling Z-Score   – adaptive threshold that follows local statistics
4. Isolation Forest  – scikit-learn ensemble method (multivariate)

All detectors return a boolean mask (True = anomaly).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)


# ── Simple Statistical Detectors ──────────────────────────────────────────────

def zscore_anomalies(series: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Flag values whose global z-score exceeds `threshold`.

    Parameters
    ----------
    series    : 1-D array of energy readings
    threshold : z-score cut-off (default 3 σ)

    Returns
    -------
    Boolean mask – True where anomaly.
    """
    mean, std = np.mean(series), np.std(series)
    if std == 0:
        return np.zeros(len(series), dtype=bool)
    z = np.abs((series - mean) / std)
    return z > threshold


def iqr_anomalies(series: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """Flag values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)


def rolling_zscore_anomalies(
    series: np.ndarray,
    window: int = 24,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Adaptive anomaly detection using a rolling z-score.

    Each point is scored relative to the rolling mean and standard
    deviation of the preceding `window` observations.
    """
    s = pd.Series(series)
    roll_mean = s.rolling(window, min_periods=1).mean()
    roll_std = s.rolling(window, min_periods=1).std().fillna(1e-9)
    z = np.abs((s - roll_mean) / roll_std)
    return (z > threshold).values


# ── Isolation Forest ──────────────────────────────────────────────────────────

def isolation_forest_anomalies(
    X: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
) -> np.ndarray:
    """
    Multivariate anomaly detection using Isolation Forest.

    Parameters
    ----------
    X             : 2-D array (n_samples, n_features)
    contamination : expected fraction of anomalies
    random_state  : reproducibility seed

    Returns
    -------
    Boolean mask – True where anomaly.
    """
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = clf.fit_predict(X)
    return preds == -1   # -1 = anomaly in sklearn convention


# ── Combined Detector ─────────────────────────────────────────────────────────

def detect_anomalies(
    series: np.ndarray,
    method: str = "rolling_zscore",
    **kwargs,
) -> dict:
    """
    Run the specified anomaly detector and return a results dict.

    Parameters
    ----------
    series  : 1-D energy consumption array
    method  : one of 'zscore', 'iqr', 'rolling_zscore', 'isolation_forest'
    kwargs  : forwarded to the selected detector

    Returns
    -------
    {
        'mask'         : boolean np.ndarray,
        'indices'      : list of anomaly indices,
        'values'       : list of anomalous values,
        'count'        : int,
        'pct'          : float (percentage of series),
        'severity'     : str ('low' / 'medium' / 'high'),
    }
    """
    dispatch = {
        "zscore": zscore_anomalies,
        "iqr": iqr_anomalies,
        "rolling_zscore": rolling_zscore_anomalies,
        "isolation_forest": lambda s, **kw: isolation_forest_anomalies(
            s.reshape(-1, 1), **kw
        ),
    }

    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'.  Choose from {list(dispatch)}")

    mask = dispatch[method](series, **kwargs)
    indices = np.where(mask)[0].tolist()
    pct = 100 * len(indices) / len(series)

    # Simple severity rating
    if pct < 1:
        severity = "low"
    elif pct < 5:
        severity = "medium"
    else:
        severity = "high"

    log.info(
        "Anomaly detection [%s] → %d anomalies (%.2f%%) – severity: %s",
        method,
        len(indices),
        pct,
        severity,
    )

    return {
        "mask": mask,
        "indices": indices,
        "values": [float(series[i]) for i in indices],
        "count": len(indices),
        "pct": round(pct, 3),
        "severity": severity,
    }


# ── Optimisation Suggestions ──────────────────────────────────────────────────

def generate_suggestions(
    series: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
    anomaly_result: dict = None,
) -> list[dict]:
    """
    Analyse consumption patterns and return actionable optimisation tips.

    Parameters
    ----------
    series        : 1-D hourly energy consumption array (kW)
    timestamps    : DatetimeIndex aligned with `series`
    anomaly_result: output of detect_anomalies()

    Returns
    -------
    List of suggestion dicts:
        {
          'category': str,
          'title'   : str,
          'detail'  : str,
          'impact'  : str ('low' / 'medium' / 'high'),
          'saving_pct': float,
        }
    """
    suggestions = []
    mean_usage = float(np.mean(series))
    peak_usage = float(np.max(series))

    # 1. Peak-hour detection (assume hours 17-21 are costly)
    if timestamps is not None:
        peak_mask = (timestamps.hour >= 17) & (timestamps.hour <= 21)
        peak_mean = float(np.mean(series[peak_mask])) if peak_mask.any() else 0
        offpeak_mean = float(np.mean(series[~peak_mask])) if (~peak_mask).any() else 0

        if peak_mean > 1.3 * offpeak_mean:
            saving = round(100 * (peak_mean - offpeak_mean) / peak_mean, 1)
            suggestions.append(
                {
                    "category": "Peak Hours",
                    "title": "Shift Usage Away From Peak Hours (5 PM – 9 PM)",
                    "detail": (
                        f"Your average peak-hour consumption is {peak_mean:.2f} kW, "
                        f"which is {saving}% higher than off-peak.  "
                        "Consider running dishwashers, washing machines, and EV chargers "
                        "after 10 PM to cut costs significantly."
                    ),
                    "impact": "high",
                    "saving_pct": saving,
                }
            )

    # 2. High overall consumption
    if mean_usage > 1.5:  # kW threshold
        suggestions.append(
            {
                "category": "Baseline Load",
                "title": "Reduce Standby / Always-On Appliances",
                "detail": (
                    f"Your average consumption is {mean_usage:.2f} kW.  "
                    "Identify standby loads such as set-top boxes, old fridges, and "
                    "gaming consoles.  Smart power strips can cut phantom load by up to 10%."
                ),
                "impact": "medium",
                "saving_pct": 10.0,
            }
        )

    # 3. Abnormal spike detection
    if anomaly_result and anomaly_result["count"] > 0:
        suggestions.append(
            {
                "category": "Anomaly",
                "title": f"Investigate {anomaly_result['count']} Abnormal Consumption Spikes",
                "detail": (
                    f"We detected {anomaly_result['count']} unusual readings "
                    f"({anomaly_result['pct']:.1f}% of your data).  "
                    "These may indicate faulty appliances, leaking HVAC, or unauthorised devices.  "
                    "Check your circuit breaker log and compare with appliance schedules."
                ),
                "impact": anomaly_result["severity"],
                "saving_pct": 5.0,
            }
        )

    # 4. Night-time consumption
    if timestamps is not None:
        night_mask = (timestamps.hour >= 0) & (timestamps.hour <= 5)
        night_mean = float(np.mean(series[night_mask])) if night_mask.any() else 0
        if night_mean > 0.5 * mean_usage:
            suggestions.append(
                {
                    "category": "Night Usage",
                    "title": "Reduce Late-Night Energy Waste",
                    "detail": (
                        f"Night-time usage is {night_mean:.2f} kW – "
                        "unusually high for sleeping hours.  "
                        "Check HVAC schedules, electric water heaters, and leave only "
                        "essential devices on standby."
                    ),
                    "impact": "medium",
                    "saving_pct": 8.0,
                }
            )

    # 5. Generic tip if no patterns found
    if not suggestions:
        suggestions.append(
            {
                "category": "General",
                "title": "Your Consumption Pattern Looks Healthy",
                "detail": (
                    "No major inefficiencies detected.  Consider a smart thermostat or "
                    "solar panels to further reduce your carbon footprint."
                ),
                "impact": "low",
                "saving_pct": 0.0,
            }
        )

    return suggestions
