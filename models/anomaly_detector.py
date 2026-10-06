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

    Each point is scored against the mean / std of the PRECEDING ``window``
    observations (the point itself is excluded, otherwise a spike inflates
    its own baseline and hides itself).  The first few points, which have too
    little history, are never flagged.
    """
    s = pd.Series(series, dtype=float)
    prev = s.shift(1)
    roll_mean = prev.rolling(window, min_periods=max(3, window // 4)).mean()
    roll_std = prev.rolling(window, min_periods=max(3, window // 4)).std()
    roll_std = roll_std.clip(lower=1e-3)          # avoid division by ~0 on flat stretches
    z = ((s - roll_mean) / roll_std).abs()
    return (z > threshold).fillna(False).to_numpy()


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
    series = np.asarray(series, dtype=float)

    # Each detector has its own parameters; map the generic ``threshold``
    # onto the right one and drop anything a detector does not understand
    # (previously a mismatched keyword raised TypeError and was swallowed).
    threshold = kwargs.pop("threshold", None)
    if method in ("zscore", "rolling_zscore") and threshold is not None:
        kwargs["threshold"] = threshold
    elif method == "iqr" and threshold is not None:
        kwargs.setdefault("factor", threshold)
    allowed = {
        "zscore": {"threshold"},
        "iqr": {"factor"},
        "rolling_zscore": {"window", "threshold"},
        "isolation_forest": {"contamination", "random_state"},
    }
    if method not in allowed:
        raise ValueError(f"Unknown method '{method}'.  Choose from {list(allowed)}")
    kwargs = {k: v for k, v in kwargs.items() if k in allowed[method]}

    dispatch = {
        "zscore": zscore_anomalies,
        "iqr": iqr_anomalies,
        "rolling_zscore": rolling_zscore_anomalies,
        "isolation_forest": lambda s, **kw: isolation_forest_anomalies(
            s.reshape(-1, 1), **kw
        ),
    }

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
    Rule-based analysis of a consumption pattern → actionable tips.

    NOTE: these are transparent heuristics, not a learned model.  Each tip has
    a ``basis`` field: 'computed' (derived from the series) or 'heuristic'
    (a typical figure from the energy-efficiency literature, not measured).

    Parameters
    ----------
    series        : 1-D hourly consumption array (kW)
    timestamps    : DatetimeIndex aligned with ``series`` (enables time-of-day tips)
    anomaly_result: output of detect_anomalies()

    Returns list of {category, title, detail, impact, saving_pct, basis}.
    """
    series = np.asarray(series, dtype=float)
    suggestions = []
    mean_usage = float(np.mean(series))

    if timestamps is not None:
        hours = np.asarray(pd.DatetimeIndex(timestamps).hour)

        # 1. Evening peak (17:00-21:59 assumed to be the expensive tariff window)
        peak_mask = (hours >= 17) & (hours <= 21)
        if peak_mask.any() and (~peak_mask).any():
            peak_mean = float(series[peak_mask].mean())
            offpeak_mean = float(series[~peak_mask].mean())
            if offpeak_mean > 0 and peak_mean > 1.3 * offpeak_mean:
                higher_pct = round(100 * (peak_mean - offpeak_mean) / offpeak_mean, 1)
                # share of total energy that sits above the off-peak level in peak hours
                excess = float(np.clip(series[peak_mask] - offpeak_mean, 0, None).sum())
                shiftable_pct = round(100 * excess / series.sum(), 1)
                suggestions.append({
                    "category": "Peak Hours",
                    "title": "Shift Usage Away From Peak Hours (5 PM – 10 PM)",
                    "detail": (
                        f"Average peak-hour load is {peak_mean:.2f} kW, {higher_pct}% above "
                        f"your off-peak average ({offpeak_mean:.2f} kW). About {shiftable_pct}% "
                        "of your total energy is this peak excess. Running dishwashers, "
                        "washing machines and EV chargers later can reduce cost on time-of-use tariffs."
                    ),
                    "impact": "high",
                    "saving_pct": shiftable_pct,
                    "basis": "computed",
                })

        # 2. Night-time load (00:00-05:59)
        night_mask = hours <= 5
        if night_mask.any():
            night_mean = float(series[night_mask].mean())
            if night_mean > 0.75 * mean_usage:
                suggestions.append({
                    "category": "Night Usage",
                    "title": "Reduce Late-Night Energy Waste",
                    "detail": (
                        f"Night-time load is {night_mean:.2f} kW – {100 * night_mean / mean_usage:.0f}% "
                        "of your average, which is high for sleeping hours. Check HVAC schedules, "
                        "water heaters and devices left running."
                    ),
                    "impact": "medium",
                    "saving_pct": 8.0,
                    "basis": "heuristic",
                })

    # 3. High overall baseline
    if mean_usage > 1.5:
        suggestions.append({
            "category": "Baseline Load",
            "title": "Reduce Standby / Always-On Appliances",
            "detail": (
                f"Average consumption is {mean_usage:.2f} kW. Look for standby loads (set-top boxes, "
                "old fridges, consoles); smart power strips typically trim phantom load by up to ~10%."
            ),
            "impact": "medium",
            "saving_pct": 10.0,
            "basis": "heuristic",
        })

    # 4. Abnormal spikes
    if anomaly_result and anomaly_result["count"] > 0:
        suggestions.append({
            "category": "Anomaly",
            "title": f"Investigate {anomaly_result['count']} Abnormal Consumption Spikes",
            "detail": (
                f"{anomaly_result['count']} unusual readings were detected "
                f"({anomaly_result['pct']:.1f}% of the data). They can indicate a faulty appliance "
                "or a device left on; compare their timestamps with your appliance schedule."
            ),
            "impact": anomaly_result["severity"],
            "saving_pct": 5.0,
            "basis": "heuristic",
        })

    # 5. Nothing found
    if not suggestions:
        suggestions.append({
            "category": "General",
            "title": "Your Consumption Pattern Looks Healthy",
            "detail": (
                "No major inefficiencies detected. A smart thermostat or rooftop solar "
                "could reduce consumption further."
            ),
            "impact": "low",
            "saving_pct": 0.0,
            "basis": "computed",
        })

    return suggestions
