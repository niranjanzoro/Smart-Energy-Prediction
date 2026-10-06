"""
data_loader.py
──────────────
Loading, cleaning, feature engineering and splitting for the hourly
household-energy forecasting task.

Design decisions (see README → "Methodology")
─────────────────────────────────────────────
* Target  : hourly mean of Global_active_power (kW).
* Features: the target's own history + calendar features + 24h/168h lags and
            24h rolling statistics.  Electrical measurements measured at the
            same time as the target (voltage, current, sub-metering) are NOT
            used – they would not be known at forecast time, and
            Global_intensity is almost a deterministic function of the target.
            This also makes true multi-step (recursive) forecasting possible.
* Split   : strictly chronological.  The scaler and the outlier-clipping
            bounds are fitted on the TRAIN period only.
* Windows : sample i uses rows [i-look_back, i) to predict row i, so no
            information from the predicted hour leaks into its inputs.

Dataset: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
"""

import logging
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_FILE = os.path.join(DATA_DIR, "household_power_consumption.txt")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_energy.csv")

RAW_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
TARGET_COL = "Global_active_power"

FEATURE_COLS = [
    "Global_active_power",          # target history (must stay first)
    "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "month",
    "rolling_mean_24h", "rolling_std_24h",
    "lag_24h", "lag_168h",
]
MIN_HISTORY = 168   # hours needed before the first valid feature row
OUTLIER_IQR_FACTOR = 4.0


# ── Download / generate raw data ─────────────────────────────────────────────

def download_dataset() -> None:
    """Download and unzip the UCI dataset if it is not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        log.info("Raw dataset already present – skipping download.")
        return

    log.info("Downloading UCI Household Power Consumption dataset …")
    zip_path = os.path.join(DATA_DIR, "hpc.zip")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Downloaded archive did not contain {os.path.basename(RAW_FILE)}. "
            "Download it manually (see README) and place it in data/."
        )
    log.info("Dataset downloaded and extracted to %s", DATA_DIR)


def generate_synthetic_dataset(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic HOURLY household-load dataset (fallback / demo data).

    The load profile has: morning and evening peaks, a different weekend shape,
    annual (winter-heating) seasonality, AR(1) autocorrelated noise, a few
    low-usage "away" days and random appliance spikes.  The target stays well
    inside the clipping range (a previous version had a bug where a trend term
    saturated the signal at its upper clip, making the target constant).
    """
    log.info("Generating synthetic hourly dataset (%d days) …", n_days)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60
    weekend = (idx.dayofweek.values >= 5).astype(float)
    doy = idx.dayofyear.values

    def bump(centre, width):
        return np.exp(-0.5 * ((hour - centre) / width) ** 2)

    base = 0.55
    weekday_shape = 0.9 * bump(7.5, 1.2) + 1.5 * bump(19.5, 2.0) + 0.25 * bump(13, 3)
    weekend_shape = 0.7 * bump(9.5, 1.8) + 1.3 * bump(19.5, 2.3) + 0.6 * bump(13.5, 2.5)
    shape = (1 - weekend) * weekday_shape + weekend * weekend_shape
    season = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365)        # winter peak

    # AR(1) noise (consumption is autocorrelated, not white noise)
    eps = rng.normal(0, 0.18, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]

    # A few "away from home" days with low usage
    away = np.zeros(n)
    for d in rng.choice(n_days, size=max(3, n_days // 60), replace=False):
        away[d * 24:(d + 1) * 24] = 1
    spikes = (rng.random(n) < 0.004) * rng.uniform(1.5, 3.5, n)

    gap = (base + shape) * season * (1 - 0.45 * away) + noise + spikes
    gap = np.clip(gap, 0.1, 8.0)

    return pd.DataFrame(
        {
            "Global_active_power": gap,
            "Global_reactive_power": np.clip(0.05 + 0.04 * gap + rng.normal(0, 0.02, n), 0, 1),
            "Voltage": np.clip(241 - 1.2 * gap + rng.normal(0, 1.5, n), 230, 250),
            "Global_intensity": np.clip(gap * 4.2 + rng.normal(0, 0.3, n), 0.2, 40),
            "Sub_metering_1": np.clip(rng.exponential(0.5, n) * (1 + 2 * bump(19, 2)), 0, 30),
            "Sub_metering_2": np.clip(rng.exponential(0.4, n) * (1 + weekend), 0, 30),
            "Sub_metering_3": np.clip(2 + 3 * gap + rng.normal(0, 1, n), 0, 30),
        },
        index=idx,
    )


# ── Loading & cleaning ───────────────────────────────────────────────────────

def load_raw(use_synthetic: bool = False) -> pd.DataFrame:
    """
    Load the raw data.

    * ``use_synthetic=True``  → hourly synthetic data.
    * ``use_synthetic=False`` → minute-level UCI file; raises
      FileNotFoundError if it is missing (no silent fallback to fake data).
    """
    if use_synthetic:
        return generate_synthetic_dataset()

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"{RAW_FILE} not found. Run `python scripts/train_models.py --real` "
            "(auto-downloads) or place the UCI file in data/."
        )
    log.info("Loading raw dataset from %s …", RAW_FILE)
    df = pd.read_csv(RAW_FILE, sep=";", na_values=["?"], low_memory=False)
    df.index = pd.to_datetime(
        df.pop("Date") + " " + df.pop("Time"), format="%d/%m/%Y %H:%M:%S"
    )
    df.index.name = "datetime"
    return df[RAW_COLS].astype(float)


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert to a complete, gap-free hourly series.

    Minute-level data is averaged per hour.  Short gaps (≤ 6 h) are linearly
    interpolated; longer gaps are filled with the value from 24 h earlier
    (same hour yesterday) and finally forward/back-filled.
    """
    hourly = df.resample("h").mean() if (df.index[1] - df.index[0]) < pd.Timedelta("1h") else df
    hourly = hourly.asfreq("h")
    n_missing = int(hourly[TARGET_COL].isna().sum())
    if n_missing:
        log.info("Filling %d missing hourly values (%.2f%%)", n_missing,
                 100 * n_missing / len(hourly))
    hourly = hourly.interpolate(limit=6, limit_area="inside")
    hourly = hourly.fillna(hourly.shift(24)).ffill().bfill()
    return hourly


def clip_outliers(y: pd.Series, train_end: pd.Timestamp,
                  factor: float = OUTLIER_IQR_FACTOR) -> tuple[pd.Series, tuple]:
    """Clip the target to [Q1 - f·IQR, Q3 + f·IQR] using TRAIN-period quartiles only."""
    train = y[y.index < train_end]
    q1, q3 = train.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = max(0.0, q1 - factor * iqr), q3 + factor * iqr
    return y.clip(lo, hi), (float(lo), float(hi))


# ── Feature engineering ──────────────────────────────────────────────────────

def build_feature_frame(y: pd.Series) -> pd.DataFrame:
    """
    Build model features from an hourly kW series with a DatetimeIndex.

    All features at row t depend only on y[≤ t] and the calendar, so the same
    function is used for training and for recursive forecasting.  The first
    MIN_HISTORY rows are dropped (lag_168h undefined).
    """
    idx = y.index
    f = pd.DataFrame(index=idx)
    f[TARGET_COL] = y.values
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["day_of_week"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["month"] = idx.month
    f["rolling_mean_24h"] = y.rolling(24, min_periods=24).mean()
    f["rolling_std_24h"] = y.rolling(24, min_periods=24).std()
    f["lag_24h"] = y.shift(24)
    f["lag_168h"] = y.shift(168)
    return f.dropna()[FEATURE_COLS]


# ── Scaling helpers ──────────────────────────────────────────────────────────

def inverse_target(y_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Convert scaled target values (feature column 0) back to kW."""
    return (np.asarray(y_scaled, dtype=float) - scaler.min_[0]) / scaler.scale_[0]


def build_sequences(data: np.ndarray, look_back: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Windows over a 2-D array: X[k] = data[k : k+look_back], y[k] = data[k+look_back, 0].
    Returns X (N, look_back, n_features), y (N,); the target index of sample k
    is ``k + look_back``.
    """
    n = len(data) - look_back
    X = np.stack([data[k:k + look_back] for k in range(n)])
    y = data[look_back:, 0]
    return X, y


# ── Public API ───────────────────────────────────────────────────────────────

def get_processed_data(
    look_back: int = 24,
    test_size: float = 0.2,
    use_synthetic: bool = False,
    save: bool = True,
) -> dict:
    """
    Full pipeline: load → hourly → split point → clip (train stats) →
    features → scale (train stats) → windows → chronological split.

    Returns a dict with:
        X_train_seq / y_train_seq, X_test_seq / y_test_seq   (scaled, LSTM shape)
        X_train_flat / y_train_flat, X_test_flat / y_test_flat (flattened)
        y_test_kw, test_index  – test targets in kW and their timestamps
        scaler, feature_cols, look_back
        y_hourly   – cleaned hourly target in kW (full series)
        split_ts   – first timestamp of the test period
        df_hourly  – all hourly raw columns (EDA)
        features   – unscaled feature frame
        source     – 'synthetic' or 'uci'
    """
    raw = load_raw(use_synthetic=use_synthetic)
    df_hourly = to_hourly(raw)

    y = df_hourly[TARGET_COL]
    split_ts = y.index[int(len(y) * (1 - test_size))]
    y_clean, bounds = clip_outliers(y, split_ts)
    log.info("Outlier clip bounds (train quartiles): %.3f – %.3f kW", *bounds)

    feats = build_feature_frame(y_clean)
    is_train_row = feats.index < split_ts

    scaler = MinMaxScaler()
    scaler.fit(feats.loc[is_train_row].values)
    scaled = scaler.transform(feats.values)

    X, y_s = build_sequences(scaled, look_back)
    target_idx = feats.index[look_back:]
    tr = target_idx < split_ts
    te = ~tr

    X_tr, y_tr, X_te, y_te = X[tr], y_s[tr], X[te], y_s[te]
    log.info("Data ready | train=%s test=%s | train ends %s, test starts %s",
             X_tr.shape, X_te.shape, target_idx[tr][-1], target_idx[te][0])

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        feats.to_csv(PROCESSED_FILE)

    return {
        "X_train_seq": X_tr, "y_train_seq": y_tr,
        "X_test_seq": X_te, "y_test_seq": y_te,
        "X_train_flat": X_tr.reshape(len(X_tr), -1), "y_train_flat": y_tr,
        "X_test_flat": X_te.reshape(len(X_te), -1), "y_test_flat": y_te,
        "y_test_kw": inverse_target(y_te, scaler),
        "test_index": target_idx[te],
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "look_back": look_back,
        "y_hourly": y_clean,
        "split_ts": split_ts,
        "df_hourly": df_hourly,
        "features": feats,
        "source": "synthetic" if use_synthetic else "uci",
        "clip_bounds": bounds,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    d = get_processed_data(use_synthetic=True, save=False)
    print("Train:", d["X_train_seq"].shape, " Test:", d["X_test_seq"].shape)
    print("Features:", d["feature_cols"])
    print(d["y_hourly"].describe().round(3))
