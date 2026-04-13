"""
data_loader.py
──────────────
Handles loading, cleaning, and preprocessing the UCI Household Power
Consumption dataset.  Produces train/test splits and normalised numpy
arrays ready for LSTM and Linear-Regression models.

Dataset source:
  https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
"""

import os
import io
import logging
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_FILE = os.path.join(DATA_DIR, "household_power_consumption.txt")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_energy.csv")

FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TARGET_COL = "Global_active_power"


# ── Download helpers ─────────────────────────────────────────────────────────

def download_dataset() -> None:
    """Download and unzip the UCI dataset if it is not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        log.info("Raw dataset already present – skipping download.")
        return

    log.info("Downloading UCI Household Power Consumption dataset …")
    zip_path = os.path.join(DATA_DIR, "hpc.zip")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    log.info("Dataset downloaded and extracted to %s", DATA_DIR)


def generate_synthetic_dataset(n_days: int = 365) -> pd.DataFrame:
    """
    Generate a realistic synthetic energy dataset when the real UCI file
    is unavailable (useful for demos / CI).

    Parameters
    ----------
    n_days : int
        Number of days of minute-level data to generate.

    Returns
    -------
    pd.DataFrame with datetime index and the same columns as the UCI dataset.
    """
    log.info("Generating synthetic dataset (%d days) …", n_days)
    rng = np.random.default_rng(42)
    n = n_days * 24 * 60  # minute-level

    time_idx = pd.date_range("2023-01-01", periods=n, freq="min")
    t = np.arange(n)

    # Daily + weekly seasonality with realistic noise
    daily = np.sin(2 * np.pi * t / (24 * 60))
    weekly = 0.3 * np.sin(2 * np.pi * t / (7 * 24 * 60))
    trend = 0.0001 * t
    noise = rng.normal(0, 0.15, n)

    gap = np.clip(1.5 + daily + weekly + trend + noise, 0.1, 8.0)

    df = pd.DataFrame(
        {
            "Global_active_power": gap,
            "Global_reactive_power": np.clip(0.1 + 0.05 * gap + rng.normal(0, 0.02, n), 0, 1),
            "Voltage": np.clip(240 + rng.normal(0, 2, n), 230, 250),
            "Global_intensity": np.clip(gap * 2.5 + rng.normal(0, 0.1, n), 0.5, 20),
            "Sub_metering_1": np.clip(rng.exponential(0.5, n), 0, 5),
            "Sub_metering_2": np.clip(rng.exponential(0.3, n), 0, 3),
            "Sub_metering_3": np.clip(5 * (daily > 0).astype(float) + rng.normal(0, 0.5, n), 0, 20),
        },
        index=time_idx,
    )
    return df


# ── Loading & Cleaning ────────────────────────────────────────────────────────

def load_raw(use_synthetic: bool = False) -> pd.DataFrame:
    """
    Load the raw UCI text file or fall back to a synthetic dataset.

    Returns
    -------
    pd.DataFrame with a DatetimeIndex at minute frequency.
    """
    if use_synthetic or not os.path.exists(RAW_FILE):
        return generate_synthetic_dataset()

    log.info("Loading raw dataset from %s …", RAW_FILE)
    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        na_values=["?"],
        low_memory=False,
    )
    df.set_index("datetime", inplace=True)
    df = df[FEATURE_COLS]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and remove obvious outliers.

    Strategy
    --------
    * Forward-fill gaps ≤ 60 consecutive minutes.
    * Drop any remaining NaN rows (they are rare in the UCI dataset).
    * Clip extreme outlier spikes beyond 4 × IQR.
    """
    log.info("Cleaning dataset – shape before: %s", df.shape)

    # Forward-fill short gaps then backward-fill at the very start
    df = df.ffill(limit=60).bfill(limit=60)
    df = df.dropna()

    # Outlier removal via IQR clipping
    for col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 4 * iqr, q3 + 4 * iqr)

    log.info("Cleaning done – shape after:  %s", df.shape)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and rolling-window features to enrich the dataset.

    New columns
    -----------
    hour, day_of_week, month, is_weekend
    rolling_mean_1h, rolling_std_1h  (60-minute window on target)
    rolling_mean_24h                 (24-hour window on target)
    lag_1h, lag_24h                  (1-hour and 24-hour lags)
    """
    df = df.copy()

    # Temporal features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Rolling statistics on target
    target = df[TARGET_COL]
    df["rolling_mean_1h"] = target.rolling(60, min_periods=1).mean()
    df["rolling_std_1h"] = target.rolling(60, min_periods=1).std().fillna(0)
    df["rolling_mean_24h"] = target.rolling(1440, min_periods=1).mean()

    # Lag features
    df["lag_1h"] = target.shift(60).bfill()
    df["lag_24h"] = target.shift(1440).bfill()

    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level data to hourly means for faster model training."""
    log.info("Resampling to hourly frequency …")
    return df.resample("h").mean()


# ── Normalisation & Splitting ─────────────────────────────────────────────────

def normalise(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit a MinMaxScaler on the training set and transform both splits.

    Returns
    -------
    train_scaled, test_scaled, scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler


def build_sequences(data: np.ndarray, look_back: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape flat 2-D array into (samples, look_back, features) for LSTM.

    Parameters
    ----------
    data      : 2-D array of shape (timesteps, features)
    look_back : number of past time-steps fed as input

    Returns
    -------
    X of shape (N, look_back, features), y of shape (N,)
    """
    X, y = [], []
    target_idx = 0  # Global_active_power is the first column
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


# ── Public API ────────────────────────────────────────────────────────────────

def get_processed_data(
    look_back: int = 24,
    test_size: float = 0.2,
    use_synthetic: bool = False,
    hourly: bool = True,
) -> dict:
    """
    Full pipeline: load → clean → feature engineering → normalise → split.

    Returns
    -------
    dict with keys:
        X_train_seq, y_train_seq  – LSTM-ready sequences
        X_test_seq,  y_test_seq
        X_train_flat, y_train_flat – flattened for LinearRegression
        X_test_flat,  y_test_flat
        scaler        – fitted MinMaxScaler
        df            – processed DataFrame (for plotting)
        feature_cols  – list of feature column names used
    """
    df = load_raw(use_synthetic=use_synthetic)
    df = clean(df)
    df = engineer_features(df)

    if hourly:
        df = resample_hourly(df)

    # All numeric columns after feature engineering
    all_cols = [c for c in df.columns if df[c].dtype != object]
    # Ensure TARGET_COL is first
    ordered = [TARGET_COL] + [c for c in all_cols if c != TARGET_COL]
    df = df[ordered]

    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    train_scaled, test_scaled, scaler = normalise(train_df, test_df)

    # Sequential data for LSTM
    X_tr_seq, y_tr_seq = build_sequences(train_scaled, look_back)
    X_te_seq, y_te_seq = build_sequences(test_scaled, look_back)

    # Flat data for Linear Regression (use same windowed approach for fairness)
    X_tr_flat = X_tr_seq.reshape(len(X_tr_seq), -1)
    X_te_flat = X_te_seq.reshape(len(X_te_seq), -1)

    log.info(
        "Data ready | train_seq=%s test_seq=%s",
        X_tr_seq.shape,
        X_te_seq.shape,
    )

    # Save processed CSV for quick reload
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_FILE)

    return {
        "X_train_seq": X_tr_seq,
        "y_train_seq": y_tr_seq,
        "X_test_seq": X_te_seq,
        "y_test_seq": y_te_seq,
        "X_train_flat": X_tr_flat,
        "y_train_flat": y_tr_seq,
        "X_test_flat": X_te_flat,
        "y_test_flat": y_te_seq,
        "scaler": scaler,
        "df": df,
        "feature_cols": ordered,
    }


if __name__ == "__main__":
    data = get_processed_data(use_synthetic=True)
    print("Dataset summary")
    print("  Training sequences:", data["X_train_seq"].shape)
    print("  Test sequences:    ", data["X_test_seq"].shape)
    print("  Features:          ", data["feature_cols"])
