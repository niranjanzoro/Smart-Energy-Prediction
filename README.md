# ⚡ Smart Energy Consumption Predictor

> AI-powered household energy forecasting using LSTM & Linear Regression, with real-time simulation, anomaly detection, and actionable optimisation suggestions.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Running the Application](#running-the-application)
7. [API Reference](#api-reference)
8. [Model Details](#model-details)
9. [Dataset](#dataset)
10. [Deployment](#deployment)
11. [Screenshots](#screenshots)
12. [Academic Notes](#academic-notes)

---

## Project Overview

This system predicts household energy consumption up to 72 hours ahead and surfaces personalised energy-saving recommendations. It compares two approaches:

| Model | Approach | Use Case |
|---|---|---|
| **LSTM** | Deep learning, time-series sequences | Complex temporal patterns |
| **Linear Regression** | Ridge regression with lag features | Fast baseline, interpretable |

Both models are evaluated using **MAE**, **RMSE**, **R²**, and **MAPE**.

---

## Features

- **Dual-model forecasting** – LSTM and Linear Regression with confidence intervals
- **Interactive dashboard** – Live charts powered by Chart.js
- **Real-time simulation** – Streams next-step predictions every 2 seconds
- **Anomaly detection** – Rolling Z-Score, IQR, and Isolation Forest methods
- **AI optimisation tips** – Pattern-based suggestions (peak shifting, standby reduction, etc.)
- **Model comparison** – Side-by-side MAE / RMSE / R² bar charts
- **REST API** – Clean Flask endpoints for all features
- **Jupyter notebook** – Full EDA walkthrough with visualisations

---

## Architecture

```
Browser (HTML/CSS/JS + Chart.js)
         │  HTTP / REST
         ▼
Flask API (backend/app.py)
    ├── /api/predict       → LSTM / Linear model inference
    ├── /api/suggestions   → AI optimisation tips
    ├── /api/anomalies     → Anomaly detection
    ├── /api/history       → Historical data
    ├── /api/simulate      → Real-time step
    └── /api/model-metrics → Stored evaluation metrics
         │
    ┌────┴─────────────────────────────┐
    │         models/                  │
    │  data_loader.py  (UCI + synth)   │
    │  lstm_model.py   (TF/Keras)      │
    │  linear_model.py (scikit-learn)  │
    │  anomaly_detector.py             │
    │  metrics.py                      │
    └──────────────────────────────────┘
```

---

## Project Structure

```
smart-energy-predictor/
│
├── data/                          # Raw and processed datasets
│   ├── household_power_consumption.txt   # UCI dataset (download separately)
│   ├── processed_energy.csv              # Generated after first run
│   └── charts/                           # Saved training charts (PNG)
│
├── models/                        # ML modules
│   ├── __init__.py
│   ├── data_loader.py             # Load, clean, feature-engineer, normalise
│   ├── lstm_model.py              # Stacked LSTM (TensorFlow/Keras)
│   ├── linear_model.py            # Ridge Regression pipeline (scikit-learn)
│   ├── anomaly_detector.py        # Z-score, IQR, Isolation Forest + suggestions
│   ├── metrics.py                 # MAE, RMSE, R², MAPE, moving average
│   └── saved/                     # Serialised model artefacts (auto-created)
│       ├── lstm_model.keras
│       ├── linear_model.pkl
│       └── scaler.pkl
│
├── backend/
│   └── app.py                     # Flask REST API (7 endpoints)
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Single-page application shell
│   └── static/
│       ├── css/style.css          # Industrial-tech dark theme
│       └── js/app.js              # All interactivity + Chart.js
│
├── notebooks/
│   └── energy_analysis.ipynb     # Full EDA + training walkthrough
│
├── scripts/
│   └── train_models.py            # CLI training script
│
├── requirements.txt
├── Procfile                       # Gunicorn (Render / Heroku)
├── render.yaml                    # Render blueprint
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- (Optional) GPU for faster LSTM training

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/smart-energy-predictor.git
cd smart-energy-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Download the real UCI dataset

The app works out-of-the-box with **synthetic data**. To use the real UCI dataset:

```bash
# The data_loader will auto-download when you run with --real flag
python scripts/train_models.py --real
```

Or manually:
1. Download from: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
2. Extract `household_power_consumption.txt` into `data/`

### 3. Train the models

```bash
# Fast run with synthetic data (no TensorFlow GPU required)
python scripts/train_models.py --synthetic --epochs 20

# Skip LSTM (fastest, no TF required)
python scripts/train_models.py --no-lstm

# Full training with real data
python scripts/train_models.py --real --epochs 50
```

Training output example:
```
INFO | Training LSTM | epochs=20  batch=64  X_train=(5832, 24, 12)
Epoch 1/20 – loss: 0.0312  val_loss: 0.0298
...
INFO | Training complete – best val_loss=0.02814

────────────────────────────────────────────────────────────
  Model Performance Comparison
────────────────────────────────────────────────────────────
Model                           MAE          RMSE            R²        MAPE %
────────────────────────────────────────────────────────────
LSTM                        0.04120       0.06310       0.92140         4.312
LinearRegression            0.08970       0.11820       0.81030         9.871
────────────────────────────────────────────────────────────
```

### 4. Start the server

```bash
python backend/app.py
# → Running on http://localhost:5000
```

### 5. Open the dashboard

Navigate to **http://localhost:5000** in your browser.

---

## Running the Application

### Development

```bash
FLASK_DEBUG=1 python backend/app.py
```

### Production (Gunicorn)

```bash
gunicorn backend.app:app --workers 2 --bind 0.0.0.0:5000
```

### Jupyter Notebook (EDA)

```bash
jupyter notebook notebooks/energy_analysis.ipynb
```

---

## API Reference

All endpoints return JSON. Base URL: `http://localhost:5000`

### `GET /api/health`
Returns API status and model availability.

```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00",
  "models": {
    "lstm_loaded": true,
    "linear_loaded": true,
    "scaler_loaded": true
  }
}
```

---

### `POST /api/predict`
Predict future energy consumption.

**Request body:**
```json
{
  "model": "linear",
  "n_steps": 24,
  "past_usage": [1.2, 0.9, 1.5, 2.1]
}
```

**Response:**
```json
{
  "timestamps": ["2024-01-15T11:00", "..."],
  "predicted": [1.32, 1.28, "..."],
  "confidence_lower": [1.21, "..."],
  "confidence_upper": [1.43, "..."],
  "model_used": "linear"
}
```

---

### `POST /api/suggestions`
Get AI-based optimisation suggestions.

**Request body (optional):**
```json
{ "series": [1.2, 0.9, 1.5, 2.8, "..."] }
```

**Response:**
```json
{
  "suggestions": [
    {
      "category": "Peak Hours",
      "title": "Shift Usage Away From Peak Hours",
      "detail": "Your peak-hour usage is 32% higher...",
      "impact": "high",
      "saving_pct": 32.0
    }
  ],
  "series_stats": { "mean_kw": 1.42, "peak_kw": 3.1, "min_kw": 0.4, "std_kw": 0.38 }
}
```

---

### `POST /api/anomalies`
Detect anomalies in a consumption series.

**Request body:**
```json
{
  "series": [1.2, 0.9, 5.8, "..."],
  "method": "rolling_zscore",
  "threshold": 3.0
}
```

---

### `GET /api/history?days=7`
Returns sample historical hourly data.

---

### `GET /api/model-metrics`
Returns stored MAE, RMSE, R², MAPE for all trained models.

---

### `POST /api/simulate`
Returns the next single predicted value for real-time streaming.

**Request body:**
```json
{ "last_values": [1.2, 1.3, 1.1] }
```

---

## Model Details

### LSTM Architecture

```
Input(shape=(24, 12))          ← 24-hour window, 12 features
   └─ LSTM(128, return_sequences=True)
   └─ Dropout(0.2)
   └─ LSTM(64)
   └─ Dropout(0.2)
   └─ Dense(32, relu)
   └─ Dense(1)                 ← kW prediction
```

- **Loss function:** Huber (robust to outliers)
- **Optimiser:** Adam with ReduceLROnPlateau
- **Early stopping:** patience=8, restores best weights

### Linear Regression

- **Pipeline:** `StandardScaler → Ridge(alpha=1.0)`
- **Input:** Flattened look-back window (24 steps × 12 features = 288 features)
- **Optional:** PolynomialFeatures(degree=2) for interaction terms

### Feature Engineering

| Feature | Description |
|---|---|
| `Global_active_power` | Target variable (kW) |
| `Global_reactive_power` | Reactive power |
| `Voltage` | Supply voltage |
| `Global_intensity` | Current intensity |
| `Sub_metering_1/2/3` | Circuit-level sub-metering |
| `hour`, `day_of_week`, `month` | Temporal features |
| `is_weekend` | Binary weekend flag |
| `rolling_mean_1h/24h` | Short/long rolling average |
| `rolling_std_1h` | Local volatility |
| `lag_1h`, `lag_24h` | 1-hour and 24-hour lags |

---

## Dataset

**UCI Household Power Consumption Dataset**
- Source: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
- Period: December 2006 – November 2010 (~2 million minute-level records)
- Features: Global active power, reactive power, voltage, current, 3× sub-metering

**Preprocessing steps:**
1. Parse datetime from separate Date and Time columns
2. Replace `?` missing values with NaN
3. Forward-fill gaps ≤ 60 minutes; drop remaining NaN
4. IQR-based outlier clipping (4× IQR fence)
5. Resample from 1-minute to hourly means
6. Add temporal + rolling + lag features
7. MinMaxScaler normalisation (fit on train only)

---

## Deployment

### Render (Recommended)

1. Push to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn backend.app:app --workers 2 --bind 0.0.0.0:$PORT`
6. Or use: `render blueprint apply` with the included `render.yaml`

### Railway

```bash
railway login
railway init
railway up
```

### Vercel (Frontend only)

Deploy the `frontend/` folder as a static site and point `API_BASE` in `app.js` to your Render backend URL.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:5000"]
```

---

## Screenshots

| Dashboard | Prediction Engine |
|---|---|
| KPI tiles, historical chart | Forecast chart with confidence bands |

| Anomaly Detection | Model Metrics |
|---|---|
| Red spikes on time-series | MAE/RMSE bar chart comparison |

---

## Academic Notes

### Why LSTM for energy forecasting?

Energy consumption is a **non-stationary time series** with strong daily and weekly seasonality. LSTMs capture long-range temporal dependencies via their cell state mechanism, making them superior to linear models when patterns are non-linear.

### Why Ridge Regression as baseline?

Ridge (L2-regularised linear regression) is a standard econometric baseline. Its performance provides a minimum bar: if the LSTM doesn't significantly outperform it, the additional complexity isn't justified.

### Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(|y - ŷ|) | Average absolute error in original units |
| RMSE | √mean((y - ŷ)²) | Penalises large errors more heavily |
| R² | 1 - SS_res/SS_tot | Fraction of variance explained (1.0 = perfect) |
| MAPE | mean(|y - ŷ|/y) × 100 | Percentage error (scale-independent) |

### Anomaly Detection Methods

| Method | Strength | Weakness |
|---|---|---|
| Z-Score | Simple, fast | Assumes normality |
| IQR Fence | Robust to outliers | Fixed threshold |
| Rolling Z-Score | Adaptive to local patterns | Requires window tuning |
| Isolation Forest | Multivariate, no distribution assumption | Black-box |

---

## License

MIT © 2024 — Free for academic and personal use.

---

*Built with Flask · TensorFlow · scikit-learn · Chart.js · UCI Energy Dataset*
