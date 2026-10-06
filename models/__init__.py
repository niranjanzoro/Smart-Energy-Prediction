"""
models/
───────
ML package for the Smart Energy Consumption Predictor.

data_loader      – load / clean / feature-engineer / split / scale
linear_model     – Ridge regression (time-series-CV tuned)
lstm_model       – stacked LSTM (TensorFlow/Keras, imported lazily)
forecaster       – recursive multi-step forecasting + horizon back-test
anomaly_detector – statistical / Isolation-Forest anomaly detection + tips
metrics          – MAE, RMSE, R², MAPE and a comparison table

TensorFlow is NOT imported here, so the API can start without it.
"""

from .data_loader import get_processed_data                                   # noqa: F401
from .metrics import evaluate_predictions, compare_models, moving_average     # noqa: F401
from .anomaly_detector import detect_anomalies, generate_suggestions          # noqa: F401
