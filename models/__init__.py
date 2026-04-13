"""
models/
───────
AI/ML module for the Smart Energy Consumption Predictor.

Exports
───────
train_all        – train both LSTM and Linear models in one call
predict_future   – generate N-step ahead forecasts
compare_models   – print performance comparison table
detect_anomalies – flag unusual consumption readings
generate_suggestions – return optimisation tips
"""

from .data_loader import get_processed_data          # noqa: F401
from .metrics import evaluate_predictions, compare_models, moving_average  # noqa: F401
from .anomaly_detector import detect_anomalies, generate_suggestions  # noqa: F401
