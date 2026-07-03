"""Prometheus metrics for the Iris API.

app.py imports the record_* helpers and calls them; /metrics serves metrics_payload().
The API runs gunicorn with ONE worker + threads, so a single registry gives
consistent counts (multi-worker would need prometheus_client multiprocess mode).
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

PREDICTIONS_TOTAL = Counter(
    "iris_predictions_total",
    "Total predictions served, by class and model version.",
    ["predicted_class", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Model inference latency in seconds.",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

PREDICTION_CONFIDENCE = Histogram(
    "iris_prediction_confidence",
    "Confidence of the predicted class.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

REQUESTS_TOTAL = Counter(
    "iris_requests_total",
    "HTTP requests, by endpoint/method/status.",
    ["endpoint", "method", "status"],
)

ERRORS_TOTAL = Counter(
    "iris_errors_total",
    "Errors by type (auth, validation, log).",
    ["type"],
)


def record_prediction(predicted_class, model_version, latency_seconds, confidence):
    PREDICTIONS_TOTAL.labels(predicted_class, model_version).inc()
    PREDICTION_LATENCY.observe(latency_seconds)
    PREDICTION_CONFIDENCE.observe(confidence)


def record_request(endpoint, method, status):
    REQUESTS_TOTAL.labels(endpoint or "unknown", method, str(status)).inc()


def record_error(kind):
    ERRORS_TOTAL.labels(kind).inc()


def metrics_payload():
    """Return (body_bytes, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
