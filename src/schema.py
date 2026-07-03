"""Request validation for the Iris prediction API.

Kept separate from app.py so the contract (feature names, bounds) lives in one place
and can be unit-tested without spinning up Flask.
"""
from __future__ import annotations

# The 4 features the model expects, in order. Mirrors database.FEATURE_COLUMNS.
FEATURE_NAMES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]

# Loose sanity bounds (cm). Not security — just reject obviously-bad input with a clear message.
FEATURE_BOUNDS = (0.0, 30.0)


class ValidationError(ValueError):
    """Raised when a /predict payload is malformed."""


def validate_features(payload) -> list[float]:
    """Validate a request body and return the 4 features as floats in canonical order."""
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")

    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        raise ValidationError(f"missing feature(s): {', '.join(missing)}")

    lo, hi = FEATURE_BOUNDS
    values = []
    for f in FEATURE_NAMES:
        v = payload[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"feature '{f}' must be a number, got {type(v).__name__}")
        v = float(v)
        if not (lo <= v <= hi):
            raise ValidationError(f"feature '{f}'={v} out of range [{lo}, {hi}]")
        values.append(v)
    return values
