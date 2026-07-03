"""Prediction logging for the Iris model.

A thin psycopg2 wrapper over the `predictions` table (see db/init/01_predictions.sql).
Connection settings come from the environment (.env), so the SAME code works whether
it runs on the host (localhost:5433) or inside the compose network (postgres:5432).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()  # load .env when present; harmless if the vars are already in the env

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5433"),
    "user": os.getenv("DB_USER", "iris"),
    "password": os.getenv("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "irisdb"),
}

# Canonical feature order — matches the columns in the predictions table.
FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


@contextmanager
def get_connection():
    """Yield a connection; commit on success, roll back on error, always close."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_prediction(
    features,
    predicted_class,
    *,
    model_version,
    confidence=None,
    latency_ms=None,
    request_id=None,
    actual_class=None,
):
    """Insert one prediction and return its new id.

    features: a dict keyed by FEATURE_COLUMNS, or a 4-length sequence in that order.
    """
    if isinstance(features, dict):
        values = [features[c] for c in FEATURE_COLUMNS]
    else:
        values = list(features)
        if len(values) != 4:
            raise ValueError(f"expected 4 features, got {len(values)}")

    sql = """
        INSERT INTO predictions (
            model_version,
            sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm,
            predicted_class, confidence, actual_class, latency_ms, request_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            (
                model_version,
                *values,
                predicted_class,
                confidence,
                actual_class,
                latency_ms,
                request_id,
            ),
        )
        return cur.fetchone()[0]


def fetch_recent(limit=5):
    """Return the most recent prediction rows (for monitoring / sanity checks)."""
    with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, created_at, model_version, predicted_class, confidence, latency_ms "
            "FROM predictions ORDER BY id DESC LIMIT %s;",
            (limit,),
        )
        return cur.fetchall()


if __name__ == "__main__":
    # Self-test: log one dummy prediction and read it back.
    new_id = log_prediction(
        {
            "sepal_length_cm": 5.1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
        },
        "setosa",
        model_version="db-selftest",
        confidence=0.97,
        latency_ms=3.1,
        request_id="db-selftest-1",
    )
    print(f"inserted id={new_id}")
    for row in fetch_recent(3):
        print(dict(row))
