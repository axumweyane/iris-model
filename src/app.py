"""Flask API serving the Iris model.

Endpoints:
  GET  /         -> service info
  GET  /health   -> liveness + readiness (model loaded, DB reachable)
  POST /predict  -> classify one flower, log the prediction, return class + confidence

Auth: POST /predict requires header  X-API-Key: <API_KEY>.
The model is loaded ONCE at startup from models/latest.json.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request

import database
import schema

load_dotenv()

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
API_KEY = os.getenv("API_KEY", "")

app = Flask(__name__)


def load_model():
    meta = json.loads((MODELS_DIR / "latest.json").read_text())
    model = joblib.load(MODELS_DIR / meta["artifact"])
    return model, meta


MODEL, META = load_model()


@app.get("/")
def index():
    return jsonify({
        "service": "iris-model",
        "model_version": META["version"],
        "endpoints": {
            "GET /health": "status",
            "POST /predict": "classify one flower (requires X-API-Key)",
        },
    })


@app.get("/health")
def health():
    db_ok, db_error = True, None
    try:
        database.fetch_recent(1)
    except Exception as e:
        db_ok, db_error = False, str(e)
    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "model_version": META["version"],
        "model_loaded": MODEL is not None,
        "db_ok": db_ok,
        "db_error": db_error,
    }), (200 if db_ok else 503)


@app.post("/predict")
def predict():
    # --- auth (fail closed) ---
    if not API_KEY or request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    # --- parse + validate ---
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "invalid or empty JSON body"}), 400
    try:
        features = schema.validate_features(payload)
    except schema.ValidationError as e:
        return jsonify({"error": str(e)}), 400

    # --- predict (timed) ---
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    t0 = time.perf_counter()
    proba = MODEL.predict_proba([features])[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    idx = int(proba.argmax())
    predicted_class = META["class_names"][idx]
    confidence = float(proba[idx])

    # --- log (a DB blip must NOT fail the prediction) ---
    logged_id, log_error = None, None
    try:
        logged_id = database.log_prediction(
            dict(zip(schema.FEATURE_NAMES, features)),
            predicted_class,
            model_version=META["version"],
            confidence=confidence,
            latency_ms=round(latency_ms, 3),
            request_id=request_id,
        )
    except Exception as e:
        log_error = str(e)

    resp = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "model_version": META["version"],
        "request_id": request_id,
        "latency_ms": round(latency_ms, 3),
        "logged_id": logged_id,
    }
    if log_error:
        resp["log_error"] = log_error
    return jsonify(resp), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
