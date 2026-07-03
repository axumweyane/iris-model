-- Prediction log for the Iris model.
-- This is the substrate every monitoring layer reads from: latency (APM),
-- input drift, prediction-rate shift, and delayed accuracy once labels arrive.
-- Runs automatically on FIRST container init via docker-entrypoint-initdb.d.

CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL   PRIMARY KEY,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version      TEXT        NOT NULL,

    -- input features (the 4 iris measurements, in cm)
    sepal_length_cm    REAL        NOT NULL,
    sepal_width_cm     REAL        NOT NULL,
    petal_length_cm    REAL        NOT NULL,
    petal_width_cm     REAL        NOT NULL,

    -- model output
    predicted_class    TEXT        NOT NULL,
    confidence         REAL        CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

    -- delayed ground truth (backfilled later -> enables accuracy-over-time)
    actual_class       TEXT,

    -- request context (latency monitoring + tracing)
    latency_ms         REAL,
    request_id         TEXT
);

COMMENT ON TABLE predictions IS 'Every model prediction logged with input, output, context, and (later) ground truth.';

-- Indexes for the monitoring queries you will actually run:
CREATE INDEX IF NOT EXISTS idx_predictions_created_at    ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions (model_version);
