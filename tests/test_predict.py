"""Tests for the Iris API: schema validation (pure) + endpoints (DB mocked)."""
import pytest

import schema


VALID = {"sepal_length_cm": 5.1, "sepal_width_cm": 3.5,
         "petal_length_cm": 1.4, "petal_width_cm": 0.2}


# ---------------- schema unit tests (no model, no DB) ----------------
class TestValidateFeatures:
    def test_valid_returns_ordered_floats(self):
        assert schema.validate_features(dict(VALID)) == [5.1, 3.5, 1.4, 0.2]

    def test_missing_feature_raises(self):
        p = dict(VALID); del p["petal_width_cm"]
        with pytest.raises(schema.ValidationError):
            schema.validate_features(p)

    def test_wrong_type_raises(self):
        p = dict(VALID); p["sepal_length_cm"] = "big"
        with pytest.raises(schema.ValidationError):
            schema.validate_features(p)

    def test_bool_rejected(self):
        p = dict(VALID); p["sepal_width_cm"] = True
        with pytest.raises(schema.ValidationError):
            schema.validate_features(p)

    def test_out_of_range_raises(self):
        p = dict(VALID); p["sepal_length_cm"] = 999.0
        with pytest.raises(schema.ValidationError):
            schema.validate_features(p)

    def test_non_dict_raises(self):
        with pytest.raises(schema.ValidationError):
            schema.validate_features([1, 2, 3, 4])


# ---------------- API tests (Flask test client, DB mocked) ----------------
@pytest.fixture
def app_client(monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module.database, "log_prediction", lambda *a, **k: 123)
    monkeypatch.setattr(app_module.database, "fetch_recent", lambda *a, **k: [])
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client(), app_module


def test_health_ok(app_client):
    client, _ = app_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_requires_api_key(app_client):
    client, _ = app_client
    r = client.post("/predict", json=VALID)
    assert r.status_code == 401


def test_predict_rejects_bad_input(app_client):
    client, mod = app_client
    r = client.post("/predict", json={"sepal_length_cm": 5.1},
                    headers={"X-API-Key": mod.API_KEY})
    assert r.status_code == 400


def test_predict_setosa_ok(app_client):
    client, mod = app_client
    r = client.post("/predict", json=VALID, headers={"X-API-Key": mod.API_KEY})
    assert r.status_code == 200
    body = r.get_json()
    assert body["predicted_class"] == "setosa"
    assert body["logged_id"] == 123           # from the mock
    assert 0.0 <= body["confidence"] <= 1.0
