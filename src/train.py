"""Train the Iris classifier and version the artifact.

- Loads Iris from scikit-learn (clean, no external file to manage).
- Trains a StandardScaler + LogisticRegression pipeline (scaling matters for logreg).
- Evaluates on a stratified held-out test split.
- Saves a versioned .pkl to models/ and writes models/latest.json, the pointer the
  serving API will read: version + artifact path + metrics + feature/class schema.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_NAMES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]


def next_version(models_dir: Path) -> str:
    """Auto-increment vN based on existing artifacts (v1, v2, ...)."""
    versions = [
        int(m.group(1))
        for p in models_dir.glob("iris_logreg_v*.pkl")
        if (m := re.search(r"_v(\d+)\.pkl$", p.name))
    ]
    return f"v{max(versions, default=0) + 1}"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_iris()
    X, y = data.data, data.target
    class_names = list(data.target_names)  # ['setosa', 'versicolor', 'virginica']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    version = next_version(MODELS_DIR)
    artifact = MODELS_DIR / f"iris_logreg_{version}.pkl"
    joblib.dump(model, artifact)

    metadata = {
        "version": version,
        "artifact": artifact.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "StandardScaler+LogisticRegression",
        "test_accuracy": round(float(acc), 4),
        "feature_names": FEATURE_NAMES,
        "class_names": class_names,
        "sklearn_version": sklearn.__version__,
    }
    (MODELS_DIR / "latest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\nSaved {artifact.name}  ->  models/latest.json now points to {version}")


if __name__ == "__main__":
    main()
