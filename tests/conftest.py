"""Test bootstrap: set env, ensure a model exists, put src/ on the path.

Runs before any test imports `app`, so the module-level model load and API_KEY
read succeed even on a fresh CI checkout (where .env and the .pkl are absent).
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dummy env (setdefault -> never clobbers a real value). DB is mocked in tests.
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5433")
os.environ.setdefault("DB_USER", "iris")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "irisdb")

# The .pkl is git-ignored; train one if it's missing so `import app` can load it.
if not (ROOT / "models" / "latest.json").exists():
    subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)

sys.path.insert(0, str(ROOT / "src"))
