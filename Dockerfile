# Production image for the Iris API.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install prod deps only (binary wheels; no build toolchain needed).
COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

COPY src/ src/

# Bake a trained model into the image so it's self-contained and runnable.
RUN python src/train.py

EXPOSE 8000
# Serve with gunicorn (2 workers). app:app is found via PYTHONPATH=/app/src.
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "--timeout", "60", "app:app"]
