FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9696 \
    MODEL_PATH=/app/models/model.pkl

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    flask>=3.0.0 \
    gunicorn>=21.0.0

COPY src/ ./src/
COPY predict.py .
COPY models/model.pkl ./models/

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; import urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 9696)}/health')" || exit 1

CMD gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 1 --timeout 120 predict:app
