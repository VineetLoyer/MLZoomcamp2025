# MBTI Personality Prediction Service
# Dockerfile for containerizing the Flask prediction service
#
# Requirements: 7.1, 7.2

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9696 \
    MODELS_DIR=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for preprocessing
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy source code
COPY src/ ./src/
COPY predict.py .

# Copy trained model artifacts
COPY models/ ./models/

# Expose the service port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')" || exit 1

# Run the prediction service with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "--timeout", "120", "predict:app"]
