FROM python:3.11-slim AS base

# System deps for scientific libs (xgboost/lightgbm use OpenMP)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r backend/requirements.txt

# Copy application code
COPY backend/ ./backend/

# Expose API port
EXPOSE 8000

# Default environment (override in compose as needed)
ENV LOG_LEVEL=INFO

# Start FastAPI via uvicorn
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Backend Dockerfile
FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps needed by scientific libs (xgboost, numpy) and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first (better layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend app code
COPY backend /app/backend

# Create data directory
RUN mkdir -p /app/backend/data

EXPOSE 8022

# Environment defaults (override via compose/env)
ENV EXCHANGE_TYPE=futures \
    ENABLE_BASE_MODELS=1 \
    ENABLE_STACKING=1 \
    GAP_FILL_ENABLED=1 \
    PREDICT_ENABLED=1

# Run uvicorn
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8022"]
