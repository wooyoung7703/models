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

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend /app/backend
# Include root-level scripts (analysis & meta retrain utilities)
COPY scripts /app/scripts
# Include root-level data (evaluation samples & meta JSON if present)
COPY data /app/data
# Duplicate BVF meta JSON into backend/data for env overrides pointing there
# Note: evaluation sample CSV is not required at runtime, so we no longer copy it.
RUN cp /app/data/bottom_vs_forecast_meta.json /app/backend/data/ 2>/dev/null || true

# Create data directory
RUN mkdir -p /app/backend/data

EXPOSE 8022

# Environment defaults (override via compose/env)
ENV EXCHANGE_TYPE=futures \
    ENABLE_BASE_MODELS=1 \
    ENABLE_STACKING=1 \
    GAP_FILL_ENABLED=1 \
    PREDICT_ENABLED=1

# Run standalone WebSocket server (no FastAPI)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -m backend.app.healthcheck || exit 1

CMD ["python", "-m", "backend.app.ws_server"]
