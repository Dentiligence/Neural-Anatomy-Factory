# ============================================================
# Dentiligence — Neural Anatomy Engine
# Production Docker Image
# ============================================================
# Multi-stage build:
#   Stage 1 (builder) : Install Python deps into a venv
#   Stage 2 (runtime) : Lean CUDA runtime image
#
# Base: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# Adjust CUDA version to match your GPU driver.
# For CPU-only: replace base with python:3.11-slim
# ============================================================

# ------ Stage 1: dependency builder -------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for numpy / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ------ Stage 2: runtime image --------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Dentiligence Core Team <engineering@dentiligence.ai>"
LABEL version="0.1.0"
LABEL description="Neural Anatomy Generation Service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_WEIGHTS_PATH=/app/weights/model.pt \
    PORT=8000

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY core/     ./core/
COPY api/      ./api/

# (Optional) pre-trained model weights — mount at runtime instead
# COPY weights/  ./weights/

# Create non-root user
RUN useradd -m -u 1000 denti && chown -R denti:denti /app
USER denti

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
