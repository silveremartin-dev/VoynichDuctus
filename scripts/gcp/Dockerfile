# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

LABEL maintainer="VoynichDuctus Research Team"
LABEL description="High-throughput digital paleography & stroke vectorization pipeline for Voynich MS and Codex Seraphinianus on GCP"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/app/cache/torch

# Install system dependencies for scientific image processing & PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    curl \
    git \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY scripts/gcp/requirements-gcp.txt /app/scripts/gcp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/scripts/gcp/requirements-gcp.txt

# Copy source tree, scripts, and package metadata
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/
RUN pip install -e .

# Copy data assets
COPY data/annotations/ /app/data/annotations/
COPY data/scans/seraphinianus/ /app/data/scans/seraphinianus/
COPY data/scans/voynich/voynichese_scans/ /app/data/scans/voynich/voynichese_scans/
COPY data/scans/voynich/yale/ /app/data/scans/voynich/yale/
COPY intermediate/test_cache/registration/ /app/intermediate/test_cache/registration/

# Create runtime scratch and output directories
RUN mkdir -p /app/output /app/cache /tmp/matplotlib && chmod -R 777 /app/output /app/cache /tmp/matplotlib

# Default entrypoint for Cloud Run Jobs / Vertex AI / Compute Engine
ENTRYPOINT ["python", "scripts/gcp/cloud_batch_runner.py"]
CMD ["--help"]
