# Multi-target Dockerfile.
#
# Build targets:
#   docker build --target cpu  -t indiclm:cpu  .   (default; no CUDA needed)
#   docker build --target gpu  -t indiclm:gpu  .   (CUDA 11.8, V100-safe fp16)
#
# The GPU target matches the WSAI cluster: torch 2.7.1+cu118, cuda 11.8.
# Use --build-arg BASE_IMAGE=... to override for other CUDA versions.

ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8

# ── CPU base (CI, local dev, CPU-only inference) ─────────────────────────────
FROM python:3.11-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY . .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

ENTRYPOINT ["indiclm"]
CMD ["doctor"]

# ── GPU base (WSAI cluster, V100 fp16 inference) ─────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu22.04 AS gpu-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch cu118 first (keeps it pinned, avoids CPU fallback)
RUN pip install --no-cache-dir \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY . .

RUN pip install --no-cache-dir -e ".[gpu,monitoring]"

# ── GPU inference image ───────────────────────────────────────────────────────
FROM gpu-base AS gpu

ENV INDICLM_DEVICE=cuda \
    INDICLM_PRECISION=fp16 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

ENTRYPOINT ["indiclm"]
CMD ["doctor"]
