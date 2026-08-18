# CPU-safe by default. Do not assume CUDA availability; a GPU-enabled
# base image is a documented alternative (see docs/training.md), not a
# requirement for this image to build or run.
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

COPY . .

# Default: run the doctor check so `docker run indiclm` gives an
# immediate, honest diagnosis of the container's environment.
ENTRYPOINT ["indiclm"]
CMD ["doctor"]
