# syntax=docker/dockerfile:1

# ---- build stage: compilers live here only -------------------------------
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first. The default PyPI wheel bundles CUDA libraries (several
# GB) that this image never uses; pass a CUDA index URL to build for a GPU.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.2.1 --index-url "$TORCH_INDEX_URL"

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# spaCy model matching spacy==3.7.x (3.8 models require spaCy 3.8).
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ---- runtime stage ---------------------------------------------------------
FROM python:3.11-slim

# Match APP_UID to the owner of the bind-mounted data/HF cache dirs on Linux
# hosts (docker build --build-arg APP_UID=$(id -u) ...).
ARG APP_UID=1000

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid "$APP_UID" app

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

WORKDIR /app
COPY app /app
RUN mkdir -p /data /hf_cache && chown app:app /data /hf_cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
