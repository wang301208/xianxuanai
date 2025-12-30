# AutoGPT agent container image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    AGENT_WORKSPACE="/app/data/workspace" \
    DATABASE_STRING="sqlite:////app/data/forge.db" \
    PORT=8000

# Install system dependencies required for runtime packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Poetry for dependency management
RUN pip install --no-cache-dir "poetry==1.7.1"

WORKDIR /app

# Only copy dependency metadata first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./
RUN poetry install --without dev --no-interaction --no-ansi --no-root

# Copy application source
COPY . .

ENV PYTHONPATH=/app

# Create an unprivileged user for running the app
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/workspace \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 CMD python - <<'PY'
import os
import sys
import urllib.request

port = os.getenv("PORT", "8000")
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=5)
except Exception as exc:  # pragma: no cover - runtime healthcheck
    sys.exit(f"healthcheck failed: {exc}")
PY

CMD ["sh", "-c", "uvicorn backend.forge.forge.app:app --host 0.0.0.0 --port ${PORT}"]
