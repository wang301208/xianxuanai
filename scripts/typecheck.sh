#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer Poetry-managed environment if available, fall back to system mypy
poetry run mypy . 2>/dev/null || mypy .
