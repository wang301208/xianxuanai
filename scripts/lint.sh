#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer Poetry-managed environment if available, fall back to system ruff
poetry run ruff check . 2>/dev/null || ruff check .
