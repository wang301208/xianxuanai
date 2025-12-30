#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest is required but not installed. Install dependencies first." >&2
  exit 1
fi

pytest_targets=("$@")
if [ ${#pytest_targets[@]} -eq 0 ]; then
  pytest_targets=(tests BrainSimulationSystem/tests)
fi

pytest \
  --cov-config=.coveragerc \
  --cov=BrainSimulationSystem \
  --cov=backend \
  --cov=algorithms \
  --cov=capability \
  --cov=modules \
  --cov=autogpts \
  --cov=scripts \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-report=html:coverage_html \
  "${pytest_targets[@]}"
