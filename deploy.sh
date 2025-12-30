#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE="forge"
PORT="8000"
NO_START=false

SETUP_ARGS=()

usage() {
  cat <<'EOF'
One-click setup + start.

Usage:
  ./deploy.sh [options]

Options:
  --service <name>        Service to start (default: forge)
  --port <port>           Service port (default: 8000)
  --no-start              Only run setup; do not start any service

Setup options (passed through to scripts/setup.sh):
  --check-only
  --skip-system-deps
  --skip-deps
  --skip-assets
  --fetch-assets
  --assets-manifest <path>
  --yes
  --min-python <version>
  --python-version <version>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)
      [[ $# -ge 2 ]] || { echo "error: --service requires a value" >&2; usage; exit 1; }
      SERVICE="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "error: --port requires a value" >&2; usage; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --no-start)
      NO_START=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # Everything else is forwarded to scripts/setup.sh
      SETUP_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "==> Running setup..."
bash "$REPO_ROOT/scripts/setup.sh" "${SETUP_ARGS[@]}"

if [[ "$NO_START" == true ]]; then
  echo "==> Setup complete (--no-start)."
  exit 0
fi

if [[ "${SETUP_ARGS[*]-}" == *"--check-only"* ]]; then
  echo "==> Setup ran in --check-only mode; skipping start."
  exit 0
fi

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  if [[ -f "$REPO_ROOT/config/.env.template" ]]; then
    cp "$REPO_ROOT/config/.env.template" "$REPO_ROOT/.env"
    echo "==> Created .env from config/.env.template. Please edit .env and add your API keys."
  else
    echo "==> Warning: config/.env.template not found; create .env manually if needed."
  fi
fi

case "$SERVICE" in
  forge)
    export PATH="$HOME/.local/bin:$HOME/.poetry/bin:$PATH"
    if ! command -v poetry >/dev/null 2>&1; then
      echo "error: poetry not found on PATH; run scripts/setup.sh first or add poetry to PATH." >&2
      exit 1
    fi
    echo "==> Starting Forge server on http://localhost:${PORT} (Ctrl+C to stop)"
    (cd "$REPO_ROOT" && PORT="$PORT" exec poetry run python -m forge)
    ;;
  *)
    echo "error: unknown service '$SERVICE' (supported: forge)" >&2
    exit 1
    ;;
esac

