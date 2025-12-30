#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Automate local deployment of the AutoGPT stack using docker compose.

Options:
  --no-build           Skip rebuilding images (compose will use existing ones)
  --run-smoke-tests    After services are healthy, run the smoke test profile
  -h, --help           Show this help message and exit
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/deploy/docker-compose.dev.yml"
ENV_FILE="${REPO_ROOT}/deploy/env/.env.dev"
SHOULD_BUILD=true
RUN_TESTS=false
PRINT_HELP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      SHOULD_BUILD=false
      shift
      ;;
    --run-smoke-tests)
      RUN_TESTS=true
      shift
      ;;
    -h|--help)
      PRINT_HELP=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      PRINT_HELP=true
      shift
      ;;
  esac
done

if [[ "$PRINT_HELP" == true ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found in PATH." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Docker Compose v2 (docker compose) or v1 (docker-compose) is required." >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Cannot find compose file at $COMPOSE_FILE" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file missing at $ENV_FILE. Copy or create it before continuing." >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Launching development stack via ${COMPOSE_CMD[*]}..."
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [[ "$SHOULD_BUILD" == true ]]; then
  COMPOSE_ARGS+=(--build)
fi

"${COMPOSE_CMD[@]}" "${COMPOSE_ARGS[@]}" up -d

echo "Services are starting. View status with: ${COMPOSE_CMD[*]} -f $COMPOSE_FILE ps"

if [[ "$RUN_TESTS" == true ]]; then
  echo "Running smoke tests profile..."
  "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" --profile tests up --build smoke-tests
fi
