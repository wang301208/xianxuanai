#!/usr/bin/env bash
# Pull the latest image(s) and recreate a docker compose service (automation helper).
#
# Usage:
#   ./compose_refresh_service.sh <project_dir> <compose_file> <service>
#
# Examples:
#   ./compose_refresh_service.sh ./deploy docker-compose.yml web
#   ./compose_refresh_service.sh . docker-compose.yml api
#
# Notes:
# - `docker pull` alone does not update a running container; compose must recreate it.
set -euo pipefail

project_dir="${1:-}"
compose_file="${2:-docker-compose.yml}"
service="${3:-}"

if [[ -z "${project_dir}" || -z "${service}" ]]; then
  echo "Usage: compose_refresh_service.sh <project_dir> <compose_file> <service>" >&2
  exit 2
fi

cd "${project_dir}"

docker compose -f "${compose_file}" pull "${service}"
docker compose -f "${compose_file}" up -d --no-deps --force-recreate --pull always "${service}"

