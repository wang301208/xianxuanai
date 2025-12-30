#!/usr/bin/env bash
# Restart a Docker container (simple wrapper for automation).
#
# Usage:
#   ./restart_container.sh <container> [timeout_seconds]
#
# Examples:
#   ./restart_container.sh my_container
#   ./restart_container.sh my_container 30
#
# Cron example (every 5 minutes):
#   */5 * * * * /path/to/restart_container.sh my_container >> /var/log/docker_restart.log 2>&1
set -euo pipefail

container="${1:-}"
timeout="${2:-10}"

if [[ -z "${container}" ]]; then
  echo "Usage: restart_container.sh <container> [timeout_seconds]" >&2
  exit 2
fi

docker restart -t "${timeout}" "${container}"

