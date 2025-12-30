#!/usr/bin/env bash
# Lightweight Docker watchdog (cron-friendly).
#
# Usage:
#   ./docker_watchdog.sh <container>
#
# Example crontab entry (check every 5 minutes):
#   */5 * * * * /path/to/docker_watchdog.sh my_container >> /var/log/docker_watchdog.log 2>&1
set -euo pipefail

container="${1:-}"
if [[ -z "${container}" ]]; then
  echo "Usage: docker_watchdog.sh <container>" >&2
  exit 2
fi

running="$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)"
if [[ "${running}" != "true" ]]; then
  echo "Container '${container}' not running; restarting..."
  docker restart "${container}"
else
  echo "Container '${container}' is running."
fi

