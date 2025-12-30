#!/bin/bash
# Restore previous model artifacts
set -e
cd "$(dirname "$0")/.."
PREV="artifacts/previous"
CUR="artifacts/current"
if [ ! -d "$PREV" ]; then
  echo "No previous artifacts to restore" >&2
  exit 1
fi
rm -rf "$CUR"
cp -r "$PREV" "$CUR"
echo "Rolled back to previous artifacts"
