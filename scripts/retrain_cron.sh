#!/bin/bash
# Run the retraining pipeline. Suitable for cron usage.
# Example crontab entry (runs daily at midnight):
# 0 0 * * * /path/to/retrain_cron.sh >> /path/to/retrain.log 2>&1
set -e
cd "$(dirname "$0")"
python -m ml.retraining_pipeline "$@"
