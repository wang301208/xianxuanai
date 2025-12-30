#!/bin/bash
set -e

pytest tests/test_algorithms.py "$@"
python scripts/benchmark_sorting.py
