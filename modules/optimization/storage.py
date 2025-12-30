from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator

# Default location for the history database
DEFAULT_HISTORY_FILE = Path(__file__).with_name("history.csv")


def append_history(
    algorithm: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> None:
    """Append a single run to the persistent history file.

    The history is stored as CSV with JSON-encoded parameter and metric
    dictionaries. Each row contains the algorithm name, parameters used and the
    resulting metrics.
    """
    fieldnames = ["algorithm", "params", "metrics"]
    exists = history_file.exists()
    with history_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "algorithm": algorithm,
                "params": json.dumps(params),
                "metrics": json.dumps(metrics),
            }
        )


def load_history(history_file: Path = DEFAULT_HISTORY_FILE) -> Iterator[Dict[str, Any]]:
    """Load all historic runs from the given file."""
    if not history_file.exists():
        return iter([])

    with history_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "algorithm": row["algorithm"],
                "params": json.loads(row["params"]),
                "metrics": json.loads(row["metrics"]),
            }
            for row in reader
        ]
    return iter(rows)
