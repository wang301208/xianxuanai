"""I/O utilities for edge devices."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class EdgeIO:
    """Simple interface for edge device input and output."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def read_sensor(self, name: str) -> Any:
        """Read sensor data from a text file.

        This function mocks sensor input by reading from ``<name>.txt`` if it exists.
        """
        sensor_file = self.data_dir / f"{name}.txt"
        if sensor_file.exists():
            return sensor_file.read_text().strip()
        return "no-data"

    def write_output(self, name: str, data: str) -> None:
        """Write output data produced by the agent."""
        out_file = self.data_dir / f"{name}.out"
        out_file.write_text(data)
