"""Helper utilities for encoding structural genomes."""

from __future__ import annotations

from typing import Dict, List


def encode_structure(topology: Dict[str, List[str]], gates: Dict[str, float]) -> Dict[str, float]:
    """Encode a module topology and gate map into architecture-friendly fields."""

    encoded: Dict[str, float] = {}
    for name, active in gates.items():
        encoded[f"module_{name}_active"] = 1.0 if float(active) >= 0.5 else 0.0
    for src, dsts in topology.items():
        for dst in dsts:
            encoded[f"edge_{src}->{dst}"] = 1.0
    active_count = sum(1 for value in gates.values() if value >= 0.5)
    connection_count = sum(len(dsts) for dsts in topology.values())
    encoded["active_modules"] = float(active_count)
    encoded["connection_count"] = float(connection_count)
    return encoded
