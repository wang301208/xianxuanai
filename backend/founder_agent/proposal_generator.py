"""Utilities for writing blueprint proposals."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

PROPOSAL_DIR = Path(__file__).resolve().parent.parent / "org_charter" / "proposals"


def generate_proposal(trends: Dict[str, float]) -> Path:
    """Persist a proposal based on *trends* and return its path."""
    proposal = {
        "timestamp": datetime.utcnow().isoformat(),
        "trends": trends,
        "summary": "Proposed blueprint adjustments based on metric trends.",
    }
    PROPOSAL_DIR.mkdir(parents=True, exist_ok=True)
    path = PROPOSAL_DIR / f"proposal_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(proposal, f, sort_keys=False)
    return path
