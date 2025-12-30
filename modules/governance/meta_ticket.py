"""Utilities to create meta-tickets for meta-skill changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

DEFAULT_DIR = Path("governance/meta_tickets")


def _slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text)


def create_meta_ticket(
    title: str, description: str, ticket_dir: Path | None = None
) -> Path:
    """Create a meta-ticket tracking a proposed meta-skill change.

    The ticket is saved as a JSON file containing the title, description and
    current status. No approval tags are recorded because meta-skill changes
    are activated automatically. A simple notification is printed to stdout so
    the message can be captured by existing communication channels.
    """
    directory = ticket_dir or DEFAULT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ticket: Dict[str, object] = {
        "title": title,
        "description": description,
        "status": "pending",
    }
    path = directory / f"{_slugify(title)}.json"
    path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")
    print(f"Meta-ticket created: {path} (auto-activation enabled)")
    return path
