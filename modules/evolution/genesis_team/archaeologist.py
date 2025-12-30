"""Archaeologist agent explores repository history."""

from __future__ import annotations

import subprocess

from .. import Agent


class Archaeologist(Agent):
    """Explores repository history to uncover past decisions."""

    def perform(self, max_entries: int = 5) -> str:
        try:
            log = subprocess.check_output(
                ["git", "log", "--pretty=format:%h %s", "-n", str(max_entries)],
                text=True,
            )
        except Exception:
            log = "Repository history unavailable."
        return log
