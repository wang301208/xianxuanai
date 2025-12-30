"""Utilities for running AutoGPT in unattended mode."""
from __future__ import annotations

from typing import Any

from autogpt.app.main import run_auto_gpt


def run_auto_loop(**kwargs: Any) -> None:
    """Run AutoGPT continuously without requiring user interaction.

    All keyword arguments are forwarded to :func:`run_auto_gpt`. The function
    ensures the agent runs in continuous mode with the initial confirmation
    prompt skipped. It keeps restarting the agent after it finishes until the
    process is interrupted.
    """
    kwargs.setdefault("continuous", True)
    kwargs.setdefault("skip_reprompt", True)

    try:
        while True:
            run_auto_gpt(**kwargs)
    except KeyboardInterrupt:
        # Allow graceful shutdown when user interrupts the process.
        return
