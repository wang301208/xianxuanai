"""Diagnostic utilities for error logging with context and stack traces."""
from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

from .auto_fixer import AutoFixFailed, AutoFixer, FixPlan, execute_with_autofix

__all__ = ["record_error", "AutoFixFailed", "AutoFixer", "FixPlan", "execute_with_autofix"]
logger = logging.getLogger("diagnostics")


def record_error(error: Exception, context: Dict[str, Any] | None = None) -> None:
    """Record an exception with stack trace and contextual information.

    Args:
        error: The exception instance to log.
        context: Optional mapping with contextual details.
    """
    context = context or {}
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    logger.error("Unhandled exception: %s | Context: %s\n%s", error, context, tb)
