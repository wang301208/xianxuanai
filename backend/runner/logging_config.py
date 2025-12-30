from __future__ import annotations

import structlog

from backend.common.logging import configure_from_settings

_configured = False


def get_logger(name: str) -> structlog.BoundLogger:
    global _configured
    if not _configured:
        configure_from_settings()
        _configured = True
    return structlog.get_logger(name)
