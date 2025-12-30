"""Shared logging setup based on structlog + standard logging."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from typing import Optional

import structlog
from pydantic import BaseModel

from ..configuration.hub import ConfigNotFoundError, ConfigurationHub
from ..configuration.loaders import get_hub


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json: bool = False
    log_dir: Optional[str] = None
    propagate: bool = False
    capture_warnings: bool = True
    app_name: str = "autoai"


_is_configured = False


def setup_logging(config: LoggingConfig | dict | None = None) -> LoggingConfig:
    """Configure stdlib logging + structlog and return the resolved config."""

    global _is_configured
    if config is None:
        resolved = LoggingConfig()
    elif isinstance(config, dict):
        resolved = LoggingConfig(**config)
    else:
        resolved = config

    root_logger = logging.getLogger()
    # Avoid duplicate handlers when re-initialising.
    root_logger.handlers.clear()
    root_logger.setLevel(resolved.level.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if resolved.log_dir:
        log_dir = Path(resolved.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f"{resolved.app_name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(resolved.capture_warnings)

    structlog_processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if resolved.json:
        structlog_processors.append(structlog.processors.JSONRenderer())
    else:
        structlog_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=[structlog.contextvars.merge_contextvars, *structlog_processors],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if not resolved.propagate:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).propagate = False

    _is_configured = True
    return resolved


def configure_from_settings(hub: ConfigurationHub | None = None) -> LoggingConfig:
    """Setup logging based on `logging` section found in the configuration hub."""

    hub = hub or get_hub()
    try:
        settings = hub.get("logging", model=LoggingConfig)
    except ConfigNotFoundError:
        settings = LoggingConfig()
    return setup_logging(settings)


def is_configured() -> bool:
    return _is_configured
