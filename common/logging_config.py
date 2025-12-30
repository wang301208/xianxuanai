from __future__ import annotations

"""Centralised logging configuration for the AutoGPT repository."""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from contextvars import ContextVar
from typing import Final, Iterable, Optional

import structlog

_CONFIGURED: bool = False
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = "keyvalue"
DEFAULT_ALERT_LEVEL: Final[int] = logging.ERROR
DEFAULT_ALERT_THROTTLE_SECONDS: Final[int] = 300

correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
request_context_var: ContextVar[dict] = ContextVar("request_context", default={})


def _resolve_level(level: Optional[str]) -> int:
    candidate = level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL
    value = logging.getLevelName(candidate.upper())
    if isinstance(value, int):
        return value
    return logging.INFO


def _resolve_format(fmt: Optional[str]) -> str:
    candidate = fmt or os.getenv("LOG_FORMAT") or DEFAULT_LOG_FORMAT
    candidate = candidate.strip().lower()
    if candidate in {"json", "keyvalue", "console"}:
        return candidate
    return DEFAULT_LOG_FORMAT


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        return record.levelno <= self.max_level


class AlertingHandler(logging.Handler):
    """Send high-priority events to an alerting sink with throttling."""

    def __init__(
        self,
        *,
        min_level: int,
        webhook_url: Optional[str],
        throttle_seconds: int,
        formatter: logging.Formatter,
    ) -> None:
        super().__init__(level=min_level)
        self.webhook_url = webhook_url
        self.throttle_seconds = throttle_seconds
        self.formatter = formatter
        self._lock = threading.Lock()
        self._last_sent_at: float = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        if not self.webhook_url:
            return

        now = time.time()
        with self._lock:
            if now - self._last_sent_at < self.throttle_seconds:
                return
            self._last_sent_at = now

        payload = {
            "level": record.levelname,
            "message": self.format(record),
            "logger": record.name,
            "correlation_id": correlation_id_var.get(),
            "request": request_context_var.get(),
        }

        if record.exc_info:
            payload["exception"] = logging.Formatter().formatException(record.exc_info)

        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)  # nosec B310 - controlled destination
        except Exception:
            # We avoid raising inside logging handlers; defer to stderr instead.
            sys.stderr.write("Failed to publish alert\n")


def bind_correlation_id(correlation_id: Optional[str]) -> None:
    """Attach a correlation ID to subsequent log entries."""

    if correlation_id:
        correlation_id_var.set(correlation_id)
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def bind_request_context(**context: str) -> None:
    """Attach request-level context to subsequent log entries."""

    cleaned = {k: v for k, v in context.items() if v is not None}
    if cleaned:
        request_context_var.set(cleaned)
        structlog.contextvars.bind_contextvars(**cleaned)


def clear_request_context() -> None:
    request_context_var.set({})
    structlog.contextvars.clear_contextvars()


def _renderer_for_format(resolved_format: str) -> structlog.types.Processor:
    if resolved_format == "json":
        return structlog.processors.JSONRenderer()
    if resolved_format == "console":
        return structlog.dev.ConsoleRenderer()
    return structlog.processors.KeyValueRenderer(
        key_order=["timestamp", "level", "event", "logger", "correlation_id"]
    )


def _build_std_handlers(resolved_level: int) -> Iterable[logging.Handler]:
    message_formatter = logging.Formatter(fmt="%(message)s")

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(message_formatter)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING - 1))

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(message_formatter)

    if resolved_level >= logging.WARNING:
        # If the resolved level filters out info logs, keep stderr only
        return [stderr_handler]
    return [stdout_handler, stderr_handler]


def _build_exporters(
    exporters: Iterable[str], formatter: logging.Formatter
) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    for name in exporters:
        name = name.strip().lower()
        if name == "syslog":
            address = os.getenv("LOG_EXPORTER_SYSLOG_ADDRESS", "localhost:514")
            host, _, port = address.partition(":")
            handler = logging.handlers.SysLogHandler(address=(host, int(port or 514)))
            handler.setFormatter(formatter)
            handlers.append(handler)
        elif name == "json_file":
            path = os.getenv("LOG_EXPORTER_JSON_PATH", "logs/observability.jsonl")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            handler = logging.FileHandler(path)
            handler.setFormatter(formatter)
            handlers.append(handler)
        elif name == "otlp":
            try:
                from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
                from opentelemetry.sdk._logs.export import (
                    BatchLogRecordProcessor,
                    OTLPLogExporter,
                )
                from opentelemetry.sdk.resources import Resource

                exporter = OTLPLogExporter(
                    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                )
                provider = LoggerProvider(
                    resource=Resource.create({"service.name": os.getenv("SERVICE_NAME", "autosuper")})
                )
                provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
                handler = LoggingHandler(level=logging.NOTSET, logger_provider=provider)
                handler.setFormatter(formatter)
                handlers.append(handler)
            except Exception:
                sys.stderr.write("OTLP exporter unavailable; skipping.\n")
        else:
            sys.stderr.write(f"Unknown log exporter '{name}', skipping.\n")

    return handlers


def _parse_exporters() -> list[str]:
    value = os.getenv("LOG_EXPORTERS", "").strip()
    if not value:
        return []
    return [part for part in value.split(",") if part]


def configure_logging(*, level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Configure structlog + stdlib logging once for the process."""

    global _CONFIGURED

    if _CONFIGURED:
        return

    resolved_level = _resolve_level(level)
    resolved_format = _resolve_format(fmt)
    std_formatter = logging.Formatter(fmt="%(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(resolved_level)

    for handler in _build_std_handlers(resolved_level):
        root_logger.addHandler(handler)

    for handler in _build_exporters(_parse_exporters(), std_formatter):
        root_logger.addHandler(handler)

    alert_handler = AlertingHandler(
        min_level=int(os.getenv("ALERT_MIN_LEVEL", DEFAULT_ALERT_LEVEL)),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        throttle_seconds=int(
            os.getenv("ALERT_THROTTLE_SECONDS", DEFAULT_ALERT_THROTTLE_SECONDS)
        ),
        formatter=std_formatter,
    )
    root_logger.addHandler(alert_handler)

    renderer = _renderer_for_format(resolved_format)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True
