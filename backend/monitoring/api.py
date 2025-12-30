from __future__ import annotations

"""Simple API for exposing stored monitoring metrics."""

import time
from typing import Any, Callable, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import structlog

from backend.common.configuration import ConfigNotFoundError, ConfigurationHub, get_hub
from backend.common.logging import configure_from_settings
from backend.service_registry import ServiceInfo, ServiceRegistry

from .storage import TimeSeriesStorage
from .evaluation import EvaluationMetrics


logger = structlog.get_logger(__name__)


class ServiceRegistrationSettings(BaseModel):
    name: str = "monitoring-api"
    host: str = "127.0.0.1"
    port: int = 8001
    scheme: str = "http"
    tags: list[str] = []
    ttl_seconds: int = 0


class MonitoringAPISettings(BaseModel):
    service_name: str = "monitoring-api"
    default_limit: int = 100
    cache_ttl_seconds: int = 0
    enabled: bool = True
    registration: Optional[ServiceRegistrationSettings] = None


class _SummaryCache:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._cached_at: float = 0.0
        self._payload: dict[str, Any] | None = None

    def get(self, loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if not self._ttl:
            return loader()
        now = time.monotonic()
        if self._payload is None or now - self._cached_at > self._ttl:
            self._payload = loader()
            self._cached_at = now
        return self._payload


def create_app(
    storage: TimeSeriesStorage | None = None,
    evaluation: EvaluationMetrics | None = None,
    *,
    hub: ConfigurationHub | None = None,
    registry: ServiceRegistry | None = None,
) -> FastAPI:
    hub = hub or get_hub()
    configure_from_settings(hub)
    try:
        settings = hub.get("monitoring.api", model=MonitoringAPISettings)
    except ConfigNotFoundError:
        settings = MonitoringAPISettings()
        logger.warning("monitoring.api settings missing; using defaults")

    if not settings.enabled:
        logger.warning("Monitoring API disabled by configuration")
        raise RuntimeError("Monitoring API disabled by configuration")

    storage = storage or TimeSeriesStorage()
    evaluation = evaluation or EvaluationMetrics()
    app = FastAPI()
    summary_cache = _SummaryCache(settings.cache_ttl_seconds)

    registration_handle: ServiceInfo | None = None
    if registry and settings.registration:
        info = ServiceInfo(
            name=settings.registration.name or settings.service_name,
            host=settings.registration.host,
            port=settings.registration.port,
            metadata={"scheme": settings.registration.scheme},
            tags=tuple(settings.registration.tags),
        )
        ttl = settings.registration.ttl_seconds or None
        registration_handle = registry.register(info, ttl_seconds=ttl)
        app.state.service_registry = registry
        app.state.service_registration = registration_handle
        logger.info(
            "monitoring.api registered",
            service_name=registration_handle.name,
            endpoint=registration_handle.endpoint(),
        )

        @app.on_event("shutdown")
        async def _deregister() -> None:
            registry.deregister(registration_handle.name, registration_handle.instance_id)
            logger.info(
                "monitoring.api deregistered",
                service_name=registration_handle.name,
            )

    @app.get("/metrics/{topic}")
    def get_events(topic: str, limit: Optional[int] = None):
        """Return recent events for *topic*."""

        actual_limit = limit or settings.default_limit
        logger.debug("metrics.fetch", topic=topic, limit=actual_limit)
        return storage.events(topic, limit=actual_limit)

    @app.get("/metrics/summary")
    def summary():
        """Return aggregated performance metrics."""

        def loader() -> dict[str, Any]:
            logger.debug("metrics.summary.refresh")
            return {
                "success_rate": storage.success_rate(),
                "bottlenecks": storage.bottlenecks(),
                "blueprint_versions": storage.blueprint_versions(),
            }

        return summary_cache.get(loader)

    @app.get("/metrics/evaluation")
    def evaluation_summary():
        """Return precision/recall, latency and fairness metrics."""

        logger.debug("metrics.evaluation")
        return evaluation.summary()

    @app.get("/metrics/explanations")
    def explanations(limit: Optional[int] = None):
        """Return logged model explanations."""

        actual_limit = limit or settings.default_limit
        logger.debug("metrics.explanations", limit=actual_limit)
        return storage.events("analysis.explanations", limit=actual_limit)

    return app
