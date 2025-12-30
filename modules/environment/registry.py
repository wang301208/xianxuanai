from __future__ import annotations

"""Runtime registry tracking worker hardware capabilities."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable

from modules.events import EventBus, publish, subscribe
from modules.events.coordination import (
    ResourceSignalEvent,
    TaskDispatchEvent,
    TaskResultEvent,
    TaskStatus,
)


@dataclass
class WorkerDescriptor:
    worker_id: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class ServiceDescriptor:
    service_id: str
    service_type: str  # e.g. "skill", "model", "graph", "vector"
    protocol: str
    endpoint: Dict[str, Any]
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "protocol": self.protocol,
            "endpoint": dict(self.endpoint),
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
            "resource_requirements": dict(self.resource_requirements),
            "metrics": dict(self.metrics),
            "last_heartbeat": self.last_heartbeat,
        }


class HardwareEnvironmentRegistry:
    """Thread-safe registry of hardware capabilities advertised by workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: Dict[str, WorkerDescriptor] = {}
        self._services: Dict[str, ServiceDescriptor] = {}

    def register(
        self,
        worker_id: str,
        capabilities: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register or update ``worker_id`` with ``capabilities``."""

        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        with self._lock:
            descriptor = self._workers.get(worker_id)
            if descriptor is None:
                descriptor = WorkerDescriptor(
                    worker_id=worker_id,
                    capabilities=dict(capabilities),
                    metadata=dict(metadata or {}),
                )
                self._workers[worker_id] = descriptor
            else:
                descriptor.capabilities = dict(capabilities)
                if metadata:
                    descriptor.metadata.update(metadata)
            descriptor.last_heartbeat = time.time()

    def update_metrics(
        self,
        worker_id: str,
        metrics: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update runtime metrics for ``worker_id``."""

        with self._lock:
            descriptor = self._workers.get(worker_id)
            if descriptor is None:
                descriptor = WorkerDescriptor(worker_id=worker_id)
                self._workers[worker_id] = descriptor
            descriptor.metrics = {str(k): float(v) for k, v in metrics.items()}
            if metadata:
                descriptor.metadata.update(metadata)
            descriptor.last_heartbeat = time.time()

    def unregister(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            descriptor = self._workers.get(worker_id)
            return descriptor.to_dict() if descriptor is not None else None

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {worker: descriptor.to_dict() for worker, descriptor in self._workers.items()}

    def list_workers(self) -> Dict[str, WorkerDescriptor]:
        with self._lock:
            return dict(self._workers)

    # ------------------------------------------------------------------ service registry
    def register_service(
        self,
        service_id: str,
        service_type: str,
        protocol: str,
        endpoint: Dict[str, Any],
        *,
        capabilities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ServiceDescriptor:
        if not service_id:
            raise ValueError("service_id must be non-empty")
        descriptor = ServiceDescriptor(
            service_id=service_id,
            service_type=service_type,
            protocol=protocol,
            endpoint=dict(endpoint),
            capabilities=dict(capabilities or {}),
            metadata=dict(metadata or {}),
            resource_requirements=dict(resource_requirements or {}),
            metrics={str(k): float(v) for k, v in (metrics or {}).items()},
        )
        with self._lock:
            self._services[service_id] = descriptor
        return descriptor

    def update_service_metrics(
        self,
        service_id: str,
        metrics: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            descriptor = self._services.get(service_id)
            if descriptor is None:
                return
            descriptor.metrics = {str(k): float(v) for k, v in metrics.items()}
            if metadata:
                descriptor.metadata.update(metadata)
            descriptor.last_heartbeat = time.time()

    def unregister_service(self, service_id: str) -> None:
        with self._lock:
            self._services.pop(service_id, None)

    def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            descriptor = self._services.get(service_id)
            return descriptor.to_dict() if descriptor is not None else None

    def snapshot_services(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {sid: descriptor.to_dict() for sid, descriptor in self._services.items()}

    def list_services(self) -> Dict[str, ServiceDescriptor]:
        with self._lock:
            return dict(self._services)


_GLOBAL_REGISTRY = HardwareEnvironmentRegistry()


def get_hardware_registry() -> HardwareEnvironmentRegistry:
    return _GLOBAL_REGISTRY


SERVICE_CATALOG_TOPIC = "service.catalog"
SERVICE_SIGNAL_TOPIC = "service.signal"
TASK_DISPATCH_TOPIC = "task.dispatch"
TASK_RESULT_TOPIC = "task.result"


def register_service(
    service_id: str,
    service_type: str,
    protocol: str,
    endpoint: Dict[str, Any],
    *,
    capabilities: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    resource_requirements: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    event_bus: Optional[EventBus] = None,
) -> ServiceDescriptor:
    registry = get_hardware_registry()
    descriptor = registry.register_service(
        service_id,
        service_type,
        protocol,
        endpoint,
        capabilities=capabilities,
        metadata=metadata,
        resource_requirements=resource_requirements,
        metrics=metrics,
    )
    if event_bus is not None:
        publish(event_bus, SERVICE_CATALOG_TOPIC, descriptor.to_dict())
    return descriptor


def unregister_service(service_id: str) -> None:
    get_hardware_registry().unregister_service(service_id)


def register_rpc_service(
    service_id: str,
    service_type: str,
    protocol: str,
    endpoint: Dict[str, Any],
    *,
    capabilities: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    resource_requirements: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    event_bus: Optional[EventBus] = None,
) -> ServiceDescriptor:
    return register_service(
        service_id,
        service_type,
        protocol,
        endpoint,
        capabilities=capabilities,
        metadata=metadata,
        resource_requirements=resource_requirements,
        metrics=metrics,
        event_bus=event_bus,
    )


def register_skill_service(
    spec: Any,
    metadata: Dict[str, Any],
    *,
    event_bus: Optional[EventBus] = None,
) -> Optional[ServiceDescriptor]:
    rpc_config = dict(metadata.get("rpc_config") or {})
    endpoint_url = rpc_config.get("endpoint") or rpc_config.get("base_url")
    if not endpoint_url:
        return None
    service_name = getattr(spec, "name", str(metadata.get("name", "skill")))
    service_id = f"skill:{service_name}"
    protocol = rpc_config.get("protocol", "http").lower()
    endpoint_payload = {
        "url": endpoint_url,
        "path": rpc_config.get("path", "/invoke"),
        "method": rpc_config.get("method", "POST"),
        "timeout": rpc_config.get("timeout"),
    }
    capabilities = {
        "input_schema": getattr(spec, "input_schema", {}),
        "output_schema": getattr(spec, "output_schema", {}),
        "tags": getattr(spec, "tags", []),
    }
    base_metadata = {
        "description": getattr(spec, "description", ""),
        "provider": getattr(spec, "provider", "builtin"),
        "version": getattr(spec, "version", "0.1.0"),
        "execution_mode": getattr(spec, "execution_mode", "local"),
    }
    extra_metadata = {
        k: v for k, v in metadata.items() if k not in {"rpc_config", "resource_requirements", "resources"}
    }
    resources = metadata.get("resource_requirements") or metadata.get("resources") or {}
    return register_rpc_service(
        service_id=service_id,
        service_type="skill",
        protocol=protocol,
        endpoint=endpoint_payload,
        capabilities=capabilities,
        metadata={**base_metadata, **extra_metadata},
        resource_requirements=resources,
        event_bus=event_bus,
    )


def register_model_service(
    model_name: str,
    rpc_config: Dict[str, Any],
    *,
    capabilities: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    resource_requirements: Optional[Dict[str, Any]] = None,
    event_bus: Optional[EventBus] = None,
) -> Optional[ServiceDescriptor]:
    endpoint_url = rpc_config.get("endpoint") or rpc_config.get("base_url")
    if not endpoint_url:
        return None
    protocol = rpc_config.get("protocol", "http").lower()
    endpoint_payload = {
        "url": endpoint_url,
        "path": rpc_config.get("path", "/infer"),
        "method": rpc_config.get("method", "POST"),
        "timeout": rpc_config.get("timeout"),
    }
    return register_rpc_service(
        service_id=f"model:{model_name}",
        service_type="model",
        protocol=protocol,
        endpoint=endpoint_payload,
        capabilities=capabilities,
        metadata=metadata,
        resource_requirements=resource_requirements,
        event_bus=event_bus,
    )


def report_resource_signal(
    worker_id: str,
    resource_signal: Dict[str, float],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    event_bus: Optional[EventBus] = None,
) -> None:
    """Update registry metrics and optionally publish a ResourceSignalEvent."""

    registry = get_hardware_registry()
    registry.update_metrics(worker_id, resource_signal, metadata=metadata)
    if event_bus is not None:
        event = ResourceSignalEvent(
            worker_id=worker_id,
            resource_signal=dict(resource_signal),
            metadata=dict(metadata or {}),
        )
        publish(event_bus, "resource.signal", event.to_dict())


def report_service_signal(
    service_id: str,
    metrics: Dict[str, float],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    event_bus: Optional[EventBus] = None,
) -> None:
    registry = get_hardware_registry()
    registry.update_service_metrics(service_id, metrics, metadata=metadata)
    if event_bus is not None:
        payload = {
            "service_id": service_id,
            "metrics": {str(k): float(v) for k, v in metrics.items()},
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        publish(event_bus, SERVICE_SIGNAL_TOPIC, payload)


def dispatch_task(
    event_bus: EventBus,
    task_id: str,
    payload: Dict[str, Any],
    *,
    assigned_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    routed: bool = False,
) -> None:
    event = TaskDispatchEvent(
        task_id=task_id,
        payload=payload,
        assigned_to=assigned_to,
        metadata=metadata,
        routed=routed,
    )
    publish(event_bus, TASK_DISPATCH_TOPIC, event.to_dict())


def publish_task_result_event(
    event_bus: EventBus,
    task_id: str,
    status: TaskStatus,
    result: Dict[str, Any],
    *,
    worker_id: Optional[str] = None,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    event = TaskResultEvent(
        task_id=task_id,
        status=status,
        result=result,
        worker_id=worker_id,
        duration=duration,
        metadata=metadata,
    )
    publish(event_bus, TASK_RESULT_TOPIC, event.to_dict())


def subscribe_resource_signals(event_bus: EventBus, handler) -> Callable[[], None]:
    return subscribe(event_bus, "resource.signal", handler)


def subscribe_service_signals(event_bus: EventBus, handler) -> Callable[[], None]:
    return subscribe(event_bus, SERVICE_SIGNAL_TOPIC, handler)


def subscribe_service_catalog(event_bus: EventBus, handler) -> Callable[[], None]:
    return subscribe(event_bus, SERVICE_CATALOG_TOPIC, handler)


def subscribe_task_dispatch(event_bus: EventBus, handler) -> Callable[[], None]:
    return subscribe(event_bus, TASK_DISPATCH_TOPIC, handler)


def subscribe_task_results(event_bus: EventBus, handler) -> Callable[[], None]:
    return subscribe(event_bus, TASK_RESULT_TOPIC, handler)
