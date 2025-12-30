from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Mapping

from modules.environment import (
    dispatch_task,
    get_hardware_registry,
    subscribe_resource_signals,
    subscribe_service_catalog,
    subscribe_service_signals,
    subscribe_task_dispatch,
    subscribe_task_results,
)


class PipelineScheduler:
    """Route task dispatch events to the appropriate RPC service pools."""

    def __init__(
        self,
        event_bus,
        *,
        pipeline: Optional[Sequence[str]] = None,
        stage_service_map: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._bus = event_bus
        self._pipeline = list(pipeline or [])
        self._stage_service_map = dict(stage_service_map or {})
        self._logger = logger or logging.getLogger(__name__)
        self._registry = get_hardware_registry()
        self._service_metrics: Dict[str, Dict[str, float]] = {}
        self._inflight: Dict[str, Dict[str, Any]] = {}
        self._subscriptions: List[Any] = []

    def start(self) -> None:
        """Begin listening for task and metric events."""

        self._subscriptions.append(
            subscribe_task_dispatch(self._bus, self._handle_dispatch)
        )
        self._subscriptions.append(
            subscribe_task_results(self._bus, self._handle_result)
        )
        self._subscriptions.append(
            subscribe_service_signals(self._bus, self._handle_service_signal)
        )
        self._subscriptions.append(
            subscribe_resource_signals(self._bus, self._handle_resource_signal)
        )
        self._subscriptions.append(
            subscribe_service_catalog(self._bus, self._handle_service_catalog)
        )

    def stop(self) -> None:
        for cancel in self._subscriptions:
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive cleanup
                self._logger.debug("Pipeline scheduler cancellation failed", exc_info=True)
        self._subscriptions.clear()

    async def _handle_dispatch(self, event: Dict[str, Any]) -> None:
        if event.get("routed"):
            return
        task_id = event.get("task_id")
        payload = dict(event.get("payload") or {})
        stage = payload.get("stage") or event.get("stage")
        if not stage and self._pipeline:
            stage = self._pipeline[0]
            payload["stage"] = stage
        service_type = self._stage_service_map.get(stage or "", stage or "skill")
        service = self._select_service(service_type)
        if service is None:
            self._logger.warning(
                "No available service for stage '%s' (type=%s)", stage, service_type
            )
            return
        metadata = dict(event.get("metadata") or {})
        metadata.update({"stage": stage, "service_id": service["service_id"]})
        self._inflight[task_id] = {
            "stage": stage,
            "service_id": service["service_id"],
        }
        dispatch_task(
            self._bus,
            task_id,
            payload,
            assigned_to=service["service_id"],
            metadata=metadata,
            routed=True,
        )

    async def _handle_result(self, event: Dict[str, Any]) -> None:
        task_id = event.get("task_id")
        if not task_id:
            return
        context = self._inflight.pop(task_id, {})
        result = event.get("result")
        if not isinstance(result, Mapping):
            self._logger.warning(
                "Task %s produced non-mapping result %r; skipping follow-up",
                task_id,
                result,
            )
            return
        result_payload = dict(result or {})
        next_stage = result_payload.get("next_stage")
        if not next_stage and self._pipeline:
            try:
                current_index = self._pipeline.index(context.get("stage"))
            except ValueError:
                current_index = -1
            if current_index != -1 and current_index + 1 < len(self._pipeline):
                next_stage = self._pipeline[current_index + 1]
        if not next_stage:
            return
        next_payload = (
            result_payload.get("next_payload")
            or result_payload.get("payload")
            or result_payload
        )
        metadata = dict(event.get("metadata") or {})
        metadata["previous_stage"] = context.get("stage")
        self.schedule_stage(task_id, next_stage, next_payload, metadata)

    async def _handle_service_signal(self, event: Dict[str, Any]) -> None:
        service_id = event.get("service_id")
        metrics = event.get("metrics") or {}
        if service_id:
            self._service_metrics[service_id] = {
                str(k): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }

    async def _handle_resource_signal(self, event: Dict[str, Any]) -> None:
        worker_id = event.get("worker_id")
        metrics = event.get("resource_signal") or {}
        if worker_id and metrics:
            numeric = {
                str(k): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            if numeric:
                self._service_metrics.setdefault(worker_id, {}).update(numeric)

    async def _handle_service_catalog(self, event: Dict[str, Any]) -> None:
        service_id = event.get("service_id")
        if service_id:
            self._service_metrics.setdefault(service_id, {})

    def schedule_stage(
        self,
        task_id: str,
        stage: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = dict(payload)
        payload.setdefault("stage", stage)
        service_type = self._stage_service_map.get(stage, stage)
        service = self._select_service(service_type)
        if service is None:
            self._logger.warning(
                "No service available for scheduled stage '%s' (type=%s)", stage, service_type
            )
            return
        meta = dict(metadata or {})
        meta.update({"stage": stage, "service_id": service["service_id"]})
        self._inflight[task_id] = {"stage": stage, "service_id": service["service_id"]}
        dispatch_task(
            self._bus,
            task_id,
            payload,
            assigned_to=service["service_id"],
            metadata=meta,
            routed=True,
        )

    def _select_service(self, service_type: str) -> Optional[Dict[str, Any]]:
        services = self._registry.list_services()
        candidates = [
            descriptor.to_dict()
            for descriptor in services.values()
            if descriptor.service_type == service_type
        ]
        if not candidates:
            return None
        best = None
        best_score = float("inf")
        for candidate in candidates:
            metrics = self._service_metrics.get(candidate["service_id"], {})
            queue_depth = metrics.get("queue_depth", 0.0)
            latency = metrics.get("p95_latency_ms", metrics.get("latency_ms", 0.0))
            utilization = max(
                metrics.get("gpu_utilization", 0.0),
                metrics.get("cpu_utilization", 0.0),
            )
            score = queue_depth + latency * 0.001 + utilization
            if score < best_score:
                best_score = score
                best = candidate
        return best
