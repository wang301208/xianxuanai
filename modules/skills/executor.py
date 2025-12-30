from __future__ import annotations

"""Sandboxed execution helpers for pluggable skills."""

import asyncio
import inspect
import threading
import os
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict, Optional, TYPE_CHECKING

from modules.execution import TaskAdapter, create_task_adapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .rpc_client import SkillRPCClient


class SkillExecutionError(RuntimeError):
    """Raised when a skill fails during sandboxed execution."""


class SkillTimeoutError(SkillExecutionError):
    """Raised when a skill exceeds its allotted execution time."""


class SkillSandbox:
    """Constrain skill execution via timeouts and concurrency limits."""

    def __init__(
        self,
        *,
        default_timeout: float = 10.0,
        max_workers: int = 8,
        task_adapter: TaskAdapter | None = None,
        event_bus: Optional[Any] = None,
        worker_id: Optional[str] = None,
        rpc_client: "SkillRPCClient | None" = None,
    ) -> None:
        self._default_timeout = max(default_timeout, 0.1)
        self._semaphore = threading.Semaphore(max_workers)
        adapter_mode = os.getenv("SKILL_TASK_ADAPTER") or None
        if task_adapter is None:
            task_adapter = create_task_adapter(
                adapter_mode,
                worker_id=worker_id or "skill-sandbox",
                event_bus=event_bus,
                max_workers=max_workers,
            )
        self._adapter = task_adapter
        self._event_bus = event_bus
        self._rpc_client: "SkillRPCClient | None" = rpc_client
        self._rpc_lock = threading.Lock()

    def shutdown(self) -> None:
        """Cleanly shutdown the sandbox executor."""

        try:
            self._adapter.shutdown()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def run(
        self,
        handler,
        payload: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute ``handler`` with sandbox constraints."""

        timeout = self._default_timeout if timeout is None or timeout <= 0 else timeout
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise SkillTimeoutError("Skill sandbox is saturated.")

        skill_metadata = dict(metadata or {})
        execution_mode = str(skill_metadata.get("execution_mode", "local") or "local").lower()
        skill_name = skill_metadata.get("name") or getattr(handler, "__name__", "skill")

        if execution_mode == "rpc":
            try:
                return self._dispatch_rpc(skill_name, payload, context or {}, skill_metadata, timeout)
            finally:
                self._semaphore.release()

        if handler is None:
            self._semaphore.release()
            raise SkillExecutionError("Skill handler is not callable.")

        task_metadata = {
            "name": skill_name,
            "category": "skill",
            "execution_mode": execution_mode,
        }

        future = self._adapter.submit(
            _invoke_handler,
            handler,
            payload,
            context or {},
            metadata=task_metadata,
        )
        future.add_done_callback(lambda _: self._semaphore.release())
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise SkillTimeoutError("Skill execution timed out.") from exc
        except TimeoutError as exc:
            future.cancel()
            raise SkillTimeoutError("Skill execution timed out.") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Skill execution failed: {exc}") from exc

    def _ensure_rpc_client(self) -> "SkillRPCClient":
        with self._rpc_lock:
            if self._rpc_client is None:
                from .rpc_client import SkillRPCClient  # local import to avoid heavy dependency at import time

                self._rpc_client = SkillRPCClient.from_env()
        return self._rpc_client

    def _dispatch_rpc(
        self,
        skill_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
        metadata: Dict[str, Any],
        timeout: float,
    ) -> Any:
        client = self._ensure_rpc_client()
        try:
            result = client.invoke(
                skill_name,
                payload,
                context=context,
                metadata=metadata,
                timeout=timeout,
            )
        except Exception as exc:
            raise SkillRPCDispatchError(f"RPC invocation failed for skill '{skill_name}': {exc}") from exc
        return result


async def _await_result(awaitable):
    return await awaitable


def _invoke_handler(handler_fn, payload_arg, context_arg) -> Any:
    result = handler_fn(payload_arg, context=context_arg or {})
    if inspect.isawaitable(result):
        return asyncio.run(_await_result(result))
    return result


class SkillRPCDispatchError(SkillExecutionError):
    """Raised when RPC delegation fails."""
