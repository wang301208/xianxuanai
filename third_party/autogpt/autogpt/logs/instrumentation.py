"""Logging utilities for runtime instrumentation."""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def log_call(func: F) -> F:
    """Log inputs, outputs and execution time for ``func``.

    The log record is emitted at INFO level with the ``runtime_metrics`` tag to
    make it easy to parse by analytics tooling. Both synchronous and
    asynchronous callables are supported.
    """

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            logger = logging.getLogger(func.__module__)
            start = time.perf_counter()
            logger.debug("Calling %s with %s %s", func.__qualname__, args, kwargs)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                payload = {
                    "function": func.__qualname__,
                    "args": args,
                    "kwargs": kwargs,
                    "duration": duration,
                    "result": result if "result" in locals() else None,
                }
                logger.info("runtime_metrics %s", json.dumps(payload, default=str))

        return cast(F, async_wrapper)

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        logger.debug("Calling %s with %s %s", func.__qualname__, args, kwargs)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            payload = {
                "function": func.__qualname__,
                "args": args,
                "kwargs": kwargs,
                "duration": duration,
                "result": result if "result" in locals() else None,
            }
            logger.info("runtime_metrics %s", json.dumps(payload, default=str))

    return cast(F, sync_wrapper)
