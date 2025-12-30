"""Utilities for unified exception logging and handling."""
from __future__ import annotations

import asyncio
import functools
import logging
import sys
import threading
from typing import Any, Callable, Coroutine, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def handle_exception(
    exc: Exception,
    logger: Optional[logging.Logger] = None,
    propagate: bool = False,
) -> None:
    """Log *exc* and optionally re-raise it.

    Args:
        exc: The exception instance to handle.
        logger: Optional logger to use; if not provided, a module-level logger is
            used.
        propagate: When ``True`` the exception is re-raised after logging.
    """

    log = logger or logging.getLogger(__name__)
    log.exception("%s", exc)
    if propagate:
        raise exc


def setup_exception_hooks(logger: Optional[logging.Logger] = None) -> None:
    """Register global hooks to log uncaught exceptions.

    This installs handlers for ``sys.excepthook``, ``threading.excepthook`` and
    the current event loop's exception handler so that any exception not handled
    elsewhere is logged through :func:`handle_exception`.
    """

    def _handle_uncaught(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        handle_exception(exc, logger)

    sys.excepthook = _handle_uncaught

    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        handle_exception(args.exc, logger)

    threading.excepthook = _handle_thread_exception

    try:
        loop = asyncio.get_event_loop()

        def _handle_async_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
            exc = context.get("exception")
            if exc:
                handle_exception(exc, logger)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_handle_async_exception)
    except RuntimeError:
        # No event loop running; ignore.
        pass


def log_async_exceptions(
    func: Callable[P, Coroutine[Any, Any, T]]
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Decorator to log exceptions raised by an async function."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            handle_exception(e)
            raise

    return wrapper


def log_thread_exceptions(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log exceptions raised in background threads."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            handle_exception(e)
            raise

    return wrapper
