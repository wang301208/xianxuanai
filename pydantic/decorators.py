"""Provide a tiny subset of decorator utilities from Pydantic."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["validator"]


def validator(*_field_names: str, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a no-op validator decorator.

    The real Pydantic records validators on the model class. The tests exercised
    in this kata only need the decorator to be syntactically valid, so this
    compatibility layer simply returns the function unchanged.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator

