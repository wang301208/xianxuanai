from __future__ import annotations

"""Simple math reasoning domain adapter."""

from .core import DomainAdapter, register_adapter


class MathDomainAdapter(DomainAdapter):
    """Evaluate basic arithmetic expressions."""

    def process(self, query: str) -> str:
        try:
            # Evaluate arithmetic expression safely using Python's eval with no builtins
            result = eval(query, {"__builtins__": {}})
        except Exception:  # pragma: no cover - invalid expression
            return "error"
        return str(result)


register_adapter("math", MathDomainAdapter)
