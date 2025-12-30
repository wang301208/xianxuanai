"""Minimal stub of the ``colorama`` package for logging helpers."""


class _Color:
    def __getattr__(self, name: str) -> str:  # pragma: no cover - debug helper
        return ""


Fore = _Color()
Style = _Color()

