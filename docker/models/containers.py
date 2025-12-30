"""Minimal Docker container type stub."""


class Container:  # pragma: no cover - simple placeholder
    def __init__(self, name: str = "stub", status: str = "exited") -> None:
        self.name = name
        self.status = status

    def start(self) -> None:  # pragma: no cover - stubbed behaviour
        self.status = "running"

    def restart(self) -> None:  # pragma: no cover - stubbed behaviour
        self.status = "running"

    def exec_run(self, *args, **kwargs):  # pragma: no cover - stubbed behaviour
        class _Result:
            exit_code = 1
            output = b"Docker is unavailable"

        return _Result()
