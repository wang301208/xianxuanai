from __future__ import annotations

"""Abstraction layer for interacting with quantum hardware backends.

This module provides a minimal wrapper around different quantum computation
backends such as Qiskit or Amazon Braket.  The implementation intentionally
keeps the interface lightweight so unit tests can mock the behaviour without
requiring the actual heavy dependencies to be installed.
"""

from typing import Any, Mapping


class QuantumHardwareInterface:
    """Facade for executing circuits on various quantum backends."""

    def __init__(self, backend: str = "local", **config: Any) -> None:
        self.backend = backend
        self.config = config
        self.client: Any | None = None

    # ------------------------------------------------------------------
    # Connection utilities
    def connect(self) -> Any:
        """Connect to the configured backend and return the client object."""

        if self.backend == "qiskit":  # pragma: no cover - requires qiskit
            from qiskit import IBMQ

            token = self.config.get("token")
            if token:
                IBMQ.enable_account(token)
            provider = IBMQ.load_account()
            backend_name = self.config.get("backend_name", "qasm_simulator")
            self.client = provider.get_backend(backend_name)
        elif self.backend == "braket":  # pragma: no cover - requires braket
            from braket.aws import AwsDevice

            arn = self.config.get("device_arn")
            self.client = AwsDevice(arn)
        else:
            # "local" mode does not require a connection
            self.client = None
        return self.client

    # ------------------------------------------------------------------
    # Execution utilities
    def run(self, circuit: Any, shots: int = 1024) -> Any:
        """Execute *circuit* on the selected backend."""

        if self.backend == "qiskit":  # pragma: no cover - requires qiskit
            from qiskit import execute

            job = execute(circuit, self.client, shots=shots)
            return job.result()
        if self.backend == "braket":  # pragma: no cover - requires braket
            task = self.client.run(circuit, shots=shots)
            return task.result()

        # Local simulation: allow ``circuit`` to be a callable or have a
        # ``simulate`` method.  If neither is available, assume it already
        # represents the result.
        if callable(circuit):
            return circuit()
        if hasattr(circuit, "simulate"):
            return circuit.simulate(shots=shots)
        return circuit

    # ------------------------------------------------------------------
    # Result handling
    def parse_counts(self, result: Any) -> Mapping[str, int] | dict[str, int]:
        """Extract measurement counts from a backend result object."""

        if hasattr(result, "get_counts"):
            return dict(result.get_counts())
        if hasattr(result, "counts"):
            return dict(result.counts)
        if hasattr(result, "measurement_counts"):
            return dict(result.measurement_counts)
        if isinstance(result, Mapping):
            return dict(result)
        return {}


__all__ = ["QuantumHardwareInterface"]
