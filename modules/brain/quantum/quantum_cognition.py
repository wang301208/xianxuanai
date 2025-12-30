"""Quantum cognition components.

This module provides lightweight representations of quantum states and a
simple network for generating entangled states.  It purposely avoids any
heavy dependencies or advanced simulation features; the goal is merely to
support unit tests that exercise basic superposition and measurement
behaviour.

Classes
-------
SuperpositionState
    Stores complex amplitudes for basis states and exposes utilities to
    obtain probabilities or a density matrix representation.
EntanglementNetwork
    Produces a small set of predefined entangled states used by the tests and
    exposes :meth:`entangle_concepts` for simple decoherence simulations.
QuantumCognition
    High level interface exposing :meth:`evaluate_probabilities`,
    :meth:`quantum_decision_making` and :meth:`quantum_memory_retrieval` which
    operate on quantum states represented by :class:`SuperpositionState` or
    density matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Union, TYPE_CHECKING, Callable, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .quantum_attention import QuantumAttention
    from .quantum_memory import QuantumMemory
    from .quantum_reasoning import QuantumReasoning
    from .quantum_ml import QuantumClassifier
    from .hardware_interface import QuantumHardwareInterface


@dataclass
class SuperpositionState:
    """Representation of a quantum superposition."""

    amplitudes: Dict[str, complex]

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(abs(a) ** 2 for a in self.amplitudes.values()))
        if not np.isclose(norm, 1):
            self.amplitudes = {k: v / norm for k, v in self.amplitudes.items()}

    def probabilities(self) -> Dict[str, float]:
        """Return measurement probabilities for each basis state."""
        return {k: float(abs(a) ** 2) for k, a in self.amplitudes.items()}

    def density_matrix(self) -> np.ndarray:
        """Return the density matrix corresponding to the state."""
        labels = sorted(self.amplitudes.keys())
        vector = np.array([self.amplitudes[l] for l in labels], dtype=complex)
        return np.outer(vector, np.conjugate(vector))


@dataclass
class EntanglementNetwork:
    """Utility for creating simple entangled states."""

    def create_bell_pair(self) -> SuperpositionState:
        """Return a Bell pair :math:`(|00\rangle + |11\rangle)/\sqrt{2}`."""
        amp = 1 / math.sqrt(2)
        return SuperpositionState({"00": amp, "01": 0, "10": 0, "11": amp})

    def entangle_concepts(
        self, concept_a: str, concept_b: str, decoherence: float = 0.0
    ) -> np.ndarray:
        """Return density matrix for two entangled *concepts*."""

        density = self.create_bell_pair().density_matrix()
        if decoherence:
            density = density.copy()
            factor = 1 - decoherence
            density[0, 3] *= factor
            density[3, 0] *= factor
        return density


class QuantumCognition:
    """High level interface for evaluating quantum cognitive states."""

    def __init__(
        self,
        network: EntanglementNetwork | None = None,
        memory: QuantumMemory | None = None,
        attention: QuantumAttention | None = None,
        reasoning: QuantumReasoning | None = None,
        search: Callable | None = None,
        classifier: "QuantumClassifier" | None = None,
        *,
        backend: str = "local",
        hardware: "QuantumHardwareInterface" | None = None,
    ) -> None:
        from .quantum_attention import QuantumAttention as QA
        from .quantum_memory import QuantumMemory as QM
        from .quantum_reasoning import QuantumReasoning as QR
        from .grover_search import grover_search
        from .quantum_ml import QuantumClassifier as QC
        from .hardware_interface import QuantumHardwareInterface

        self.network = network or EntanglementNetwork()
        self.memory = memory or QM()
        self.attention = attention or QA()
        self.reasoning = reasoning or QR()
        self.search = search or grover_search
        self.classifier = classifier or QC()

        self.backend = backend
        self.hardware = hardware
        if self.backend != "local" and self.hardware is None:
            self.hardware = QuantumHardwareInterface(self.backend)
        self.task_queue: list[tuple[Any, int]] = []

    def evaluate_probabilities(
        self, input_state: Union[SuperpositionState, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate measurement probabilities for *input_state*."""

        if isinstance(input_state, SuperpositionState):
            return input_state.probabilities()

        matrix = np.asarray(input_state, dtype=complex)
        diag = np.real_if_close(np.diag(matrix))
        size = diag.shape[0]
        num_qubits = int(math.log2(size)) if size else 0
        labels = [format(i, f"0{num_qubits}b") for i in range(size)]
        return {label: float(diag[i]) for i, label in enumerate(labels)}

    def make_decision(
        self,
        options: Mapping[str, Iterable[complex]],
        rng: np.random.Generator | None = None,
    ) -> tuple[str, Dict[str, float]]:
        """Backward compatible alias for :meth:`quantum_decision_making`."""

        return self.quantum_decision_making(options, rng)

    def quantum_decision_making(
        self,
        options: Mapping[str, Iterable[complex]],
        rng: np.random.Generator | None = None,
    ) -> tuple[str, Dict[str, float]]:
        """Perform a quantum-style decision over *options*."""

        probs = self.reasoning.interference(options)
        labels = list(probs.keys())
        generator = rng or np.random.default_rng()
        choice = generator.choice(labels, p=list(probs.values()))
        return choice, probs

    def quantum_memory_retrieval(
        self,
        weights: Mapping[str, complex],
        focus: Mapping[str, float] | None = None,
    ) -> SuperpositionState:
        """Retrieve a superposition from memory optionally applying attention."""

        state = self.memory.superposition(weights)
        if focus:
            state = self.attention.apply(state, focus)
        return state

    def grover_search(self, items: Iterable, oracle) -> object:
        """Run Grover search over *items* using *oracle* function."""

        return self.search(list(items), oracle)

    def train_classifier(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train internal quantum classifier on dataset."""

        self.classifier.train(X, y)

    def classify(self, sample: Iterable[float]) -> int:
        """Classify ``sample`` using the trained quantum classifier."""

        return self.classifier.predict(sample)

    # ------------------------------------------------------------------
    # Hardware/backend integration
    def add_task(self, circuit: Any, shots: int = 1024) -> None:
        """Queue a quantum task for later execution."""

        self.task_queue.append((circuit, shots))

    def run_tasks(self) -> list[Mapping[str, int]]:
        """Execute all queued tasks and return parsed measurement results."""

        results: list[Mapping[str, int]] = []
        for circuit, shots in self.task_queue:
            if self.hardware:
                raw = self.hardware.run(circuit, shots=shots)
                results.append(self.hardware.parse_counts(raw))
            else:
                raw = self._run_local(circuit, shots)
                results.append(self._parse_local(raw))
        self.task_queue.clear()
        return results

    def _run_local(self, circuit: Any, shots: int) -> Any:
        if callable(circuit):
            return circuit()
        if hasattr(circuit, "simulate"):
            return circuit.simulate(shots=shots)
        return circuit

    def _parse_local(self, result: Any) -> Mapping[str, int]:
        if isinstance(result, Mapping):
            return dict(result)
        if hasattr(result, "get_counts"):
            return dict(result.get_counts())
        return {"result": result}


__all__ = ["SuperpositionState", "EntanglementNetwork", "QuantumCognition"]
