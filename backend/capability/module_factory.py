from __future__ import annotations

"""Evolution-driven capability module generation.

This module provides :class:`EvolutionaryModuleFactory`, which monitors problem
signatures that lack specialised support and spawns new modules on demand. A
lightweight genetic algorithm searches over a small neural controller template;
the best blueprint is materialised as a runtime-loadable module and registered
with :mod:`backend.capability.module_registry`.
"""

from dataclasses import dataclass, field
import logging
import math
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple
import uuid
import random

from .module_registry import available_modules, register_module
from .runtime_loader import RuntimeModuleManager

try:  # Optional dependency during unit tests
    from modules.interface import ModuleInterface
except Exception:  # pragma: no cover - fall back to a no-op base class
    class ModuleInterface:  # type: ignore[too-many-ancestors]
        """Fallback ModuleInterface used when the real interface is unavailable."""

        dependencies: List[str] = []

        def initialize(self) -> None:  # pragma: no cover - default no-op
            pass

        def shutdown(self) -> None:  # pragma: no cover - default no-op
            pass

try:  # Prefer the project's GA implementation when available
    from modules.evolution.evolving_cognitive_architecture import GAConfig, GeneticAlgorithm
except Exception:  # pragma: no cover - fallback keeps tests self-contained
    @dataclass
    class GAConfig:
        population_size: int = 20
        generations: int = 5
        mutation_rate: float = 0.3
        mutation_sigma: float = 0.1

    class GeneticAlgorithm:
        def __init__(
            self,
            fitness_fn: Callable[[Dict[str, float]], float],
            config: GAConfig | None = None,
            seed: int | None = None,
        ) -> None:
            self.fitness_fn = fitness_fn
            self.config = config or GAConfig()
            self._rng = random.Random(seed) if seed is not None else random

        def _mutate(self, individual: Dict[str, float]) -> None:
            for key in individual:
                if self._rng.random() < self.config.mutation_rate:
                    delta = self._rng.gauss(0.0, self.config.mutation_sigma)
                    new_value = individual[key] + delta
                    if "learning_rate" in key:
                        individual[key] = max(1e-6, new_value)
                    elif "memory_budget" in key:
                        individual[key] = min(max(new_value, 0.0), 10.0)
                    else:
                        individual[key] = new_value

        def evolve(
            self, seed: Dict[str, float]
        ) -> Tuple[Dict[str, float], float, List[Tuple[Dict[str, float], float]]]:
            best = seed.copy()
            best_score = self.fitness_fn(best)
            history: List[Tuple[Dict[str, float], float]] = [(best.copy(), best_score)]

            population: List[Dict[str, float]] = []
            for _ in range(self.config.population_size):
                ind = seed.copy()
                self._mutate(ind)
                population.append(ind)

            for _ in range(self.config.generations):
                scored = [(ind, self.fitness_fn(ind)) for ind in population]
                scored.sort(key=lambda x: x[1], reverse=True)
                best_candidate, best_candidate_score = scored[0]
                history.append((best_candidate.copy(), best_candidate_score))
                if best_candidate_score > best_score:
                    best, best_score = best_candidate.copy(), best_candidate_score
                population = []
                for _ in range(self.config.population_size):
                    child = best.copy()
                    self._mutate(child)
                    population.append(child)

            return best, best_score, history

if TYPE_CHECKING:
    from modules.evolution.dynamic_architecture import DynamicArchitectureExpander

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


SimulationDataset = Iterable[Tuple[float, float]]


@dataclass
class ModuleBlueprint:
    """Representation of an evolved module candidate."""

    signature: str
    parameters: Dict[str, float]
    fitness: float
    error: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleSpec:
    """Materialised module specification registered in the system."""

    name: str
    blueprint: ModuleBlueprint
    quality: float
    history: List[Tuple[Dict[str, float], float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


_ACTIVATIONS: Dict[int, Callable[[float], float]] = {
    0: lambda x: x,
    1: math.tanh,
    2: lambda x: max(0.0, x),
    3: math.sin,
}


_DEFAULT_BLUEPRINT_TEMPLATE: Dict[str, float] = {
    "layer1_weight": 0.1,
    "layer1_bias": 0.0,
    "layer2_weight": 0.1,
    "layer2_bias": 0.0,
    "activation": 0.0,
    "residual_weight": 0.0,
    "poly_w0": 0.0,
    "poly_w1": 0.0,
    "poly_w2": 0.0,
    "importance_weight": 0.0,
    "learning_rate": 0.01,
    "memory_budget_scalar": 1.0,
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "module"


def _decode_activation(value: float) -> Callable[[float], float]:
    index = int(round(value))
    return _ACTIVATIONS[index % len(_ACTIVATIONS)]


def _polynomial_basis(x: float) -> Tuple[float, float, float]:
    return (1.0, x, x * x)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Runtime module implementation
# ---------------------------------------------------------------------------


class EvolvedCapabilityModule(ModuleInterface):
    """Runtime module executing an evolved controller blueprint."""

    dependencies: List[str] = []

    def __init__(self, blueprint: ModuleBlueprint) -> None:
        self.blueprint = blueprint
        self._active = False
        self._hyperparameters = {
            "importance_weight": self.blueprint.parameters.get("importance_weight", 1.0),
            "learning_rate": self.blueprint.parameters.get("learning_rate", 0.01),
            "memory_budget": self.blueprint.parameters.get("memory_budget_scalar", 1.0),
        }

    def initialize(self) -> None:  # pragma: no cover - trivial hook
        self._active = True

    def shutdown(self) -> None:  # pragma: no cover - trivial hook
        self._active = False

    def __call__(self, value: float) -> float:
        return self.forward(value)

    def forward(self, value: float) -> float:
        params = self.blueprint.parameters
        w0 = params["layer1_weight"]
        b0 = params["layer1_bias"]
        w1 = params["layer2_weight"]
        b1 = params["layer2_bias"]
        residual = params["residual_weight"]
        activation = _decode_activation(params["activation"])
        importance = self._hyperparameters["importance_weight"]
        gate = max(0.0, 1.0 + math.tanh(importance))
        effective_residual = residual * gate

        basis = _polynomial_basis(value)
        poly = (
            params["poly_w0"] * basis[0]
            + params["poly_w1"] * basis[1]
            + params["poly_w2"] * basis[2]
        )

        hidden = activation(w0 * value + b0)
        output = w1 * hidden + b1
        return output + effective_residual * value + poly

    def evaluate_batch(self, inputs: Iterable[float]) -> List[float]:
        return [self.forward(x) for x in inputs]

    def get_hyperparameters(self) -> Dict[str, float]:
        """Expose evolved hyperparameters to downstream controllers."""

        return dict(self._hyperparameters)


# ---------------------------------------------------------------------------
# Factory orchestrator
# ---------------------------------------------------------------------------


class EvolutionaryModuleFactory:
    """Generate specialised modules for recurring problem signatures."""

    def __init__(
        self,
        ga_config: Optional[GAConfig] = None,
        seed: Optional[int] = None,
        blueprint_template: Optional[Callable[[], Dict[str, float]] | Dict[str, float]] = None,
    ) -> None:
        self._ga_config = ga_config or GAConfig(population_size=20, generations=6, mutation_sigma=0.4)
        self._seed = seed
        self._catalog: Dict[str, ModuleSpec] = {}
        self._rng_seed_sequence = seed
        self._blueprint_template_factory = self._resolve_blueprint_template(blueprint_template)

    # ------------------------------------------------------------------ #
    def _resolve_blueprint_template(
        self, template: Optional[Callable[[], Dict[str, float]] | Dict[str, float]]
    ) -> Callable[[], Dict[str, float]]:
        if template is None:
            return lambda: {}
        if callable(template):  # type: ignore[return-value]
            def _factory() -> Dict[str, float]:
                overrides = template()
                return dict(overrides)

            return _factory

        base_template = dict(template)

        def _factory_from_mapping() -> Dict[str, float]:
            return dict(base_template)

        return _factory_from_mapping

    # ------------------------------------------------------------------ #
    def _initial_blueprint(self) -> Dict[str, float]:
        overrides = self._blueprint_template_factory()
        blueprint = {**_DEFAULT_BLUEPRINT_TEMPLATE}
        blueprint.update(overrides)
        return blueprint

    # ------------------------------------------------------------------ #
    def _fitness_fn(
        self,
        dataset: List[Tuple[float, float]],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Dict[str, float]], float]:
        context_data: Dict[str, Any] = dict(context or {})
        metrics = {
            key: float(value)
            for key, value in (context_data.get("architectural_metrics", {}) or {}).items()
            if isinstance(value, (int, float))
        }
        metric_weights = {
            key: float(value)
            for key, value in (context_data.get("metric_weights", {}) or {}).items()
            if isinstance(value, (int, float))
        }
        # Ensure core hyperparameters always participate in the fitness signal.
        metrics.setdefault(
            "importance_weight",
            float(
                context_data.get(
                    "target_importance", _DEFAULT_BLUEPRINT_TEMPLATE["importance_weight"]
                )
            ),
        )
        metric_weights.setdefault(
            "importance_weight", float(context_data.get("importance_weight_penalty", 0.0))
        )
        metrics.setdefault(
            "learning_rate",
            float(
                context_data.get(
                    "target_learning_rate", _DEFAULT_BLUEPRINT_TEMPLATE["learning_rate"]
                )
            ),
        )
        metric_weights.setdefault(
            "learning_rate", float(context_data.get("learning_rate_penalty", 0.0))
        )
        metrics.setdefault(
            "memory_budget_scalar",
            float(context_data.get("memory_budget", _DEFAULT_BLUEPRINT_TEMPLATE["memory_budget_scalar"])),
        )
        metric_weights.setdefault(
            "memory_budget_scalar", float(context_data.get("memory_pressure", 0.0))
        )
        reward_metrics = {
            key: float(value)
            for key, value in (context_data.get("reward_metrics", {}) or {}).items()
            if isinstance(value, (int, float))
        }
        reward_scale = float(context_data.get("reward_weight", 1.0))

        def score(params: Dict[str, float]) -> float:
            mse = self._mean_squared_error(dataset, params)
            complexity = sum(abs(params[k]) for k in params if "weight" in k) * 0.01
            metric_penalty = 0.0
            for name, target in metrics.items():
                weight = metric_weights.get(name, 0.0)
                if weight == 0.0:
                    continue
                diff = params.get(name, target) - target
                metric_penalty += weight * diff * diff

            metric_reward = sum(
                reward_scale * metric_weights.get(name, 1.0) * reward_metrics[name]
                for name in reward_metrics
            )

            return -(mse + complexity + metric_penalty) + metric_reward

        return score

    # ------------------------------------------------------------------ #
    def _mean_squared_error(self, dataset: List[Tuple[float, float]], params: Dict[str, float]) -> float:
        activation = _decode_activation(params["activation"])
        total = 0.0
        for x, target in dataset:
            hidden = activation(params["layer1_weight"] * x + params["layer1_bias"])
            output = params["layer2_weight"] * hidden + params["layer2_bias"]
            basis = _polynomial_basis(x)
            poly = (
                params["poly_w0"] * basis[0]
                + params["poly_w1"] * basis[1]
                + params["poly_w2"] * basis[2]
            )
            prediction = output + params["residual_weight"] * x + poly
            error = prediction - target
            total += error * error
        return total / max(1, len(dataset))

    # ------------------------------------------------------------------ #
    def _refine_polynomial_weights(
        self, params: Dict[str, float], dataset: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        try:
            import numpy as np
        except Exception:  # pragma: no cover - fallback when numpy unavailable
            return self._refine_polynomial_weights_manual(params, dataset)

        activation = _decode_activation(params["activation"])
        rows: List[List[float]] = []
        targets: List[float] = []
        for x, target in dataset:
            basis = list(_polynomial_basis(x))
            hidden = activation(params["layer1_weight"] * x + params["layer1_bias"])
            network = params["layer2_weight"] * hidden + params["layer2_bias"]
            network += params["residual_weight"] * x
            rows.append(basis)
            targets.append(target - network)

        matrix = np.asarray(rows, dtype=float)
        vector = np.asarray(targets, dtype=float)
        try:
            weights, _, _, _ = np.linalg.lstsq(matrix, vector, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - ill-conditioned case
            return params

        refined = params.copy()
        refined["poly_w0"], refined["poly_w1"], refined["poly_w2"] = (
            float(weights[0]),
            float(weights[1]),
            float(weights[2]),
        )
        return refined

    # ------------------------------------------------------------------ #
    def _refine_polynomial_weights_manual(
        self, params: Dict[str, float], dataset: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        activation = _decode_activation(params["activation"])
        s_x0 = float(len(dataset))
        s_x1 = 0.0
        s_x2 = 0.0
        s_x3 = 0.0
        s_x4 = 0.0
        b0 = 0.0
        b1 = 0.0
        b2 = 0.0
        for x, target in dataset:
            hidden = activation(params["layer1_weight"] * x + params["layer1_bias"])
            network = params["layer2_weight"] * hidden + params["layer2_bias"]
            importance = params.get("importance_weight", 0.0)
            gate = max(0.0, 1.0 + math.tanh(importance))
            network += params["residual_weight"] * gate * x
            residual = target - network
            s_x1 += x
            x2 = x * x
            s_x2 += x2
            x3 = x2 * x
            s_x3 += x3
            s_x4 += x2 * x2
            b0 += residual
            b1 += residual * x
            b2 += residual * x2

        matrix = [
            [s_x0, s_x1, s_x2],
            [s_x1, s_x2, s_x3],
            [s_x2, s_x3, s_x4],
        ]
        vector = [b0, b1, b2]
        solution = self._solve_linear_system_3x3(matrix, vector)
        if solution is None:
            return params

        refined = params.copy()
        refined["poly_w0"], refined["poly_w1"], refined["poly_w2"] = solution
        return refined

    # ------------------------------------------------------------------ #
    @staticmethod
    def _solve_linear_system_3x3(
        matrix: List[List[float]], vector: List[float]
    ) -> Optional[Tuple[float, float, float]]:
        a = [row[:] for row in matrix]
        b = list(vector)
        n = 3
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
            if abs(a[pivot][i]) < 1e-12:
                return None
            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                b[i], b[pivot] = b[pivot], b[i]
            pivot_val = a[i][i]
            for j in range(i, n):
                a[i][j] /= pivot_val
            b[i] /= pivot_val
            for r in range(n):
                if r == i:
                    continue
                factor = a[r][i]
                if factor == 0.0:
                    continue
                for j in range(i, n):
                    a[r][j] -= factor * a[i][j]
                b[r] -= factor * b[i]
        return float(b[0]), float(b[1]), float(b[2])

    # ------------------------------------------------------------------ #
    def _materialize(
        self,
        signature: str,
        params: Dict[str, float],
        fitness: float,
        dataset: List[Tuple[float, float]],
        history: List[Tuple[Dict[str, float], float]],
        context: Optional[Dict[str, Any]],
    ) -> ModuleSpec:
        mse = self._mean_squared_error(dataset, params)
        quality = 1.0 / (1.0 + mse)
        slug = _slugify(signature)
        name = f"auto_{slug}_{uuid.uuid4().hex[:8]}"
        hyperparameters = {
            "importance_weight": params.get("importance_weight", 1.0),
            "learning_rate": params.get("learning_rate", 0.01),
            "memory_budget": params.get("memory_budget_scalar", 1.0),
        }
        metadata = {"dataset_size": len(dataset), "hyperparameters": hyperparameters}
        if context:
            metadata["context"] = dict(context)
        blueprint = ModuleBlueprint(
            signature=signature,
            parameters=params,
            fitness=fitness,
            error=mse,
            metadata=metadata,
        )
        register_module(name, lambda blueprint=blueprint: EvolvedCapabilityModule(blueprint))
        spec = ModuleSpec(name=name, blueprint=blueprint, quality=quality, history=history)
        logger.info("Registered evolved module %s for '%s' with quality %.4f", name, signature, quality)
        return spec

    # ------------------------------------------------------------------ #
    def _prepare_dataset(self, dataset: SimulationDataset) -> List[Tuple[float, float]]:
        prepared = [(float(x), float(y)) for x, y in dataset]
        if not prepared:
            raise ValueError("Simulation dataset is empty; cannot evolve module.")
        return prepared

    # ------------------------------------------------------------------ #
    def catalog(self) -> Dict[str, ModuleSpec]:
        """Return a snapshot of the generated module catalog."""
        return dict(self._catalog)

    # ------------------------------------------------------------------ #
    def has_module(self, signature: str) -> bool:
        """Return True if ``signature`` already has an evolved module."""
        return signature in self._catalog

    # ------------------------------------------------------------------ #
    def spawn_module(
        self,
        signature: str,
        dataset: SimulationDataset,
        min_quality: float = 0.6,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModuleSpec:
        """Generate (or return) a module specialised for ``signature``.

        Parameters
        ----------
        signature:
            Problem signature describing the capability gap.
        dataset:
            Iterable of ``(input, target)`` samples used to evaluate candidates.
        min_quality:
            Minimum acceptable quality (in [0, 1]). If the best candidate falls
            below this threshold a :class:`RuntimeError` is raised.
        context:
            Optional dictionary providing architectural metrics and target
            hyperparameters that influence the evolutionary search.
        """

        if signature in self._catalog:
            return self._catalog[signature]

        prepared = self._prepare_dataset(dataset)
        fitness_fn = self._fitness_fn(prepared, context=context)
        ga = GeneticAlgorithm(fitness_fn, self._ga_config, seed=self._seed)
        seed_blueprint = self._initial_blueprint()
        best_params, best_fitness, history = ga.evolve(seed_blueprint)
        best_params = self._refine_polynomial_weights(best_params, prepared)
        best_fitness = fitness_fn(best_params)

        spec = self._materialize(signature, best_params, best_fitness, prepared, history, context)
        if spec.quality < min_quality:
            raise RuntimeError(
                f"Generated module quality {spec.quality:.3f} below threshold {min_quality:.3f}"
            )

        self._catalog[signature] = spec
        return spec

    # ------------------------------------------------------------------ #
    def available(self) -> List[str]:
        """List registered module names, including evolved modules."""
        return available_modules()


# ---------------------------------------------------------------------------
# Growth co-ordinator
# ---------------------------------------------------------------------------


class ModuleGrowthController:
    """Coordinate module generation, loading and architectural integration."""

    def __init__(
        self,
        factory: EvolutionaryModuleFactory,
        loader: RuntimeModuleManager,
        expander: "DynamicArchitectureExpander | None" = None,
    ) -> None:
        self._factory = factory
        self._loader = loader
        self._expander = expander

    def ensure_module(
        self,
        signature: str,
        dataset: SimulationDataset,
        parent: Optional[str] = None,
        min_quality: float = 0.6,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModuleSpec:
        """Ensure a specialised module exists and is wired into the topology."""

        spec = self._factory.spawn_module(
            signature, dataset, min_quality=min_quality, context=context
        )
        module = self._loader.load(spec.name)

        if self._expander is not None:
            self._expander.add_module(spec.name, module)
            if parent is not None:
                self._expander.connect(parent, spec.name)

        return spec


__all__ = [
    "ModuleBlueprint",
    "ModuleSpec",
    "EvolutionaryModuleFactory",
    "EvolvedCapabilityModule",
    "ModuleGrowthController",
]
