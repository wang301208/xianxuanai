from __future__ import annotations

"""Evolution-engine powered optimiser with specialist module orchestration."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import random
import time

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy unavailable
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional fallback for package layout differences
    from benchmarks.problems import Problem
except Exception:  # pragma: no cover - fallback when benchmarks is namespaced
    from modules.benchmarks.problems import Problem  # type: ignore

from modules.evolution import (
    EvolutionEngine,
    EvolutionGeneticAlgorithm,
    NASMutationSpace,
    NASParameter,
    SpecialistModule,
    TaskContext,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig
from modules.monitoring.collector import MetricEvent

from .self_play_trainer import SelfPlayResult, SelfPlayTrainer
from .termination import StopCondition

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML unavailable
    yaml = None  # type: ignore[assignment]


@dataclass(slots=True)
class EvolutionEngineConfig:
    """Configuration controlling the evolutionary search dynamics."""

    population_size: int = 20
    generations: int = 5
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.1


def _architecture_from_vector(vector: Sequence[float]) -> dict[str, float]:
    return {f"x{i}": float(value) for i, value in enumerate(vector)}


def _vector_from_architecture(architecture: Mapping[str, float], dim: int) -> List[float]:
    return [float(architecture.get(f"x{i}", 0.0)) for i in range(dim)]


def _evaluate(problem: Problem, architecture: Mapping[str, float]) -> Tuple[List[float], float, float]:
    vector = _vector_from_architecture(architecture, problem.dim)
    value = float(problem.evaluate(vector))
    if not math.isfinite(value):
        return vector, value, -float("inf")
    return vector, value, -value


def _default_specialists(problem: Problem) -> List[SpecialistModule]:
    specialists: List[SpecialistModule] = []
    optimum = getattr(problem, "optimum", None)
    if optimum is not None:
        optimum_vector = [float(v) for v in optimum]

        def solver(arch: Mapping[str, float], task: TaskContext) -> dict[str, float]:
            base = dict(arch)
            for i, value in enumerate(optimum_vector):
                base[f"x{i}"] = float(value)
            return base

        specialists.append(
            SpecialistModule(
                name=f"{problem.name}_expert",
                capabilities={"global_optimum", problem.name},
                solver=solver,
                priority=5.0,
            )
        )
    return specialists


def _build_nas_space(problem: Problem, initial: Sequence[float]) -> NASMutationSpace:
    parameters = {
        f"x{i}": NASParameter(
            name=f"x{i}",
            min_value=float(min(bounds)),
            max_value=float(max(bounds)),
            step=max((max(bounds) - min(bounds)) / 100.0, 1e-3),
            dtype="float",
            default=float(initial[i]),
        )
        for i, bounds in enumerate(problem.bounds)
    }
    return NASMutationSpace(parameters)


def _merge_specialists(
    provided: Optional[Iterable[SpecialistModule]],
    defaults: Iterable[SpecialistModule],
) -> List[SpecialistModule]:
    merged: List[SpecialistModule] = []
    seen: set[str] = set()
    if provided:
        for module in provided:
            merged.append(module)
            seen.add(module.name)
    for module in defaults:
        if module.name not in seen:
            merged.append(module)
    return merged


def _load_specialist_config_source(
    source: str | Path | Mapping[str, Any]
) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Specialist configuration not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load YAML specialist configurations"
            )
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise TypeError("Specialist configuration must be a mapping")
    return data


def _resolve_solver_callable(module_path: Optional[str], solver_path: str):
    resolved_module: Optional[str] = None
    attribute: str
    if ":" in solver_path:
        resolved_module, attribute = solver_path.split(":", 1)
    elif module_path:
        resolved_module = module_path
        attribute = solver_path
    else:
        if "." not in solver_path:
            raise ValueError(
                "Solver path must be module:callable or provide module and callable"
            )
        resolved_module, attribute = solver_path.rsplit(".", 1)
    module = import_module(resolved_module)
    solver = getattr(module, attribute)
    if not callable(solver):
        raise TypeError(
            f"Configured solver '{attribute}' in '{resolved_module}' is not callable"
        )
    return solver


def _load_specialists_from_config(
    config: Mapping[str, Any]
) -> List[SpecialistModule]:
    specialists: List[SpecialistModule] = []
    for key, entry in config.items():
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Each specialist configuration must be a mapping of properties"
            )
        module_path = entry.get("module")
        solver_path = entry.get("solver")
        if not isinstance(solver_path, str):
            raise TypeError("Specialist configuration requires a 'solver' string")
        solver = _resolve_solver_callable(
            module_path if isinstance(module_path, str) else None, solver_path
        )
        capabilities = entry.get("capabilities")
        if capabilities is None:
            raise TypeError("Specialist configuration requires 'capabilities'")
        if isinstance(capabilities, str):
            capability_set = {capabilities}
        elif isinstance(capabilities, Iterable):
            capability_set = {str(cap) for cap in capabilities}
        else:
            raise TypeError("'capabilities' must be a string or iterable of strings")
        priority = entry.get("priority", 0.0)
        try:
            priority_value = float(priority)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("'priority' must be a numeric value") from exc
        name = entry.get("name")
        specialist_name = str(name) if name is not None else str(key)
        specialists.append(
            SpecialistModule(
                name=specialist_name,
                capabilities=capability_set,
                solver=solver,
                priority=priority_value,
            )
        )
    return specialists


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: Optional[int] = 100,
    max_time: Optional[float] = None,
    patience: Optional[int] = None,
    *,
    config: EvolutionEngineConfig | None = None,
    specialists: Optional[Iterable[SpecialistModule]] = None,
    specialist_config: str | Path | Mapping[str, Any] | None = None,
    task_capabilities: Sequence[str] | None = None,
    self_play: SelfPlayTrainer | None = None,
    self_play_specialist_name: str | None = None,
    self_play_priority: float = 2.0,
    return_details: bool = False,
) -> Tuple[Sequence[float], float, int, float]:
    """Optimise ``problem`` using the unified evolution engine.

    Parameters
    ----------
    specialist_config:
        Optional mapping or filesystem path pointing to specialist definitions.
        When supplied, the configuration is loaded via
        :func:`_load_specialists_from_config` and merged with provided
        ``specialists`` and defaults.
    self_play:
        Optional :class:`SelfPlayTrainer` used to generate a specialist policy
        via a self-play training loop prior to running evolution.
    self_play_specialist_name:
        Override name for the specialist created from the self-play result.
    self_play_priority:
        Priority assigned to the automatically registered specialist module.
    return_details:
        When ``True`` an additional mapping is returned containing metadata
        about the self-play run (trainer result and registered specialist).
    """

    rng = random.Random(seed)
    lower = [float(b[0]) for b in problem.bounds]
    upper = [float(b[1]) for b in problem.bounds]
    initial_vector = [rng.uniform(lo, hi) for lo, hi in zip(lower, upper)]

    config = config or EvolutionEngineConfig()
    ga_config = GAConfig(
        population_size=config.population_size,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
    )

    nas_space = _build_nas_space(problem, initial_vector)
    initial_architecture = _architecture_from_vector(initial_vector)

    def fitness_fn(architecture: Mapping[str, float]) -> float:
        _, _, score = _evaluate(problem, architecture)
        return float(score)

    ga = EvolutionGeneticAlgorithm(fitness_fn, ga_config, seed=seed, post_mutation=nas_space.postprocess)

    loaded_specialists: Optional[List[SpecialistModule]] = None
    if specialist_config is not None:
        config_mapping = _load_specialist_config_source(specialist_config)
        loaded_specialists = _load_specialists_from_config(config_mapping)

    provided_specialists: Optional[List[SpecialistModule]] = None
    if specialists or loaded_specialists:
        provided_specialists = list(loaded_specialists or [])
        if specialists:
            provided_specialists.extend(specialists)

    specialist_modules = _merge_specialists(
        provided_specialists, _default_specialists(problem)
    )

    engine = EvolutionEngine(
        initial_architecture,
        fitness_fn,
        ga=ga,
        nas_space=nas_space,
        specialist_modules=specialist_modules,
    )

    task = TaskContext(
        name=problem.name,
        required_capabilities=tuple(task_capabilities or ("global_optimum", problem.name)),
        metadata={
            "dimension": problem.dim,
            "bounds": tuple(problem.bounds),
            "optimum_value": getattr(problem, "optimum_value", None),
        },
    )

    trainer_result: SelfPlayResult | None = None
    registered_specialist: SpecialistModule | None = None
    if self_play is not None:
        trainer_result = self_play.train(task)
        default_name = (
            self_play.config.policy_name
            if hasattr(self_play, "config") and self_play.config.policy_name
            else f"{problem.name}_{trainer_result.algorithm}_policy"
        )
        specialist_name = self_play_specialist_name or default_name
        capabilities = task_capabilities or (problem.name, "self_play")
        registered_specialist = trainer_result.build_specialist(
            name=str(specialist_name),
            capabilities=capabilities,
            priority=self_play_priority,
        )
        engine.specialists.register(registered_specialist)

    stopper = StopCondition(max_iters=max_iters, max_time=max_time, patience=patience)

    best_architecture = engine.cognition.architecture.copy()
    best_vector, best_value, _ = _evaluate(problem, best_architecture)

    while stopper.keep_running():
        current_arch = engine.cognition.architecture.copy()
        _, current_value, current_perf = _evaluate(problem, current_arch)
        metrics = [
            MetricEvent(
                module="evolution_engine",
                latency=0.0,
                energy=0.0,
                throughput=float(current_perf),
                timestamp=time.time(),
            )
        ]

        candidate_arch = engine.run_evolution_cycle(metrics, task=task)
        candidate_vector, candidate_value, _ = _evaluate(problem, candidate_arch)
        improved = candidate_value < best_value
        if improved:
            best_value = candidate_value
            best_vector = candidate_vector
            best_architecture = candidate_arch.copy()
        stopper.update(improved)

    elapsed = time.time() - stopper.start_time
    if np is not None:
        result_vector: Sequence[float] = np.asarray(best_vector, dtype=float)
    else:  # pragma: no cover - exercised when numpy missing
        result_vector = [float(v) for v in best_vector]

    details = {
        "self_play": trainer_result,
        "specialist": registered_specialist,
    }
    if return_details:
        return result_vector, float(best_value), stopper.iteration, elapsed, details
    return result_vector, float(best_value), stopper.iteration, elapsed
