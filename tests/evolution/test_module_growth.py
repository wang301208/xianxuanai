import math
from statistics import mean

from backend.capability.module_factory import (
    EvolutionaryModuleFactory,
    ModuleBlueprint,
    ModuleGrowthController,
    EvolvedCapabilityModule,
)
from backend.capability.runtime_loader import RuntimeModuleManager
from modules.evolution.evolving_cognitive_architecture import GAConfig


from typing import Callable, Dict, List, Optional, Set


class StubExpander:
    def __init__(self, modules: Dict[str, Callable[[float], float]], connections: Optional[Dict[str, List[str]]] = None) -> None:
        self.modules = modules
        self.connections = connections or {name: [] for name in modules}

    def add_module(self, name: str, module) -> None:
        self.modules[name] = module
        self.connections.setdefault(name, [])

    def connect(self, src: str, dst: str) -> None:
        self.connections.setdefault(src, [])
        if dst not in self.connections[src]:
            self.connections[src].append(dst)

    def run(self, start: str, value: float) -> float:
        visited: set[str] = set()

        def _run(node: str, current: float) -> float:
            visited.add(node)
            output = self.modules[node](current)
            for nxt in self.connections.get(node, []):
                if nxt not in visited:
                    output = _run(nxt, output)
            return output

        return _run(start, value)


def _target_function(x: float) -> float:
    return 2.0 * x * x - 3.0 * x + 1.0


def _dataset() -> list[tuple[float, float]]:
    return [(float(val), _target_function(float(val))) for val in (-3, -2, -1, 0, 1, 2, 3)]


def _mean_squared_error(pairs: list[tuple[float, float]]) -> float:
    errors = [(pred - target) ** 2 for pred, target in pairs]
    return mean(errors)


def test_evolutionary_factory_registers_specialised_module():
    factory = EvolutionaryModuleFactory(
        ga_config=GAConfig(population_size=25, generations=8, mutation_sigma=0.25, mutation_rate=0.4),
        seed=42,
    )
    spec = factory.spawn_module("quadratic-anomaly", _dataset(), min_quality=0.2)

    manager = RuntimeModuleManager()
    module = manager.load(spec.name)

    samples = [(-4.0, _target_function(-4.0)), (1.5, _target_function(1.5)), (3.5, _target_function(3.5))]
    evaluated = [(module(x), target) for x, target in samples]
    mse = _mean_squared_error(evaluated)
    assert mse < 1.0


def test_growth_controller_wires_module_into_architecture():
    factory = EvolutionaryModuleFactory(
        ga_config=GAConfig(population_size=20, generations=6, mutation_sigma=0.35),
        seed=123,
    )
    manager = RuntimeModuleManager()

    expander = StubExpander(modules={"root": lambda x: x})
    controller = ModuleGrowthController(factory, manager, expander)

    spec = controller.ensure_module("quadratic-routing", _dataset(), parent="root", min_quality=0.2)
    assert spec.name in expander.modules

    value = 2.25
    result = expander.run("root", value)
    expected = _target_function(value)
    assert math.isfinite(result)
    assert abs(result - expected) < 1.5


def test_genome_includes_hyperparameters_and_mutates_with_context():
    context = {
        "target_importance": 0.3,
        "importance_weight_penalty": 5.0,
        "target_learning_rate": 0.05,
        "learning_rate_penalty": 5.0,
        "memory_budget": 0.4,
        "memory_pressure": 3.0,
        "architectural_metrics": {
            "importance_weight": 0.3,
            "learning_rate": 0.05,
            "memory_budget_scalar": 0.4,
        },
        "metric_weights": {
            "importance_weight": 5.0,
            "learning_rate": 5.0,
            "memory_budget_scalar": 3.0,
        },
    }
    factory = EvolutionaryModuleFactory(
        ga_config=GAConfig(population_size=30, generations=10, mutation_sigma=0.3, mutation_rate=0.45),
        seed=99,
    )
    baseline = factory._initial_blueprint()
    spec = factory.spawn_module(
        "quadratic-hyper-meta", _dataset(), min_quality=0.2, context=context
    )

    params = spec.blueprint.parameters
    for key in ("importance_weight", "learning_rate", "memory_budget_scalar"):
        assert key in params
    targets = {
        "importance_weight": context["target_importance"],
        "learning_rate": context["target_learning_rate"],
        "memory_budget_scalar": context["memory_budget"],
    }
    improvements = 0
    for key, target in targets.items():
        diff = abs(params[key] - target)
        baseline_diff = abs(baseline[key] - target)
        assert diff <= baseline_diff + 0.05
        if diff < baseline_diff:
            improvements += 1

    assert improvements >= 1

    # Confirm evolutionary history explored variations in new genes.
    mutated = False
    for candidate, _ in spec.history[1:]:
        if any(abs(candidate[key] - baseline[key]) > 1e-6 for key in targets):
            mutated = True
            break
    assert mutated, "expected genome fields to mutate during search"

    # Hyperparameters should be surfaced via metadata.
    hyper = spec.blueprint.metadata["hyperparameters"]
    assert math.isclose(hyper["learning_rate"], params["learning_rate"], rel_tol=1e-6)
    assert spec.blueprint.metadata["context"]["target_importance"] == context["target_importance"]


def test_evolved_module_applies_importance_gate_and_exposes_hyperparameters():
    params = {
        "layer1_weight": 0.0,
        "layer1_bias": 0.0,
        "layer2_weight": 0.0,
        "layer2_bias": 0.0,
        "activation": 0.0,
        "residual_weight": 1.5,
        "poly_w0": 0.0,
        "poly_w1": 0.0,
        "poly_w2": 0.0,
        "importance_weight": 2.0,
        "learning_rate": 0.05,
        "memory_budget_scalar": 0.5,
    }
    blueprint = ModuleBlueprint(
        signature="manual-test",
        parameters=params,
        fitness=0.0,
        error=0.0,
        metadata={},
    )
    module = EvolvedCapabilityModule(blueprint)

    gate = max(0.0, 1.0 + math.tanh(params["importance_weight"]))
    value = 2.0
    expected = gate * params["residual_weight"] * value
    assert math.isclose(module.forward(value), expected, rel_tol=1e-6)

    hypers = module.get_hyperparameters()
    assert math.isclose(hypers["learning_rate"], params["learning_rate"], rel_tol=1e-12)
    assert math.isclose(hypers["memory_budget"], params["memory_budget_scalar"], rel_tol=1e-12)

    # A second module with low importance should down-gate the residual response.
    low_params = dict(params)
    low_params["importance_weight"] = -3.0
    low_blueprint = ModuleBlueprint(
        signature="manual-test",
        parameters=low_params,
        fitness=0.0,
        error=0.0,
        metadata={},
    )
    low_module = EvolvedCapabilityModule(low_blueprint)
    assert module.forward(value) > low_module.forward(value)
