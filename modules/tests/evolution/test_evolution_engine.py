import os
import random
import sys
import types

sys.path.insert(0, os.path.abspath(os.getcwd()))

if "psutil" not in sys.modules:  # pragma: no cover - test helper stub
    class _DummyProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def cpu_percent(self, interval=None):
            return 0.0

        def memory_percent(self):
            return 0.0

    psutil_stub = types.SimpleNamespace(
        Process=lambda pid: _DummyProcess(pid),
        NoSuchProcess=RuntimeError,
        AccessDenied=RuntimeError,
    )
    sys.modules["psutil"] = psutil_stub


if "fastapi" not in sys.modules:  # pragma: no cover - test helper stub
    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.routes = {}

        def get(self, path):
            def decorator(func):
                self.routes[("GET", path)] = func
                return func

            return decorator

        def post(self, path):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

    sys.modules["fastapi"] = types.SimpleNamespace(FastAPI=_DummyFastAPI)


if "matplotlib" not in sys.modules:  # pragma: no cover - test helper stub
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = matplotlib_stub
else:
    matplotlib_stub = sys.modules["matplotlib"]


if "matplotlib.pyplot" not in sys.modules:  # pragma: no cover - test helper stub
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):
        return None

    for attr in (
        "figure",
        "plot",
        "close",
        "subplots",
        "tight_layout",
        "savefig",
        "imshow",
        "title",
        "xlabel",
        "ylabel",
        "legend",
    ):
        setattr(pyplot_stub, attr, _noop)

    sys.modules["matplotlib.pyplot"] = pyplot_stub
    setattr(matplotlib_stub, "pyplot", pyplot_stub)


if "numpy" not in sys.modules:  # pragma: no cover - test helper stub
    numpy_stub = types.ModuleType("numpy")

    class _DummyArray(list):
        def tolist(self):
            return list(self)

    def _as_array(value, dtype=None):
        if isinstance(value, _DummyArray):
            return value
        if isinstance(value, (list, tuple)):
            return _DummyArray(value)
        return _DummyArray([value])

    def _zeros(shape, dtype=None):
        length = 0
        if isinstance(shape, int):
            length = shape
        elif isinstance(shape, (list, tuple)) and shape:
            length = int(shape[0])
        return _DummyArray([0.0] * max(length, 0))

    numpy_stub.ndarray = _DummyArray
    numpy_stub.array = _as_array
    numpy_stub.asarray = _as_array
    numpy_stub.zeros = _zeros
    numpy_stub.ones = lambda shape, dtype=None: _DummyArray([1.0] * (shape if isinstance(shape, int) else int(shape[0]) if isinstance(shape, (list, tuple)) and shape else 0))
    numpy_stub.float32 = float
    numpy_stub.float64 = float
    numpy_stub.int32 = int
    numpy_stub.dot = lambda a, b: 0.0
    numpy_stub.linalg = types.SimpleNamespace(norm=lambda x: 0.0)

    sys.modules["numpy"] = numpy_stub


if "yaml" not in sys.modules:  # pragma: no cover - test helper stub
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda data: {}
    yaml_stub.safe_dump = lambda data, *args, **kwargs: ""
    sys.modules["yaml"] = yaml_stub


if "PIL" not in sys.modules:  # pragma: no cover - test helper stub
    pil_stub = types.ModuleType("PIL")
    sys.modules["PIL"] = pil_stub
else:
    pil_stub = sys.modules["PIL"]


if "PIL.Image" not in sys.modules:  # pragma: no cover - test helper stub
    image_stub = types.ModuleType("PIL.Image")

    class _DummyImage:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            return None

        def resize(self, *args, **kwargs):
            return self

        def convert(self, *args, **kwargs):
            return self

        def copy(self):
            return self

        def load(self):
            return None

    image_stub.Image = _DummyImage
    image_stub.open = lambda *args, **kwargs: _DummyImage()
    image_stub.new = lambda *args, **kwargs: _DummyImage()
    image_stub.fromarray = lambda *args, **kwargs: _DummyImage()

    sys.modules["PIL.Image"] = image_stub
    pil_stub.Image = image_stub

from modules.evolution import (
    EvolutionEngine,
    EvolutionGeneticAlgorithm,
    SpecialistModule,
    TaskContext,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig
from modules.monitoring.collector import MetricEvent


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def test_multi_cycle_evolution_and_rollback():
    random.seed(0)
    ga = EvolutionGeneticAlgorithm(
        fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5)
    )
    engine = EvolutionEngine({"weight": 0.0}, fitness_fn, ga)

    # Run several evolution cycles using feedback derived from current performance
    for step in range(5):
        perf = fitness_fn(engine.cognition.architecture)
        metrics = [
            MetricEvent(
                module="evolve",
                latency=0.0,
                energy=0.0,
                throughput=perf,
                timestamp=float(step),
            )
        ]
        engine.run_evolution_cycle(metrics)

    performances = [rec.performance for rec in engine.history()]
    assert performances[-1] > performances[0]

    # Verify rollback restores initial architecture
    initial_arch = engine.history()[0].architecture
    engine.rollback(0)
    assert engine.cognition.architecture == initial_arch


def test_specialist_module_selected_when_capability_matches():
    random.seed(1)
    ga = EvolutionGeneticAlgorithm(
        fitness_fn, GAConfig(population_size=6, generations=3, mutation_sigma=0.3)
    )
    engine = EvolutionEngine({"weight": -0.5}, fitness_fn, ga)

    specialist = SpecialistModule(
        name="strategy_master",
        capabilities={"strategy", "planning"},
        solver=lambda arch, task: {**arch, "weight": 1.0},
        priority=1.0,
    )
    engine.register_specialist_module(specialist)

    metrics = [
        MetricEvent(
            module="evolve",
            latency=0.0,
            energy=0.0,
            throughput=-0.25,
            timestamp=0.0,
        )
    ]
    task = TaskContext(name="strategy optimisation", required_capabilities=("strategy",))

    engine.run_evolution_cycle(metrics, task=task)

    assert engine.cognition.architecture["weight"] == 1.0
    record = engine.history()[-1]
    assert record.metrics["source"] == "specialist"
    assert record.metrics["specialist_module"] == "strategy_master"


def test_specialist_module_ignored_without_matching_capability():
    random.seed(2)
    ga = EvolutionGeneticAlgorithm(
        fitness_fn, GAConfig(population_size=6, generations=3, mutation_sigma=0.3)
    )
    engine = EvolutionEngine({"weight": -0.5}, fitness_fn, ga)

    specialist = SpecialistModule(
        name="vision_master",
        capabilities={"vision"},
        solver=lambda arch, task: {**arch, "weight": 1.0},
        priority=2.0,
    )
    engine.register_specialist_module(specialist)

    metrics = [
        MetricEvent(
            module="evolve",
            latency=0.0,
            energy=0.0,
            throughput=-0.25,
            timestamp=0.0,
        )
    ]
    task = TaskContext(name="strategy optimisation", required_capabilities=("strategy",))

    engine.run_evolution_cycle(metrics, task=task)

    record = engine.history()[-1]
    assert record.metrics["source"] == "genetic"
    assert "specialist_module" not in record.metrics
