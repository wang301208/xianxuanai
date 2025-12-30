import os
import sys

import math
import random
import pytest
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

if "psutil" not in sys.modules:  # pragma: no cover - stub for optional dependency
    class _DummyProcess:
        def __init__(self, pid: int | None = None) -> None:
            self.pid = pid or 0

        def cpu_percent(self, interval=None):
            return 0.0

        def memory_percent(self):
            return 0.0

        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

    sys.modules["psutil"] = types.SimpleNamespace(
        Process=lambda pid=None: _DummyProcess(pid),
        NoSuchProcess=RuntimeError,
        AccessDenied=RuntimeError,
    )


if "fastapi" not in sys.modules:  # pragma: no cover - stub FastAPI when missing
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


if "matplotlib" not in sys.modules:  # pragma: no cover - simple matplotlib stub
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = matplotlib_stub
else:
    matplotlib_stub = sys.modules["matplotlib"]


if "matplotlib.pyplot" not in sys.modules:  # pragma: no cover - pyplot stub
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


if "yaml" not in sys.modules:  # pragma: no cover - minimal yaml stub
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda data: {}
    yaml_stub.safe_dump = lambda data, *args, **kwargs: ""
    sys.modules["yaml"] = yaml_stub


if "PIL" not in sys.modules:  # pragma: no cover - PIL stubs
    pil_stub = types.ModuleType("PIL")
    sys.modules["PIL"] = pil_stub
else:
    pil_stub = sys.modules["PIL"]


if "PIL.Image" not in sys.modules:
    image_stub = types.ModuleType("PIL.Image")

    class _DummyImage:
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


if "numpy" not in sys.modules:  # pragma: no cover - lightweight stub
    numpy_stub = types.ModuleType("numpy")

    class _Array(list):
        def tolist(self):
            return list(self)

        def _binary_op(self, other, op):
            if isinstance(other, (list, tuple, _Array)):
                return _Array(op(float(a), float(b)) for a, b in zip(self, other))
            if isinstance(other, (int, float)):
                return _Array(op(float(a), float(other)) for a in self)
            return NotImplemented

        def __mul__(self, other):
            return self._binary_op(other, lambda a, b: a * b)

        __rmul__ = __mul__

        def __add__(self, other):
            return self._binary_op(other, lambda a, b: a + b)

        __radd__ = __add__

        def __sub__(self, other):
            return self._binary_op(other, lambda a, b: a - b)

        def __rsub__(self, other):
            if isinstance(other, (int, float)):
                return _Array(float(other) - float(a) for a in self)
            if isinstance(other, (list, tuple, _Array)):
                return _Array(float(a) - float(b) for a, b in zip(other, self))
            return NotImplemented

    def _as_array(value):
        if isinstance(value, _Array):
            return value
        if isinstance(value, (list, tuple)):
            return _Array(float(v) for v in value)
        return _Array([float(value)])

    numpy_stub.ndarray = _Array
    numpy_stub.array = lambda value, dtype=None: _as_array(value)
    numpy_stub.asarray = lambda value, dtype=None: _as_array(value)
    numpy_stub.minimum = lambda a, b: _Array(min(float(x), float(y)) for x, y in zip(_as_array(a), _as_array(b)))
    numpy_stub.maximum = lambda a, b: _Array(max(float(x), float(y)) for x, y in zip(_as_array(a), _as_array(b)))
    numpy_stub.sum = lambda value: float(sum(float(v) for v in _as_array(value)))
    numpy_stub.square = lambda value: _Array(float(v) * float(v) for v in _as_array(value))
    numpy_stub.cos = lambda value: _Array(math.cos(float(v)) for v in _as_array(value))
    numpy_stub.pi = math.pi
    numpy_stub.float32 = float
    numpy_stub.float64 = float
    numpy_stub.int32 = int
    numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float))
    numpy_stub.bool_ = bool
    numpy_stub.zeros = lambda shape, dtype=None: _Array(0.0 for _ in range(shape if isinstance(shape, int) else int(shape[0])))
    numpy_stub.ones = lambda shape, dtype=None: _Array(1.0 for _ in range(shape if isinstance(shape, int) else int(shape[0])))

    class _Random:
        def __init__(self, seed=None):
            self._rng = random.Random(seed)

        def uniform(self, low, high):
            lows = _as_array(low)
            highs = _as_array(high)
            return _Array(self._rng.uniform(float(lo), float(hi)) for lo, hi in zip(lows, highs))

        def normal(self, mean, std, size=None):
            means = _as_array(mean)
            stds = _as_array(std)
            count = size[0] if size else 1
            samples = []
            for _ in range(count):
                samples.append(
                    _Array(self._rng.gauss(float(m), float(s)) for m, s in zip(means, stds))
                )
            return _Array(samples)

    numpy_stub.random = types.SimpleNamespace(default_rng=lambda seed=None: _Random(seed))
    numpy_stub.argmin = lambda values: min(range(len(values)), key=lambda i: values[i])

    sys.modules["numpy"] = numpy_stub

from backend.algorithms import evolution_engine
from modules.benchmarks.problems import Sphere
from modules.evolution import SpecialistModule


def test_evolution_engine_algorithm_selects_specialist():
    problem = Sphere(dim=2, bound=2.0)

    specialist = SpecialistModule(
        name="sphere_opt_master",
        capabilities={"global_optimum", problem.name},
        solver=lambda arch, task: {f"x{i}": 0.0 for i in range(problem.dim)},
        priority=10.0,
    )

    best_x, best_val, iterations, elapsed = evolution_engine.optimize(
        problem,
        seed=42,
        max_iters=5,
        specialists=[specialist],
        task_capabilities=("global_optimum", problem.name),
    )

    if hasattr(best_x, "tolist"):
        vector = [float(v) for v in best_x.tolist()]
    else:
        vector = [float(v) for v in best_x]

    for value in vector:
        assert math.isclose(value, 0.0, abs_tol=1e-6)
    assert best_val == pytest.approx(problem.optimum_value, abs=1e-6)
    assert iterations > 0
    assert elapsed >= 0.0
    assert specialist.usage_count >= 1
