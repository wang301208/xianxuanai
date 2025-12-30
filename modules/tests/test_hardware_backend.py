import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from ml.backends import get_backend  # noqa: E402


def test_default_backend_cpu():
    backend = get_backend()
    assert backend.name == 'cpu'


def test_gpu_fallback_to_cpu():
    backend = get_backend('gpu')
    assert backend.name == 'cpu'


def test_cpu_matmul():
    backend = get_backend('cpu')
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = backend.matmul(a, b)
    np.testing.assert_array_equal(result, np.array([[19, 22], [43, 50]]))
