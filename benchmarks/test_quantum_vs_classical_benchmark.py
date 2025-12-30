import numpy as np
from modules.brain.quantum.grover_search import grover_search
from modules.brain.quantum.quantum_ml import QuantumClassifier


def classical_search(target, items):
    for item in items:
        if item == target:
            return item
    raise ValueError("target not found")


def classical_classifier_train(X, y):
    centroids = {label: X[y == label].mean(axis=0) for label in np.unique(y)}

    def predict(sample):
        return min(centroids, key=lambda c: np.linalg.norm(sample - centroids[c]))

    return predict


# --- correctness checks ---------------------------------------------------------

def test_search_correctness():
    data = list(range(2**10))
    target = data[-1]
    assert classical_search(target, data) == grover_search(data, lambda x: x == target)


def test_classifier_correctness():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    qc = QuantumClassifier()
    qc.train(X, y)
    classical_predict = classical_classifier_train(X, y)
    sample = np.array([0.2, 0.2])
    assert qc.predict(sample) == classical_predict(sample)


# --- benchmarks ----------------------------------------------------------------

def test_classical_search_benchmark(benchmark):
    data = list(range(2**10))
    target = data[-1]
    benchmark(lambda: classical_search(target, data))


def test_grover_search_benchmark(benchmark):
    data = list(range(2**10))
    target = data[-1]
    benchmark(lambda: grover_search(data, lambda x: x == target))


def test_classical_classifier_benchmark(benchmark):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    classical_predict = classical_classifier_train(X, y)
    sample = np.array([0.2, 0.2])
    benchmark(classical_predict, sample)


def test_quantum_classifier_benchmark(benchmark):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    qc = QuantumClassifier()
    qc.train(X, y)
    sample = np.array([0.2, 0.2])
    benchmark(qc.predict, sample)
