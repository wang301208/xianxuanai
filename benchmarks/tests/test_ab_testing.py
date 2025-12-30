"""Tests for the A/B testing utilities."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from benchmarks.ab_testing import ABTestConfig, run_ab_test, significance_test


def algo_random(X, y):
    model = DummyClassifier(strategy="uniform", random_state=0)
    model.fit(X, y)
    return model.predict(X)


def algo_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model.predict(X)


def test_knn_beats_random() -> None:
    X, y = make_classification(n_samples=200, n_features=20, n_informative=15, random_state=0)
    config = ABTestConfig(algo_a=algo_random, algo_b=algo_knn, data=(X, y), name_a="random", name_b="knn")
    result = run_ab_test(config)
    assert result.algo_b.accuracy >= result.algo_a.accuracy
    t_stat, p_val = significance_test(result.algo_a.correctness, result.algo_b.correctness)
    assert p_val < 0.05
