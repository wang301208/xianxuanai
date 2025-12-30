import math

from modules.benchmark.ab_testing import ABTester


def test_ab_tester_runs(tmp_path):
    def model_a(x):
        return x + 1

    def model_b(x):
        return x * 2

    def scorer(result):
        return float(result)

    tester = ABTester(scorer)
    result = tester.run("A", model_a, "B", model_b, tasks=[1, 2, 3])

    summary = result.summary()
    assert math.isclose(summary["A"], 3.0)  # mean of [2,3,4]
    assert math.isclose(summary["B"], 4.0)  # mean of [2,4,6]
