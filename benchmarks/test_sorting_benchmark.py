import random
from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort


def _generate_data(size: int = 200) -> list[int]:
    random.seed(0)
    return random.sample(range(size * 5), size)


def test_bubble_sort_benchmark(benchmark) -> None:
    data = _generate_data()
    algo = BubbleSort()
    benchmark(algo.execute, data)


def test_quick_sort_benchmark(benchmark) -> None:
    data = _generate_data()
    algo = QuickSort()
    benchmark(algo.execute, data)
