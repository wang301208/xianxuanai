import pytest

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort
from algorithms.sorting.basic.selection_sort import SelectionSort
from algorithms.sorting.advanced.merge_sort import MergeSort
from algorithms.sorting.advanced.heap_sort import HeapSort


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), SelectionSort(), MergeSort(), HeapSort()])
def test_sorting_basic(algorithm):
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert algorithm.execute(data) == sorted(data)
    assert data == original


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), SelectionSort(), MergeSort(), HeapSort()])
def test_sorting_empty_and_single(algorithm):
    assert algorithm.execute([]) == []
    assert algorithm.execute([1]) == [1]


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), SelectionSort(), MergeSort(), HeapSort()])
def test_sorting_type_error(algorithm):
    with pytest.raises(TypeError):
        algorithm.execute([1, "a"])
