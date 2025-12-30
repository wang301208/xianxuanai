import pytest

from algorithms.sorting.advanced.merge_sort import MergeSort


def test_merge_sort_basic():
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert MergeSort().execute(data) == sorted(data)
    assert data == original


def test_merge_sort_empty_list():
    assert MergeSort().execute([]) == []


def test_merge_sort_duplicates():
    data = [3, 1, 2, 3, 1]
    assert MergeSort().execute(data) == sorted(data)


def test_merge_sort_sorted_sequence():
    data = [1, 2, 3, 4, 5]
    assert MergeSort().execute(data) == data
