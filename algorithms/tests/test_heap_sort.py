from algorithms.sorting.advanced.heap_sort import HeapSort


def test_heap_sort_ascending():
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert HeapSort().execute(data) == sorted(data)
    assert data == original


def test_heap_sort_descending():
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert HeapSort().execute(data, reverse=True) == sorted(data, reverse=True)
    assert data == original


def test_heap_sort_empty_and_single():
    sorter = HeapSort()
    assert sorter.execute([]) == []
    assert sorter.execute([], reverse=True) == []
    assert sorter.execute([1]) == [1]
    assert sorter.execute([1], reverse=True) == [1]
