from algorithms.sorting.basic.selection_sort import SelectionSort


def test_selection_sort_sorted():
    data = [1, 2, 3, 4, 5]
    original = list(data)
    assert SelectionSort().execute(data) == [1, 2, 3, 4, 5]
    assert data == original


def test_selection_sort_reverse():
    data = [5, 4, 3, 2, 1]
    original = list(data)
    assert SelectionSort().execute(data) == [1, 2, 3, 4, 5]
    assert data == original


def test_selection_sort_duplicates():
    data = [3, 1, 2, 3, 1]
    original = list(data)
    assert SelectionSort().execute(data) == [1, 1, 2, 3, 3]
    assert data == original
