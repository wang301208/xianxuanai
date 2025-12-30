import pytest

from algorithms.searching.basic.linear_search import LinearSearch
from algorithms.searching.basic.binary_search import BinarySearch


@pytest.mark.parametrize("searcher", [LinearSearch(), BinarySearch()])
def test_search_found(searcher):
    data = [1, 3, 5, 7, 9]
    target = 7
    assert searcher.execute(data, target) == data.index(target)


def test_binary_search_boundaries():
    searcher = BinarySearch()
    data = [1, 3, 5, 7]
    assert searcher.execute(data, 1) == 0
    assert searcher.execute(data, 7) == len(data) - 1


def test_search_not_found():
    data = [1, 3, 5]
    assert LinearSearch().execute(data, 7) == -1
    assert BinarySearch().execute(data, 7) == -1


def test_search_empty_list():
    assert LinearSearch().execute([], 1) == -1
    assert BinarySearch().execute([], 1) == -1


@pytest.mark.parametrize("searcher", [LinearSearch(), BinarySearch()])
def test_search_type_error(searcher):
    with pytest.raises(TypeError):
        searcher.execute(None, 1)
