import pytest
from algorithms.template import TemplateAlgorithm


def test_template_algorithm_sum():
    data = [1, 2, 3, 4]
    assert TemplateAlgorithm().execute(data) == 10


def test_template_algorithm_non_list_raises_type_error():
    with pytest.raises(TypeError):
        TemplateAlgorithm().execute("1234")


def test_template_algorithm_non_numeric_raises_value_error():
    with pytest.raises(ValueError):
        TemplateAlgorithm().execute([1, "two", 3])
