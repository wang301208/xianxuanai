import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from algorithms.data_structures.basic.hash_table import HashTable


class Collide:
    """自定义对象，使其哈希值相同以触发冲突。"""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 42

    def __eq__(self, other):
        return isinstance(other, Collide) and self.value == other.value


def test_hash_table_collision_handling():
    table = HashTable(initial_capacity=4)
    k1 = Collide("a")
    k2 = Collide("b")

    table.put(k1, 1)
    table.put(k2, 2)

    assert table.get(k1) == 1
    assert table.get(k2) == 2

    table.delete(k1)
    assert table.get(k1) is None
    assert table.get(k2) == 2


def test_hash_table_resize():
    table = HashTable(initial_capacity=4)
    for i in range(5):
        table.put(i, i)

    assert table.get(4) == 4
    assert len(table.buckets) > 4
