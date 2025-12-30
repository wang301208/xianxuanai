import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from algorithms.data_structures.basic.linked_list import LinkedList


def test_linked_list_operations():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)

    assert ll.find(2).value == 2
    assert ll.delete(2)
    assert ll.find(2) is None
    assert ll.to_list() == [1, 3]
    assert not ll.delete(4)
