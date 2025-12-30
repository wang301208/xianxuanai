import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from algorithms.data_structures.advanced.binary_tree import BinaryTree


def test_binary_tree_operations():
    bt = BinaryTree()
    for value in [5, 3, 7, 2, 4, 6, 8]:
        bt.insert(value)

    assert bt.search(4)
    assert not bt.search(10)
    assert bt.inorder_traversal() == [2, 3, 4, 5, 6, 7, 8]
