"""Simple B-tree index for ordered key-value pairs."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ...base import Algorithm


@dataclass
class BTreeNode:
    t: int
    leaf: bool = True
    keys: List[Any] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    children: List[BTreeNode] = field(default_factory=list)


class BTreeIndex(Algorithm):
    """B-tree based index supporting logarithmic search and insertion.

    Parameters
    ----------
    t: int, optional
        Minimum degree of the B-tree. Each node can store ``2*t - 1`` keys.

    Operations
    ----------
    search(key): ``O(log n)`` time.
    insert(key, value): ``O(log n)`` time.
    Space complexity: ``O(n)`` where ``n`` is the number of stored items.
    """

    def __init__(self, t: int = 2) -> None:
        self.t = t
        self.root = BTreeNode(t)

    # Search ---------------------------------------------------------------
    def search(self, key: Any, node: Optional[BTreeNode] = None) -> Any | None:
        node = node or self.root
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return node.values[i]
        if node.leaf:
            return None
        return self.search(key, node.children[i])

    # Insertion ------------------------------------------------------------
    def insert(self, key: Any, value: Any) -> None:
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, leaf=False, children=[root])
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key, value)
        else:
            self._insert_non_full(root, key, value)

    def _split_child(self, parent: BTreeNode, index: int) -> None:
        t = self.t
        node = parent.children[index]
        new = BTreeNode(t, leaf=node.leaf)
        parent.keys.insert(index, node.keys[t - 1])
        parent.values.insert(index, node.values[t - 1])
        parent.children.insert(index + 1, new)
        new.keys = node.keys[t:]
        new.values = node.values[t:]
        node.keys = node.keys[: t - 1]
        node.values = node.values[: t - 1]
        if not node.leaf:
            new.children = node.children[t:]
            node.children = node.children[:t]

    def _insert_non_full(self, node: BTreeNode, key: Any, value: Any) -> None:
        i = bisect.bisect_left(node.keys, key)
        if node.leaf:
            node.keys.insert(i, key)
            node.values.insert(i, value)
            return
        if len(node.children[i].keys) == 2 * self.t - 1:
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
        self._insert_non_full(node.children[i], key, value)

    def execute(self, *args, **kwargs) -> List[Any]:
        """Return ordered keys stored in the B-tree."""
        return self.root.keys
