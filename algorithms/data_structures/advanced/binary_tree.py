from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from algorithms.base import Algorithm


@dataclass
class Node:
    """二叉树节点类。
    
    每个节点包含一个值和指向左右子节点的引用。
    这是构建二叉搜索树的基本单元。
    
    属性:
        value: 节点存储的值
        left: 指向左子节点的引用，如果没有左子节点则为 None
        right: 指向右子节点的引用，如果没有右子节点则为 None
    """
    value: Any
    left: Optional[Node] = None
    right: Optional[Node] = None


class BinaryTree(Algorithm):
    """二叉搜索树实现。

    二叉搜索树是一种特殊的二叉树，对于每个节点：
    - 左子树中所有节点的值都小于该节点的值
    - 右子树中所有节点的值都大于该节点的值
    - 左右子树也都是二叉搜索树
    
    主要操作：
        - insert: 插入新值到树中
        - search: 查找指定值是否存在
        - inorder_traversal: 中序遍历获取有序序列
    
    时间复杂度:
        - insert: O(h) - h为树的高度，平均O(log n)，最坏O(n)
        - search: O(h) - h为树的高度，平均O(log n)，最坏O(n)
        - inorder_traversal: O(n) - 需要访问所有n个节点
    空间复杂度: O(n) - 存储n个节点
    
    优点:
        - 查找、插入、删除操作平均时间复杂度为O(log n)
        - 中序遍历可以得到有序序列
        - 结构简单，易于理解和实现
    
    缺点:
        - 在最坏情况下（完全不平衡）退化为链表，时间复杂度为O(n)
        - 不是自平衡的，需要额外的平衡操作来保证性能
    """
=======

    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def insert(self, value: Any) -> None:
        """向二叉搜索树中插入一个值。
        
        参数:
            value: 要插入的值
            
        时间复杂度: O(h) - h为树的高度
        
        示例:
            >>> tree = BinaryTree()
            >>> tree.insert(5)
            >>> tree.insert(3)
            >>> tree.insert(7)
        """
        if self.root is None:
            self.root = Node(value)
            return
        self._insert(self.root, value)

    def _insert(self, node: Node, value: Any) -> None:
        """递归插入辅助方法。
        
        根据二叉搜索树的性质，将值插入到正确的位置：
        - 如果值小于当前节点值，插入到左子树
        - 如果值大于等于当前节点值，插入到右子树
        
        参数:
            node: 当前节点
            value: 要插入的值
        """
        if value < node.value:
            if node.left is None:
                node.left = Node(value)  # 在左子树的空位置插入新节点
            else:
                self._insert(node.left, value)  # 递归插入到左子树
        else:
            if node.right is None:
                node.right = Node(value)  # 在右子树的空位置插入新节点
            else:
                self._insert(node.right, value)  # 递归插入到右子树
=======

    def search(self, value: Any) -> bool:
        """在二叉搜索树中查找指定值。
        
        参数:
            value: 要查找的值
            
        返回:
            bool: 如果找到返回True，否则返回False
            
        时间复杂度: O(h) - h为树的高度
        
        示例:
            >>> tree = BinaryTree()
            >>> tree.insert(5)
            >>> tree.search(5)  # True
            >>> tree.search(10) # False
        """
        return self._search(self.root, value)

    def _search(self, node: Optional[Node], value: Any) -> bool:
        """递归查找辅助方法。
        
        利用二叉搜索树的有序性质进行高效查找：
        - 如果当前节点为空，说明未找到
        - 如果值等于当前节点值，找到目标
        - 如果值小于当前节点值，在左子树中查找
        - 如果值大于当前节点值，在右子树中查找
        
        参数:
            node: 当前节点
            value: 要查找的值
            
        返回:
            bool: 是否找到目标值
        """
        if node is None:
            return False  # 到达空节点，未找到
        if value == node.value:
            return True   # 找到目标值
        if value < node.value:
            return self._search(node.left, value)   # 在左子树中查找
        return self._search(node.right, value)      # 在右子树中查找
=======

    def inorder_traversal(self) -> List[Any]:
        """中序遍历二叉搜索树，返回有序的值列表。
        
        中序遍历的顺序是：左子树 -> 根节点 -> 右子树
        对于二叉搜索树，中序遍历的结果是按升序排列的。
        
        返回:
            List[Any]: 按升序排列的所有节点值
            
        时间复杂度: O(n) - 需要访问所有n个节点
        空间复杂度: O(h) - 递归调用栈的深度为树的高度h
        
        示例:
            >>> tree = BinaryTree()
            >>> tree.insert(5)
            >>> tree.insert(3)
            >>> tree.insert(7)
            >>> tree.insert(1)
            >>> tree.inorder_traversal()  # [1, 3, 5, 7]
        """
        result: List[Any] = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[Node], result: List[Any]) -> None:
        """中序遍历的递归辅助方法。
        
        按照 左子树 -> 根节点 -> 右子树 的顺序遍历。
        
        参数:
            node: 当前节点
            result: 存储遍历结果的列表
        """
        if node:
            self._inorder(node.left, result)    # 先遍历左子树
            result.append(node.value)           # 访问根节点
            self._inorder(node.right, result)   # 最后遍历右子树

    def execute(self, *args, **kwargs) -> List[Any]:
        """返回树的中序遍历结果。
        
        这是Algorithm基类要求的方法实现。
        
        返回:
            List[Any]: 树中所有值的有序列表
        """
        return self.inorder_traversal()
=======
