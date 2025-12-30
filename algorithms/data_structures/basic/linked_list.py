from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List

from algorithms.base import Algorithm


@dataclass
class Node:
    """链表节点类。
    
    每个节点包含一个值和指向下一个节点的引用。
    这是构建链表的基本单元。
    
    属性:
        value: 节点存储的值
        next: 指向下一个节点的引用，如果是最后一个节点则为 None
    """
    value: Any
    next: Optional[Node] = None


class LinkedList(Algorithm):
    """单向链表实现。

    链表是一种线性数据结构，其中元素不存储在连续的内存位置。
    每个元素（节点）包含数据和指向下一个节点的引用。
    
    主要操作：
        - append: 在链表末尾添加元素
        - find: 查找包含指定值的节点
        - delete: 删除包含指定值的第一个节点
        - to_list: 转换为普通列表
    
    时间复杂度:
        - append: O(n) - 需要遍历到末尾
        - find: O(n) - 可能需要遍历整个链表
        - delete: O(n) - 可能需要遍历整个链表
    空间复杂度: O(n) - n 为存储元素的数量
    
    优点:
        - 动态大小，可以在运行时增长或缩小
        - 插入和删除操作高效（如果有节点引用）
        - 内存使用灵活
    
    缺点:
        - 不支持随机访问，必须顺序遍历
        - 额外的内存开销（存储指针）
        - 不利于缓存局部性
    """

    def __init__(self) -> None:
        """初始化空链表。"""
        self.head: Optional[Node] = None

    def append(self, value: Any) -> None:
        """在链表末尾添加一个值。
        
        参数:
            value: 要添加的值
            
        时间复杂度: O(n) - 需要遍历到链表末尾
        
        示例:
            >>> linked_list = LinkedList()
            >>> linked_list.append(1)
            >>> linked_list.append(2)
        """
        new_node = Node(value)
        if not self.head:  # 如果链表为空，新节点成为头节点
            self.head = new_node
            return
        
        # 遍历到链表末尾
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node  # 将新节点连接到末尾

    def find(self, value: Any) -> Optional[Node]:
        """查找包含指定值的第一个节点。
        
        参数:
            value: 要查找的值
            
        返回:
            Optional[Node]: 包含该值的节点，如果未找到则返回 None
            
        时间复杂度: O(n) - 最坏情况下需要遍历整个链表
        
        示例:
            >>> linked_list = LinkedList()
            >>> linked_list.append(1)
            >>> linked_list.append(2)
            >>> node = linked_list.find(2)
            >>> print(node.value if node else "Not found")  # 输出: 2
        """
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def delete(self, value: Any) -> bool:
        """删除包含指定值的第一个节点。
        
        参数:
            value: 要删除的值
            
        返回:
            bool: 如果成功删除返回 True，否则返回 False
            
        时间复杂度: O(n) - 最坏情况下需要遍历整个链表
        
        示例:
            >>> linked_list = LinkedList()
            >>> linked_list.append(1)
            >>> linked_list.append(2)
            >>> success = linked_list.delete(1)
            >>> print(success)  # 输出: True
        """
        current = self.head
        prev: Optional[Node] = None
        
        while current:
            if current.value == value:
                if prev:  # 删除的不是头节点
                    prev.next = current.next
                else:  # 删除的是头节点
                    self.head = current.next
                return True
            prev = current
            current = current.next
        return False  # 未找到要删除的值

    def to_list(self) -> List[Any]:
        """将链表转换为普通列表。
        
        返回:
            List[Any]: 包含链表所有值的列表
            
        时间复杂度: O(n) - 需要遍历整个链表
        
        示例:
            >>> linked_list = LinkedList()
            >>> linked_list.append(1)
            >>> linked_list.append(2)
            >>> print(linked_list.to_list())  # 输出: [1, 2]
        """
        result: List[Any] = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

    def execute(self, *args, **kwargs) -> List[Any]:
        """返回链表的列表表示。
        
        这是 Algorithm 基类要求的方法实现。
        
        返回:
            List[Any]: 链表的列表表示
        """
        return self.to_list()
