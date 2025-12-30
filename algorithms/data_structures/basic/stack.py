from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from algorithms.base import Algorithm


@dataclass
class Stack(Algorithm):
    """简单的后进先出（LIFO）栈实现。

    栈是一种基础的数据结构，遵循后进先出的原则。
    只能在栈顶进行插入和删除操作，就像一摞盘子一样。
    
    主要操作：
        - push: 将元素压入栈顶
        - pop: 弹出栈顶元素
        - peek: 查看栈顶元素但不移除
        - is_empty: 检查栈是否为空
    
    时间复杂度:
        - push: O(1) - 常数时间插入
        - pop: O(1) - 常数时间删除
        - peek: O(1) - 常数时间查看
    空间复杂度: O(n) - n 为存储元素的数量
    
    应用场景:
        - 函数调用栈
        - 表达式求值
        - 括号匹配
        - 撤销操作
    """

    items: List[Any]

    def __init__(self) -> None:
        """初始化空栈。"""
        self.items = []

    def push(self, item: Any) -> None:
        """将元素压入栈顶。
        
        参数:
            item: 要压入栈的元素
            
        示例:
            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.push(2)
        """
        self.items.append(item)

    def pop(self) -> Any:
        """弹出并返回栈顶元素。
        
        返回:
            Any: 栈顶元素，如果栈为空则返回 None
            
        示例:
            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.push(2)
            >>> stack.pop()  # 返回 2
            >>> stack.pop()  # 返回 1
        """
        return self.items.pop() if self.items else None

    def peek(self) -> Any:
        """查看栈顶元素但不移除。
        
        返回:
            Any: 栈顶元素，如果栈为空则返回 None
            
        示例:
            >>> stack = Stack()
            >>> stack.push(1)
            >>> stack.peek()  # 返回 1，但不移除
            >>> stack.peek()  # 仍然返回 1
        """
        return self.items[-1] if self.items else None

    def is_empty(self) -> bool:
        """检查栈是否为空。
        
        返回:
            bool: 如果栈为空返回 True，否则返回 False
        """
        return not self.items

    def execute(self, *args, **kwargs) -> List[Any]:
        """返回当前栈的快照。
        
        返回:
            List[Any]: 栈中所有元素的副本列表
        """
        return list(self.items)
