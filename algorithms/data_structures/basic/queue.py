from collections import deque
from typing import Any, Deque, List

from algorithms.base import Algorithm


class Queue(Algorithm):
    """基于 collections.deque 的简单先进先出（FIFO）队列实现。

    队列是一种基础的数据结构，遵循先进先出的原则。
    元素从队尾入队，从队头出队，就像排队买票一样。
    
    主要操作：
        - enqueue: 将元素加入队尾
        - dequeue: 从队头移除元素
        - peek: 查看队头元素但不移除
        - is_empty: 检查队列是否为空
    
    时间复杂度:
        - enqueue: O(1) - 常数时间入队
        - dequeue: O(1) - 常数时间出队
        - peek: O(1) - 常数时间查看
    空间复杂度: O(n) - n 为存储元素的数量
    
    应用场景:
        - 广度优先搜索（BFS）
        - 任务调度
        - 缓冲区管理
        - 打印队列
    
    实现说明:
        使用 collections.deque 作为底层数据结构，
        提供了高效的两端操作性能。
    """

    def __init__(self) -> None:
        """初始化空队列。"""
        self.items: Deque[Any] = deque()

    def enqueue(self, item: Any) -> None:
        """将元素加入队尾。
        
        参数:
            item: 要加入队列的元素
            
        示例:
            >>> queue = Queue()
            >>> queue.enqueue(1)
            >>> queue.enqueue(2)
        """
        self.items.append(item)

    def dequeue(self) -> Any:
        """从队头移除并返回元素。
        
        返回:
            Any: 队头元素，如果队列为空则返回 None
            
        示例:
            >>> queue = Queue()
            >>> queue.enqueue(1)
            >>> queue.enqueue(2)
            >>> queue.dequeue()  # 返回 1
            >>> queue.dequeue()  # 返回 2
        """
        return self.items.popleft() if self.items else None

    def peek(self) -> Any:
        """查看队头元素但不移除。
        
        返回:
            Any: 队头元素，如果队列为空则返回 None
            
        示例:
            >>> queue = Queue()
            >>> queue.enqueue(1)
            >>> queue.peek()  # 返回 1，但不移除
            >>> queue.peek()  # 仍然返回 1
        """
        return self.items[0] if self.items else None

    def is_empty(self) -> bool:
        """检查队列是否为空。
        
        返回:
            bool: 如果队列为空返回 True，否则返回 False
        """
        return not self.items

    def execute(self, *args, **kwargs) -> List[Any]:
        """返回当前队列的快照。
        
        返回:
            List[Any]: 队列中所有元素的副本列表
        """
        return list(self.items)
