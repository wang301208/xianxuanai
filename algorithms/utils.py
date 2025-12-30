"""算法的通用数据结构和辅助函数。

本模块提供了算法实现中常用的数据结构和工具函数，
包括元素交换函数和简单的图数据结构。
"""
from typing import Any, Dict, List


def swap(items: List[Any], i: int, j: int) -> None:
    """在列表中原地交换两个元素的位置。

    这是一个常用的辅助函数，用于排序算法中交换元素位置。
    
    参数:
        items: 要操作的列表
        i: 第一个元素的索引
        j: 第二个元素的索引
        
    时间复杂度: O(1) - 常数时间操作
    空间复杂度: O(1) - 不需要额外空间
    
    示例:
        >>> arr = [1, 2, 3]
        >>> swap(arr, 0, 2)
        >>> print(arr)  # [3, 2, 1]
    """
    items[i], items[j] = items[j], items[i]


class Graph:
    """使用邻接表实现的简单无向图。
    
    这是一个基础的图数据结构实现，使用字典存储邻接表，
    适用于图算法的实现和测试。
    
    属性:
        adjacency: 存储图的邻接表，键为节点，值为邻居节点列表
    """

    def __init__(self) -> None:
        """初始化空图。"""
        self.adjacency: Dict[Any, List[Any]] = {}

    def add_edge(self, u: Any, v: Any) -> None:
        """在节点 u 和 v 之间添加一条无向边。

        由于是无向图，会在两个节点的邻接表中都添加对方。
        
        参数:
            u: 第一个节点
            v: 第二个节点
            
        时间复杂度: O(1) - 平均情况下的常数时间
        空间复杂度: O(1) - 每条边占用常数空间
        
        示例:
            >>> graph = Graph()
            >>> graph.add_edge('A', 'B')
            >>> print(graph.neighbors('A'))  # ['B']
        """
        self.adjacency.setdefault(u, []).append(v)
        self.adjacency.setdefault(v, []).append(u)

    def neighbors(self, node: Any) -> List[Any]:
        """返回指定节点的所有邻居节点。
        
        参数:
            node: 要查询邻居的节点
            
        返回:
            List[Any]: 邻居节点列表，如果节点不存在则返回空列表
            
        时间复杂度: O(1) - 字典查找
        空间复杂度: O(1) - 返回已存在的列表引用
        """
        return self.adjacency.get(node, [])
