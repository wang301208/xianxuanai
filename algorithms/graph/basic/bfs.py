"""Breadth-first search algorithm."""
from collections import deque
from ...base import Algorithm
from ...utils import Graph
from typing import Any, List


class BreadthFirstSearch(Algorithm):
    """广度优先搜索算法实现。
    
    广度优先搜索（BFS）是一种图遍历算法，它按层次顺序访问图中的节点。
    算法从起始节点开始，首先访问所有距离为1的邻居节点，然后访问距离为2的节点，
    以此类推，直到访问完所有可达的节点。
    
    算法特点：
        - 使用队列（FIFO）来实现
        - 按距离递增的顺序访问节点
        - 能找到最短路径（对于无权图）
        - 适用于层次遍历、最短路径查找等
    
    应用场景：
        - 最短路径查找（无权图）
        - 层次遍历
        - 连通性检测
        - 社交网络中的朋友推荐
        - 网络爬虫的广度抓取
    """

    def execute(self, graph: Graph, start: Any) -> List[Any]:
        """从指定起始节点开始执行广度优先搜索。

        参数:
            graph: 要遍历的图对象
            start: 搜索的起始节点
            
        返回:
            List[Any]: 按BFS顺序访问的节点列表
            
        时间复杂度: O(V + E) - V为顶点数，E为边数，每个顶点和边都被访问一次
        空间复杂度: O(V) - 队列和访问标记集合最多存储所有顶点
        
        算法步骤:
            1. 将起始节点加入队列并标记为已访问
            2. 当队列不为空时：
               - 从队列前端取出一个节点
               - 将该节点加入访问序列
               - 将其所有未访问的邻居加入队列并标记为已访问
        
        示例:
            >>> graph = Graph()
            >>> graph.add_edge('A', 'B')
            >>> graph.add_edge('A', 'C')
            >>> graph.add_edge('B', 'D')
            >>> bfs = BreadthFirstSearch()
            >>> result = bfs.execute(graph, 'A')
            >>> print(result)  # 可能输出: ['A', 'B', 'C', 'D']
        """
        visited: List[Any] = []              # 存储访问顺序的列表
        queue: deque[Any] = deque([start])   # 使用双端队列实现FIFO
        seen = {start}                       # 记录已访问节点，避免重复访问
        
        while queue:  # 当队列不为空时继续遍历
            node = queue.popleft()           # 从队列前端取出节点（FIFO）
            visited.append(node)             # 将节点加入访问序列
            
            # 遍历当前节点的所有邻居
            for neighbor in graph.neighbors(node):
                if neighbor not in seen:     # 如果邻居未被访问过
                    seen.add(neighbor)       # 标记邻居为已访问
                    queue.append(neighbor)   # 将邻居加入队列等待处理
        
        return visited
