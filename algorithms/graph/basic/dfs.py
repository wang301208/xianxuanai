"""Depth-first search algorithm."""
from ...base import Algorithm
from ...utils import Graph
from typing import Any, List, Set


class DepthFirstSearch(Algorithm):
    """深度优先搜索算法实现。
    
    深度优先搜索（DFS）是一种图遍历算法，它尽可能深地搜索图的分支。
    算法从起始节点开始，沿着一条路径一直走到底，然后回溯到上一个节点，
    继续探索其他未访问的路径。
    
    算法特点：
        - 使用递归或栈来实现
        - 优先访问深层节点
        - 适用于寻找路径、检测环、拓扑排序等
    
    应用场景：
        - 路径查找
        - 连通性检测
        - 拓扑排序
        - 强连通分量检测
        - 迷宫求解
    """

    def execute(self, graph: Graph, start: Any) -> List[Any]:
        """从指定起始节点开始执行深度优先搜索。

        参数:
            graph: 要遍历的图对象
            start: 搜索的起始节点
            
        返回:
            List[Any]: 按DFS顺序访问的节点列表
            
        时间复杂度: O(V + E) - V为顶点数，E为边数
        空间复杂度: O(V) - 需要存储访问状态和递归调用栈
        
        示例:
            >>> graph = Graph()
            >>> graph.add_edge('A', 'B')
            >>> graph.add_edge('A', 'C')
            >>> graph.add_edge('B', 'D')
            >>> dfs = DepthFirstSearch()
            >>> result = dfs.execute(graph, 'A')
            >>> print(result)  # 可能输出: ['A', 'B', 'D', 'C']
        """
        visited: List[Any] = []  # 存储访问顺序的列表
        seen: Set[Any] = set()   # 记录已访问节点的集合，用于快速查找
        self._dfs(graph, start, visited, seen)
        return visited

    def _dfs(self, graph: Graph, node: Any, visited: List[Any], seen: Set[Any]) -> None:
        """深度优先搜索的递归实现。
        
        这是DFS算法的核心递归函数，实现了深度优先的遍历逻辑：
        1. 标记当前节点为已访问
        2. 将当前节点加入访问序列
        3. 对每个未访问的邻居节点递归调用DFS
        
        参数:
            graph: 图对象
            node: 当前访问的节点
            visited: 按访问顺序存储节点的列表
            seen: 已访问节点的集合，用于避免重复访问
        """
        seen.add(node)           # 标记当前节点为已访问
        visited.append(node)     # 将节点加入访问序列
        
        # 遍历当前节点的所有邻居
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:  # 如果邻居未被访问过
                # 递归访问邻居节点，实现深度优先
                self._dfs(graph, neighbor, visited, seen)
