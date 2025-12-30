"""Dijkstra shortest path algorithm using a priority queue."""
from __future__ import annotations

import heapq
from typing import Any, Dict, List, Tuple

from ...base import Algorithm


class Dijkstra(Algorithm):
    """单源最短路径的 Dijkstra 算法实现。

    该算法使用优先队列（最小堆）处理带权无向图，
    计算源点到图中各节点的最短距离。如果某个节点不可达，
    则其距离为 ``float('inf')``。
    """

    def execute(
        self, graph: Dict[Any, List[Tuple[Any, float]]], start: Any
    ) -> Dict[Any, float]:
        """计算源点到其他所有节点的最短距离。

        参数:
            graph: 图的邻接表表示，键为节点，值为邻居及权重的列表。
            start: 源点。

        返回:
            Dict[Any, float]: 源点到每个节点的最短距离映射。
        """
        # 初始化距离字典，所有节点距离设为正无穷
        distances: Dict[Any, float] = {node: float("inf") for node in graph}
        distances[start] = 0.0

        # 最小堆，存储待处理的节点及其当前已知的最短距离
        pq: List[Tuple[float, Any]] = [(0.0, start)]

        while pq:
            current_dist, node = heapq.heappop(pq)
            if current_dist > distances[node]:
                continue

            # 遍历邻居节点，尝试更新距离
            for neighbor, weight in graph.get(node, []):
                new_dist = current_dist + weight
                if new_dist < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        return distances
