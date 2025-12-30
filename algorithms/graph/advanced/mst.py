"""Minimum Spanning Tree algorithm using Kruskal's method."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...base import Algorithm


class KruskalMST(Algorithm):
    """基于 Kruskal 算法的最小生成树实现。

    该算法接受一个无向图的邻接表表示，返回生成树的边集合及总权重。
    如果图不连通，则抛出 ``ValueError``。
    """

    def execute(
        self, graph: Dict[Any, List[Tuple[Any, float]]]
    ) -> Tuple[List[Tuple[Any, Any, float]], float]:
        """计算图的最小生成树。

        参数:
            graph: 图的邻接表，键为节点，值为邻接节点与权重的列表。

        返回:
            Tuple[List[Tuple[Any, Any, float]], float]:
                - 生成树的边集合，每条边表示为 ``(u, v, weight)``。
                - 生成树的总权重。

        异常:
            ValueError: 如果图不是连通的。
        """
        # 收集所有边，避免重复 (u,v) 与 (v,u)
        seen = set()
        edges: List[Tuple[Any, Any, float]] = []
        for u, neighbors in graph.items():
            for v, weight in neighbors:
                edge = (u, v) if (u, v) not in seen and (v, u) not in seen else None
                if edge:
                    seen.add(edge)
                    edges.append((u, v, weight))

        # 按权重排序
        edges.sort(key=lambda e: e[2])

        parent = {node: node for node in graph}
        rank = {node: 0 for node in graph}

        def find(x: Any) -> Any:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: Any, y: Any) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        mst_edges: List[Tuple[Any, Any, float]] = []
        total_weight = 0.0
        for u, v, w in edges:
            if union(u, v):
                mst_edges.append((u, v, w))
                total_weight += w

        if len(mst_edges) != len(graph) - 1:
            raise ValueError("Graph is not connected")

        return mst_edges, total_weight
