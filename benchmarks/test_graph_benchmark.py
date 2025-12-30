import random
from algorithms.graph.basic.bfs import BreadthFirstSearch
from algorithms.graph.basic.dfs import DepthFirstSearch
from algorithms.utils import Graph


def _generate_graph(nodes: int = 200, edges_per_node: int = 3) -> Graph:
    random.seed(0)
    g = Graph()
    for i in range(nodes):
        for j in range(1, edges_per_node + 1):
            g.add_edge(i, (i + j) % nodes)
    return g


def test_bfs_benchmark(benchmark) -> None:
    graph = _generate_graph()
    bfs = BreadthFirstSearch()
    benchmark(bfs.execute, graph, 0)


def test_dfs_benchmark(benchmark) -> None:
    graph = _generate_graph()
    dfs = DepthFirstSearch()
    benchmark(dfs.execute, graph, 0)
