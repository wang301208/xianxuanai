from algorithms.causal.causal_graph import CausalGraph


def test_build_graph_structure():
    graph = CausalGraph()
    graph.add_node("A")
    graph.add_node("B", lambda a: a)
    graph.add_edge("A", "B")

    assert "A" in graph.nodes and "B" in graph.nodes
    assert ("A", "B") in graph.edges
    assert "A" in graph.nodes["B"]["parents"]


def test_intervene_and_infer_chain():
    graph = CausalGraph()
    graph.add_node("X")
    graph.add_node("Y", lambda x: x + 1)
    graph.add_node("Z", lambda y: y * 2)
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")

    graph.intervene("X", 2)
    assert graph.infer("Z") == 6

    graph.intervene("X", 3)
    assert graph.infer("Z") == 8
