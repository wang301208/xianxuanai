from backend.knowledge.vector_store import LocalVectorStore


def test_local_vector_store_search_includes_similarity_and_does_not_mutate_metadata():
    store = LocalVectorStore(3, use_faiss=False)

    meta_a = {"id": "a"}
    meta_b = {"id": "b"}
    meta_c = {"id": "c"}
    store.add([1.0, 0.0, 0.0], meta_a)
    store.add([0.0, 1.0, 0.0], meta_b)
    store.add([-1.0, 0.0, 0.0], meta_c)

    hits = store.search([1.0, 0.0, 0.0], top_k=3)
    assert [hit["id"] for hit in hits] == ["a", "b", "c"]
    assert all(isinstance(hit.get("similarity"), float) for hit in hits)
    assert hits[0]["similarity"] > hits[1]["similarity"] > hits[2]["similarity"]

    assert "similarity" not in meta_a
    assert "similarity" not in meta_b
    assert "similarity" not in meta_c

