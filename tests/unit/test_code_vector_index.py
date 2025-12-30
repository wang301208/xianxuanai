from pathlib import Path

from modules.knowledge.code_vector_index import CodeVectorIndex


def test_code_vector_index_build_search_persist_and_references(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo(x: int) -> int:\n"
        "    \"\"\"demo function\"\"\"\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    (repo_root / "other.py").write_text(
        "def bar(y: int) -> int:\n"
        "    return y * 2\n",
        encoding="utf-8",
    )

    persist_path = tmp_path / "code_index.json"
    index = CodeVectorIndex(root=repo_root, persist_path=persist_path, max_files=20, use_faiss=False)
    stats = index.build(save=True)
    assert stats["chunks_indexed"] >= 2
    assert persist_path.exists()

    chunks = index._extract_chunks(repo_root / "algo.py")
    assert chunks
    query = index._chunk_to_embedding_text(chunks[0])
    hits = index.search(query, top_k=3)
    assert hits
    assert hits[0]["path"] == "algo.py"
    assert hits[0]["symbol"] == "foo"
    assert hits[0]["kind"] == "function"
    assert hits[0].get("similarity") is not None

    index2 = CodeVectorIndex(root=repo_root, persist_path=persist_path, max_files=20, use_faiss=False)
    assert index2.load() is True
    hits2 = index2.search(query, top_k=1)
    assert hits2 and hits2[0]["path"] == "algo.py"

    refs = index2.hits_as_references(hits2, max_chars=200)
    assert refs and refs[0]["url"] == "algo.py"
    assert refs[0]["snippet"]

