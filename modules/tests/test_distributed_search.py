import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from capability.distributed_search import map_reduce_search, spark_search  # noqa: E402


def _write_index(path):
    data = [
        {"id": "a", "embedding": [1.0, 0.0]},
        {"id": "b", "embedding": [0.0, 1.0]},
        {"id": "c", "embedding": [1.0, 1.0]},
    ]
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def test_spark_search(tmp_path):
    index_file = tmp_path / "index.jsonl"
    _write_index(index_file)

    results = spark_search(str(index_file), [1.0, 0.0], n_results=2)
    assert results[0][0] == "a"
    assert results[1][0] == "c"


def test_map_reduce_search(tmp_path):
    index_file = tmp_path / "index.jsonl"
    _write_index(index_file)

    results = map_reduce_search(str(index_file), [1.0, 0.0], n_results=2)
    assert results[0][0] == "a"
    assert results[1][0] == "c"
