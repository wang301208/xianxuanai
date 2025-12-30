import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from backend.memory.long_term import LongTermMemory


def test_bulk_insert_benchmark(tmp_path, benchmark):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    data = [("cat", f"content {i}") for i in range(1000)]

    def bulk_insert():
        with mem.batch():
            for cat, content in data:
                mem.add(cat, content)

    benchmark(bulk_insert)
    mem.close()


def test_bulk_retrieval_benchmark(tmp_path, benchmark):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    with mem.batch():
        for i in range(1000):
            mem.add("cat", f"content {i}", tags=["tag"])

    def retrieve():
        list(mem.get(category="cat", tags=["tag"]))

    benchmark(retrieve)
    mem.close()

