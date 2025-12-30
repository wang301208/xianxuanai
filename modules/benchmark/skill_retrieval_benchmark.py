"""Micro-benchmark for sequential vs threaded skill retrieval."""
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
import sys

# Ensure repository root on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from capability.librarian import Librarian


class DummyIndex:
    def __init__(self, ids: List[str]):
        self.ids = ids

    def query(self, embedding, n_results, vector_type="text"):
        return {"ids": [self.ids[:n_results]]}


def _create_skills(base: Path, n: int) -> List[str]:
    skills_dir = base / "skills"
    skills_dir.mkdir()
    names = []
    # Make skill files moderately large to better showcase I/O parallelism
    payload = "x" * 10000
    for i in range(n):
        name = f"skill{i}"
        (skills_dir / f"{name}.py").write_text(
            f"def {name}():\n    return {i}\n# {payload}\n", encoding="utf-8"
        )
        (skills_dir / f"{name}.json").write_text("{}", encoding="utf-8")
        names.append(name)
    return names


def benchmark(n: int = 100) -> dict[str, float]:
    with TemporaryDirectory() as tmp:
        repo = Path(tmp)
        names = _create_skills(repo, n)
        lib = Librarian(str(repo))
        lib.index = DummyIndex(names)
        # Simulate I/O latency to highlight parallel speedup
        original_get_skill = lib.get_skill

        def delayed(name: str):
            time.sleep(0.005)
            return original_get_skill(name)

        lib.get_skill = delayed
        embedding = [0.0]

        start = time.perf_counter()
        lib.search(embedding, n_results=n, return_content=True, max_workers=1)
        sequential = time.perf_counter() - start

        start = time.perf_counter()
        lib.search(embedding, n_results=n, return_content=True)
        parallel = time.perf_counter() - start

        return {"sequential": sequential, "parallel": parallel}


if __name__ == "__main__":
    results = benchmark()
    print(
        f"sequential: {results['sequential']:.4f}s, parallel: {results['parallel']:.4f}s"
    )
