import numpy as np
from time import perf_counter
from dataclasses import dataclass


@dataclass
class MemoryItem:
    e_summary: np.ndarray
    e_weighted: np.ndarray
    e_chunks: np.ndarray


def calculate_scores(memory: MemoryItem, query: np.ndarray, strategy: str):
    base = memory.e_weighted if strategy == "weighted" else memory.e_summary
    summary_score = float(np.dot(base, query))
    chunk_scores = np.dot(memory.e_chunks, query).tolist()
    return max([summary_score, *chunk_scores]), summary_score, chunk_scores


def build_memory_items(n_items: int, n_chunks: int, dim: int) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    for _ in range(n_items):
        e_chunks = np.random.rand(n_chunks, dim).astype(np.float32)
        weights = [len(str(i)) for i in range(n_chunks)]
        e_weighted = np.average(e_chunks, axis=0, weights=weights)
        e_summary = e_weighted + np.random.normal(0, 0.01, size=dim)
        items.append(MemoryItem(e_summary=e_summary, e_weighted=e_weighted, e_chunks=e_chunks))
    return items


def build_queries(items: list[MemoryItem], n_queries: int) -> list[tuple[int, np.ndarray]]:
    queries = []
    n_items = len(items)
    for _ in range(n_queries):
        idx = np.random.randint(0, n_items)
        chunk_idx = np.random.randint(0, items[idx].e_chunks.shape[0])
        queries.append((idx, items[idx].e_chunks[chunk_idx]))
    return queries


def evaluate(items: list[MemoryItem], queries, strategy: str) -> tuple[float, float]:
    start = perf_counter()
    correct = 0
    for target, q in queries:
        scores = [calculate_scores(m, q, strategy)[0] for m in items]
        if int(np.argmax(scores)) == target:
            correct += 1
    elapsed = perf_counter() - start
    return correct / len(queries), elapsed


def main() -> None:
    np.random.seed(42)
    items = build_memory_items(n_items=50, n_chunks=3, dim=10)
    queries = build_queries(items, n_queries=100)

    acc_sum, t_sum = evaluate(items, queries, "summary")
    acc_wgt, t_wgt = evaluate(items, queries, "weighted")

    print("Embedding Strategy Comparison:")
    print(f"Summary vector  - accuracy: {acc_sum:.2%}, time: {t_sum:.4f}s")
    print(f"Weighted average - accuracy: {acc_wgt:.2%}, time: {t_wgt:.4f}s")


if __name__ == "__main__":
    main()
