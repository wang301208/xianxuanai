from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from .vector_store import VectorMemoryStore


if TYPE_CHECKING:  # pragma: no cover - circular import avoidance for typing
    from .lifecycle import MemoryLifecycleManager


logger = logging.getLogger(__name__)


@dataclass
class MemoryDecayPolicy:
    """Parameters controlling decay and consolidation."""

    half_life: float = 7 * 24 * 3600  # seconds (one week)
    min_weight: float = 0.05
    archive_threshold: float = 0.15
    promote_threshold: float = 0.6
    min_usage_for_promotion: int = 3
    consolidate_interval: float = 3600.0
    episodic_summary_interval: float = 3 * 3600.0
    episodic_summary_window: int = 20
    episodic_summary_min_records: int = 5
    vector_optimization_interval: float = 6 * 3600.0
    vector_optimization_sample_size: int = 256
    vector_redundancy_similarity: float = 0.96
    vector_novelty_top_k: int = 10


class MemoryMaintenanceDaemon:
    """Background maintenance for vector memory stores."""

    def __init__(
        self,
        store: VectorMemoryStore,
        *,
        policy: MemoryDecayPolicy | None = None,
        clock=time,
        lifecycle: "MemoryLifecycleManager" | None = None,
        episode_log_path: Path | str | None = None,
        episode_summarizer: Callable[[List[Dict[str, Any]]], Dict[str, Any]] | None = None,
        random_seed: Optional[int] = None,
    ):
        self.store = store
        self.policy = policy or MemoryDecayPolicy()
        self.clock = clock
        self._last_consolidation = 0.0
        self._lifecycle = lifecycle
        self._episode_log_path = Path(episode_log_path) if episode_log_path else None
        self._episode_summarizer = episode_summarizer or self._default_episode_summarizer
        self._episode_buffer: List[Dict[str, Any]] = []
        self._episode_log_position = 0
        self._episode_log_size = 0
        self._last_episode_summary = 0.0
        self._last_vector_optimization = 0.0
        self._rng = random.Random(random_seed)

    def tick(self) -> Dict[str, Any]:
        """Run a maintenance pass."""

        events: Dict[str, Any] = {
            "decayed": [],
            "archived": [],
            "promoted": [],
            "summaries": [],
            "episodic_summaries": [],
            "vector_optimization": {},
        }
        now = self.clock.time()
        for record in self.store.iter_records():
            metadata = record.setdefault("metadata", {})
            usage = int(metadata.get("usage", 0))
            last_access = float(metadata.get("last_access", metadata.get("created_at", now)))
            weight = float(metadata.get("weight", 1.0))
            age = max(0.0, now - last_access)
            decayed_weight = self._decay_weight(weight, age)
            metadata["weight"] = max(self.policy.min_weight, decayed_weight)
            metadata["last_access"] = last_access

            if usage >= self.policy.min_usage_for_promotion and decayed_weight >= self.policy.promote_threshold:
                metadata["promoted"] = True
                events["promoted"].append(record["id"])

            if decayed_weight < self.policy.archive_threshold:
                metadata["archived"] = True
                events["archived"].append(record["id"])
            else:
                events["decayed"].append(record["id"])

        if now - self._last_consolidation >= self.policy.consolidate_interval:
            self._persist_metadata()
            self._last_consolidation = now
        if self._lifecycle is not None:
            summaries = self._lifecycle.run_summary_consolidation(timestamp=now)
            if summaries:
                events["summaries"].extend(summaries)
        episodic = self._run_episode_consolidation(now)
        if episodic:
            events["summaries"].extend(episodic)
            events["episodic_summaries"].extend(episodic)
        optimization = self._optimize_vector_memory(now)
        if optimization.get("archived"):
            events["archived"].extend(optimization["archived"])
        if optimization:
            events["vector_optimization"] = optimization
        if not events["episodic_summaries"]:
            events.pop("episodic_summaries", None)
        if not events["vector_optimization"]:
            events.pop("vector_optimization", None)
        return events

    def _decay_weight(self, weight: float, age: float) -> float:
        import math

        lambda_decay = math.log(2) / max(1.0, self.policy.half_life)
        decayed = weight * math.exp(-lambda_decay * age)
        return decayed

    def _persist_metadata(self) -> None:
        self.store._persist()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ episodic consolidation
    def _run_episode_consolidation(self, now: float) -> List[str]:
        if self._episode_log_path is None:
            return []
        interval = max(60.0, self.policy.episodic_summary_interval)
        if self._last_episode_summary and (now - self._last_episode_summary) < interval:
            return []
        self._fetch_new_episodes()
        if not self._episode_buffer or len(self._episode_buffer) < self.policy.episodic_summary_min_records:
            return []

        batch_size = min(len(self._episode_buffer), self.policy.episodic_summary_window)
        batch = self._episode_buffer[:batch_size]
        try:
            summary_payload = self._episode_summarizer(batch)
        except Exception:
            logger.exception("Failed to summarise episodic experiences")
            return []

        summary_text = str(summary_payload.get("text", "")).strip()
        if not summary_text:
            return []

        metadata = dict(summary_payload.get("metadata", {}))
        metadata.setdefault("type", "episodic_digest")
        metadata.setdefault("source", "experience_hub")
        metadata.setdefault("summary_strategy", "episodic_concatenate")
        metadata["episode_count"] = len(batch)
        metadata["created_at"] = now
        metadata["last_access"] = now
        metadata.setdefault("usage", 0)
        metadata.setdefault("weight", 1.0)
        if "importance" not in metadata:
            success_rate = float(metadata.get("success_rate", 0.5))
            metadata["importance"] = min(0.95, 0.55 + success_rate * 0.35)

        cycles = []
        task_ids = set()
        timestamps: List[float] = []
        for episode in batch:
            meta = episode.get("metadata", {}) or {}
            cycle = meta.get("cycle")
            if cycle is not None:
                try:
                    cycles.append(int(cycle))
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    continue
            task_id = episode.get("task_id")
            if task_id:
                task_ids.add(str(task_id))
            ts = float(episode.get("_timestamp", now))
            timestamps.append(ts)
        if timestamps:
            metadata["window_start"] = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat()
            metadata["window_end"] = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()
        if cycles:
            metadata["source_cycles"] = sorted(set(cycles))
        if task_ids:
            metadata["source_task_ids"] = sorted(task_ids)

        key_facts_raw = summary_payload.get("key_facts", []) or []
        key_facts = [str(fact).strip() for fact in key_facts_raw if str(fact).strip()]
        if key_facts:
            metadata["key_facts"] = key_facts[:50]

        record_id = self.store.add_text(summary_text, metadata=metadata)
        self._episode_buffer = self._episode_buffer[batch_size:]
        self._last_episode_summary = now
        return [record_id]

    def _fetch_new_episodes(self) -> None:
        if self._episode_log_path is None:
            return
        path = self._episode_log_path
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        except OSError:
            logger.debug("Unable to stat episode log at %s", path, exc_info=True)
            return
        if stat.st_size < self._episode_log_size:
            self._episode_log_position = 0
            self._episode_buffer.clear()
        self._episode_log_size = stat.st_size
        try:
            with path.open("r", encoding="utf-8") as handle:
                start = self._episode_log_position
                if start > 0:
                    for _ in range(start):
                        skipped = handle.readline()
                        if skipped == "":
                            start = 0
                            self._episode_log_position = 0
                            handle.seek(0)
                            break
                for idx, line in enumerate(handle, start=start):
                    stripped = line.strip()
                    if not stripped:
                        self._episode_log_position = idx + 1
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed episode entry at %s line %s", path, idx + 1)
                        self._episode_log_position = idx + 1
                        continue
                    payload["_timestamp"] = self._parse_episode_timestamp(payload)
                    self._episode_buffer.append(payload)
                    self._episode_log_position = idx + 1
        except FileNotFoundError:
            return
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to read episodes from %s", path)

    def _parse_episode_timestamp(self, payload: Dict[str, Any]) -> float:
        raw = payload.get("created_at") or payload.get("metadata", {}).get("created_at")
        if raw is None:
            return self.clock.time()
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return self.clock.time()

    def _default_episode_summarizer(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not episodes:
            return {"text": "", "metadata": {}}
        total_reward = 0.0
        total_steps = 0
        success_count = 0
        lines: List[str] = []
        key_facts: List[str] = []
        for episode in episodes:
            meta = episode.get("metadata", {}) or {}
            summary = str(meta.get("summary") or "").strip()
            if not summary:
                summary = (
                    f"policy={episode.get('policy_version')} reward={episode.get('total_reward', 0.0):.2f}"
                    f" steps={episode.get('steps', 0)} success={bool(episode.get('success'))}"
                )
            created = episode.get("created_at") or meta.get("created_at")
            if created:
                lines.append(f"{created}: {summary}")
            else:
                lines.append(summary)
            if summary and summary not in key_facts:
                key_facts.append(summary)
            try:
                total_reward += float(episode.get("total_reward") or 0.0)
            except (TypeError, ValueError):
                total_reward += 0.0
            try:
                total_steps += int(episode.get("steps") or 0)
            except (TypeError, ValueError):
                total_steps += 0
            if bool(episode.get("success")):
                success_count += 1
        count = len(episodes)
        avg_reward = total_reward / count if count else 0.0
        avg_steps = total_steps / count if count else 0.0
        success_rate = success_count / count if count else 0.0
        timestamps = [float(ep.get("_timestamp", self.clock.time())) for ep in episodes]
        if timestamps:
            start_iso = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()
        else:  # pragma: no cover - defensive guard
            now_iso = datetime.fromtimestamp(self.clock.time(), tz=timezone.utc).isoformat()
            start_iso = end_iso = now_iso
        metadata = {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "window_start": start_iso,
            "window_end": end_iso,
        }
        return {
            "text": "\n".join(lines).strip(),
            "metadata": metadata,
            "key_facts": key_facts[:25],
        }

    # ------------------------------------------------------------------ vector optimisation
    def _optimize_vector_memory(self, now: float) -> Dict[str, Any]:
        interval = max(300.0, self.policy.vector_optimization_interval)
        if self._last_vector_optimization and (now - self._last_vector_optimization) < interval:
            return {}
        vectors = getattr(self.store, "_vectors", None)
        if not isinstance(vectors, np.ndarray) or vectors.size == 0:
            self._last_vector_optimization = now
            return {}
        records = list(self.store.iter_records())
        total_records = len(records)
        if total_records < 2:
            self._last_vector_optimization = now
            return {}

        matrix = np.asarray(vectors, dtype=np.float32)
        count = min(matrix.shape[0], total_records)
        if count < 2:
            self._last_vector_optimization = now
            return {}
        matrix = matrix[:count]
        records = records[:count]

        sample_size = min(count, max(8, self.policy.vector_optimization_sample_size))
        if count > sample_size:
            indices = sorted(self._rng.sample(range(count), sample_size))
            matrix = matrix[indices]
            records = [records[i] for i in indices]
        else:
            sample_size = count

        # normalise vectors to unit length
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        centered = normalized - normalized.mean(axis=0, keepdims=True)

        explained_variance: List[float] = []
        try:
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
            variance = singular_values**2
            total_variance = float(np.sum(variance))
            if total_variance > 1e-12:
                explained_variance = (variance / total_variance).tolist()
        except Exception:
            logger.debug("Vector optimisation SVD failed", exc_info=True)

        similarity = np.clip(np.dot(normalized, normalized.T), -1.0, 1.0)
        redundancy_threshold = float(self.policy.vector_redundancy_similarity)
        archived_ids: List[str] = []
        redundant_pairs: List[tuple[str, str]] = []
        archive_candidates: set[str] = set()

        def _score(meta: Dict[str, Any]) -> float:
            importance = float(meta.get("importance", meta.get("weight", 0.5)))
            weight = float(meta.get("weight", 1.0))
            usage = float(meta.get("usage", 0))
            return importance + 0.5 * weight + 0.1 * usage

        for i in range(sample_size):
            for j in range(i):
                if similarity[i, j] < redundancy_threshold:
                    continue
                meta_i = records[i].setdefault("metadata", {})
                meta_j = records[j].setdefault("metadata", {})
                if meta_i.get("archived") or meta_j.get("archived"):
                    continue
                score_i = _score(meta_i)
                score_j = _score(meta_j)
                if score_i >= score_j:
                    winner = records[i]["id"]
                    loser = records[j]["id"]
                else:
                    winner = records[j]["id"]
                    loser = records[i]["id"]
                if loser in archive_candidates:
                    continue
                archive_candidates.add(loser)
                redundant_pairs.append((winner, loser))

        for record_id in archive_candidates:
            try:
                if self.store.archive_record(record_id):
                    archived_ids.append(record_id)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Failed to archive redundant memory %s", record_id)

        novelty_scores: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            if record["id"] in archive_candidates:
                continue
            row = similarity[idx]
            if row.size <= 1:
                max_sim = 0.0
            else:
                max_sim = float(np.max(np.delete(row, idx)))
            novelty = float(max(0.0, 1.0 - max_sim))
            novelty_scores.append({"id": record["id"], "score": novelty})

        novelty_scores.sort(key=lambda item: item["score"], reverse=True)
        top_k = min(len(novelty_scores), max(1, self.policy.vector_novelty_top_k))
        novel_updates: List[Dict[str, Any]] = []
        for rank, item in enumerate(novelty_scores[:top_k], start=1):
            record_id = item["id"]
            update = {
                "novelty_score": float(item["score"]),
                "novelty_rank": rank,
                "last_optimized_at": now,
            }
            if self.store.update_metadata(record_id, update):
                novel_updates.append({"id": record_id, "score": update["novelty_score"]})

        if archived_ids or novel_updates:
            self._persist_metadata()
        self._last_vector_optimization = now
        return {
            "archived": archived_ids,
            "novel": novel_updates,
            "redundant_pairs": redundant_pairs,
            "sample_size": sample_size,
            "total_records": total_records,
            "explained_variance": explained_variance[: min(3, len(explained_variance))],
        }
