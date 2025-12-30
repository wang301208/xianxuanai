from __future__ import annotations

"""Persistent meta-experience store for self-improvement attempts.

The SelfImprovementManager can record the outcome of each improvement attempt
and later consult these "meta experiences" to prioritise remediation actions.

Storage format: JSONL (one JSON object per line).
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def metric_group(metric: str) -> str:
    token = str(metric or "").strip()
    if not token:
        return "unknown"
    return token.split("_", 1)[0].strip().lower() or "unknown"


def _objective_gain(record: Mapping[str, Any]) -> float | None:
    baseline = _safe_float(record.get("baseline"))
    average = _safe_float(record.get("average"))
    if baseline is None or average is None:
        return None
    direction = str(record.get("direction") or "").strip().lower()
    if direction not in {"increase", "decrease"}:
        direction = "increase"
    return (average - baseline) if direction == "increase" else (baseline - average)


@dataclass(frozen=True)
class KindStats:
    kind: str
    trials: int
    successes: int
    success_rate: float
    avg_gain: float
    score: float


class ImprovementExperienceStore:
    """Abstract interface for experience persistence and retrieval."""

    def record(self, record: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def rank_kinds(self, *, metric: str, kinds: Sequence[str]) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def stats_for(self, *, metric: str, kinds: Sequence[str]) -> Dict[str, KindStats]:  # pragma: no cover - interface
        raise NotImplementedError


class NullExperienceStore(ImprovementExperienceStore):
    def record(self, record: Mapping[str, Any]) -> None:
        return

    def rank_kinds(self, *, metric: str, kinds: Sequence[str]) -> list[str]:
        del metric
        return list(kinds)

    def stats_for(self, *, metric: str, kinds: Sequence[str]) -> Dict[str, KindStats]:
        del metric
        return {}


class InMemoryExperienceStore(ImprovementExperienceStore):
    def __init__(self, records: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._records: list[dict[str, Any]] = [dict(r) for r in (records or []) if isinstance(r, Mapping)]
        self._lock = threading.RLock()
        self._min_trials = 1

    def record(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            return
        with self._lock:
            self._records.append(dict(record))

    def rank_kinds(self, *, metric: str, kinds: Sequence[str]) -> list[str]:
        stats = self.stats_for(metric=metric, kinds=kinds)
        if not stats:
            return list(kinds)
        known = [k for k in kinds if k in stats and stats[k].trials > 0]
        if not known:
            return list(kinds)
        ranked_known = sorted(known, key=lambda k: stats[k].score, reverse=True)
        unknown = [k for k in kinds if k not in ranked_known]
        return ranked_known + unknown

    def stats_for(self, *, metric: str, kinds: Sequence[str]) -> Dict[str, KindStats]:
        kinds_set = {str(k) for k in kinds if str(k)}
        if not kinds_set:
            return {}
        with self._lock:
            records = list(self._records)
        return _compute_stats(records, metric=metric, kinds=kinds_set, min_trials=self._min_trials)


class JsonlExperienceStore(ImprovementExperienceStore):
    def __init__(
        self,
        *,
        path: str | os.PathLike[str],
        enabled: bool = True,
        max_records: int = 500,
        min_trials: int = 1,
        score_gain_weight: float = 0.15,
    ) -> None:
        self._path = Path(path)
        self._enabled = bool(enabled)
        self._max_records = max(10, int(max_records))
        self._min_trials = max(1, int(min_trials))
        self._score_gain_weight = max(0.0, float(score_gain_weight))
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls) -> ImprovementExperienceStore:
        enabled = _env_bool("SELF_IMPROVEMENT_EXPERIENCE_ENABLED", True)
        if not enabled:
            return NullExperienceStore()
        root = Path(os.getenv("WORKSPACE_ROOT") or Path.cwd()).resolve()
        default_path = root / "memory" / "self_improvement_experience.jsonl"
        path = Path(os.getenv("SELF_IMPROVEMENT_EXPERIENCE_PATH") or default_path)
        max_records = _env_int("SELF_IMPROVEMENT_EXPERIENCE_MAX_RECORDS", 500)
        min_trials = _env_int("SELF_IMPROVEMENT_EXPERIENCE_MIN_TRIALS", 1)
        weight = float(os.getenv("SELF_IMPROVEMENT_EXPERIENCE_GAIN_WEIGHT") or 0.15)
        return cls(path=path, enabled=True, max_records=max_records, min_trials=min_trials, score_gain_weight=weight)

    def record(self, record: Mapping[str, Any]) -> None:
        if not self._enabled or not isinstance(record, Mapping):
            return
        payload = dict(record)
        payload.setdefault("time", time.time())
        metric = str(payload.get("metric") or "").strip()
        if metric and "metric_group" not in payload:
            payload["metric_group"] = metric_group(metric)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to append self-improvement experience", exc_info=True)

    def rank_kinds(self, *, metric: str, kinds: Sequence[str]) -> list[str]:
        stats = self.stats_for(metric=metric, kinds=kinds)
        if not stats:
            return list(kinds)
        known = [k for k in kinds if k in stats and stats[k].trials > 0]
        if not known:
            return list(kinds)
        ranked_known = sorted(known, key=lambda k: stats[k].score, reverse=True)
        unknown = [k for k in kinds if k not in ranked_known]
        return ranked_known + unknown

    def stats_for(self, *, metric: str, kinds: Sequence[str]) -> Dict[str, KindStats]:
        if not self._enabled:
            return {}
        kinds_set = {str(k) for k in kinds if str(k)}
        if not kinds_set:
            return {}
        records = list(self._read_recent_records())
        return _compute_stats(
            records,
            metric=str(metric),
            kinds=kinds_set,
            min_trials=self._min_trials,
            gain_weight=self._score_gain_weight,
        )

    def _read_recent_records(self) -> Iterable[Dict[str, Any]]:
        with self._lock:
            path = self._path
        if not path.exists():
            return []
        buffer: deque[Dict[str, Any]] = deque(maxlen=self._max_records)
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        buffer.append(obj)
        except Exception:  # pragma: no cover - best effort
            return []
        return list(buffer)


def _compute_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    kinds: set[str],
    min_trials: int,
    gain_weight: float = 0.15,
) -> Dict[str, KindStats]:
    metric = str(metric or "").strip()
    if not metric:
        return {}
    group = metric_group(metric)

    candidates: list[Mapping[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        if not rec.get("evaluated", False):
            continue
        if str(rec.get("kind") or "") not in kinds:
            continue
        if str(rec.get("metric") or "") == metric:
            candidates.append(rec)

    if not candidates:
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            if not rec.get("evaluated", False):
                continue
            if str(rec.get("kind") or "") not in kinds:
                continue
            if str(rec.get("metric_group") or "") == group:
                candidates.append(rec)

    if not candidates:
        return {}

    by_kind: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in candidates:
        kind = str(rec.get("kind") or "")
        if kind:
            by_kind[kind].append(rec)

    stats: Dict[str, KindStats] = {}
    for kind, items in by_kind.items():
        trials = len(items)
        successes = sum(1 for r in items if bool(r.get("success")))
        success_rate = float(successes) / float(trials) if trials > 0 else 0.0

        gains = []
        for r in items:
            gain = _objective_gain(r)
            if gain is not None:
                gains.append(float(gain))
        avg_gain = sum(gains) / float(len(gains)) if gains else 0.0

        score = success_rate + float(gain_weight) * avg_gain
        stats[kind] = KindStats(
            kind=kind,
            trials=int(trials),
            successes=int(successes),
            success_rate=float(success_rate),
            avg_gain=float(avg_gain),
            score=float(score),
        )

    # If we have only tiny samples overall, avoid overfitting.
    if sum(s.trials for s in stats.values()) < max(1, int(min_trials)):
        return {}
    return stats


__all__ = [
    "ImprovementExperienceStore",
    "NullExperienceStore",
    "InMemoryExperienceStore",
    "JsonlExperienceStore",
    "KindStats",
    "metric_group",
]
