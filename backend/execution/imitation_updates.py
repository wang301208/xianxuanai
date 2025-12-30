"""Utilities for lightweight online imitation (behavior cloning) updates."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def _safe_timestamp(entry: Mapping[str, Any]) -> float:
    try:
        return float(entry.get("timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0


def apply_online_imitation_updates(
    imitation_model: Any,
    demonstrations: Iterable[MutableMapping[str, Any]] | deque[MutableMapping[str, Any]],
    *,
    max_samples: int = 16,
    min_samples: int = 1,
) -> dict[str, float]:
    """Train an imitation model from recent demonstration samples.

    Demonstration entries are expected to contain at least ``state`` and ``action``.
    Trained samples are marked with ``trained=True`` to avoid duplicate updates.

    The imitation model may implement either:
    - ``observe(state, action, **kwargs) -> Mapping``; or
    - ``observe_batch(batch, **kwargs) -> Mapping | Sequence[Mapping]`` where ``batch`` is
      a list of demonstration mappings.
    """

    if imitation_model is None:
        return {}

    max_samples = int(max(1, max_samples))
    min_samples = int(max(1, min_samples))
    candidates = [e for e in demonstrations if isinstance(e, MutableMapping)]
    candidates.sort(key=_safe_timestamp, reverse=True)

    selected: list[MutableMapping[str, Any]] = []
    for entry in candidates:
        if len(selected) >= max_samples:
            break
        if entry.get("trained"):
            continue
        action = entry.get("action") or entry.get("command")
        state = entry.get("state")
        if not action or not isinstance(state, Mapping):
            continue
        selected.append(entry)

    if len(selected) < min_samples:
        return {}

    count = 0
    loss_total = 0.0
    loss_seen = 0
    entropy_total = 0.0
    entropy_seen = 0
    accuracy_total = 0.0
    accuracy_seen = 0

    def _accumulate(summary: Mapping[str, Any]) -> None:
        nonlocal loss_total, loss_seen, entropy_total, entropy_seen, accuracy_total, accuracy_seen

        loss = summary.get("loss")
        if loss is not None:
            loss_total += float(loss)
            loss_seen += 1
        entropy = summary.get("entropy")
        if entropy is not None:
            entropy_total += float(entropy)
            entropy_seen += 1
        accuracy = summary.get("accuracy")
        if accuracy is not None:
            accuracy_total += float(accuracy)
            accuracy_seen += 1

    observe_batch = getattr(imitation_model, "observe_batch", None)
    if callable(observe_batch):
        try:
            summary = observe_batch(list(selected))
        except Exception:
            summary = None
        if isinstance(summary, Mapping):
            _accumulate(summary)
            count = len(selected)
            for entry in selected:
                entry["trained"] = True
        elif isinstance(summary, Sequence):
            for item in summary:
                if isinstance(item, Mapping):
                    _accumulate(item)
                    count += 1
            if count:
                for entry in selected[:count]:
                    entry["trained"] = True

    if count == 0:
        observe = getattr(imitation_model, "observe", None)
        if not callable(observe):
            return {}
        for entry in selected:
            state = entry.get("state")
            action = entry.get("action") or entry.get("command")
            if not action or not isinstance(state, Mapping):
                continue
            summary = observe(state, action)
            if isinstance(summary, Mapping):
                _accumulate(summary)
            entry["trained"] = True
            count += 1

    if count == 0:
        return {}

    metrics: dict[str, float] = {"imitation_updates": float(count)}
    if loss_seen:
        metrics["imitation_loss"] = loss_total / float(loss_seen)
    if entropy_seen:
        metrics["imitation_entropy"] = entropy_total / float(entropy_seen)
    if accuracy_seen:
        metrics["imitation_accuracy"] = accuracy_total / float(accuracy_seen)
    return metrics


__all__ = ["apply_online_imitation_updates"]
