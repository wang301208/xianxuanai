"""Utilities for lightweight online training during the learning cycle."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def _safe_timestamp(entry: Mapping[str, Any]) -> float:
    try:
        return float(entry.get("timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0


def apply_online_model_updates(
    predictive_model: Any,
    working_memory: Iterable[MutableMapping[str, Any]] | deque[MutableMapping[str, Any]],
    *,
    max_samples: int = 8,
) -> dict[str, float]:
    """Feed recent perceptions back into the predictive model for online updates.

    The helper reuses short-term working memory entries marked as ``type`` ==
    ``"perception"`` that have not yet been trained (``trained`` flag absent or
    falsy). Entries are processed from most recent to oldest until ``max_samples``
    is reached. Each trained entry is marked with ``trained=True`` to avoid
    duplicate updates across cycles.

    Returns a small metrics dictionary capturing how many samples were
    processed, along with averaged reconstruction and prediction losses when
    available from the model summaries.
    """

    if predictive_model is None:
        return {}

    count = 0
    reconstruction_total = 0.0
    prediction_total = 0.0
    prediction_error_total = 0.0
    reconstruction_seen = 0
    prediction_seen = 0
    error_seen = 0

    def _accumulate(summary: Mapping[str, Any]) -> None:
        nonlocal reconstruction_total, prediction_total, prediction_error_total
        nonlocal reconstruction_seen, prediction_seen, error_seen

        recon = summary.get("reconstruction_loss")
        if recon is not None:
            reconstruction_total += float(recon)
            reconstruction_seen += 1
        pred = summary.get("prediction_loss")
        if pred is not None:
            prediction_total += float(pred)
            prediction_seen += 1
        pred_err = summary.get("prediction_error")
        if pred_err is not None:
            prediction_error_total += float(pred_err)
            error_seen += 1

    # Sort newest-first by timestamp so we refresh the freshest context first.
    sorted_entries = sorted(
        [e for e in working_memory if isinstance(e, MutableMapping)],
        key=_safe_timestamp,
        reverse=True,
    )

    selected: list[MutableMapping[str, Any]] = []
    for entry in sorted_entries:
        if len(selected) >= int(max_samples):
            break
        if entry.get("type") != "perception":
            continue
        if entry.get("trained"):
            continue
        selected.append(entry)

    if not selected:
        return {}

    data_batch = [entry.get("data") or {} for entry in selected]
    metadata_batch: list[dict[str, Any]] = []
    for entry in selected:
        metadata: dict[str, Any] = {"timestamp": entry.get("timestamp")}
        meta = entry.get("metadata")
        if isinstance(meta, Mapping):
            metadata.update(meta)
        metadata_batch.append(metadata)

    observe_batch = getattr(predictive_model, "observe_batch", None)
    if callable(observe_batch):
        try:
            summary = observe_batch(data_batch, metadata=metadata_batch)  # type: ignore[misc]
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
        for entry, data, metadata in zip(selected, data_batch, metadata_batch):
            summary = predictive_model.observe(data, metadata=metadata)  # type: ignore[arg-type]
            if isinstance(summary, Mapping):
                _accumulate(summary)
            entry["trained"] = True
            count += 1

    if count == 0:
        return {}

    metrics: dict[str, float] = {"online_updates": float(count)}
    if reconstruction_seen:
        metrics["online_reconstruction_loss"] = reconstruction_total / reconstruction_seen
    if prediction_seen:
        metrics["online_prediction_loss"] = prediction_total / prediction_seen
    if error_seen:
        metrics["online_prediction_error"] = prediction_error_total / error_seen
    return metrics
