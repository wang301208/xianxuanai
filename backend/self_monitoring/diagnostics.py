from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..memory.long_term import LongTermMemory
from .monitor import RecoveryDecision

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .health import SensorReading


@dataclass
class DiagnosticReport:
    """Outcome of a diagnostic investigation."""

    sensor: str
    status: str
    actions: List[str] = field(default_factory=list)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemDiagnostics:
    """Diagnose anomalies and attempt automated recovery procedures."""

    def __init__(self, *, log_reader: Optional[Any] = None) -> None:
        self._log_reader = log_reader

    # ------------------------------------------------------------------
    def diagnose(
        self,
        sensor: str,
        reading: "SensorReading",
        decision: Optional[RecoveryDecision] = None,
    ) -> DiagnosticReport:
        """Analyse an anomalous reading and attempt corrective actions."""

        category = getattr(reading, "category", None) or reading.metadata.get("category")
        actions: List[str] = []
        metadata: Dict[str, Any] = {}
        notes: List[str] = []
        status = "observed"

        if category == "memory":
            memory_obj = reading.metadata.get("memory")
            path = reading.metadata.get("path") or getattr(memory_obj, "path", None)
            backup_path = reading.metadata.get("backup_path")
            new_memory, repaired, mem_actions = self.ensure_memory_store(
                memory_obj,
                path=path,
                backup_path=backup_path,
            )
            metadata["memory"] = new_memory
            actions.extend(mem_actions)
            status = "recovered" if repaired else "verified"
            notes.append("Long-term memory store checked")
        elif category == "model":
            model_path = reading.metadata.get("model_path")
            snapshot_dir = reading.metadata.get("snapshot_dir")
            snapshot_name = reading.metadata.get("snapshot_name")
            try:
                restored, target, model_actions = self.restore_model_snapshot(
                    model_path=model_path,
                    snapshot_dir=snapshot_dir,
                    snapshot_name=snapshot_name,
                )
                actions.extend(model_actions)
                metadata["model_path"] = str(target) if target else None
                status = "restored" if restored else "unresolved"
                notes.append("Model snapshot recovery attempted")
            except Exception as exc:  # pragma: no cover - defensive
                status = "failed"
                actions.append(f"model_restore_error:{exc}")
                notes.append(str(exc))
        elif category == "sensor":
            reset_fn = reading.metadata.get("reset")
            if callable(reset_fn):
                try:
                    reset_fn()
                    actions.append("sensor_reset_invoked")
                    status = "recovered"
                except Exception as exc:  # pragma: no cover - defensive
                    actions.append(f"sensor_reset_failed:{exc}")
                    status = "failed"
                    notes.append(str(exc))
            else:
                status = "queued"
                actions.append("sensor_followup_required")
        else:
            if decision is not None and decision.should_retry:
                status = "retrying"
            else:
                status = "logged"
            actions.append("no_automatic_repair")

        if self._log_reader:
            try:
                recent_logs = list(self._log_reader(20))
                metadata["logs"] = recent_logs
                actions.append("logs_collected")
            except Exception:  # pragma: no cover - defensive
                pass

        notes_text = reading.message or ""
        if notes:
            if notes_text:
                notes_text += " | "
            notes_text += "; ".join(notes)

        return DiagnosticReport(
            sensor=sensor,
            status=status,
            actions=actions,
            notes=notes_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def ensure_memory_store(
        self,
        memory: Optional[LongTermMemory],
        *,
        path: Optional[str | Path] = None,
        backup_path: Optional[str | Path] = None,
    ) -> Tuple[LongTermMemory, bool, List[str]]:
        """Verify the memory store is writable, restoring from backup if needed."""

        actions: List[str] = []
        if memory is not None:
            try:
                memory.store(
                    "probe",
                    metadata={"category": "_health_check", "tags": ("system",)},
                )
                actions.append("memory_write_success")
                return memory, False, actions
            except sqlite3.Error:
                actions.append("memory_write_failed")
            except Exception:
                actions.append("memory_write_failed")
            try:
                memory.close()
            except Exception:
                pass

        if path is None:
            raise ValueError("path is required to reinitialise memory store")

        target_path = Path(path)
        repaired = False

        if backup_path is not None:
            backup = Path(backup_path)
            if backup.exists():
                shutil.copy2(backup, target_path)
                actions.append(f"memory_restored_from_backup:{backup}")
                repaired = True

        if not target_path.exists():
            repaired = True
        else:
            try:
                sqlite3.connect(target_path).close()
            except sqlite3.Error:
                target_path.unlink(missing_ok=True)
                repaired = True

        new_memory = LongTermMemory(target_path)
        actions.append("memory_reinitialised")
        return new_memory, repaired, actions

    def restore_model_snapshot(
        self,
        *,
        model_path: Optional[str | Path],
        snapshot_dir: Optional[str | Path],
        snapshot_name: Optional[str | Path] = None,
    ) -> Tuple[bool, Optional[Path], List[str]]:
        """Restore a model file from the most recent or specified snapshot."""

        if model_path is None or snapshot_dir is None:
            raise ValueError("model_path and snapshot_dir are required")

        actions: List[str] = []
        model_target = Path(model_path)
        snapshots_root = Path(snapshot_dir)

        if not snapshots_root.exists():
            raise FileNotFoundError(f"Snapshot directory {snapshots_root} not found")

        if snapshot_name is not None:
            snapshot_candidate = snapshots_root / snapshot_name
            if not snapshot_candidate.exists():
                raise FileNotFoundError(f"Snapshot {snapshot_candidate} not found")
        else:
            candidates = [p for p in snapshots_root.iterdir() if p.is_file()]
            if not candidates:
                raise FileNotFoundError(f"No snapshots found in {snapshots_root}")
            snapshot_candidate = max(candidates, key=lambda p: p.stat().st_mtime)

        model_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_candidate, model_target)
        actions.append(f"model_restored:{snapshot_candidate.name}")
        return True, model_target, actions
