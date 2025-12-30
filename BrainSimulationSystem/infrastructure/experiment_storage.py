from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import gzip
import shutil
import getpass
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore


_INVALID_PATH_CHARS = set('<>:"/\\|?*')


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding=encoding)
    os.replace(tmp_path, path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_path_component(value: Optional[str], *, fallback: str, max_length: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback

    normalized = []
    for ch in text:
        if ch in _INVALID_PATH_CHARS or ord(ch) < 32:
            normalized.append("_")
        elif ch.isspace():
            normalized.append("_")
        else:
            normalized.append(ch)
    result = "".join(normalized).strip(" ._")
    if not result:
        result = fallback
    if len(result) > max_length:
        result = result[:max_length]
    return result


def _default_experiment_dir_name(experiment_id: str, name: Optional[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    exp = _sanitize_path_component(experiment_id, fallback="experiment", max_length=64)
    nm = _sanitize_path_component(name, fallback="run", max_length=64)
    return f"experiment_{exp}_{ts}_{nm}"


def _choose_unique_dir_name(base_dir: Path, candidate: str) -> str:
    base_dir = Path(base_dir)
    if not (base_dir / candidate).exists():
        return candidate
    for i in range(1, 1000):
        alt = f"{candidate}_{i:03d}"
        if not (base_dir / alt).exists():
            return alt
    raise RuntimeError(f"Failed to choose a unique experiment directory name under {base_dir}")


@dataclass(frozen=True)
class ExperimentDirs:
    root: Path
    config: Path
    data: Path
    results: Path
    model: Path
    sim: Path
    inputs: Path
    outputs: Path
    checkpoints: Path
    logs: Path
    analysis: Path
    visualizations: Path


class ExperimentCatalog:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "catalog.jsonl"

    def append(self, record: Dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)


class ExperimentStorage:
    META_SCHEMA_VERSION = "exp-meta-v1"
    MANIFEST_SCHEMA_VERSION = "exp-manifest-v1"
    PROVENANCE_SCHEMA_VERSION = "exp-provenance-v1"

    def __init__(
        self,
        base_dir: str | Path,
        experiment_id: str,
        *,
        experiment_dir_name: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self._meta_overrides = dict(meta or {})

        auto_dir_name = experiment_dir_name is None
        chosen_dir_name = experiment_dir_name or _default_experiment_dir_name(experiment_id, name)
        if auto_dir_name:
            chosen_dir_name = _choose_unique_dir_name(self.base_dir, chosen_dir_name)
        self.experiment_dir_name = chosen_dir_name

        root = self.base_dir / chosen_dir_name
        config = root / "config"
        data = root / "data"
        results = root / "results"
        self.dirs = ExperimentDirs(
            root=root,
            config=config,
            data=data,
            results=results,
            model=config / "model",
            sim=config / "sim",
            inputs=config / "inputs",
            outputs=results,
            checkpoints=data / "checkpoints",
            logs=results / "logs",
            analysis=results / "analysis",
            visualizations=results / "visualizations",
        )

        self.meta_path = self.dirs.root / "meta.json"
        self.manifest_path = self.dirs.root / "manifest.json"
        self.provenance_path = self.dirs.root / "provenance.json"

        self._manifest: Dict[str, Any] = {}
        self._meta: Dict[str, Any] = {}
        self._manifest_lock = threading.Lock()

    def initialize(self) -> None:
        for directory in (
            self.dirs.root,
            self.dirs.config,
            self.dirs.data,
            self.dirs.results,
            self.dirs.model,
            self.dirs.sim,
            self.dirs.inputs,
            self.dirs.outputs,
            self.dirs.checkpoints,
            self.dirs.logs,
            self.dirs.analysis,
            self.dirs.visualizations,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._meta = self._load_or_create_meta()
        self._manifest = self._load_or_create_manifest()
        self._write_provenance_if_missing()

    def mark_status(self, status: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._meta = self._load_or_create_meta()
        self._meta["status"] = status
        self._meta["updated_at"] = _utc_now_iso()
        if extra:
            self._meta.update(extra)
        _atomic_write_text(self.meta_path, json.dumps(self._meta, ensure_ascii=False, indent=2))

    def finalize(self, status: str, *, summary: Optional[Dict[str, Any]] = None) -> None:
        extra: Dict[str, Any] = {"ended_at": _utc_now_iso()}
        if summary is not None:
            extra["summary"] = summary
        self.mark_status(status, extra=extra)

        ExperimentCatalog(self.base_dir).append(
            {
                "schema_version": "exp-catalog-v1",
                "experiment_id": self.experiment_id,
                "name": self.name,
                "description": self.description,
                "path": str(self.dirs.root),
                "status": status,
                "created_at": self._meta.get("created_at"),
                "ended_at": self._meta.get("ended_at"),
                "project": self._meta.get("project"),
                "responsible": self._meta.get("responsible"),
                "tags": self._meta.get("tags"),
            }
        )

    def write_json(
        self,
        relative_path: str | Path,
        data: Any,
        *,
        kind: str,
        format: str = "json",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target = self.dirs.root / relative_path
        _atomic_write_text(target, json.dumps(data, ensure_ascii=False, indent=2))
        self.register_file(target, kind=kind, format=format, metadata=metadata)
        return target

    def write_yaml(
        self,
        relative_path: str | Path,
        data: Any,
        *,
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("需要安装 PyYAML 库来支持 YAML 写入") from e

        target = self.dirs.root / relative_path
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        _atomic_write_text(target, text)
        self.register_file(target, kind=kind, format="yaml", metadata=metadata)
        return target

    def write_jsonl_gz(
        self,
        relative_path: str | Path,
        records: Iterable[Dict[str, Any]],
        *,
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, target)
        self.register_file(target, kind=kind, format="jsonl.gz", metadata=metadata)
        return target

    def write_csv(
        self,
        relative_path: str | Path,
        rows: Iterable[Dict[str, Any]],
        *,
        kind: str,
        fieldnames: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        import csv

        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        buffered_rows: Optional[list[Dict[str, Any]]] = None
        if fieldnames is None:
            buffered_rows = list(rows)
            field_set: list[str] = []
            for row in buffered_rows:
                for key in row.keys():
                    if key not in field_set:
                        field_set.append(key)
            fieldnames = field_set

        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            if not fieldnames:
                handle.write("")
            else:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                if buffered_rows is not None:
                    for row in buffered_rows:
                        writer.writerow(row)
                else:
                    for row in rows:
                        writer.writerow(row)

        os.replace(tmp_path, target)
        self.register_file(target, kind=kind, format="csv", metadata=metadata)
        return target

    def write_text(
        self,
        relative_path: str | Path,
        text: str,
        *,
        kind: str,
        format: str = "text",
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target = self.dirs.root / relative_path
        _atomic_write_text(target, text, encoding=encoding)
        self.register_file(target, kind=kind, format=format, metadata=metadata)
        return target

    def write_npz(
        self,
        relative_path: str | Path,
        arrays: Dict[str, Any],
        *,
        kind: str,
        compressed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        import numpy as np

        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        with tmp_path.open("wb") as f:
            if compressed:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
        os.replace(tmp_path, target)
        self.register_file(target, kind=kind, format="npz", metadata=metadata)
        return target

    def write_timeseries_npz(
        self,
        relative_path: str | Path,
        series: Dict[str, Any],
        *,
        kind: str,
        compressed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        import numpy as np

        arrays = {name: np.asarray(values) for name, values in (series or {}).items()}
        return self.write_npz(
            relative_path,
            arrays,
            kind=kind,
            compressed=compressed,
            metadata=metadata,
        )

    def write_hdf5(
        self,
        relative_path: str | Path,
        datasets: Dict[str, Any],
        *,
        kind: str,
        compression: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        if h5py is None:
            raise ImportError("需要安装 h5py 库来支持 HDF5 写入")

        import numpy as np

        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        with h5py.File(tmp_path, "w") as handle:
            for name, value in (datasets or {}).items():
                array = np.asarray(value)
                if compression:
                    handle.create_dataset(name, data=array, compression="gzip")
                else:
                    handle.create_dataset(name, data=array)
            handle.attrs["schema_version"] = "hdf5-datasets-v1"

        os.replace(tmp_path, target)
        self.register_file(target, kind=kind, format="hdf5", metadata=metadata)
        return target

    def write_pytables_table(
        self,
        relative_path: str | Path,
        rows: Iterable[Dict[str, Any]],
        *,
        kind: str,
        fieldnames: Optional[list[str]] = None,
        table_name: str = "table",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write a structured table via PyTables (optional dependency).

        This is intended for large, queryable, structured logs (e.g. training logs).
        """
        try:
            import numpy as np
            import tables  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("需要安装 tables(PyTables) 库来支持 HDF5 表格写入") from e

        rows_list = list(rows)
        if fieldnames is None:
            if not rows_list:
                raise ValueError("fieldnames is required when rows is empty")
            fieldnames = list(rows_list[0].keys())
        if not fieldnames:
            raise ValueError("fieldnames must not be empty")

        def infer_dtype(name: str) -> tuple[str, Any]:
            values = [row.get(name) for row in rows_list]
            non_null = [v for v in values if v is not None]
            if not non_null:
                return name, "S1"

            if any(isinstance(v, str) for v in non_null):
                max_len = max(len(str(v).encode("utf-8")) for v in non_null)
                return name, f"S{max(1, max_len)}"

            if any(isinstance(v, (dict, list, tuple)) for v in non_null):
                max_len = max(len(json.dumps(v, ensure_ascii=False).encode("utf-8")) for v in non_null)
                return name, f"S{max(1, max_len)}"

            if any(isinstance(v, (float, np.floating)) for v in non_null):
                return name, np.float64

            if any(isinstance(v, (bool, np.bool_)) for v in non_null):
                if any(v is None for v in values):
                    return name, np.int8
                return name, np.bool_

            if any(isinstance(v, (int, np.integer)) for v in non_null):
                if any(v is None for v in values):
                    return name, np.float64
                return name, np.int64

            max_len = max(len(str(v).encode("utf-8")) for v in non_null)
            return name, f"S{max(1, max_len)}"

        dtype = np.dtype([infer_dtype(name) for name in fieldnames])
        table_data = np.zeros(len(rows_list), dtype=dtype)

        for i, row in enumerate(rows_list):
            for name in fieldnames:
                value = row.get(name)
                column_kind = table_data.dtype[name].kind
                if value is None:
                    if column_kind == "f":
                        table_data[name][i] = np.nan
                    elif column_kind in {"i", "u"}:
                        table_data[name][i] = 0
                    elif column_kind == "b":
                        table_data[name][i] = False
                    elif column_kind == "S":
                        table_data[name][i] = b""
                    else:
                        table_data[name][i] = 0
                    continue

                if column_kind == "f":
                    table_data[name][i] = float(value)
                elif column_kind in {"i", "u"}:
                    table_data[name][i] = int(value)
                elif column_kind == "b":
                    table_data[name][i] = bool(value)
                elif column_kind == "S":
                    if isinstance(value, (dict, list, tuple)):
                        text = json.dumps(value, ensure_ascii=False)
                    else:
                        text = str(value)
                    table_data[name][i] = text.encode("utf-8")
                else:
                    table_data[name][i] = str(value).encode("utf-8")

        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")

        with tables.open_file(str(tmp_path), mode="w") as handle:
            handle.root._v_attrs["schema_version"] = "pytables-table-v1"
            handle.root._v_attrs["experiment_id"] = str(self.experiment_id)
            if metadata:
                handle.root._v_attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)

            table = handle.create_table(
                where="/",
                name=str(table_name),
                description=table_data.dtype,
                expectedrows=max(1, len(table_data)),
            )
            if len(table_data):
                table.append(table_data)
                table.flush()

        os.replace(tmp_path, target)
        self.register_file(target, kind=kind, format="hdf5+pytables", metadata=metadata)
        return target

    def export_spikes_to_nwb(
        self,
        relative_path: str | Path,
        *,
        times,
        senders,
        session_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        try:
            from datetime import datetime  # local import to avoid import cycles

            import numpy as np
            from pynwb import NWBFile, NWBHDF5IO  # type: ignore
            from pynwb.base import TimeSeries  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("需要安装 pynwb 库来支持 NWB 导出") from e

        meta: Dict[str, Any] = {}
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        times_arr = np.asarray(times, dtype=np.float64)
        senders_arr = np.asarray(senders, dtype=np.int64)

        identifier = meta.get("experiment_id") or self.experiment_id
        description = session_description or meta.get("description") or meta.get("name") or "brain simulation"
        start = datetime.now(timezone.utc)

        experimenter = None
        responsible = meta.get("responsible") or {}
        if isinstance(responsible, dict):
            experimenter = responsible.get("name") or responsible.get("user")
        if experimenter:
            experimenter = [str(experimenter)]

        nwbfile = NWBFile(
            session_description=str(description),
            identifier=str(identifier),
            session_start_time=start,
            experimenter=experimenter,
        )

        # Store the raw (time, sender) event list as an acquisition TimeSeries for round-tripping.
        nwbfile.add_acquisition(
            TimeSeries(
                name="spike_events",
                data=senders_arr,
                unit="neuron_id",
                timestamps=times_arr,
                description="Spike event list: timestamps with neuron id (sender) as data.",
            )
        )

        # Group spike times by unit id.
        order = np.argsort(senders_arr, kind="stable")
        senders_sorted = senders_arr[order]
        times_sorted = times_arr[order]

        unique_ids, start_idx = np.unique(senders_sorted, return_index=True)
        end_idx = list(start_idx[1:]) + [len(senders_sorted)]

        for unit_id, a, b in zip(unique_ids.tolist(), start_idx.tolist(), end_idx):
            nwbfile.add_unit(id=int(unit_id), spike_times=times_sorted[a:b])

        target = self.dirs.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with NWBHDF5IO(str(tmp_path), "w") as io:
            io.write(nwbfile)
        os.replace(tmp_path, target)

        self.register_file(target, kind="nwb", format="nwb", metadata=metadata)
        return target

    def export_network_to_sonata_stub(
        self,
        relative_dir: str | Path,
        *,
        network_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Best-effort SONATA-like export.

        This writes minimal node/edge CSV tables based on configuration only.
        For full SONATA compliance, provide real node/edge attributes from the
        simulator/network implementation.
        """

        sonata_dir = Path(relative_dir)
        num_neurons = int(network_config.get("num_neurons", 0))

        nodes_path = sonata_dir / "nodes.csv"
        node_types_path = sonata_dir / "node_types.csv"
        edges_path = sonata_dir / "edges.csv"
        edge_types_path = sonata_dir / "edge_types.csv"
        config_path = sonata_dir / "sonata_config.json"

        def iter_nodes():
            for i in range(num_neurons):
                yield {"node_id": i, "node_type_id": 0}

        self.write_csv(
            nodes_path,
            iter_nodes(),
            kind="sonata_nodes",
            fieldnames=["node_id", "node_type_id"],
            metadata=metadata,
        )

        self.write_csv(
            node_types_path,
            [
                {
                    "node_type_id": 0,
                    "model_type": "point_neuron",
                    "threshold": network_config.get("threshold"),
                }
            ],
            kind="sonata_node_types",
            fieldnames=["node_type_id", "model_type", "threshold"],
            metadata=metadata,
        )

        self.write_csv(
            edges_path,
            [],
            kind="sonata_edges",
            fieldnames=["edge_id", "source_node_id", "target_node_id", "edge_type_id", "syn_weight"],
            metadata=metadata,
        )

        self.write_csv(
            edge_types_path,
            [
                {
                    "edge_type_id": 0,
                    "model_template": "static_synapse",
                    "weight_mean": network_config.get("weight_mean", network_config.get("synaptic_weight")),
                    "weight_std": network_config.get("weight_std", network_config.get("synaptic_weight_std")),
                }
            ],
            kind="sonata_edge_types",
            fieldnames=["edge_type_id", "model_template", "weight_mean", "weight_std"],
            metadata=metadata,
        )

        self.write_json(
            config_path,
            {
                "schema_version": "sonata-stub-v1",
                "experiment_id": self.experiment_id,
                "network": {
                    "nodes": str(nodes_path).replace("\\", "/"),
                    "node_types": str(node_types_path).replace("\\", "/"),
                    "edges": str(edges_path).replace("\\", "/"),
                    "edge_types": str(edge_types_path).replace("\\", "/"),
                },
                "note": "This is a minimal SONATA-like export derived from config only.",
            },
            kind="sonata_config",
            metadata=metadata,
        )

        return {
            "sonata_dir": str(sonata_dir).replace("\\", "/"),
            "nodes": str(nodes_path).replace("\\", "/"),
            "node_types": str(node_types_path).replace("\\", "/"),
            "edges": str(edges_path).replace("\\", "/"),
            "edge_types": str(edge_types_path).replace("\\", "/"),
            "config": str(config_path).replace("\\", "/"),
            "num_neurons": str(num_neurons),
        }

    def copy_into(
        self,
        source_path: str | Path,
        relative_dest_dir: str | Path,
        *,
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[Path]:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(str(source))

        dest_dir = self.dirs.root / relative_dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        if source.is_file():
            dest_file = dest_dir / source.name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_file)
            self.register_file(
                dest_file,
                kind=kind,
                format=dest_file.suffix.lstrip(".") or "file",
                metadata={"source": str(source), **(metadata or {})},
            )
            copied.append(dest_file)
            return copied

        if source.is_dir():
            root_dir = dest_dir / source.name
            root_dir.mkdir(parents=True, exist_ok=True)
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(source)
                dest_file = root_dir / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest_file)
                self.register_file(
                    dest_file,
                    kind=kind,
                    format=dest_file.suffix.lstrip(".") or "file",
                    metadata={
                        "source": str(path),
                        "source_root": str(source),
                        "source_relpath": rel.as_posix(),
                        **(metadata or {}),
                    },
                )
                copied.append(dest_file)
            return copied

        raise ValueError(f"Unsupported source path type: {source}")

    def register_file(
        self,
        path: str | Path,
        *,
        kind: str,
        format: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._manifest_lock:
            file_path = Path(path)
            rel = file_path.relative_to(self.dirs.root).as_posix()
            stat = file_path.stat()
            artifact = {
                "path": rel,
                "kind": kind,
                "format": format,
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": _sha256_file(file_path),
                "metadata": metadata or {},
            }

            self._manifest = self._load_or_create_manifest()
            artifacts: list[dict[str, Any]] = list(self._manifest.get("artifacts", []))
            artifacts = [a for a in artifacts if a.get("path") != rel]
            artifacts.append(artifact)
            self._manifest["artifacts"] = sorted(artifacts, key=lambda a: a.get("path", ""))
            self._manifest["updated_at"] = _utc_now_iso()
            _atomic_write_text(
                self.manifest_path, json.dumps(self._manifest, ensure_ascii=False, indent=2)
            )

    def register_files(
        self,
        paths: Iterable[str | Path],
        *,
        kind: str,
        format: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        for path in paths:
            self.register_file(path, kind=kind, format=format, metadata=metadata)

    def _load_or_create_meta(self) -> Dict[str, Any]:
        changed = False
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            meta = {
                "schema_version": self.META_SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "name": self.name,
                "description": self.description,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "status": "created",
            }
            changed = True

        meta.setdefault("schema_version", self.META_SCHEMA_VERSION)
        meta.setdefault("experiment_id", self.experiment_id)
        if self.name is not None:
            meta.setdefault("name", self.name)
        if self.description is not None:
            meta.setdefault("description", self.description)
        meta.setdefault("created_at", _utc_now_iso())
        meta.setdefault("updated_at", _utc_now_iso())
        meta.setdefault("status", "created")

        meta.setdefault("project", self._get_project_info())
        meta.setdefault("responsible", self._get_default_responsible())
        meta.setdefault("tags", [])
        meta.setdefault("metadata", {})
        meta.setdefault("storage", {})
        if isinstance(meta.get("storage"), dict):
            meta["storage"].setdefault("directory_name", self.experiment_dir_name)
            meta["storage"].setdefault("layout", "config-data-results-v1")
        meta.setdefault("host", platform.node())
        meta.setdefault("git", self._get_git_info())

        changed = self._apply_meta_overrides(meta) or changed

        if changed:
            _atomic_write_text(self.meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def _apply_meta_overrides(self, meta: Dict[str, Any]) -> bool:
        if not self._meta_overrides:
            return False

        changed = False
        for key, value in self._meta_overrides.items():
            if value is None:
                continue
            if isinstance(value, dict) and isinstance(meta.get(key), dict):
                for inner_key, inner_value in value.items():
                    if inner_value is None:
                        continue
                    if meta[key].get(inner_key) != inner_value:
                        meta[key][inner_key] = inner_value
                        changed = True
            else:
                if meta.get(key) != value:
                    meta[key] = value
                    changed = True
        return changed

    def _get_project_info(self) -> Dict[str, Any]:
        project: Dict[str, Any] = {"name": "BrainSimulationSystem", "version": None}
        try:
            import BrainSimulationSystem  # type: ignore

            project["version"] = getattr(BrainSimulationSystem, "__version__", None)
        except Exception:
            project["version"] = None
        return project

    def _get_default_responsible(self) -> Dict[str, Any]:
        user = None
        try:
            user = getpass.getuser()
        except Exception:
            user = os.environ.get("USERNAME") or os.environ.get("USER")
        return {"user": user}

    def _load_or_create_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))

        manifest = {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "artifacts": [],
        }
        _atomic_write_text(self.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    def _write_provenance_if_missing(self) -> None:
        if self.provenance_path.exists():
            return

        provenance = {
            "schema_version": self.PROVENANCE_SCHEMA_VERSION,
            "created_at": _utc_now_iso(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "git": self._get_git_info(),
        }
        _atomic_write_text(self.provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2))

    def _get_git_info(self) -> Dict[str, Any]:
        try:
            commit = (
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(self.dirs.root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                .stdout.strip()
            )
            is_dirty = (
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(self.dirs.root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                .stdout.strip()
                != ""
            )
            return {"commit": commit or None, "dirty": is_dirty}
        except Exception:
            return {"commit": None, "dirty": None}
