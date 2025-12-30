"""Filesystem sandbox / transaction overlay for ToolEnvironmentBridge.

This sandbox implements a conservative *overlay* model:
  - Write/modify/create/delete operations are applied to a sandbox directory.
  - Reads/listing prefer sandboxed content when present.
  - Deletions are represented as tombstones so the sandbox view hides originals.

It is designed for "transaction-like" execution: an agent can safely perform
file operations and inspect results without touching the real files until a
separate commit step is performed (which should be gated by approvals).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _root_tag(root: Path) -> str:
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass
class FilesystemSandboxConfig:
    enabled: bool = False
    root: Optional[str] = None
    namespace_by_root: bool = True
    keep_history: bool = True  # when False, reset clears all content after commit


@dataclass
class SandboxStatus:
    root: str
    enabled: bool
    changed: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)


class FilesystemSandbox:
    """Overlay sandbox for filesystem operations."""

    def __init__(self, root: Path, *, allowed_roots: Iterable[Path]) -> None:
        self.root = Path(root).resolve()
        self._allowed_roots = [Path(p).resolve() for p in allowed_roots]
        self._tag_to_root = { _root_tag(r): r for r in self._allowed_roots }
        self._tombstones: Set[Tuple[str, str]] = set()  # (tag, rel_posix)
        self._changed: Set[Tuple[str, str]] = set()  # (tag, rel_posix)
        self.root.mkdir(parents=True, exist_ok=True)

    def _map(self, path: Path) -> Tuple[str, Path, Path]:
        resolved = Path(path).resolve()
        for root in self._allowed_roots:
            try:
                rel = resolved.relative_to(root)
            except Exception:
                continue
            tag = _root_tag(root)
            sandbox_path = self.root / tag / rel
            return tag, resolved, sandbox_path
        raise ValueError("path_not_in_allowed_roots")

    def _key(self, tag: str, resolved: Path, *, root: Optional[Path] = None) -> Tuple[str, str]:
        base = root or self._tag_to_root.get(tag)
        if base is None:
            return tag, resolved.as_posix()
        try:
            rel = resolved.relative_to(base)
        except Exception:
            rel = Path(resolved.name)
        return tag, rel.as_posix()

    def status(self) -> SandboxStatus:
        changed = [f"{tag}:{rel}" for tag, rel in sorted(self._changed)]
        deleted = [f"{tag}:{rel}" for tag, rel in sorted(self._tombstones)]
        return SandboxStatus(root=str(self.root), enabled=True, changed=changed, deleted=deleted)

    # ------------------------------------------------------------------ #
    # Overlay operations
    # ------------------------------------------------------------------ #
    def exists(self, path: Path) -> bool:
        tag, resolved, sandbox_path = self._map(path)
        key = self._key(tag, resolved)
        if key in self._tombstones:
            return False
        return sandbox_path.exists() or resolved.exists()

    def read_text(self, path: Path, *, encoding: str = "utf-8", errors: str = "replace") -> Tuple[str, Dict[str, Any]]:
        tag, resolved, sandbox_path = self._map(path)
        key = self._key(tag, resolved)
        if key in self._tombstones:
            return "", {"error": "file_missing", "sandboxed": True, "tombstoned": True, "path": str(resolved)}
        source = "original"
        target = resolved
        if sandbox_path.exists():
            source = "sandbox"
            target = sandbox_path
        text = target.read_text(encoding=encoding, errors=errors)
        return text, {"sandboxed": True, "source": source, "path": str(resolved), "sandbox_path": str(sandbox_path)}

    def read_bytes(self, path: Path) -> Tuple[bytes, Dict[str, Any]]:
        tag, resolved, sandbox_path = self._map(path)
        key = self._key(tag, resolved)
        if key in self._tombstones:
            return b"", {"error": "file_missing", "sandboxed": True, "tombstoned": True, "path": str(resolved)}
        source = "original"
        target = resolved
        if sandbox_path.exists():
            source = "sandbox"
            target = sandbox_path
        data = target.read_bytes()
        return data, {
            "sandboxed": True,
            "source": source,
            "path": str(resolved),
            "sandbox_path": str(sandbox_path),
            "bytes": int(len(data)),
        }

    def list_dir(self, path: Path, *, max_entries: int = 50) -> Tuple[List[str], Dict[str, Any]]:
        tag, resolved, sandbox_path = self._map(path)
        original_entries: Dict[str, bool] = {}
        sandbox_entries: Dict[str, bool] = {}

        original_is_dir = resolved.exists() and resolved.is_dir()
        sandbox_is_dir = sandbox_path.exists() and sandbox_path.is_dir()

        if not original_is_dir and not sandbox_is_dir:
            if resolved.exists() and not resolved.is_dir():
                raise NotADirectoryError(str(resolved))
            raise FileNotFoundError(str(resolved))

        if original_is_dir:
            for entry in resolved.iterdir():
                original_entries[entry.name] = entry.is_dir()

        if sandbox_is_dir:
            for entry in sandbox_path.iterdir():
                sandbox_entries[entry.name] = entry.is_dir()

        merged: Dict[str, bool] = dict(original_entries)
        merged.update(sandbox_entries)

        # Apply tombstones (only files are supported by the bridge right now).
        for name in list(merged.keys()):
            rel_key = self._key(tag, (resolved / name))
            if rel_key in self._tombstones and not merged.get(name, False):
                merged.pop(name, None)

        names = sorted(merged.keys())
        rendered = [n + ("/" if merged.get(n) else "") for n in names[: max(0, int(max_entries))]]
        return rendered, {"sandboxed": True, "path": str(resolved), "sandbox_path": str(sandbox_path), "entries": rendered}

    def write_text(
        self,
        path: Path,
        text: str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        append: bool = False,
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        tag, resolved, sandbox_path = self._map(path)
        key = self._key(tag, resolved)
        self._tombstones.discard(key)
        if create_parents:
            sandbox_path.parent.mkdir(parents=True, exist_ok=True)

        if append and not sandbox_path.exists() and resolved.exists():
            sandbox_path.write_text(resolved.read_text(encoding=encoding, errors=errors), encoding=encoding, errors=errors)

        mode = "a" if append else "w"
        with sandbox_path.open(mode, encoding=encoding, errors=errors, newline="") as handle:
            handle.write(str(text))

        rel_key = self._key(tag, resolved)
        self._changed.add(rel_key)
        return {"sandboxed": True, "path": str(resolved), "sandbox_path": str(sandbox_path), "append": bool(append)}

    def mkdir(self, path: Path, *, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        tag, resolved, sandbox_path = self._map(path)
        sandbox_path.mkdir(parents=parents, exist_ok=exist_ok)
        rel_key = self._key(tag, resolved)
        self._changed.add(rel_key)
        return {"sandboxed": True, "path": str(resolved), "sandbox_path": str(sandbox_path)}

    def delete_file(self, path: Path) -> Dict[str, Any]:
        tag, resolved, sandbox_path = self._map(path)
        rel_key = self._key(tag, resolved)
        self._tombstones.add(rel_key)
        if sandbox_path.exists() and sandbox_path.is_file():
            try:
                sandbox_path.unlink()
            except Exception:
                pass
        return {"sandboxed": True, "path": str(resolved), "sandbox_path": str(sandbox_path), "tombstoned": True}

    # ------------------------------------------------------------------ #
    # Commit / reset
    # ------------------------------------------------------------------ #
    def commit(self) -> Dict[str, Any]:
        """Apply sandbox changes onto the original roots."""

        copied = 0
        deleted = 0
        missing_roots: List[str] = []

        for tag_dir in self.root.iterdir():
            if not tag_dir.is_dir():
                continue
            tag = tag_dir.name
            root = self._tag_to_root.get(tag)
            if root is None:
                missing_roots.append(tag)
                continue

            for file_path in tag_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(tag_dir)
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest)
                copied += 1

            for tomb_tag, rel_posix in list(self._tombstones):
                if tomb_tag != tag:
                    continue
                target = root / Path(rel_posix)
                if target.exists() and target.is_file():
                    try:
                        target.unlink()
                        deleted += 1
                    except Exception:
                        pass

        return {
            "sandbox_root": str(self.root),
            "copied_files": copied,
            "deleted_files": deleted,
            "unknown_root_tags": missing_roots,
        }

    def reset(self) -> Dict[str, Any]:
        """Clear sandbox state and files."""

        try:
            if self.root.exists():
                shutil.rmtree(self.root)
        finally:
            self.root.mkdir(parents=True, exist_ok=True)
            self._tombstones.clear()
            self._changed.clear()
        return {"sandbox_root": str(self.root), "reset": True}


__all__ = ["FilesystemSandbox", "FilesystemSandboxConfig", "SandboxStatus"]
