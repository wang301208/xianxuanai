#!/usr/bin/env python3
"""
Manage large model/data assets required by the repo.

This script is intentionally dependency-free (stdlib only) so it can run during bootstrap.

Manifest format (JSON):
{
  "version": 1,
  "assets": [
    {
      "id": "brainsim_decision_policy",
      "path": "BrainSimulationSystem/models/rl/decision_policy.zip",
      "optional": true,
      "sha256": "hex" | "",
      "groups": ["brainsim", "rl"],
      "source": {
        "type": "url",
        "url": "https://...",
        "headers": {"Authorization": "Bearer ${HUGGINGFACE_API_TOKEN}"},
        "size_bytes": 123456789
      },
      "extract": {
        "type": "zip|tar",
        "destination": "models/some_dir"
      },
      "instructions": "How to obtain this asset (e.g. set URL/token or run a training script)."
    }
  ]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _configure_io() -> None:
    # Avoid crashing on Windows consoles that cannot encode emoji.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(template: str) -> tuple[str, list[str]]:
    missing: list[str] = []

    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        value = os.getenv(var)
        if value is None:
            missing.append(var)
            return ""
        return value

    return _ENV_PATTERN.sub(repl, template), missing


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = dest_dir.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            resolved = member_path.resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                raise ValueError(f"Refusing to extract outside destination: {member.filename}")
        zf.extractall(dest_dir)


def _safe_extract_tar(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = dest_dir.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            member_path = dest_dir / member.name
            resolved = member_path.resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                raise ValueError(f"Refusing to extract outside destination: {member.name}")
        tf.extractall(dest_dir)


def _http_head_size(url: str, headers: dict[str, str]) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (validated URL source)
            length = resp.headers.get("Content-Length")
            return int(length) if length and length.isdigit() else None
    except urllib.error.HTTPError as exc:
        if exc.code in {405, 403, 404}:
            return None
        raise


def _download_file(
    *,
    url: str,
    dest: Path,
    headers: dict[str, str],
    expected_sha256: str | None,
    expected_size: int | None,
    show_progress: bool = True,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".partial")

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    downloaded = 0
    last_update = 0.0

    def maybe_report(force: bool = False) -> None:
        nonlocal last_update
        if not show_progress:
            return
        now = time.time()
        if not force and (now - last_update) < 0.2:
            return
        last_update = now
        elapsed = max(now - start_time, 1e-6)
        speed = downloaded / elapsed
        if expected_size:
            pct = min(downloaded / expected_size * 100.0, 100.0)
            msg = (
                f"\r{pct:6.2f}%  {_human_bytes(downloaded)}/{_human_bytes(expected_size)}"
                f"  {_human_bytes(int(speed))}/s"
            )
        else:
            msg = f"\r{_human_bytes(downloaded)}  {_human_bytes(int(speed))}/s"
        sys.stderr.write(msg)
        sys.stderr.flush()

    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (validated URL source)
        content_len = resp.headers.get("Content-Length")
        if expected_size is None and content_len and content_len.isdigit():
            expected_size = int(content_len)

        with tmp_path.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                maybe_report()

    maybe_report(force=True)
    if show_progress:
        sys.stderr.write("\n")

    if expected_size is not None and downloaded != expected_size:
        raise ValueError(
            f"Download size mismatch for {dest} (got {downloaded} bytes, expected {expected_size})."
        )

    if expected_sha256:
        actual = _sha256_file(tmp_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {dest} (got {actual}).")

    tmp_path.replace(dest)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    rel_path: str
    optional: bool
    sha256: str | None
    groups: tuple[str, ...]
    source: dict[str, Any] | None
    extract: dict[str, Any] | None
    instructions: str | None

    def resolved_path(self, root: Path) -> Path:
        path = Path(self.rel_path)
        return path if path.is_absolute() else (root / path)


def _load_manifest(path: Path) -> list[Asset]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError("Unsupported manifest version (expected 1).")
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, list):
        raise ValueError("Manifest must contain an 'assets' array.")

    assets: list[Asset] = []
    for item in assets_raw:
        if not isinstance(item, dict):
            raise ValueError("Each asset entry must be an object.")
        asset_id = str(item.get("id") or "").strip()
        rel_path = str(item.get("path") or "").strip()
        if not asset_id or not rel_path:
            raise ValueError("Each asset must have non-empty 'id' and 'path'.")
        optional = bool(item.get("optional", True))
        sha256 = str(item.get("sha256") or "").strip() or None
        groups = tuple(str(g) for g in (item.get("groups") or []) if str(g).strip())
        source = item.get("source") if isinstance(item.get("source"), dict) else None
        extract = item.get("extract") if isinstance(item.get("extract"), dict) else None
        instructions = str(item.get("instructions") or "").strip() or None
        assets.append(
            Asset(
                asset_id=asset_id,
                rel_path=rel_path,
                optional=optional,
                sha256=sha256,
                groups=groups,
                source=source,
                extract=extract,
                instructions=instructions,
            )
        )
    return assets


def _iter_selected(assets: Iterable[Asset], only: set[str], groups: set[str]) -> list[Asset]:
    selected: list[Asset] = []
    for asset in assets:
        if only and asset.asset_id not in only:
            continue
        if groups and not (set(asset.groups) & groups):
            continue
        selected.append(asset)
    return selected


def _asset_status(asset: Asset, root: Path) -> tuple[str, str | None]:
    path = asset.resolved_path(root)
    if not path.exists():
        return "missing", None
    if asset.sha256:
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            return "error", f"unable to read file: {exc}"
        if actual.lower() != asset.sha256.lower():
            return "bad_hash", f"sha256 mismatch (got {actual})"
    return "ok", None


def _resolve_url_and_headers(source: dict[str, Any]) -> tuple[str | None, dict[str, str], list[str]]:
    if str(source.get("type") or "").strip() != "url":
        return None, {}, []

    url_template = str(source.get("url") or "").strip()
    if not url_template:
        return None, {}, []

    url, missing = _expand_env_vars(url_template)
    headers: dict[str, str] = {}
    headers_raw = source.get("headers") if isinstance(source.get("headers"), dict) else {}
    for key, val in headers_raw.items():
        key_s = str(key)
        val_s = str(val)
        expanded, miss = _expand_env_vars(val_s)
        missing.extend(miss)
        if expanded:
            headers[key_s] = expanded
    return url, headers, sorted(set(missing))


def _maybe_extract(asset: Asset, root: Path) -> None:
    if not asset.extract:
        return
    kind = str(asset.extract.get("type") or "").strip().lower()
    destination = str(asset.extract.get("destination") or "").strip()
    if not kind or not destination:
        raise ValueError(f"Asset {asset.asset_id} has invalid extract configuration.")

    archive = asset.resolved_path(root)
    dest_dir = (root / destination).resolve()
    if kind == "zip":
        _safe_extract_zip(archive, dest_dir)
    elif kind == "tar":
        _safe_extract_tar(archive, dest_dir)
    else:
        raise ValueError(f"Unsupported extract type '{kind}' for asset {asset.asset_id}.")


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = Path(args.manifest)
    manifest_path = manifest if manifest.is_absolute() else (root / manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    assets = _load_manifest(manifest_path)
    selected = _iter_selected(assets, set(args.only or []), set(args.group or []))
    if not selected:
        print("No matching assets in manifest.")
        return 0

    missing_required = False
    missing_any = False
    for asset in selected:
        status, detail = _asset_status(asset, root)
        path = asset.resolved_path(root)
        if status == "ok":
            print(f"[OK] {asset.asset_id}: {path}")
        else:
            missing_any = True
            prefix = "WARN" if asset.optional else "ERROR"
            print(f"[{prefix}] {asset.asset_id}: {path} ({status}{': ' + detail if detail else ''})")
            if not asset.optional:
                missing_required = True
            if asset.instructions:
                print(f"       hint: {asset.instructions}")

    if args.strict:
        return 1 if missing_any else 0
    return 1 if missing_required else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = Path(args.manifest)
    manifest_path = manifest if manifest.is_absolute() else (root / manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    assets = _load_manifest(manifest_path)
    selected = _iter_selected(assets, set(args.only or []), set(args.group or []))
    if not selected:
        print("No matching assets in manifest.")
        return 0

    failed_required = False
    failed_any = False
    for asset in selected:
        status, detail = _asset_status(asset, root)
        path = asset.resolved_path(root)
        if status == "ok":
            print(f"[OK] {asset.asset_id}: {path}")
            continue

        url, headers, missing_env = _resolve_url_and_headers(asset.source or {})
        if not url:
            msg = f"Missing download URL for asset {asset.asset_id}."
            prefix = "WARN" if asset.optional else "ERROR"
            print(f"[{prefix}] {msg}")
            if asset.instructions:
                print(f"       hint: {asset.instructions}")
            failed_any = True
            if not asset.optional:
                failed_required = True
            continue
        if missing_env:
            prefix = "WARN" if asset.optional else "ERROR"
            print(
                f"[{prefix}] Missing env var(s) required to download {asset.asset_id}: {', '.join(missing_env)}"
            )
            if asset.instructions:
                print(f"       hint: {asset.instructions}")
            failed_any = True
            if not asset.optional:
                failed_required = True
            continue

        expected_size = None
        if isinstance(asset.source, dict):
            size_val = asset.source.get("size_bytes")
            if isinstance(size_val, int) and size_val > 0:
                expected_size = size_val
        if expected_size is None:
            try:
                expected_size = _http_head_size(url, headers)
            except Exception:
                expected_size = None

        size_str = _human_bytes(expected_size) if expected_size else "unknown"
        print(f"Asset: {asset.asset_id}")
        print(f"  target: {path}")
        print(f"  source: {url}")
        print(f"  size:   {size_str}")
        if asset.sha256:
            print(f"  sha256: {asset.sha256}")

        if not args.yes:
            answer = input("Download now? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Skipped.")
                failed_any = True
                if not asset.optional:
                    failed_required = True
                continue

        try:
            _download_file(
                url=url,
                dest=path,
                headers=headers,
                expected_sha256=asset.sha256,
                expected_size=expected_size,
            )
            _maybe_extract(asset, root)
            print(f"[OK] Downloaded {asset.asset_id} -> {path}")
        except Exception as exc:
            prefix = "WARN" if asset.optional else "ERROR"
            print(f"[{prefix}] Failed to fetch {asset.asset_id}: {exc}")
            try:
                if path.exists() and path.is_file():
                    pass
                else:
                    partial = path.with_suffix(path.suffix + ".partial")
                    if partial.exists():
                        partial.unlink()
            except Exception:
                pass
            failed_any = True
            if not asset.optional:
                failed_required = True

    if args.strict:
        return 1 if (failed_required or failed_any) else 0
    return 1 if failed_required else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fetch_assets.py", description="Check/fetch model & data assets.")
    parser.add_argument(
        "--root",
        default=str(_repo_root()),
        help="Repo root directory used to resolve relative paths (default: repo root).",
    )
    parser.add_argument(
        "--manifest",
        default="config/assets.json",
        help="Path to assets manifest (default: config/assets.json).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--root",
            default=argparse.SUPPRESS,
            help="Repo root directory used to resolve relative paths (can be set before or after the command).",
        )
        sub.add_argument(
            "--manifest",
            default=argparse.SUPPRESS,
            help="Path to assets manifest (can be set before or after the command).",
        )
        sub.add_argument("--only", action="append", default=[], help="Only process a specific asset id (repeatable).")
        sub.add_argument("--group", action="append", default=[], help="Only process a specific group (repeatable).")
        sub.add_argument(
            "--strict",
            action="store_true",
            help="Fail the command if any selected asset is missing/unavailable.",
        )

    check = subparsers.add_parser("check", help="Check whether assets exist and match hashes (if provided).")
    add_common(check)
    check.set_defaults(func=cmd_check)

    fetch = subparsers.add_parser("fetch", help="Download missing assets from configured sources.")
    add_common(fetch)
    fetch.add_argument("--yes", action="store_true", help="Do not prompt; assume 'yes' for downloads.")
    fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_io()
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
