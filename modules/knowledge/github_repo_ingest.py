from __future__ import annotations

"""Helpers for ingesting GitHub repositories as a local code corpus.

This module is dependency-light (stdlib only) so it can be used by both runtime
services and sandboxed tool bridges. Network access is performed via the
provided ``download_fn`` to keep the module testable/offline by default.
"""

import hashlib
import io
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

DownloadFn = Callable[[str, int, float, Dict[str, str]], bytes]


def parse_github_repo(value: str) -> Tuple[str, str]:
    """Return (owner, repo) for a GitHub repo identifier.

    Accepts:
    - ``owner/repo``
    - ``https://github.com/owner/repo`` (and variants with trailing slashes)
    - ``git@github.com:owner/repo.git``
    """

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("missing_repo")

    # SSH style: git@github.com:owner/repo(.git)?
    match = re.match(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", raw, flags=re.IGNORECASE)
    if match:
        return match.group("owner"), match.group("repo")

    # URL style.
    if raw.startswith("http://") or raw.startswith("https://"):
        stripped = raw.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        match = re.match(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)$", stripped, flags=re.IGNORECASE)
        if match:
            repo = match.group("repo")
            if repo.endswith(".git"):
                repo = repo[: -len(".git")]
            return match.group("owner"), repo

    # Plain owner/repo
    if "/" in raw:
        owner, repo = raw.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if repo.endswith(".git"):
            repo = repo[: -len(".git")]
        if owner and repo and "/" not in owner and "/" not in repo:
            return owner, repo

    raise ValueError("invalid_github_repo")


def build_codeload_zip_url(owner: str, repo: str, ref: str) -> str:
    """Build GitHub codeload zip URL for a repo/ref."""

    ref_value = str(ref or "").strip() or "main"
    owner_value = str(owner or "").strip()
    repo_value = str(repo or "").strip()
    if not owner_value or not repo_value:
        raise ValueError("invalid_repo")
    return f"https://codeload.github.com/{owner_value}/{repo_value}/zip/{ref_value}"


def _default_download(url: str, max_bytes: int, timeout_s: float, headers: Dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    limit = max(1, int(max_bytes))
    with urllib.request.urlopen(request, timeout=float(timeout_s)) as resp:  # noqa: S310 - controlled by callers
        chunks: list[bytes] = []
        size = 0
        while True:
            block = resp.read(64 * 1024)
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if limit > 0 and size > limit:
                raise ValueError("download_too_large")
        return b"".join(chunks)


def _stable_id(*parts: str) -> str:
    payload = "|".join(str(p or "") for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _ensure_under_root(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except Exception as exc:
        raise ValueError("path_traversal") from exc
    return resolved


def safe_extract_zip_bytes(
    zip_bytes: bytes,
    *,
    dest_dir: Path,
    max_files: int = 20_000,
    max_unzipped_bytes: int = 120_000_000,
    allow_suffixes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Extract zip bytes into ``dest_dir`` with basic zip-slip/bomb protection."""

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = dest_dir.resolve()
    allow_suffix_set = None
    if allow_suffixes is not None:
        allow_suffix_set = {str(s).lower() for s in allow_suffixes if str(s).strip()}

    extracted_files = 0
    extracted_bytes = 0
    skipped_files = 0
    started = time.time()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = zf.infolist()
        if max_files > 0 and len(members) > max_files:
            raise ValueError("zip_too_many_entries")

        for info in members:
            name = str(getattr(info, "filename", "") or "")
            if not name or name.endswith("/"):
                continue

            rel = PurePosixPath(name)
            # Reject absolute paths, traversal, Windows drive tricks, and odd separators.
            if rel.is_absolute():
                raise ValueError("zip_path_traversal")
            if any(part in {"..", ""} for part in rel.parts):
                raise ValueError("zip_path_traversal")
            if any("\\" in part or ":" in part for part in rel.parts):
                raise ValueError("zip_path_traversal")

            if allow_suffix_set is not None:
                suffix = rel.suffix.lower()
                if suffix and suffix not in allow_suffix_set:
                    skipped_files += 1
                    continue

            target = _ensure_under_root(resolved_root.joinpath(*rel.parts), resolved_root)
            target.parent.mkdir(parents=True, exist_ok=True)

            file_size = int(getattr(info, "file_size", 0) or 0)
            extracted_bytes += file_size
            if max_unzipped_bytes > 0 and extracted_bytes > max_unzipped_bytes:
                raise ValueError("zip_unzipped_too_large")

            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(64 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

            extracted_files += 1
            if max_files > 0 and extracted_files >= max_files:
                break

    duration_s = round(time.time() - started, 4)
    return {
        "dest_dir": str(resolved_root),
        "extracted_files": int(extracted_files),
        "extracted_bytes": int(extracted_bytes),
        "skipped_files": int(skipped_files),
        "duration_s": duration_s,
    }


def detect_single_top_level_dir(dest_dir: Path) -> Optional[Path]:
    """Return the single top-level directory inside ``dest_dir`` if it exists."""

    dest_dir = Path(dest_dir)
    if not dest_dir.exists() or not dest_dir.is_dir():
        return None
    children = [p for p in dest_dir.iterdir() if p.name not in {".", ".."}]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return None


@dataclass(frozen=True)
class GitHubRepoIngestConfig:
    dest_root: Path
    max_download_bytes: int = 30_000_000
    max_unzipped_bytes: int = 120_000_000
    max_extract_files: int = 20_000
    extract_suffixes: Sequence[str] | None = (".py", ".md", ".txt")
    timeout_s: float = 20.0
    user_agent: str = "CodeCorpusIngest/1.0"


@dataclass(frozen=True)
class GitHubRepoIngestResult:
    repo: str
    ref: str
    url: str
    extracted_dir: str
    repo_root: str
    download_bytes: int
    extract_stats: Dict[str, Any]
    manifest_path: Optional[str] = None


class GitHubRepoIngestor:
    def __init__(self, *, download_fn: DownloadFn | None = None) -> None:
        self._download = download_fn or _default_download

    def ingest(
        self,
        repo: str,
        *,
        ref: str = "main",
        config: GitHubRepoIngestConfig,
        force: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GitHubRepoIngestResult:
        owner, name = parse_github_repo(repo)
        ref_value = str(ref or "").strip() or "main"
        url = build_codeload_zip_url(owner, name, ref_value)

        dest_root = Path(config.dest_root).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        stable = _stable_id(owner, name, ref_value)
        base = f"{owner}_{name}_{stable}".lower()
        if force:
            base = f"{base}_{int(time.time())}"
        extracted_dir = dest_root / base
        extracted_dir.mkdir(parents=True, exist_ok=True)

        headers = {"User-Agent": config.user_agent}
        zip_bytes = self._download(url, int(config.max_download_bytes), float(config.timeout_s), headers)
        stats = safe_extract_zip_bytes(
            zip_bytes,
            dest_dir=extracted_dir,
            max_files=int(config.max_extract_files),
            max_unzipped_bytes=int(config.max_unzipped_bytes),
            allow_suffixes=config.extract_suffixes,
        )

        repo_root = detect_single_top_level_dir(extracted_dir) or extracted_dir

        manifest_path = extracted_dir / ".ingest.json"
        payload: Dict[str, Any] = {
            "source": "github",
            "repo": f"{owner}/{name}",
            "ref": ref_value,
            "url": url,
            "download_bytes": int(len(zip_bytes)),
            "extracted_dir": str(extracted_dir),
            "repo_root": str(repo_root),
            "extracted_at": time.time(),
            "stats": dict(stats),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        try:
            manifest_path.write_text(
                json_dumps(payload) + "\n",
                encoding="utf-8",
            )
            manifest_rendered: Optional[str] = str(manifest_path)
        except Exception:
            manifest_rendered = None

        return GitHubRepoIngestResult(
            repo=f"{owner}/{name}",
            ref=ref_value,
            url=url,
            extracted_dir=str(extracted_dir),
            repo_root=str(repo_root),
            download_bytes=int(len(zip_bytes)),
            extract_stats=dict(stats),
            manifest_path=manifest_rendered,
        )


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


__all__ = [
    "GitHubRepoIngestConfig",
    "GitHubRepoIngestResult",
    "GitHubRepoIngestor",
    "parse_github_repo",
    "build_codeload_zip_url",
    "safe_extract_zip_bytes",
    "detect_single_top_level_dir",
]
