from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from modules.knowledge.github_repo_ingest import parse_github_repo, safe_extract_zip_bytes


def _zip_bytes(names: list[str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, "print('x')\n")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("org/repo", ("org", "repo")),
        ("https://github.com/org/repo", ("org", "repo")),
        ("https://github.com/org/repo/", ("org", "repo")),
        ("git@github.com:org/repo.git", ("org", "repo")),
    ],
)
def test_parse_github_repo(raw: str, expected: tuple[str, str]) -> None:
    assert parse_github_repo(raw) == expected


def test_safe_extract_blocks_zip_slip(tmp_path: Path) -> None:
    for name in ("../evil.py", "/abs/evil.py", "C:evil.py", "repo/..\\evil.py"):
        payload = _zip_bytes([name])
        with pytest.raises(ValueError):
            safe_extract_zip_bytes(payload, dest_dir=tmp_path / "out")


def test_safe_extract_suffix_filter(tmp_path: Path) -> None:
    payload = _zip_bytes(["repo/a.py", "repo/b.bin"])
    stats = safe_extract_zip_bytes(payload, dest_dir=tmp_path / "out", allow_suffixes=[".py"])
    assert stats["extracted_files"] == 1
    assert stats["skipped_files"] == 1
