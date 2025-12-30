import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.security.safe_modification import SafeModificationSandbox


def test_rollback_on_commit_failure(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    file = target / "data.txt"
    file.write_text("original")

    sandbox = SafeModificationSandbox(target)
    with sandbox:
        sandbox.apply_change("data.txt", "changed")

        # Simulate failure while copying back to target
        import modules.security.safe_modification as sm
        original_copy2 = sm.shutil.copy2

        def broken_copy(src, dst, *args, **kwargs):
            # fail only when writing back to the target file
            if Path(dst) == file:
                raise IOError("disk full")
            return original_copy2(src, dst, *args, **kwargs)

        monkeypatch.setattr(sm.shutil, "copy2", broken_copy)

        with pytest.raises(IOError):
            sandbox.commit()

    # The original file should remain untouched after rollback
    assert file.read_text() == "original"


def test_prevent_path_traversal(tmp_path):
    target = tmp_path / "target"
    target.mkdir()

    with SafeModificationSandbox(target) as sandbox:
        with pytest.raises(ValueError):
            sandbox.apply_change("../evil.txt", "bad")

    assert not (target.parent / "evil.txt").exists()
