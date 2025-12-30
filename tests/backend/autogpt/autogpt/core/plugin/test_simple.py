import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.getcwd(), "backend", "autogpt"))
from third_party.autogpt.autogpt.core.plugin.simple import SimplePluginService


def _write_plugin(path: Path, class_name: str = "MyPlugin") -> None:
    path.write_text(f"class {class_name}:\n    pass\n")


def test_load_from_file_path_py(tmp_path):
    plugin_file = tmp_path / "my_plugin.py"
    _write_plugin(plugin_file, "MyPlugin")
    plugin_cls = SimplePluginService.load_from_file_path(str(plugin_file))
    assert plugin_cls.__name__ == "MyPlugin"


def test_load_from_file_path_zip(tmp_path):
    source = tmp_path / "plug.py"
    _write_plugin(source, "ZipPlugin")
    zip_path = tmp_path / "plugin.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(source, arcname="plug.py")
    plugin_cls = SimplePluginService.load_from_file_path(str(zip_path))
    assert plugin_cls.__name__ == "ZipPlugin"


def test_load_from_file_path_directory(tmp_path):
    pkg = tmp_path / "dirplug"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("class DirPlugin:\n    pass\n")
    plugin_cls = SimplePluginService.load_from_file_path(str(pkg))
    assert plugin_cls.__name__ == "DirPlugin"


def test_resolve_name_to_path_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_file = workspace / "my_plugin.py"
    _write_plugin(plugin_file)
    monkeypatch.setattr(SimplePluginService, "WORKSPACE_PLUGIN_DIR", workspace)
    path = SimplePluginService.resolve_name_to_path("my_plugin", "file")
    assert path == str(plugin_file)


def test_resolve_name_to_path_core(tmp_path, monkeypatch):
    root = tmp_path / "my_core" / "plugins"
    root.mkdir(parents=True)
    (root.parent / "__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    (root / "dummycore.py").write_text("")
    sys.path.insert(0, str(tmp_path))
    monkeypatch.setattr(SimplePluginService, "CORE_PLUGIN_PACKAGE", "my_core.plugins")
    try:
        path = SimplePluginService.resolve_name_to_path("dummycore", "import")
        assert path == "my_core.plugins.dummycore"
    finally:
        sys.path.remove(str(tmp_path))


def test_resolve_name_to_path_auto_package(tmp_path):
    pkg_root = tmp_path / "auto_gpt_plugins"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    (pkg_root / "dummy.py").write_text("")
    sys.path.insert(0, str(tmp_path))
    try:
        path = SimplePluginService.resolve_name_to_path("dummy", "import")
        assert path == "auto_gpt_plugins.dummy"
    finally:
        sys.path.remove(str(tmp_path))
