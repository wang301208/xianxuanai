import textwrap
from pathlib import Path

from importlib.metadata import EntryPoint, EntryPoints
from watchdog.events import FileModifiedEvent

from algorithms import plugin_loader


def _write_plugin(tmp_path: Path, body: str) -> Path:
    file = tmp_path / "tmp_plugin.py"
    file.write_text(textwrap.dedent(body))
    return file


def test_load_update_and_rollback(tmp_path, monkeypatch):
    plugin_file = _write_plugin(
        tmp_path,
        """
        def value():
            return 1
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint(name="tmp", value="tmp_plugin", group="autogpt.algorithms")
    monkeypatch.setattr(plugin_loader, "entry_points", lambda: EntryPoints([ep]))

    loader = plugin_loader.AlgorithmPluginLoader()
    loader.load_plugins()
    assert loader.modules["tmp"].value() == 1

    plugin_file.write_text("def value():\n    return 2\n")
    loader.on_modified(FileModifiedEvent(str(plugin_file)))
    assert loader.modules["tmp"].value() == 2

    plugin_file.write_text("def value():\n    return 3\n\nbroken =")
    loader.on_modified(FileModifiedEvent(str(plugin_file)))
    assert loader.modules["tmp"].value() == 2
