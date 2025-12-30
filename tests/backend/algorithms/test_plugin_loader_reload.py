import logging
import textwrap
from pathlib import Path

from importlib.metadata import EntryPoint, EntryPoints
from watchdog.events import FileModifiedEvent

from algorithms import plugin_loader


def _write_plugin(tmp_path: Path, body: str) -> Path:
    file = tmp_path / "tmp_plugin_reload.py"
    file.write_text(textwrap.dedent(body))
    return file


def test_reload_failure_rolls_back_and_logs(tmp_path, monkeypatch, caplog):
    plugin_file = _write_plugin(
        tmp_path,
        """
        def value():
            return 1
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint(name="tmp_reload", value="tmp_plugin_reload", group="autogpt.algorithms")
    monkeypatch.setattr(plugin_loader, "entry_points", lambda: EntryPoints([ep]))

    loader = plugin_loader.AlgorithmPluginLoader()
    loader.load_plugins()
    assert loader.modules["tmp_reload"].value() == 1

    plugin_file.write_text("def value():\n    return 2\n")
    loader.on_modified(FileModifiedEvent(str(plugin_file)))
    assert loader.modules["tmp_reload"].value() == 2

    plugin_file.write_text(
        "import nonexistent_module\n\n" "def value():\n    return 3\n"
    )
    with caplog.at_level(logging.ERROR):
        loader.on_modified(FileModifiedEvent(str(plugin_file)))
    assert loader.modules["tmp_reload"].value() == 2
    assert "Failed to reload plugin tmp_reload" in caplog.text
