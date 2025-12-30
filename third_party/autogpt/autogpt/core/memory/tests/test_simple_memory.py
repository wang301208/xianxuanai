import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from autogpt.core.memory.simple import SimpleMemory
from autogpt.core.workspace.simple import SimpleWorkspace


def _make_memory(tmp_path: Path) -> SimpleMemory:
    logger = logging.getLogger("simple_memory_test")
    ws_settings = SimpleWorkspace.default_settings.copy(deep=True)
    ws_settings.configuration.root = str(tmp_path)
    workspace = SimpleWorkspace(ws_settings, logger)
    mem_settings = SimpleMemory.default_settings.copy(deep=True)
    return SimpleMemory(mem_settings, logger, workspace)


def test_summarize_and_retrieve(tmp_path):
    memory = _make_memory(tmp_path)
    messages = [
        "the cat sat on the mat",
        "dog chased the cat",
        "sun is bright",
        "I like turtles",
    ]
    for m in messages:
        memory.add(m)
    memory.summarize_and_archive(max_history_length=2)

    assert memory.get() == ["sun is bright", "I like turtles"]

    cat_result = memory.get(query="cat", limit=1)
    assert cat_result and "cat" in cat_result[0]

    turtle_result = memory.get(query="turtles", limit=1)
    assert turtle_result == ["I like turtles"]
