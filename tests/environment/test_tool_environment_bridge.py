from BrainSimulationSystem.environment.base import EnvironmentAdapter, ObservationTransformer
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge


def test_tool_environment_read_file_allowed(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")

    env = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    adapter = EnvironmentAdapter(env, transformer=ObservationTransformer())
    adapter.reset()

    packet, reward, terminated, info = adapter.step({"type": "read_file", "path": str(target)})
    assert packet.text == "hello"
    assert reward > 0
    assert terminated is False
    assert info["path"].endswith("hello.txt")


def test_tool_environment_read_file_blocked(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")

    env = ToolEnvironmentBridge(allowed_roots=[tmp_path / "subdir"])
    adapter = EnvironmentAdapter(env)
    adapter.reset()

    packet, reward, terminated, info = adapter.step({"type": "read_file", "path": str(target)})
    assert packet.text in (None, "")
    assert reward < 0
    assert terminated is False
    assert info["blocked"] is True


def test_tool_environment_shell_disabled_by_default():
    env = ToolEnvironmentBridge(allowed_shell_prefixes=[])
    adapter = EnvironmentAdapter(env)
    adapter.reset()

    packet, reward, terminated, info = adapter.step({"type": "shell", "command": ["python", "-c", "print(1)"]})
    assert packet.text in (None, "")
    assert reward < 0
    assert terminated is False
    assert info["blocked"] is True

