import asyncio
from pathlib import Path

import yaml

from backend.agent_factory import _parse_blueprint
from backend.forge.forge.sdk.db import AgentDB


def _make_blueprint(tmp_path: Path) -> Path:
    data = {"role_name": "test", "core_prompt": "test"}
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def test_parse_blueprint_benchmark(benchmark, tmp_path):
    path = _make_blueprint(tmp_path)
    _parse_blueprint(path)  # warm cache
    benchmark(_parse_blueprint, path)


def test_agentdb_create_task_benchmark(benchmark, tmp_path):
    db_path = tmp_path / "bench.sqlite3"
    db = AgentDB(f"sqlite:///{db_path}")

    async def _create():
        await db.create_task("input")

    benchmark(lambda: asyncio.run(_create()))
