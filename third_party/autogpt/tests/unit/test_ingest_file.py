import numpy
import numpy
import pytest
from pytest_mock import MockerFixture

import autogpt.commands.file_operations as file_ops
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding
from autogpt.config import Config
from autogpt.file_storage import FileStorage


class DummyAgent:
    def __init__(self, workspace: FileStorage, config: Config):
        self.workspace = workspace
        self.legacy_config = config


class SimpleMemory:
    def __init__(self):
        self.items: list[MemoryItem] = []

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, x):
        return x in self.items

    def __len__(self):
        return len(self.items)

    def add(self, item: MemoryItem):
        self.items.append(item)

    def discard(self, item: MemoryItem):
        if item in self.items:
            self.items.remove(item)


@pytest.fixture
def mock_embedding() -> Embedding:
    return numpy.full((1,), 0.0255, numpy.float32)


@pytest.fixture
def memory_json_file():
    return SimpleMemory()


@pytest.mark.asyncio
async def test_ingest_file_updates_memory(
    storage: FileStorage,
    config: Config,
    memory_json_file,
    mocker: MockerFixture,
    mock_embedding: Embedding,
):
    agent = DummyAgent(storage, config)

    async def make_memory(content: str, path: str, cfg: Config):
        return MemoryItem(
            raw_content=content,
            summary="",
            chunk_summaries=[""],
            chunks=[content],
            e_summary=mock_embedding,
            e_weighted=mock_embedding,
            e_chunks=[mock_embedding],
            metadata={"location": path, "source_type": "text_file"},
        )

    mocker.patch_object(
        file_ops.MemoryItemFactory,
        "from_text_file",
        side_effect=make_memory,
    )

    await storage.write_file("test.txt", "first")
    file_ops.ingest_file("test.txt", memory_json_file, agent)
    assert len(memory_json_file) == 1
    assert next(iter(memory_json_file)).raw_content == "first"

    await storage.write_file("test.txt", "second")
    file_ops.ingest_file("test.txt", memory_json_file, agent)
    assert len(memory_json_file) == 1
    assert next(iter(memory_json_file)).raw_content == "second"


@pytest.mark.asyncio
async def test_ingest_file_handles_file_types(
    storage: FileStorage,
    config: Config,
    memory_json_file,
    mocker: MockerFixture,
    mock_embedding: Embedding,
):
    agent = DummyAgent(storage, config)

    async def make_text_memory(content: str, path: str, cfg: Config):
        return MemoryItem(
            raw_content=content,
            summary="",
            chunk_summaries=[""],
            chunks=[content],
            e_summary=mock_embedding,
            e_weighted=mock_embedding,
            e_chunks=[mock_embedding],
            metadata={"location": path, "source_type": "text_file"},
        )

    async def make_code_memory(content: str, path: str, cfg: Config):
        return MemoryItem(
            raw_content=content,
            summary="",
            chunk_summaries=[""],
            chunks=[content],
            e_summary=mock_embedding,
            e_weighted=mock_embedding,
            e_chunks=[mock_embedding],
            metadata={"location": path, "source_type": "code_file"},
        )

    text_mock = mocker.patch_object(
        file_ops.MemoryItemFactory,
        "from_text_file",
        side_effect=make_text_memory,
    )
    code_mock = mocker.patch_object(
        file_ops.MemoryItemFactory,
        "from_code_file",
        side_effect=make_code_memory,
    )

    await storage.write_file("note.txt", "hello")
    file_ops.ingest_file("note.txt", memory_json_file, agent)
    assert text_mock.call_count == 1
    assert len(memory_json_file) == 1

    await storage.write_file("script.py", "print('hi')")
    file_ops.ingest_file("script.py", memory_json_file, agent)
    assert code_mock.call_count == 1
    assert len(memory_json_file) == 2

    with storage.open_file("data.bin", mode="w", binary=True) as f:
        f.write(b"\x00\x01")
    file_ops.ingest_file("data.bin", memory_json_file, agent)
    assert text_mock.call_count == 1
    assert code_mock.call_count == 1
    assert len(memory_json_file) == 2

