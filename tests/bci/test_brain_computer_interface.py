import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.bci import BrainComputerInterface, MockDeviceAdapter


def test_stream_basic():
    adapter = MockDeviceAdapter([1, 2, 3])
    bci = BrainComputerInterface(adapter)

    async def runner():
        stream = bci.stream()
        return [await stream.__anext__() for _ in range(3)]

    results = asyncio.run(runner())
    assert results == [1, 2, 3]


def test_send_and_receive():
    adapter = MockDeviceAdapter([])
    bci = BrainComputerInterface(adapter)

    async def runner():
        await bci.send("ping")
        stream = bci.stream()
        return await stream.__anext__()

    assert asyncio.run(runner()) == "ping"


def test_error_recovery():
    adapter = MockDeviceAdapter(["ok", "after"], fail_at=1)
    bci = BrainComputerInterface(adapter)

    async def runner():
        stream = bci.stream()
        first = await stream.__anext__()
        second = await stream.__anext__()
        return first, second

    assert asyncio.run(runner()) == ("ok", "after")
