"""Brain-computer interface abstractions.

This module provides an asynchronous ``BrainComputerInterface`` that interacts
with EEG, EMG, and robotic arm devices through pluggable device adapters. It
supports error recovery and ships with a ``MockDeviceAdapter`` for testing.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Iterable, Optional


class DeviceAdapter:
    """Abstract adapter for BCI peripheral devices."""

    async def read(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    async def write(self, data: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def reset(self) -> None:  # pragma: no cover - interface
        """Reset the device after an error."""
        return None


class BrainComputerInterface:
    """Coordinates asynchronous I/O with a BCI device.

    Data from the device can be consumed via the asynchronous ``stream``
    generator. If a device error occurs, the interface attempts to reset the
    adapter and continue streaming.
    """

    def __init__(self, device: DeviceAdapter) -> None:
        self.device = device
        self._listeners: list[Callable[[Any], None]] = []

    def add_listener(self, callback: Callable[[Any], None]) -> None:
        """Register a callback invoked for every received item."""

        self._listeners.append(callback)

    async def stream(self) -> AsyncIterator[Any]:
        """Yield data from the device indefinitely with error recovery."""

        while True:
            try:
                data = await self.device.read()
            except Exception:  # pragma: no cover - error path tested separately
                await self.device.reset()
                continue
            for cb in self._listeners:
                cb(data)
            yield data

    async def send(self, data: Any) -> None:
        """Send data to the device."""

        await self.device.write(data)


class MockDeviceAdapter(DeviceAdapter):
    """Simple in-memory adapter for testing and simulation.

    Parameters
    ----------
    data_stream:
        Iterable providing initial data to be read from the device.
    fail_at:
        Optional index at which ``read`` will raise ``RuntimeError`` to simulate
        a device failure. After a call to ``reset`` the adapter resumes normal
        operation.
    """

    def __init__(self, data_stream: Iterable[Any], fail_at: Optional[int] = None) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in data_stream:
            self.queue.put_nowait(item)
        self.fail_at = fail_at
        self._count = 0

    async def read(self) -> Any:
        if self.fail_at is not None and self._count >= self.fail_at:
            # simulate failure before consuming next item
            self._count = 0
            raise RuntimeError("simulated device failure")
        self._count += 1
        return await self.queue.get()

    async def write(self, data: Any) -> None:
        await self.queue.put(data)

    async def reset(self) -> None:
        self._count = 0
