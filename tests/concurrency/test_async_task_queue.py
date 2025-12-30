import asyncio
import threading
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.common.async_utils import AsyncTaskQueue, run_async


def test_queue_thread_safety():
    queue = AsyncTaskQueue(workers=8)
    counter = 0
    lock = asyncio.Lock()

    async def increment():
        nonlocal counter
        # 模拟异步 I/O
        await asyncio.sleep(0.01)
        async with lock:
            counter += 1

    def submit_many():
        for _ in range(25):
            queue.submit(increment())

    threads = [threading.Thread(target=submit_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    queue.wait_all()
    assert counter == 100


def test_queue_performance():
    queue = AsyncTaskQueue(workers=8)

    async def io_task():
        await asyncio.sleep(0.05)

    # 顺序执行
    start = time.perf_counter()
    for _ in range(20):
        run_async(io_task())
    sequential = time.perf_counter() - start

    # 并发执行
    start = time.perf_counter()
    for _ in range(20):
        queue.submit(io_task())
    queue.wait_all()
    concurrent = time.perf_counter() - start

    assert concurrent < sequential
