"""AutoGPT 异步编程工具模块。

本模块提供了 AutoGPT 项目中异步编程的辅助工具，
主要解决在不同上下文中执行异步代码的兼容性问题。

主要功能:
    - 智能的协程执行管理
    - 事件循环兼容性处理
    - 线程安全的异步操作

应用场景:
    - 在同步代码中调用异步函数
    - 处理嵌套事件循环的情况
    - 确保异步操作的线程安全性
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable


def run_async(coro: Awaitable[Any]) -> Any:
    """智能执行协程，兼容各种事件循环环境。
    
    这个函数能够在不同的异步环境中正确执行协程，
    自动处理事件循环的存在与否，确保代码的兼容性。
    
    执行策略:
        - 如果当前线程没有运行事件循环，使用 asyncio.run() 创建新循环
        - 如果当前线程已有运行中的事件循环，使用线程安全的方式调度协程
    
    参数:
        coro: 要执行的协程对象（Awaitable）
        
    返回:
        Any: 协程执行的结果
        
    异常:
        可能抛出协程执行过程中的任何异常
        
    使用场景:
        1. 在同步函数中调用异步 API
        2. 在 Jupyter Notebook 等已有事件循环的环境中执行异步代码
        3. 在测试代码中混合同步和异步操作
        4. 在回调函数中执行异步操作
        
    使用示例:
        # 在同步代码中调用异步函数
        async def fetch_data():
            # 异步获取数据的逻辑
            return "data"
            
        # 同步调用
        result = run_async(fetch_data())
        print(result)  # 输出: "data"
        
    技术细节:
        - 使用 asyncio.get_running_loop() 检测当前事件循环
        - 使用 asyncio.run_coroutine_threadsafe() 确保线程安全
        - 自动处理 RuntimeError 异常来判断循环状态
    """
    try:
        # 尝试获取当前线程中运行的事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的事件循环，创建新的事件循环来执行协程
        return asyncio.run(coro)
    else:
        # 已有运行中的事件循环，使用线程安全的方式调度协程
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # 等待协程完成并返回结果
        return future.result()


class AsyncTaskQueue:
    """线程安全的异步任务队列。

    该队列通过专用事件循环和 :class:`asyncio.Queue` 实现任务调度，
    允许从任意线程提交协程而无需显式加锁。内部使用
    ``asyncio.run_coroutine_threadsafe`` 将任务放入队列，从而利用
    ``asyncio`` 的无锁队列特性保证线程安全。
    """

    def __init__(self, workers: int = 4) -> None:
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[Awaitable[Any]] = asyncio.Queue()
        self._workers = workers
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # 启动指定数量的工作协程
        for _ in range(self._workers):
            self._submit_to_loop(self._worker())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit_to_loop(self, coro: Awaitable[Any]) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _worker(self) -> None:
        while True:
            task = await self._queue.get()
            try:
                await task
            finally:
                self._queue.task_done()

    def submit(self, coro: Awaitable[Any]) -> None:
        """提交协程任务到队列。"""

        self._submit_to_loop(self._queue.put(coro))

    def wait_all(self) -> None:
        """等待所有已提交的任务完成。"""

        self._submit_to_loop(self._queue.join()).result()
