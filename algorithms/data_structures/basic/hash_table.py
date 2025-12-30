from __future__ import annotations

from typing import Any, List, Tuple

from algorithms.base import Algorithm


class HashTable(Algorithm):
    """基于拉链法实现的哈希表。

    哈希表是一种通过哈希函数将键映射到桶（bucket）的数据结构。
    当多个键映射到同一个桶时，使用拉链法（链表）处理冲突。

    主要操作：
        - put: 插入或更新键值对
        - get: 根据键获取对应的值
        - delete: 删除键值对

    当负载因子（元素数量 / 桶数量）超过设定阈值时，哈希表会自动扩容。
    """

    def __init__(self, initial_capacity: int = 8, load_factor: float = 0.75) -> None:
        """初始化哈希表。

        参数:
            initial_capacity: 初始桶数量
            load_factor: 负载因子阈值，超过后触发扩容
        """
        self.capacity = max(1, initial_capacity)
        self.load_factor = load_factor
        self.buckets: List[List[Tuple[Any, Any]]] = [list() for _ in range(self.capacity)]
        self.size = 0

    def _hash(self, key: Any) -> int:
        """计算键的哈希索引。"""
        return hash(key) % self.capacity

    def put(self, key: Any, value: Any) -> None:
        """插入或更新键值对。"""
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1
        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key: Any) -> Any:
        """根据键获取对应的值。"""
        index = self._hash(key)
        bucket = self.buckets[index]
        for k, v in bucket:
            if k == key:
                return v
        return None

    def delete(self, key: Any) -> bool:
        """删除键值对。

        返回:
            bool: 如果删除成功返回 True，否则返回 False
        """
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        return False

    def _resize(self) -> None:
        """扩容哈希表，将桶数量翻倍并重新哈希已有元素。"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [list() for _ in range(self.capacity)]
        old_size = self.size
        self.size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
        self.size = old_size

    def execute(self, *args, **kwargs) -> dict[Any, Any]:
        """返回当前哈希表中的所有键值对。"""
        return {k: v for bucket in self.buckets for k, v in bucket}
