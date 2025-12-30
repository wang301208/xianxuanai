"""最近最少使用（LRU）缓存实现。"""
from collections import OrderedDict
from typing import Any

from ...base import Algorithm


class LRUCache(Algorithm):
    """最近最少使用（LRU）缓存实现。

    LRU缓存是一种常用的缓存淘汰策略，当缓存容量满时，
    优先淘汰最近最少使用的数据项。这种策略基于局部性原理，
    认为最近使用的数据在未来被使用的概率更高。

    算法原理：
        - 使用有序字典（OrderedDict）来维护访问顺序
        - 每次访问时将对应项移到末尾（表示最近使用）
        - 当容量超限时，删除字典开头的项（最少使用的）

    属性:
        capacity: 缓存的最大容量
        cache: 存储键值对的有序字典

    主要操作:
        - get(key): 根据键获取值，时间复杂度 O(1)，空间复杂度 O(1)
        - put(key, value): 插入或更新键值对，时间复杂度 O(1)，空间复杂度 O(1)

    应用场景:
        - 操作系统页面置换
        - CPU缓存管理
        - 数据库缓冲池
        - Web缓存系统
        - 应用程序内存管理
    """
=======

    def __init__(self, capacity: int) -> None:
        """初始化LRU缓存。
        
        参数:
            capacity: 缓存的最大容量，必须大于0
        """
        self.capacity = capacity
        self.cache: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any | None:
        """根据键获取对应的值。
        
        如果键存在，将其移动到有序字典的末尾（标记为最近使用）。
        如果键不存在，返回None。
        
        参数:
            key: 要查找的键
            
        返回:
            Any | None: 对应的值，如果键不存在则返回None
            
        时间复杂度: O(1) - OrderedDict的查找和移动操作都是常数时间
        
        示例:
            >>> cache = LRUCache(2)
            >>> cache.put('a', 1)
            >>> cache.put('b', 2)
            >>> cache.get('a')  # 返回1，并将'a'标记为最近使用
        """
        if key not in self.cache:
            return None
        # 将访问的键移到末尾，表示最近使用
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """插入或更新键值对。
        
        如果键已存在，更新其值并移到末尾。
        如果键不存在，插入新的键值对。
        当缓存超过容量时，删除最少使用的项（字典开头的项）。
        
        参数:
            key: 要插入或更新的键
            value: 对应的值
            
        时间复杂度: O(1) - 所有操作都是常数时间
        
        示例:
            >>> cache = LRUCache(2)
            >>> cache.put('a', 1)
            >>> cache.put('b', 2)
            >>> cache.put('c', 3)  # 'a'被淘汰，因为它是最少使用的
        """
        if key in self.cache:
            # 如果键已存在，移到末尾表示最近使用
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # 如果超过容量，删除最少使用的项（字典开头的项）
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # last=False表示删除开头的项

    def execute(self, *args, **kwargs) -> dict[Any, Any]:
        """返回当前缓存状态的快照。
        
        这是Algorithm基类要求的方法实现。
        
        返回:
            dict[Any, Any]: 当前缓存中所有键值对的字典
        """
        return dict(self.cache)
=======
