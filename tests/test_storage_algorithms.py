from algorithms.storage.basic.lru_cache import LRUCache
from algorithms.storage.basic.lfu_cache import LFUCache
from algorithms.storage.advanced.btree_index import BTreeIndex


def test_lru_cache_behavior():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")  # evicts key 2
    assert cache.get(2) is None
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"


def test_lfu_cache_behavior():
    cache = LFUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")  # evicts key 2 (lowest freq)
    assert cache.get(2) is None
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"


def test_btree_index():
    index = BTreeIndex()
    for i in range(1, 6):
        index.insert(i, str(i))
    for i in range(1, 6):
        assert index.search(i) == str(i)
    assert index.search(6) is None
