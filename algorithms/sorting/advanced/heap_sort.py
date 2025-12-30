"""堆排序算法实现。"""
from ...base import Algorithm
from typing import List, Any


class HeapSort(Algorithm):
    """使用堆排序算法对列表进行排序。

    堆排序通过构建最大堆并不断交换堆顶元素，
    从而在原地实现排序。
    """

    def execute(self, data: List[Any], reverse: bool = False) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表
            reverse: 是否按降序排序，默认为 False(升序)

        返回:
            List[Any]: 排序后的数据副本

        时间复杂度: O(n log n)
        空间复杂度: O(1)
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        n = len(arr)

        # 构建最大堆
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)

        # 依次将堆顶元素放到末尾，并调整堆
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self._heapify(arr, i, 0)

        if reverse:
            arr.reverse()
        return arr

    def _heapify(self, arr: List[Any], n: int, i: int) -> None:
        """维护以 i 为根的子树的最大堆性质。"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest)
