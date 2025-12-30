"""归并排序算法实现。"""
from ...base import Algorithm
from typing import List, Any


class MergeSort(Algorithm):
    """使用归并排序算法对列表进行排序。

    归并排序是一种高效、稳定的分治排序算法，它将数组分为两半，
    分别排序后再合并。
    """

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表

        返回:
            List[Any]: 排序后的数据副本

        时间复杂度: O(n log n)
        空间复杂度: O(n)
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        return self._merge_sort(arr)

    def _merge_sort(self, arr: List[Any]) -> List[Any]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self._merge_sort(arr[:mid])
        right = self._merge_sort(arr[mid:])
        return self._merge(left, right)

    def _merge(self, left: List[Any], right: List[Any]) -> List[Any]:
        merged: List[Any] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged
