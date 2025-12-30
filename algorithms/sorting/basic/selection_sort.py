"""选择排序算法实现。"""
from ...utils import swap
from ...base import Algorithm
from typing import List, Any


class SelectionSort(Algorithm):
    """使用选择排序算法对列表进行排序。

    选择排序通过在未排序部分中寻找最小元素，
    将其交换到当前排序位置，逐步构建有序序列。
    """

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表

        返回:
            List[Any]: 排序后的数据副本

        时间复杂度: O(n^2) - 需要进行 n*(n-1)/2 次比较
        空间复杂度: O(1) - 只需要常数额外空间
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        n = len(arr)

        for i in range(n):
            # 假设当前索引 i 处的元素为最小值
            min_index = i
            # 在未排序部分中寻找更小的元素
            for j in range(i + 1, n):
                if arr[j] < arr[min_index]:
                    min_index = j
            # 将找到的最小元素交换到当前位置
            if min_index != i:
                swap(arr, i, min_index)
        return arr
