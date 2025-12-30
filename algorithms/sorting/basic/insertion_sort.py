"""插入排序算法实现。"""
from ...base import Algorithm
from typing import List, Any


class InsertionSort(Algorithm):
    """使用插入排序算法对列表进行排序。

    插入排序通过逐个将元素插入到已排序子序列中，
    以构建最终的排序数组。适合小规模数据或部分有序的数据。
    """

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表

        返回:
            List[Any]: 排序后的数据副本

        时间复杂度:
            - 最坏情况: O(n^2) - 需要移动大量元素
            - 最好情况: O(n) - 当输入已经有序
        空间复杂度: O(1) - 原地排序
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组

        # 从第二个元素开始，逐个插入到前面已排序的子序列
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            # 将大于 key 的元素向后移动，为 key 腾出位置
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        return arr
