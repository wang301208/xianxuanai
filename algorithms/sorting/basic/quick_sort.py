"""快速排序算法实现。"""
from ...utils import swap
from ...base import Algorithm
from typing import List, Any


class QuickSort(Algorithm):
    """使用快速排序算法对列表进行排序。
    
    快速排序是一种高效的分治排序算法，通过选择一个基准元素，
    将数组分为两部分，然后递归地对两部分进行排序。
    """

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表
            
        返回:
            List[Any]: 排序后的数据副本
            
        时间复杂度:
            - 平均情况: O(n log n)
            - 最坏情况: O(n^2) - 当数组已经有序或逆序时
            - 最好情况: O(n log n)
        空间复杂度: O(log n) - 由于递归调用栈
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        self._quicksort(arr, 0, len(arr) - 1)
        return arr

    def _quicksort(self, arr: List[Any], low: int, high: int) -> None:
        """递归执行快速排序。
        
        参数:
            arr: 待排序的数组
            low: 排序范围的起始索引
            high: 排序范围的结束索引
        """
        if low < high:
            # 获取分区点，分区点左边的元素都小于等于基准值，右边的都大于基准值
            pivot = self._partition(arr, low, high)
            # 递归排序左半部分
            self._quicksort(arr, low, pivot - 1)
            # 递归排序右半部分
            self._quicksort(arr, pivot + 1, high)

    def _partition(self, arr: List[Any], low: int, high: int) -> int:
        """分区函数，将数组分为两部分。
        
        选择最后一个元素作为基准值，将小于等于基准值的元素放在左边，
        大于基准值的元素放在右边。
        
        参数:
            arr: 待分区的数组
            low: 分区范围的起始索引
            high: 分区范围的结束索引
            
        返回:
            int: 基准值的最终位置索引
        """
        pivot = arr[high]  # 选择最后一个元素作为基准值
        i = low - 1  # 小于基准值的元素的索引指针
        
        # 遍历数组，将小于等于基准值的元素移到左边
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                swap(arr, i, j)  # 交换元素位置
        
        # 将基准值放到正确的位置
        swap(arr, i + 1, high)
        return i + 1  # 返回基准值的位置
