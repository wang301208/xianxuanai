"""冒泡排序算法实现。"""
from ...utils import swap
from ...base import Algorithm
from typing import List, Any


class BubbleSort(Algorithm):
    """使用冒泡排序算法对列表进行排序。
    
    冒泡排序是一种简单的排序算法，通过重复遍历列表，
    比较相邻元素并交换它们（如果顺序错误）。
    较大的元素会像气泡一样"冒泡"到列表的末尾。
    """

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表
            
        返回:
            List[Any]: 排序后的数据副本
            
        时间复杂度: O(n^2) - 需要进行 n*(n-1)/2 次比较
        空间复杂度: O(1) - 只需要常数额外空间
        
        算法特点:
            - 稳定排序：相等元素的相对位置不会改变
            - 原地排序：只需要常数额外空间
            - 简单易懂：适合教学和小数据集
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        n = len(arr)
        
        # 外层循环：控制排序的轮数
        for i in range(n):
            # 内层循环：在未排序部分进行相邻元素比较
            # 每轮结束后，最大元素会"冒泡"到正确位置
            for j in range(0, n - i - 1):
                # 如果前一个元素大于后一个元素，则交换它们
                if arr[j] > arr[j + 1]:
                    swap(arr, j, j + 1)
        return arr
