"""二分搜索算法实现。"""
from ...base import Algorithm
from typing import Any, List


class BinarySearch(Algorithm):
    """使用二分搜索技术在已排序列表中查找元素。
    
    二分搜索是一种高效的搜索算法，通过反复将搜索区间对半分割，
    快速定位目标元素。前提条件是数据必须已经排序。
    
    算法原理：
        1. 比较中间元素与目标值
        2. 如果相等，返回索引
        3. 如果中间元素小于目标值，搜索右半部分
        4. 如果中间元素大于目标值，搜索左半部分
        5. 重复直到找到目标或搜索区间为空
    """

    def execute(self, data: List[Any], target: Any) -> int:
        """在已排序的数据中查找目标值的索引。

        参数:
            data: 已排序的数据列表（必须是升序）
            target: 要查找的目标值
            
        返回:
            int: 目标值的索引，如果未找到则返回 -1
            
        时间复杂度: O(log n) - 每次搜索范围减半
        空间复杂度: O(1) - 只使用常数额外空间
        
        注意:
            输入数据必须已经排序，否则结果不可靠
            
        示例:
            >>> searcher = BinarySearch()
            >>> searcher.execute([1, 3, 5, 7, 9], 5)
            2
            >>> searcher.execute([1, 3, 5, 7, 9], 6)
            -1
        """
        low, high = 0, len(data) - 1  # 初始化搜索范围的左右边界
        
        while low <= high:  # 当搜索范围有效时继续
            mid = (low + high) // 2  # 计算中间位置索引
            
            if data[mid] == target:  # 找到目标值
                return mid
            elif data[mid] < target:  # 中间值小于目标，搜索右半部分
                low = mid + 1
            else:  # 中间值大于目标，搜索左半部分
                high = mid - 1
                
        return -1  # 未找到目标值
