"""线性搜索算法实现。"""
from ...base import Algorithm
from typing import Any, List


class LinearSearch(Algorithm):
    """通过顺序扫描查找目标值的索引。
    
    线性搜索（也称为顺序搜索）是最简单的搜索算法，
    通过逐个检查列表中的每个元素来查找目标值。
    
    算法特点：
        - 简单易懂，实现直观
        - 不要求数据预先排序
        - 适用于小数据集或无序数据
        - 在最坏情况下需要检查所有元素
    """

    def execute(self, data: List[Any], target: Any) -> int:
        """在数据列表中查找目标值的索引。

        参数:
            data: 要搜索的数据列表（可以是无序的）
            target: 要查找的目标值
            
        返回:
            int: 目标值的索引，如果未找到则返回 -1
            
        时间复杂度: 
            - 最好情况: O(1) - 目标值在第一个位置
            - 平均情况: O(n/2) - 目标值在中间位置
            - 最坏情况: O(n) - 目标值在最后或不存在
        空间复杂度: O(1) - 只使用常数额外空间
        
        示例:
            >>> searcher = LinearSearch()
            >>> searcher.execute([3, 1, 4, 1, 5], 4)
            2
            >>> searcher.execute([3, 1, 4, 1, 5], 9)
            -1
        """
        # 遍历列表中的每个元素及其索引
        for i, value in enumerate(data):
            if value == target:  # 找到目标值
                return i
        return -1  # 遍历完成后仍未找到目标值
