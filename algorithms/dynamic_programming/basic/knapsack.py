"""使用二维动态规划解决 0-1 背包问题。"""
from typing import List, Tuple

from ...base import Algorithm


class Knapsack(Algorithm):
    """求解 0-1 背包问题的最大价值，并返回选择的物品索引。

    给定一组物品，每个物品都有重量和价值，在限定的背包容量内，
    选择一组物品使得总价值最大且每个物品最多选择一次。

    使用二维 DP 表格，其中 dp[i][w] 表示前 i 个物品在容量 w 下的最大价值。
    通过回溯 dp 表格可以获得被选中的物品集合。
    """

    def execute(self, weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
        """返回背包能够获得的最大价值以及所选择物品的索引列表。

        参数:
            weights: 物品的重量列表
            values: 物品的价值列表
            capacity: 背包的最大承重

        返回:
            Tuple[int, List[int]]: 最大价值和选择的物品索引（0 开始）

        时间复杂度: O(n * capacity)
        空间复杂度: O(n * capacity)
        """
        if len(weights) != len(values):
            raise ValueError("weights and values must have the same length")

        n = len(weights)
        # 创建 DP 表格，初始化为 0
        dp: List[List[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

        # 填充 DP 表格
        for i in range(1, n + 1):
            wt = weights[i - 1]
            val = values[i - 1]
            for w in range(capacity + 1):
                if wt <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt] + val)
                else:
                    dp[i][w] = dp[i - 1][w]

        # 回溯找到选择的物品
        w = capacity
        selected: List[int] = []
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= weights[i - 1]

        selected.reverse()
        return dp[n][capacity], selected
