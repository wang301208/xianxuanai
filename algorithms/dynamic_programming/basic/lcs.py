"""使用动态规划计算最长公共子序列。"""
from ...base import Algorithm
from typing import List


class LongestCommonSubsequence(Algorithm):
    """计算最长公共子序列的长度。
    
    最长公共子序列（LCS）问题是计算机科学中的经典问题。
    给定两个序列，找出它们的最长公共子序列的长度。
    
    子序列定义：
        - 子序列是从原序列中删除一些（可能为零）元素后得到的序列
        - 子序列中元素的相对顺序必须保持不变
        - 例如："ACE" 是 "ABCDE" 的子序列
    
    问题示例：
        s1 = "ABCDGH"
        s2 = "AEDFHR"
        LCS = "ADH"，长度为 3
    
    动态规划解法：
        使用二维表格 dp[i][j] 表示 s1[0...i-1] 和 s2[0...j-1] 的 LCS 长度
    """

    def execute(self, s1: str, s2: str) -> int:
        """返回两个字符串的最长公共子序列长度。

        参数:
            s1: 第一个字符串
            s2: 第二个字符串
            
        返回:
            int: 最长公共子序列的长度
            
        时间复杂度: O(m * n) - m 和 n 分别是两个字符串的长度
        空间复杂度: O(m * n) - 需要二维 DP 表格
        
        算法思路:
            1. 创建 (m+1) × (n+1) 的 DP 表格
            2. dp[i][j] 表示 s1[0...i-1] 和 s2[0...j-1] 的 LCS 长度
            3. 状态转移方程：
               - 如果 s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
               - 否则: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        示例:
            >>> lcs = LongestCommonSubsequence()
            >>> lcs.execute("ABCDGH", "AEDFHR")  # 返回 3 (ADH)
            >>> lcs.execute("AGGTAB", "GXTXAYB")  # 返回 4 (GTAB)
        """
        m, n = len(s1), len(s2)
        
        # 创建 DP 表格，初始化为 0
        # dp[i][j] 表示 s1[0...i-1] 和 s2[0...j-1] 的 LCS 长度
        dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充 DP 表格
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    # 字符匹配，LCS 长度增加 1
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    # 字符不匹配，取两种情况的最大值
                    # 要么跳过 s1 的当前字符，要么跳过 s2 的当前字符
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        
        # 返回整个字符串的 LCS 长度
        return dp[m][n]
