"""使用动态规划计算斐波那契数列。"""
from ...base import Algorithm


class Fibonacci(Algorithm):
    """计算第 n 个斐波那契数。
    
    斐波那契数列是一个经典的数学序列，其中每个数字是前两个数字的和。
    序列开始为：0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    
    数学定义：
        F(0) = 0
        F(1) = 1
        F(n) = F(n-1) + F(n-2) for n > 1
    
    本实现使用动态规划的思想，通过迭代方式避免了递归的重复计算，
    大大提高了效率并减少了空间复杂度。
    """

    def execute(self, n: int) -> int:
        """返回第 n 个斐波那契数。

        参数:
            n: 要计算的斐波那契数的位置（从0开始）
            
        返回:
            int: 第 n 个斐波那契数
            
        时间复杂度: O(n) - 需要迭代 n 次
        空间复杂度: O(1) - 只使用常数额外空间
        
        算法优势:
            - 相比递归实现，避免了指数级的时间复杂度
            - 空间效率高，不需要额外的存储空间
            - 可以处理较大的 n 值
        
        示例:
            >>> fib = Fibonacci()
            >>> fib.execute(0)  # 返回 0
            >>> fib.execute(1)  # 返回 1
            >>> fib.execute(5)  # 返回 5
            >>> fib.execute(10) # 返回 55
        """
        # 基础情况：F(0) = 0, F(1) = 1
        if n <= 1:
            return n
        
        # 使用两个变量保存前两个斐波那契数
        prev, curr = 0, 1  # prev = F(0), curr = F(1)
        
        # 从第2个数开始迭代计算
        for _ in range(2, n + 1):
            # 计算下一个斐波那契数并更新变量
            prev, curr = curr, prev + curr
            
        return curr
