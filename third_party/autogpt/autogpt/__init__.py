"""AutoGPT 主模块初始化文件。

本模块负责 AutoGPT 应用程序的初始化设置，包括测试环境的随机种子配置。
在测试或持续集成环境中，设置固定的随机种子以确保测试结果的可重现性。
"""

import os
import random
import sys

# 检测是否在测试环境或持续集成环境中运行
if "pytest" in sys.argv or "pytest" in sys.modules or os.getenv("CI"):
    print("Setting random seed to 42")
    # 设置固定的随机种子，确保测试结果的可重现性
    # 这对于涉及随机性的测试用例非常重要
    random.seed(42)
