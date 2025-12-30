"""AutoGPT: 基于 GPT 的 AI 助手主入口文件。

本文件是 AutoGPT 应用程序的主入口点，当作为模块运行时会启动命令行界面。
AutoGPT 是一个自主的 AI 代理，能够执行各种任务并与用户进行交互。

使用方法:
    python -m autogpt

功能特性:
    - 自主任务执行
    - 智能决策制定
    - 多种能力集成
    - 用户交互界面
"""

import autogpt.app.cli

if __name__ == "__main__":
    # 启动 AutoGPT 的命令行界面
    # 这将解析命令行参数并启动相应的功能
    autogpt.app.cli.cli()
