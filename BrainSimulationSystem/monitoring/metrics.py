# -*- coding: utf-8 -*-
"""
监控导出模块（最小占位）
- 将网络统计信息导出为 JSON 文件
- 默认不启用，避免影响现有测试与运行
"""

import json
import os
from typing import Dict, Any

def export_metrics(stats: Dict[str, Any], export_path: str) -> bool:
    """导出监控指标到指定路径（JSON 格式）"""
    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False