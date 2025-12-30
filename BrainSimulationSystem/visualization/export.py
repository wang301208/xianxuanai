# -*- coding: utf-8 -*-
"""
可视化导出占位（默认关闭，非侵入式）
- 将关键统计指标导出为 CSV，便于后续观察与绘图
- 保持最小实现，不依赖外部绘图库
"""

import os
from typing import Dict, Any

def export_csv(stats: Dict[str, Any], export_path: str) -> bool:
    """导出统计为 CSV 文件（最小实现，中文注释）"""
    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        lines = []
        # 网络规模
        ns = stats.get("network_size", {})
        lines.append("metric,value")
        lines.append(f"brain_regions,{ns.get('brain_regions', 0)}")
        lines.append(f"cortical_columns,{ns.get('cortical_columns', 0)}")
        lines.append(f"long_range_connections,{ns.get('long_range_connections', 0)}")
        # 全局指标
        lines.append(f"total_neurons,{stats.get('total_neurons', 0)}")
        lines.append(f"total_spikes,{stats.get('total_spikes', 0)}")
        lines.append(f"global_spike_rate,{stats.get('global_spike_rate', 0.0)}")
        lines.append(f"active_columns,{stats.get('active_columns', 0)}")
        lines.append(f"network_synchrony,{stats.get('network_synchrony', 0.0)}")
        lines.append(f"mean_update_time,{stats.get('mean_update_time', 0.0)}")
        # 神经形态后端状态
        lines.append(f"neuromorphic_active,{1 if stats.get('neuromorphic_active', False) else 0}")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    except Exception:
        return False