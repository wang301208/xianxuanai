# -*- coding: utf-8 -*-
"""
配置校验占位（默认只记录警告，不中断运行）
- 校验关键字段是否存在与类型合理
- 返回 warnings 列表
"""

from typing import Dict, Any, List

def validate_config(cfg: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # scope 校验
    scope = cfg.get("scope", {})
    if not isinstance(scope.get("brain_regions", []), list) or not scope.get("brain_regions"):
        warnings.append("scope.brain_regions 缺失或为空：将退回默认脑区映射。")
    if "columns_per_region" not in scope:
        warnings.append("scope.columns_per_region 缺失：将使用默认值。")

    # 皮层结构校验
    cstruct = cfg.get("cortical_structure", {})
    layers = cstruct.get("layers", [])
    if not isinstance(layers, list) or not layers:
        warnings.append("cortical_structure.layers 缺失或为空：皮层柱层初始化可能退回占位。")

    # 连接模式校验
    conn = cfg.get("connectivity_patterns", {})
    lc = conn.get("local_connectivity", {})
    if "intra_columnar" not in lc:
        warnings.append("connectivity_patterns.local_connectivity.intra_columnar 缺失：局部连接退回默认概率。")
    if "long_range_connectivity" not in conn or "cortico_cortical" not in conn.get("long_range_connectivity", {}):
        warnings.append("connectivity_patterns.long_range_connectivity.cortico_cortical 缺失：长程连接退回默认概率。")

    # 神经形态硬件校验
    nm = cfg.get("neuromorphic", {})
    hp = nm.get("hardware_platforms", {})
    if "intel_loihi" not in hp or "spinnaker" not in hp:
        warnings.append("neuromorphic.hardware_platforms 缺失：硬件后端初始化将被跳过。")

    # 仿真与监控
    sim = cfg.get("simulation", {})
    if "save_interval" not in sim:
        warnings.append("simulation.save_interval 缺失：周期性保存逻辑将被禁用。")
    mon = cfg.get("monitoring", {})
    if "enabled" not in mon:
        warnings.append("monitoring.enabled 缺失：监控导出默认关闭。")

    # 运行时
    rt = cfg.get("runtime", {})
    if "timestep_ms" not in rt:
        warnings.append("runtime.timestep_ms 缺失：时间步长将使用默认值。")
    if "checkpoint" not in rt:
        warnings.append("runtime.checkpoint 缺失：检查点保存将被禁用。")

    return warnings