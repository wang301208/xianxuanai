# -*- coding: utf-8 -*-
"""
硬件后端设备发现与最小映射占位（默认关闭）
- 目标：提供 Loihi / SpiNNaker / BrainScaleS 的可用性探测与元数据采集
- 非侵入式：仅返回元数据，不进行设备初始化或运行
"""

from typing import Dict, Any

def discover_devices() -> Dict[str, Any]:
    """探测可用的神经形态设备并返回元数据（占位实现）"""
    devices = {
        "loihi": {"available": False, "chips": 0, "cores_per_chip": 0},
        "spinnaker": {"available": False, "boards": 0, "cores_per_board": 18},
        "brainscales": {"available": False, "wafer_available": False}
    }
    try:
        import nengo_loihi  # noqa
        import nengo  # noqa
        devices["loihi"]["available"] = True
        # 占位芯片数量（需真实查询接口时再完善）
        devices["loihi"]["chips"] = 1
        devices["loihi"]["cores_per_chip"] = 128
    except Exception:
        pass

    try:
        import spynnaker8 as sim  # noqa
        devices["spinnaker"]["available"] = True
        devices["spinnaker"]["boards"] = 1
    except Exception:
        pass

    try:
        import pybrainscales  # 假设库名，占位
        devices["brainscales"]["available"] = True
        devices["brainscales"]["wafer_available"] = True
    except Exception:
        pass

    return devices

def suggest_mapping(network_stats: Dict[str, Any]) -> Dict[str, Any]:
    """基于网络规模建议最小映射策略（占位）"""
    total_neurons = int(network_stats.get("network_size", {}).get("cortical_columns", 0))
    return {
        "loihi": {"suggested_columns": min(10, total_neurons)},
        "spinnaker": {"suggested_columns": min(10, max(0, total_neurons - 10))},
        "brainscales": {"suggested_columns": 0}
    }