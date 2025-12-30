# -*- coding: utf-8 -*-
"""
解剖连接与细胞多样性库（最小可插拔基础模块）
- 提供脑区的细胞类型比例、层内/层间/长程连接概率的集中管理
- 供 network.py 在启用时加载使用，以替换/修正默认的占位参数
- 默认不启用（避免影响现有测试），后续按配置接入

注意：
- 本模块为数据接口与轻逻辑聚合，不直接修改网络结构
- 后续可扩展为读取外部数据库/文献参数，并支持版本化

"""

from typing import Dict, Any, List, Tuple

# 细胞类型比例（数据为占位，后续可替换为文献与数据库来源）
CELL_TYPE_RATIOS: Dict[str, Dict[str, float]] = {
    # 皮层区域：锥体细胞与中间神经元的比例，分层可选
    "PREFRONTAL_CORTEX": {"pyramidal": 0.8, "interneuron": 0.2},
    "PRIMARY_VISUAL_CORTEX": {"pyramidal": 0.75, "interneuron": 0.25},
    "PRIMARY_MOTOR_CORTEX": {"pyramidal": 0.78, "interneuron": 0.22},
    "PRIMARY_SOMATOSENSORY_CORTEX": {"pyramidal": 0.76, "interneuron": 0.24},
    # 海马结构（可细分到亚区）
    "HIPPOCAMPUS_CA1": {"pyramidal": 0.85, "interneuron": 0.15},
    "HIPPOCAMPUS_CA3": {"pyramidal": 0.82, "interneuron": 0.18},
    "DENTATE_GYRUS": {"pyramidal": 0.2, "interneuron": 0.8},  # 颗粒细胞占比高
}

# 层内/层间连接概率（占位参数）
LAYER_CONNECTIVITY: Dict[str, Dict[str, float]] = {
    "intra_columnar": {
        "default": 0.02,
        "L2": 0.025,
        "L3": 0.022,
        "L4": 0.03,
        "L5": 0.018,
        "L6": 0.015
    },
    "layer_specific": {
        "L2_to_L5": 0.012,
        "L3_to_L5": 0.011,
        "L4_to_L2": 0.016,
        "L5_to_L2": 0.009,
        "L6_to_L4": 0.007
    }
}

# 长程连接概率（皮层-皮层、皮层-海马、皮层-丘脑等）
LONG_RANGE_CONNECTIVITY: Dict[str, float] = {
    "cortico_cortical": 0.2,
    "cortico_hippocampal": 0.15,
    "cortico_thalamic": 0.25,
    "hippocampo_thalamic": 0.1
}

def get_cell_type_ratios(region_name: str) -> Dict[str, float]:
    """获取指定脑区的细胞类型比例（未定义则返回默认值）"""
    return CELL_TYPE_RATIOS.get(region_name.upper(), {"pyramidal": 0.8, "interneuron": 0.2})

def get_layer_connectivity() -> Dict[str, Dict[str, float]]:
    """获取层内与层间连接概率集"""
    return LAYER_CONNECTIVITY

def get_long_range_connectivity() -> Dict[str, float]:
    """获取长程连接概率集"""
    return LONG_RANGE_CONNECTIVITY

def suggest_population_scaling(region_name: str, base_density: int) -> Dict[str, int]:
    """
    根据脑区细胞比例建议不同类型的数量缩放（不改变总量，只分配结构）
    - base_density：基础密度（可来自 L1-L6 配置）
    返回：{"pyramidal": N1, "interneuron": N2}
    """
    ratios = get_cell_type_ratios(region_name)
    total = max(1, base_density)
    pyramidal_n = int(total * ratios["pyramidal"])
    interneuron_n = total - pyramidal_n
    return {"pyramidal": pyramidal_n, "interneuron": interneuron_n}

def anatomy_metadata() -> Dict[str, Any]:
    """导出当前解剖参数元数据，用于统计与可视化"""
    return {
        "regions_with_ratios": list(CELL_TYPE_RATIOS.keys()),
        "layer_connectivity_keys": list(LAYER_CONNECTIVITY.keys()),
        "long_range_keys": list(LONG_RANGE_CONNECTIVITY.keys())
    }