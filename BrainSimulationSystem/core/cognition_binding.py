# -*- coding: utf-8 -*-
"""
认知/记忆模块与生理脑区绑定（最小可插拔基础模块）
- 提供认知与记忆子模块到具体生理回路的映射策略
- 默认不启用，避免影响现有测试；后续通过配置按需接入 network.py

设计思路：
- 以功能图谱为参考，将执行控制、工作记忆、情绪调节等，绑定到 PFC、ACC、海马-丘脑环路等
- 可输出绑定元数据供 network_statistics 展示

"""

from typing import Dict, Any, List, Tuple

# 功能到脑区的基础映射（占位）
FUNCTIONAL_BINDINGS: Dict[str, List[str]] = {
    "executive_control": ["PREFRONTAL_CORTEX", "ANTERIOR_CINGULATE_CORTEX"],
    "working_memory": ["PREFRONTAL_CORTEX", "HIPPOCAMPUS_CA1", "HIPPOCAMPUS_CA3"],
    "episodic_memory": ["HIPPOCAMPUS_CA1", "DENTATE_GYRUS", "THALAMUS_MD"],
    "visual_attention": ["PRIMARY_VISUAL_CORTEX", "POSTERIOR_PARIETAL_CORTEX", "THALAMUS_LGN"],
    "motor_planning": ["PRIMARY_MOTOR_CORTEX", "STRIATUM", "SUBTHALAMIC_NUCLEUS"],
    "emotion_regulation": ["AMYGDALA", "NUCLEUS_ACCUMBENS", "PREFRONTAL_CORTEX"]
}

def get_functional_bindings() -> Dict[str, List[str]]:
    """获取功能到脑区的基础映射"""
    return FUNCTIONAL_BINDINGS

def suggest_bindings_for_config(scope_regions: List[str]) -> Dict[str, List[str]]:
    """
    根据配置中的脑区集合建议可行的功能绑定（过滤不存在的脑区）
    返回：功能 -> [已存在的脑区]
    """
    region_set = set(r.upper() for r in scope_regions)
    suggested: Dict[str, List[str]] = {}
    for func, regions in FUNCTIONAL_BINDINGS.items():
        filtered = [r for r in regions if r.upper() in region_set]
        if filtered:
            suggested[func] = filtered
    return suggested

def binding_metadata() -> Dict[str, Any]:
    """导出绑定元数据"""
    return {
        "functional_domains": list(FUNCTIONAL_BINDINGS.keys()),
        "total_bindings": sum(len(v) for v in FUNCTIONAL_BINDINGS.values())
    }