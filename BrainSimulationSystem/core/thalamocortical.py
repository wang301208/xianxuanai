# -*- coding: utf-8 -*-
"""
丘脑-皮层环路（最小占位实现）
- 提供丘脑中继核的轻量模型与与皮层柱的基本投射接口
- 仅用于结构落地与连接打通，参数与动力学后续精细化
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

class ThalamicRelay:
    """丘脑中继核（最小实现）"""
    def __init__(self, relay_id: int, nucleus: str, position: Tuple[float, float, float], size: int = 50):
        self.relay_id = relay_id
        self.nucleus = nucleus  # 如 LGN/MD/VPL/VPM 等
        self.position = position
        self.size = size
        # 简化：仅用索引代表中继神经元
        self.neurons: List[int] = list(range(size))
        self.logger = logging.getLogger(f"ThalamicRelay_{relay_id}")

    def activity(self, dt_ms: float) -> Dict[int, float]:
        """生成丘脑中继的占位活动（返回对每个中继神经元的电流注入值）"""
        # 简化为弱随机驱动
        currents = {nid: float(np.random.uniform(0.0, 0.05)) for nid in self.neurons}
        return currents

def connect_thalamus_to_cortex(relay: ThalamicRelay,
                               cortical_columns: Dict[int, Any],
                               synapse_manager,
                               targets: List[int],
                               glut_weight_range: Tuple[float, float] = (0.1, 0.5)) -> List[int]:
    """将丘脑中继核连接到指定皮层柱（使用突触管理器创建兴奋性长程突触）"""
    syn_ids: List[int] = []
    import numpy as np
    from .synapses import create_glutamate_synapse_config

    for col_id in targets:
        column = cortical_columns.get(col_id)
        if not column or not column.neurons:
            continue
        # 在目标柱内随机选择一些神经元作为投射目标
        post_candidates = list(column.neurons.keys())
        if not post_candidates:
            continue
        post_id = int(np.random.choice(post_candidates))
        # 简化：从丘脑中继随机选择一个“代表神经元”作为突触前
        pre_id = int(np.random.choice(relay.neurons))
        syn_cfg = create_glutamate_synapse_config(weight=float(np.random.uniform(*glut_weight_range)),
                                                  learning_enabled=True)
        syn_id = synapse_manager.create_synapse(pre_id, post_id, syn_cfg)
        syn_ids.append(syn_id)
    return syn_ids