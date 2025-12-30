"""
突触模型公共接口 - 重构版本
使用统一的突触模型实现，消除重复定义
"""
from typing import Dict, Any
from .neuron_base import SynapseBase
from .synapse_models import (
    StaticSynapse, DynamicSynapse, NMDAReceptorSynapse, 
    GABAReceptorSynapse, GapJunction, ModulatorySynapse,
    SynapseType
)
from .synapse_manager import (
    SynapseManager,
    create_synapse_manager,
    create_glutamate_synapse_config,
    create_gaba_synapse_config,
)
from .stp_synapse import STPSynapse

# 突触类型映射
_synapse_type_mapping = {
    'static': SynapseType.STATIC,
    'dynamic': SynapseType.DYNAMIC,
    'nmda': SynapseType.NMDA,
    'gaba': SynapseType.GABA,
    'gap': SynapseType.GAP_JUNCTION,
    'modulatory': SynapseType.MODULATORY
}

# 突触类映射
_synapse_class_mapping = {
    SynapseType.STATIC: StaticSynapse,
    SynapseType.DYNAMIC: DynamicSynapse,
    SynapseType.NMDA: NMDAReceptorSynapse,
    SynapseType.GABA: GABAReceptorSynapse,
    SynapseType.GAP_JUNCTION: GapJunction,
    SynapseType.MODULATORY: ModulatorySynapse
}

def create_synapse(synapse_type: str, pre_id: int, post_id: int, params: Dict[str, Any]):
    """
    创建突触的工厂函数（向后兼容）
    
    Args:
        synapse_type: 突触类型字符串
        pre_id: 突触前神经元ID
        post_id: 突触后神经元ID
        params: 参数字典
        
    Returns:
        突触实例
    """
    if synapse_type not in _synapse_type_mapping:
        raise ValueError(f"不支持的突触类型: {synapse_type}")
    
    synapse_type_enum = _synapse_type_mapping[synapse_type]
    synapse_class = _synapse_class_mapping[synapse_type_enum]
    
    # 创建突触ID（基于前后神经元ID）
    synapse_id = hash(f"{pre_id}_{post_id}") % 1000000
    
    return synapse_class(synapse_id, synapse_type_enum, params)

# 向后兼容的适配器类
class SynapseAdapter:
    """突触适配器，提供向后兼容的接口"""
    
    def __init__(self, synapse_type: str, pre_id: int, post_id: int, params: Dict[str, Any]):
        self.synapse = create_synapse(synapse_type, pre_id, post_id, params)
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = params.get('weight', 1.0)
        self.delay = params.get('delay', 0.0)
    
    def transmit(self, pre_spike: bool, dt: float, current_time: float = None) -> float:
        """传输脉冲（向后兼容接口）"""
        return self.synapse.update(dt, pre_spike)
    
    def reset(self):
        """重置突触状态"""
        self.synapse.reset()
    
    def set_weight(self, new_weight: float):
        """设置突触权重"""
        self.weight = new_weight
        # 更新底层突触的权重
        if hasattr(self.synapse, '_weight'):
            self.synapse._weight = new_weight

# 导出统一模型类
StaticSynapse = StaticSynapse
DynamicSynapse = DynamicSynapse
NMDAReceptorSynapse = NMDAReceptorSynapse
GABAReceptorSynapse = GABAReceptorSynapse
GapJunction = GapJunction
ModulatorySynapse = ModulatorySynapse

Synapse = SynapseBase

__all__ = [
    "create_synapse", "SynapseAdapter", "StaticSynapse", "DynamicSynapse",
    "NMDAReceptorSynapse", "GABAReceptorSynapse", "GapJunction", "ModulatorySynapse",
    "SynapseManager", "create_synapse_manager", "create_glutamate_synapse_config", "create_gaba_synapse_config",
    "Synapse", "STPSynapse"
]
