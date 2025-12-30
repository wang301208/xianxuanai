"""
点神经元模型适配器模块 - 重构版本
使用统一的神经元模型实现，消除重复定义
"""
from typing import Dict, Any
from multi_neuron_models import (
    NeuronType, create_neuron, get_default_parameters,
    LIFNeuron, IzhikevichNeuron, AdExNeuron, HodgkinHuxleyNeuron
)

# 适配器类，提供向后兼容的接口
class PointNeuronAdapter:
    """点神经元适配器，将统一模型转换为旧接口"""
    
    def __init__(self, neuron_type: NeuronType, neuron_id: int, params: Dict[str, Any]):
        """
        初始化适配器
        
        Args:
            neuron_type: 神经元类型枚举
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典
        """
        self.neuron = create_neuron(neuron_type, neuron_id, params)
        self.id = neuron_id
        self.params = params
    
    def reset(self) -> None:
        """重置神经元状态"""
        self.neuron.reset()
    
    def update(self, input_current: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            input_current: 输入电流
            dt: 时间步长
            
        Returns:
            是否发放脉冲
        """
        self.neuron.I_ext = input_current
        return self.neuron.update(dt, 0.0)  # 简化时间处理
    
    @property
    def voltage(self) -> float:
        """获取当前膜电位"""
        return self.neuron.V
    
    def adjust_threshold(self, adjustment: float):
        """调整激活阈值"""
        if hasattr(self.neuron, 'V_thresh'):
            self.neuron.V_thresh += adjustment
        elif hasattr(self.neuron, 'threshold'):
            self.neuron.threshold += adjustment

# 向后兼容的工厂函数
def create_point_neuron(neuron_type: str, neuron_id: int, params: Dict[str, Any]) -> PointNeuronAdapter:
    """
    创建点神经元（向后兼容）
    
    Args:
        neuron_type: 神经元类型字符串
        neuron_id: 神经元ID
        params: 参数
        
    Returns:
        神经元适配器实例
    """
    type_mapping = {
        'lif': NeuronType.LIF,
        'izhikevich': NeuronType.IZHIKEVICH,
        'adex': NeuronType.ADAPTIVE_EXPONENTIAL,
        'hh': NeuronType.HODGKIN_HUXLEY
    }
    
    if neuron_type not in type_mapping:
        raise ValueError(f"不支持的神经元类型: {neuron_type}")
    
    return PointNeuronAdapter(type_mapping[neuron_type], neuron_id, params)

# 导出统一模型类以保持兼容性
LIFNeuron = LIFNeuron
IzhikevichNeuron = IzhikevichNeuron
AdExNeuron = AdExNeuron
HodgkinHuxleyNeuron = HodgkinHuxleyNeuron