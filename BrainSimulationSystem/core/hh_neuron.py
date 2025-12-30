"""
Hodgkin-Huxley (HH) 神经元模型适配器
使用统一的多神经元模型实现
"""
from typing import Dict, Any
from .multi_neuron_models import HodgkinHuxleyNeuron, NeuronType

# 向后兼容的HH神经元类
class HodgkinHuxleyNeuronAdapter(HodgkinHuxleyNeuron):
    """
    Hodgkin-Huxley神经元适配器，提供向后兼容的接口
    """
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化HH神经元适配器
        
        Args:
            neuron_id: 神经元ID
            params: 参数字典
        """
        super().__init__(neuron_id, params)
        self.id = neuron_id
        self.params = params
    
    def update(self, I_ext: float, dt: float) -> bool:
        """
        更新神经元状态（向后兼容接口）
        
        Args:
            I_ext: 外部电流
            dt: 时间步长
            
        Returns:
            是否发放脉冲
        """
        self.I_ext = I_ext
        return super().update(dt, 0.0)  # 简化时间处理
    
    @property
    def voltage(self) -> float:
        """获取膜电位"""
        return self.V

# 导出统一实现
HodgkinHuxleyNeuron = HodgkinHuxleyNeuron