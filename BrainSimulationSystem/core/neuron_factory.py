"""
神经元创建工厂和包装类
Neuron creation factory and wrapper classes.
"""
from typing import Dict, Any, Optional
import numpy as np

from .point_neurons import Neuron, LIFNeuron, IzhikevichNeuron, AdExNeuron
from .hh_neuron import HodgkinHuxleyNeuron

class PositionalNeuron:
    """
    一个包装类，为任何神经元模型添加3D空间位置信息。
    """
    
    def __init__(self, neuron: Neuron, position: Optional[np.ndarray] = None):
        """
        初始化带位置的神经元。

        Args:
            neuron (Neuron): 基础的神经元实例。
            position (Optional[np.ndarray]): 神经元的3D坐标 (μm)。
        """
        self.neuron = neuron
        self.position = position if position is not None else np.zeros(3)
        
    def __getattr__(self, name: str) -> Any:
        """
        将所有其他属性访问请求转发给内部的神经元实例，
        使得包装类对其外部行为透明。
        """
        return getattr(self.neuron, name)

def create_neuron(neuron_type: str, neuron_id: int, params: Dict[str, Any], position: Optional[np.ndarray] = None) -> PositionalNeuron:
    """
    工厂函数，用于创建并包装指定类型的神经元。

    Args:
        neuron_type (str): 神经元类型 ('lif', 'izhikevich', 'adex', 'hh')。
        neuron_id (int): 神经元的唯一标识符。
        params (Dict[str, Any]): 特定于模型的参数字典。
        position (Optional[np.ndarray]): 神经元的3D位置。

    Returns:
        PositionalNeuron: 一个包含创建的神经元及其位置的包装实例。

    Raises:
        ValueError: 如果指定的神经元类型不受支持。
    """
    neuron_classes = {
        'lif': LIFNeuron,
        'izhikevich': IzhikevichNeuron,
        'adex': AdExNeuron,
        'hh': HodgkinHuxleyNeuron,
    }
    
    if neuron_type not in neuron_classes:
        raise ValueError(f"不支持的神经元类型: {neuron_type}")
    
    # 验证HH模型的参数
    if neuron_type == 'hh':
        required = ['C_m', 'g_Na', 'g_K', 'g_L', 'E_Na', 'E_K', 'E_L']
        if not all(p in params for p in required):
            raise ValueError(f"HH模型缺少必要的参数。需要: {required}")
            
    # 创建基础神经元实例
    base_neuron = neuron_classes[neuron_type](neuron_id, params)
    
    # 使用PositionalNeuron进行包装
    return PositionalNeuron(base_neuron, position)