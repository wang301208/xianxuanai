"""
神经元模型公共接口 - 重构版本
统一从核心模块导入，消除重复定义
"""

from .neuron_base import NeuronBase, NeuronType
from .multi_neuron_models import (
    LIFNeuron,
    IzhikevichNeuron,
    AdExNeuron,
    HodgkinHuxleyNeuron,
    create_neuron
)

# 定义对外暴露的公共API
__all__ = [
    "NeuronBase",
    "LIFNeuron",
    "IzhikevichNeuron",
    "AdExNeuron",
    "HodgkinHuxleyNeuron",
    "create_neuron",
    "NeuronType",
    "Neuron"
]
Neuron = NeuronBase
