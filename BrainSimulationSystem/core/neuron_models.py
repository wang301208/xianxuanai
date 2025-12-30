"""
神经元模型统一导入模块
提供从multi_neuron_models.py导入统一神经元模型的接口
"""

from .multi_neuron_models import (
    LIFNeuron,
    HodgkinHuxleyNeuron,
    AdExNeuron,
    IzhikevichNeuron,
    MultiCompartmentNeuron,
    create_neuron,
    get_default_parameters,
    NeuronType
)

# 为保持向后兼容性，提供别名
from .neuron_base import NeuronBase
PyramidalNeuron = MultiCompartmentNeuron
Interneuron = LIFNeuron

__all__ = [
    'LIFNeuron',
    'HodgkinHuxleyNeuron',
    'AdExNeuron',
    'IzhikevichNeuron',
    'MultiCompartmentNeuron',
    'create_neuron',
    'get_default_parameters',
    'NeuronType',
    'NeuronBase',
    'PyramidalNeuron',
    'Interneuron'
]