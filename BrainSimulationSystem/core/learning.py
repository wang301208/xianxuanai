"""
学习和记忆模块公共接口
Public API for Learning and Memory Modules

该模块从learning_rules子目录中导入各种学习规则，
并提供一个工厂函数来创建它们。
This module imports various learning rules from the learning_rules subdirectory
and provides a factory function to create them.
"""
from typing import Dict, Any

from .network import NeuralNetwork
from .learning_rules.base import LearningRule
from .learning_rules.stdp import STDPLearning
from .learning_rules.neuromodulated_stdp import NeuromodulatedSTDPLearning
from .learning_rules.hebbian import HebbianLearning
from .learning_rules.bcm import BCMLearning
from .learning_rules.oja import OjaLearning
from .learning_rules.homeostatic import HomeostaticPlasticity

# 学习规则注册表
_learning_rule_classes = {
    'stdp': STDPLearning,
    'neuromodulated_stdp': NeuromodulatedSTDPLearning,
    'hebbian': HebbianLearning,
    'bcm': BCMLearning,
    'oja': OjaLearning,
    'homeostatic': HomeostaticPlasticity,
}

def create_learning_rule(rule_type: str, network: NeuralNetwork, params: Dict[str, Any]) -> LearningRule:
    """
    创建指定类型的学习规则
    
    Args:
        rule_type: 学习规则类型
        network: 神经网络实例
        params: 学习参数字典
        
    Returns:
        创建的学习规则实例
        
    Raises:
        ValueError: 如果指定的学习规则类型不支持
    """
    if rule_type not in _learning_rule_classes:
        raise ValueError(f"不支持的学习规则类型: {rule_type}")
    
    rule_class = _learning_rule_classes[rule_type]
    return rule_class(network, params)

__all__ = [
    "LearningRule",
    "STDPLearning",
    "NeuromodulatedSTDPLearning",
    "HebbianLearning",
    "BCMLearning",
    "OjaLearning",
    "HomeostaticPlasticity",
    "create_learning_rule"
]
