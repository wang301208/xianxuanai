"""
学习规则基类
Base Class for Learning Rules
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class LearningRule(ABC):
    """学习规则基类，定义所有学习规则的通用接口"""
    
    def __init__(self, network, params: Dict[str, Any]):
        """
        初始化学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典
        """
        self.network = network
        self.params = params
    
    @abstractmethod
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        根据网络状态更新权重或神经元参数
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        pass