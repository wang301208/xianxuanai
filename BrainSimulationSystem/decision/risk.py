"""
风险决策模块

实现基于前景理论的神经风险计算系统
"""

from typing import Dict
import numpy as np

class RiskCalculator:
    def __init__(self):
        # 风险参数配置
        self.parameters = {
            'loss_aversion': 2.5,  # 损失厌恶系数
            'probability_weighting': 0.8,  # 概率权重系数
            'risk_sensitivity': 1.2  # 风险敏感度
        }
        
        # 个性调节因子
        self.personality_factors = {
            'risk_taking': 0.5,
            'impulsivity': 0.3
        }
    
    def calculate_utility(self, prospect: Dict) -> float:
        """计算风险前景效用值"""
        # 价值函数
        gains = prospect.get('gains', 0)
        losses = prospect.get('losses', 0)
        value = gains - self.parameters['loss_aversion'] * losses
        
        # 概率权重
        prob = self._weight_probability(prospect.get('probability', 0.5))
        
        return value * prob
    
    def _weight_probability(self, p: float) -> float:
        """应用概率权重函数"""
        gamma = self.parameters['probability_weighting']
        return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))
    
    def adjust_for_personality(self, personality: Dict):
        """根据个性特征调整风险参数"""
        # 风险寻求倾向
        self.parameters['loss_aversion'] = max(1, 
            self.parameters['loss_aversion'] - personality.get('risk_taking', 0) * 0.5)
        
        # 冲动性影响
        self.parameters['risk_sensitivity'] = min(2,
            self.parameters['risk_sensitivity'] + personality.get('impulsivity', 0) * 0.3)
    
    def get_risk_profile(self) -> Dict:
        """获取当前风险计算配置"""
        return {
            'current_loss_aversion': self.parameters['loss_aversion'],
            'effective_sensitivity': self.parameters['risk_sensitivity'] * 
                                   self.personality_factors['risk_taking']
        }