"""
性别自定义模块

实现完全可配置的性别维度系统，与原有性别角色模型协同工作
"""

from typing import Dict, Optional
import numpy as np

class GenderCustomizer:
    def __init__(self):
        # 多维度配置
        self.config = {
            'biological': {
                'type': 'female',  # 默认女性
                'strength': 0.8    # 默认强度
            },
            'identity': {
                'self_defined': 'female',
                'on_spectrum': 0.8  # 偏向女性
            },
            'expression': {
                'masculinity': 0.3,
                'femininity': 0.7,
                'neutrality': 0.5
            }
        }
        
        # 自定义影响因子
        self.influence_factors = {
            'decision': 0.3,
            'personality': 0.2,
            'social': 0.4
        }
    
    def set_biological(self, bio_type: str, strength: float = 1.0):
        """设置生物性别参数"""
        valid_types = ['male', 'female', 'intersex', 'unspecified']
        if bio_type in valid_types:
            self.config['biological'] = {
                'type': bio_type,
                'strength': np.clip(strength, 0, 1)
            }
    
    def set_identity(self, identity: str, spectrum: Optional[float] = None):
        """设置性别认同"""
        self.config['identity']['self_defined'] = identity
        if spectrum is not None:
            self.config['identity']['on_spectrum'] = np.clip(spectrum, 0, 1)
    
    def set_expression(self, traits: Dict[str, float]):
        """设置性别表达特征"""
        for trait, value in traits.items():
            if trait in self.config['expression']:
                self.config['expression'][trait] = np.clip(value, 0, 1)
    
    def calculate_influence(self, domain: str) -> float:
        """计算对特定系统的影响因子"""
        bio = self.config['biological']['strength'] * 0.3
        identity = self.config['identity']['on_spectrum'] * 0.4
        expression = sum(self.config['expression'].values()) / 3 * 0.3
        
        base = (bio + identity + expression) / 3
        return base * self.influence_factors.get(domain, 0.3)
    
    def get_full_config(self) -> Dict:
        """获取完整配置"""
        return {
            'config': self.config,
            'influence': self.influence_factors
        }
    
    def reset_to_default(self):
        """重置为女性默认配置"""
        self.__init__()