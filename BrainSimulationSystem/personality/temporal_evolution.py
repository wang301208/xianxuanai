"""
个性参数时间演化模型

实现基于社会学习理论和神经可塑性的个性发展系统
"""

import numpy as np
from typing import Dict

class PersonalityEvolver:
    def __init__(self):
        # 基础五大人格特质 (初始均值)
        self.base_traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.3
        }
        
        # 社会影响因子
        self.social_factors = {
            'education': 0.1,
            'career': 0.15,
            'relationships': 0.2
        }
        
        # 时间衰减系数
        self.temporal_decay = 0.98
        
    def update(self, timestep: float = 1.0) -> Dict:
        """更新个性参数"""
        # 随机波动 (正态分布)
        noise = np.random.normal(0, 0.05, 5)
        
        # 社会影响累积
        social_effect = sum(self.social_factors.values()) / 3
        
        # 更新每个特质
        traits = list(self.base_traits.keys())
        for i, trait in enumerate(traits):
            # 基础变化 + 社会影响 + 随机波动
            delta = (0.5 - self.base_traits[trait]) * 0.1
            delta += social_effect * 0.3
            delta += noise[i]
            
            # 应用时间步长和衰减
            self.base_traits[trait] = np.clip(
                self.base_traits[trait] + delta * timestep * self.temporal_decay,
                0, 1
            )
        
        return self.base_traits.copy()