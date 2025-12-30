"""
性格动态演化系统

实现基于生物-心理-社会模型的性格参数时间演化
"""

import numpy as np
from typing import Dict, Tuple

class PersonalityDynamics:
    def __init__(self):
        # 基础特质参数 (均值±标准差)
        self.base_traits = {
            'openness': (0.5, 0.15),
            'conscientiousness': (0.6, 0.1), 
            'extraversion': (0.4, 0.2),
            'agreeableness': (0.55, 0.12),
            'neuroticism': (0.3, 0.18)
        }
        
        # 社会影响权重
        self.social_weights = {
            'education': 0.2,
            'occupation': 0.25,
            'relationships': 0.35,
            'trauma': -0.5
        }
        
        # 生物约束范围
        self.biological_constraints = {
            'openness': (0.2, 0.9),
            'neuroticism': (0.1, 0.7)
        }

    def update(self, timestep: float, env_factors: Dict) -> Dict[str, float]:
        """更新性格参数"""
        new_traits = {}
        
        for trait, (mean, std) in self.base_traits.items():
            # 1. 基础值 + 随机波动
            base = np.random.normal(mean, std * 0.5)
            
            # 2. 社会影响计算
            social_effect = sum(
                weight * env_factors.get(factor, 0)
                for factor, weight in self.social_weights.items()
            )
            
            # 3. 应用时间衰减
            delta = social_effect * timestep * 0.1
            constrained_min, constrained_max = self.biological_constraints.get(trait, (0.1, 0.9))
            
            new_value = np.clip(base + delta, constrained_min, constrained_max)
            new_traits[trait] = round(new_value, 3)
            
            # 更新基准值 (缓慢漂移)
            self.base_traits[trait] = (mean * 0.95 + new_value * 0.05, std)
            
        return new_traits