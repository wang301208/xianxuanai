"""
社会情境决策模块

实现基于社会规范和价值判断的决策系统
"""

from typing import Dict
import numpy as np

class SocialDecisionSystem:
    def __init__(self):
        # 社会规范权重
        self.norm_weights = {
            'reciprocity': 0.6,  # 互惠性
            'fairness': 0.8,     # 公平性
            'authority': 0.4,    # 权威服从
            'conformity': 0.5    # 从众倾向
        }
        
        # 社会价值参数
        self.value_parameters = {
            'ingroup_bias': 0.7,  # 群体内偏袒
            'social_distance': 0.3  # 社会距离影响
        }
    
    def evaluate_social_context(self, context: Dict) -> float:
        """评估社会情境影响"""
        # 规范一致性评分
        norm_score = sum(
            context.get(k, 0) * w 
            for k, w in self.norm_weights.items()
        ) / sum(self.norm_weights.values())
        
        # 社会价值评估
        value_score = (
            context.get('ingroup', 0) * self.value_parameters['ingroup_bias'] -
            context.get('distance', 0) * self.value_parameters['social_distance']
        )
        
        return np.clip(norm_score + value_score * 0.5, 0, 1)
    
    def update_norms(self, experience: Dict):
        """根据社会经验更新规范权重"""
        for norm in self.norm_weights:
            if norm in experience:
                delta = experience[norm] * 0.1 - 0.05  # 标准化调整
                self.norm_weights[norm] = np.clip(
                    self.norm_weights[norm] + delta, 0.1, 1.0)
    
    def get_social_profile(self) -> Dict:
        """获取当前社会决策参数"""
        return {
            'dominant_norm': max(self.norm_weights, key=self.norm_weights.get),
            'ingroup_bias': self.value_parameters['ingroup_bias']
        }