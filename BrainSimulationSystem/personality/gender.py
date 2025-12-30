"""
性别角色动态内化系统

实现基于社会反馈的性别角色学习与调整机制
"""

from typing import Dict, List
import numpy as np

class GenderRoleInternalizer:
    def __init__(self):
        # 初始性别角色模板
        self.role_schemas = {
            'assertiveness': 0.5,
            'expressiveness': 0.5,
            'independence': 0.5,
            'nurturance': 0.5
        }
        
        # 社会学习参数
        self.learning_parameters = {
            'conformity_bias': 0.6,  # 从众倾向
            'authority_weight': 0.4,  # 权威影响
            'peer_influence': 0.3,    # 同伴影响
            'self_consistency': 0.7  # 自我一致性保持
        }
        
        # 内化历史记录
        self.internalization_history = []
    
    def process_social_feedback(self, feedback: Dict):
        """处理社会反馈并调整角色内化"""
        # 计算综合社会压力
        social_pressure = (
            feedback.get('authority_approval', 0) * self.learning_parameters['authority_weight'] +
            feedback.get('peer_acceptance', 0) * self.learning_parameters['peer_influence']
        ) * self.learning_parameters['conformity_bias']
        
        # 应用角色调整
        for role, value in feedback.get('role_expectations', {}).items():
            if role in self.role_schemas:
                current = self.role_schemas[role]
                # 平衡社会压力与自我一致性
                adjustment = social_pressure * (value - current) * 0.1
                self.role_schemas[role] = np.clip(
                    current + adjustment,
                    0, 1
                )
        
        # 记录内化状态
        self.internalization_history.append(self.role_schemas.copy())
    
    def get_current_schemas(self) -> Dict:
        """获取当前内化的性别角色模式"""
        return self.role_schemas.copy()
    
    def calculate_dissonance(self) -> float:
        """计算角色认知失调程度"""
        if len(self.internalization_history) < 2:
            return 0.0
        
        # 计算最近两次变化的差异
        last = self.internalization_history[-1]
        prev = self.internalization_history[-2]
        changes = [abs(last[r] - prev[r]) for r in self.role_schemas]
        return sum(changes) / len(changes)
    
    def reset_to_baseline(self, biological_sex: str):
        """重置为生物性别基准"""
        baselines = {
            'male': {'assertiveness': 0.6, 'nurturance': 0.4},
            'female': {'assertiveness': 0.4, 'nurturance': 0.6},
            'neutral': {}
        }.get(biological_sex, {})
        
        for role, value in baselines.items():
            if role in self.role_schemas:
                self.role_schemas[role] = value