"""
个性发展轨迹模型

实现基于年龄阶段和生活事件的个性参数动态变化系统
"""

from typing import Dict, List

class PersonalityDevelopmentTracker:
    def __init__(self):
        # 基于大五人格的发展曲线参数
        self.developmental_curves = {
            'openness': {
                'peak_age': 25,
                'growth_rate': 0.015,
                'decline_rate': 0.01
            },
            'conscientiousness': {
                'linear_growth': 0.02,
                'plateau_age': 50
            },
            'extraversion': {
                'teen_peak': 18,
                'midlife_dip': 40,
                'recovery_rate': 0.005
            },
            'agreeableness': {
                'late_bloom': 0.01,
                'social_acceleration': 0.03
            },
            'neuroticism': {
                'decline_start': 30,
                'decline_rate': 0.015
            }
        }
        
        # 生活事件影响矩阵
        self.life_event_impacts = {
            'career_advancement': {
                'conscientiousness': +0.1,
                'neuroticism': -0.05
            },
            'major_trauma': {
                'neuroticism': +0.3,
                'openness': -0.15
            },
            'long_term_relationship': {
                'agreeableness': +0.2,
                'extraversion': +0.1
            }
        }
    
    def calculate_trajectory(self, current_age: int, life_history: List) -> Dict:
        """计算当前个性发展状态"""
        traits = {}
        
        # 应用基础发展曲线
        for trait, curve in self.developmental_curves.items():
            if 'peak_age' in curve:
                # 钟形曲线特质
                age_diff = current_age - curve['peak_age']
                traits[trait] = max(0, min(1, 
                    0.5 + curve['growth_rate'] * curve['peak_age'] - 
                    abs(age_diff) * curve['decline_rate']
                ))
            elif 'linear_growth' in curve:
                # 线性增长特质
                traits[trait] = min(1, 0.3 + current_age * curve['linear_growth'])
        
        # 应用生活事件影响
        for event in life_history:
            impacts = self.life_event_impacts.get(event['type'], {})
            for trait, delta in impacts.items():
                traits[trait] = max(0, min(1, traits.get(trait, 0.5) + delta * event.get('intensity', 1)))
        
        return traits
    
    def predict_future_trend(self, current_age: int, years: int) -> Dict:
        """预测未来个性发展趋势"""
        return {
            trait: min(1, value * (1 + 0.01 * years))  # 简化预测模型
            for trait, value in self.calculate_trajectory(current_age, []).items()
        }