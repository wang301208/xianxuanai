"""
个性特质系统增强版

包含社会文化影响因素的个性发展模型
"""

from typing import Dict, List

class PersonalitySystem:
    def __init__(self):
        # 大五人格基础特质
        self.traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        # 新增社会影响参数
        self.social_factors = {
            'cultural_norms': 0.5,  # 文化规范内化程度
            'peer_pressure': 0.3,   # 同伴压力敏感度
            'authority_influence': 0.4,  # 权威影响系数
            'social_learning_rate': 0.2  # 社会学习速率
        }
        
        # 社会情境记忆
        self.social_memory = []
    
    def process_social_experience(self, experience: Dict):
        """处理社会经验并更新个性"""
        # 记录社会经验
        self.social_memory.append(experience)
        
        # 文化规范影响
        if 'cultural_value' in experience:
            delta = experience['cultural_value'] * self.social_factors['social_learning_rate']
            self.social_factors['cultural_norms'] = max(0, min(1, 
                self.social_factors['cultural_norms'] + delta))
        
        # 同伴压力影响
        if 'peer_consensus' in experience:
            delta = (experience['peer_consensus'] - 0.5) * self.social_factors['peer_pressure']
            self.traits['agreeableness'] = max(0, min(1,
                self.traits['agreeableness'] + delta))
    
    def get_social_influence_score(self) -> float:
        """计算当前社会影响综合指数"""
        weights = {
            'cultural_norms': 0.4,
            'peer_pressure': 0.3,
            'authority_influence': 0.3
        }
        return sum(
            self.social_factors[factor] * weight 
            for factor, weight in weights.items()
        ) / sum(weights.values())
    
    def predict_conformity(self, situation: Dict) -> float:
        """预测在给定社会情境中的从众倾向"""
        authority_presence = situation.get('authority', False)
        peer_uniformity = situation.get('peer_uniformity', 0.5)
        
        base = self.social_factors['peer_pressure'] * peer_uniformity
        if authority_presence:
            base += self.social_factors['authority_influence'] * 0.5
        return min(1, base * 1.2)