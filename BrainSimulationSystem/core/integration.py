"""
个性与性别动态系统集成

将时间演化模型与内化过程连接为统一系统
"""

from personality.temporal_evolution import PersonalityEvolver
from personality.gender_internalization import GenderInternalizer

class IntegratedSystem:
    def __init__(self):
        self.personality = PersonalityEvolver()
        self.gender = GenderInternalizer()
        self.time_elapsed = 0.0
    
    def step(self, hours: float = 1.0):
        """推进系统时间"""
        self.time_elapsed += hours
        
        # 每24小时更新一次个性
        if self.time_elapsed >= 24:
            traits = self.personality.update(hours / 24)
            self.time_elapsed = 0
            
            # 个性变化影响性别认知
            gender_strength = 0.05 * traits['openness']
            gender_state = self.gender.internalize(gender_strength)
            
            return {
                'traits': traits,
                'gender_state': gender_state
            }
        return None