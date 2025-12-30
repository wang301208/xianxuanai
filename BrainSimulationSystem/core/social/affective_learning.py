"""
社会情感学习模块

实现镜像神经模拟、共情发展和道德判断
"""

class MirrorNeuronSystem:
    def __init__(self):
        self.mirror_ratio = 0.7      # 镜像强度(0-1)
        self.emotional_resonance = 0.5  # 情绪共鸣系数
        
    def simulate(self, observed_action: Dict) -> float:
        """模拟观察到的动作
        
        Args:
            observed_action: 包含motor_pattern和reward的动作描述
            
        Returns:
            模拟奖励值(0-1)
        """
        # 运动模式匹配度
        motor_similarity = self._calc_motor_similarity(observed_action['motor_pattern'])
        
        # 综合镜像响应
        return self.mirror_ratio * motor_similarity * observed_action['reward']
        
    def _calc_motor_similarity(self, pattern: List[float]) -> float:
        """计算运动模式相似度"""
        # 简化实现 - 实际应使用运动皮层表征
        return min(1.0, sum(p**2 for p in pattern)/len(pattern))
        

class SocialAffectiveLearner:
    def __init__(self):
        self.mirror_system = MirrorNeuronSystem()
        self.empathy_level = 0.5
        self.moral_judgment = 0.6
        
    def observe_action(self, actor: Dict, action: Dict) -> float:
        """观察学习与共情反应
        
        Args:
            actor: 包含emotional_intensity的施动者状态
            action: 包含motor_pattern和reward的动作描述
            
        Returns:
            学习到的奖励预期值
        """
        # 基础镜像学习
        base_reward = self.mirror_system.simulate(action)
        
        # 共情调节
        empathy_modulation = 1 + (self.empathy_level - 0.5) * actor['emotional_intensity']
        learned_value = base_reward * empathy_modulation
        
        # 道德判断调节
        if action.get('unethical', False):
            learned_value *= (1 - self.moral_judgment)
            
        return min(1.0, max(0.0, learned_value))
        
    def update_empathy(self, social_feedback: float):
        """根据社会反馈更新共情水平"""
        self.empathy_level += 0.1 * (social_feedback - 0.5)
        self.empathy_level = max(0.1, min(0.9, self.empathy_level))
        
    def moral_development(self, ethical_dilemma: Dict):
        """通过道德困境发展道德判断"""
        # 简化实现 - 实际应使用道德推理模型
        if ethical_dilemma['harm'] > 0.5:
            self.moral_judgment = min(1.0, self.moral_judgment + 0.05)