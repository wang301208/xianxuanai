"""
学习系统集成模块

包含基础学习机制与情绪调节的完整集成
"""

from typing import Dict
from ..learning.synaptic import SynapticLearning
from ..learning.emotional_modulation import EmotionalModulator

class LearningSystem:
    def __init__(self):
        # 初始化核心组件
        self.synaptic_learner = SynapticLearning()
        self.emotion_modulator = EmotionalModulator()
        
        # 学习状态跟踪
        self.learning_state = {
            'last_emotion': None,
            'consolidation_queue': []
        }
    
    def process_experience(self, experience: Dict) -> Dict:
        """处理新经验并生成记忆痕迹"""
        # 情绪评估
        emotion = self._assess_emotion(experience)
        
        # 情绪调节编码
        modulated_exp = self.emotion_modulator.modulate_encoding(
            experience, emotion)
            
        # 突触学习
        memory = self.synaptic_learner.encode(modulated_exp)
        
        # 情绪情境标记
        memory.update({
            'emotion_context': emotion,
            'encoding_time': self._get_current_time()
        })
        
        # 加入巩固队列
        self.learning_state['consolidation_queue'].append(memory)
        return memory
    
    def consolidate_memories(self):
        """离线记忆巩固处理"""
        for memory in self.learning_state['consolidation_queue']:
            # 情绪调节巩固
            stress_level = memory.get('stress', 0)
            self.emotion_modulator.modulate_consolidation(memory, stress_level)
            
            # 执行巩固
            self.synaptic_learner.consolidate(memory)
        
        # 清空队列
        self.learning_state['consolidation_queue'] = []
    
    def _assess_emotion(self, data: Dict) -> str:
        """从输入数据提取情绪特征"""
        valence = data.get('valence', 0)
        arousal = data.get('arousal', 0)
        
        if arousal > 0.7:
            return 'fear' if valence < -0.5 else 'joy'
        elif valence < -0.3:
            return 'sadness'
        return 'neutral'
    
    def _get_current_time(self) -> float:
        """获取当前仿真时间"""
        # 实际实现应从仿真时钟获取
        import time
        return time.time()

# 系统单例
learning_system = LearningSystem()