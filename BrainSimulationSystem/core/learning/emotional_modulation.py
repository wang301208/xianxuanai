"""
情绪调节学习增强模块

实现情绪状态对记忆编码和巩固的调控机制
"""

from typing import Dict
import numpy as np

class EmotionalModulator:
    def __init__(self):
        # 神经调节物质参数
        self.neuromodulators = {
            'dopamine': {
                'baseline': 0.5,
                'learning_gain': 1.2  # 多巴胺学习增益
            },
            'norepinephrine': {
                'alert_effect': 0.7  # 警觉性增强
            },
            'cortisol': {
                'consolidation_suppress': 0.6  # 巩固抑制阈值
            }
        }
        
        # 情绪-记忆映射参数
        self.emotion_weights = {
            'fear': {'encoding': 1.3, 'retrieval': 0.8},
            'joy': {'encoding': 1.5, 'retrieval': 1.2},
            'sadness': {'encoding': 0.9, 'retrieval': 1.1}
        }
    
    def modulate_encoding(self, memory: Dict, emotion: str) -> Dict:
        """情绪调节的记忆编码增强"""
        # 获取情绪参数
        emotion_params = self.emotion_weights.get(emotion, {})
        
        # 应用编码增强
        memory['weight'] *= emotion_params.get('encoding', 1.0)
        
        # 神经调节物质效应
        if emotion == 'fear':
            memory['noradrenergic_tag'] = True
        elif emotion == 'joy':
            memory['dopaminergic_tag'] = True
            
        return memory
    
    def modulate_consolidation(self, memory: Dict, stress_level: float) -> Dict:
        """应激水平调节记忆巩固"""
        if stress_level > self.neuromodulators['cortisol']['consolidation_suppress']:
            memory['consolidation_rate'] *= 0.7
            memory['priority'] = max(0, memory['priority'] - 0.3)
        return memory
    
    def modulate_retrieval(self, memory: Dict, current_emotion: str) -> float:
        """情绪一致性检索增强"""
        emotion_match = memory.get('emotion_tag', 'neutral') == current_emotion
        return memory['strength'] * (1.2 if emotion_match else 1.0)
    
    def update_neuromodulator_levels(self, experience: Dict):
        """根据经验更新神经调节物质水平"""
        if experience.get('reward', 0) > 0:
            self.neuromodulators['dopamine']['baseline'] = min(
                1.0, self.neuromodulators['dopamine']['baseline'] + 0.1)
        
        if experience.get('threat', 0) > 0.5:
            self.neuromodulators['norepinephrine']['alert_effect'] = min(
                1.0, self.neuromodulators['norepinephrine']['alert_effect'] + 0.15)