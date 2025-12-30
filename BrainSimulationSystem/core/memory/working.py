"""
情绪增强的工作记忆系统

实现受情绪状态调节的工作记忆处理流程
"""

from typing import Dict, List
from ..cognition.interface import CognitiveEmotionInterface

class WorkingMemory:
    def __init__(self, interface: CognitiveEmotionInterface):
        self.interface = interface
        self.slots = []
        self.current_emotion = None
        
    def encode(self, stimulus: Dict) -> Dict:
        """情绪调节的信息编码"""
        # 更新当前情绪状态
        self.current_emotion = stimulus.get('emotion', {})
        
        # 应用情绪调节
        attention_boost = self.interface.apply_emotion_effects(
            'attention', self.current_emotion)
        processed = {
            'content': stimulus['content'],
            'salience': stimulus.get('salience', 1.0) * attention_boost,
            'duration': self._calculate_duration(stimulus),
            'emotion_context': self.current_emotion
        }
        
        # 维护工作记忆槽
        self._manage_slots(processed)
        return processed
        
    def _calculate_duration(self, stimulus: Dict) -> float:
        """计算信息保持时间"""
        base_duration = 2.0  # 秒
        cognitive_load = stimulus.get('cognitive_load', 0.5)
        return base_duration * (1 + cognitive_load)
        
    def _manage_slots(self, item: Dict):
        """管理工作记忆槽"""
        if len(self.slots) >= 7:  # 容量限制
            self.slots.pop(0)
        self.slots.append(item)
        
    def get_contents(self) -> List[Dict]:
        """获取当前工作记忆内容"""
        return self.slots
        
    def apply_cognitive_control(self, strategy: Dict):
        """应用认知控制策略"""
        if not self.current_emotion:
            return
            
        regulation = self.interface.regulate_emotion(strategy)
        if 'intensity_reduction' in regulation:
            self.current_emotion['intensity'] *= (
                1 - regulation['intensity_reduction'])