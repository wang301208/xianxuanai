"""
情绪调节系统

基于PAD三维情绪模型调节记忆过程
"""

import numpy as np
from typing import Dict
from .events import EventBus

class AmygdalaModel:
    """杏仁核情绪处理模型"""
    
    def __init__(self):
        self.threat_sensitivity = 0.7
        self.reward_sensitivity = 0.6
        self.emotion_state = np.zeros(3)  # PAD向量
        
        EventBus.subscribe('sensory_input', self.evaluate_stimulus)
        
    def evaluate_stimulus(self, stimulus: Dict):
        """评估刺激的情绪意义"""
        # 威胁检测
        threat_level = self._calc_threat(stimulus)
        
        # 奖赏检测
        reward_level = self._calc_reward(stimulus)
        
        # 更新PAD情绪状态
        self.emotion_state = np.array([
            reward_level - threat_level,            # 效价(Pleasure)
            max(threat_level, reward_level),        # 唤醒度(Arousal)
            1 - abs(reward_level - threat_level)   # 控制度(Dominance)
        ])
        
        EventBus.publish('emotion_update', 
                        pad_state=self.emotion_state)
        
    def _calc_threat(self, stimulus: Dict) -> float:
        """计算威胁程度"""
        return min(1, stimulus.get('intensity', 0) * self.threat_sensitivity)
        
    def _calc_reward(self, stimulus: Dict) -> float:
        """计算奖赏价值"""
        return min(1, stimulus.get('novelty', 0) * self.reward_sensitivity)

class EmotionModulation:
    """情绪-记忆调节接口"""
    
    def __init__(self):
        # 创伤记忆阈值
        self.trauma_threshold = 0.8  # 威胁敏感度阈值
        self.suppression_factor = 0.3  # 抑制系数
        
        # 优先巩固参数
        self.priority_tags = {
            'high_arousal': 0.7,
            'positive_valence': 0.6,
            'novelty': 0.5
        }
        self.modulators = {
            'dopamine': 0.5,    # 多巴胺水平 [0,1]
            'norepinephrine': 0.5,  # 去甲肾上腺素
            'serotonin': 0.5    # 血清素
        }
        
        EventBus.subscribe('memory_encoding', self.on_encoding)
        EventBus.subscribe('memory_retrieval', self.on_retrieval)
        
    def on_encoding(self, memory: Dict):
        """编码阶段情绪调节"""
        # 标记优先巩固记忆
        memory['priority'] = self._calc_priority(memory)
        
        # 健康状态下的情绪增强
        memory['strength'] *= (1 + self.modulators['dopamine'] * 0.3)
            
        # 多巴胺增强记忆强度
        memory['strength'] *= (1 + self.modulators['dopamine'] * 0.5)
        
    def _calc_priority(self, memory: Dict) -> float:
        """计算记忆优先级"""
        priority = 0.0
        if memory.get('arousal', 0) > self.priority_tags['high_arousal']:
            priority += 1.0
        if memory.get('valence', 0) > self.priority_tags['positive_valence']:
            priority += 0.8
        if memory.get('novelty', 0) > self.priority_tags['novelty']:
            priority += 0.5
        return min(1.0, priority)
        
    def on_retrieval(self, memory: Dict):
        """检索阶段情绪调节"""
        # 创伤记忆抑制机制
        if memory.get('is_trauma', False) and not memory.get('suppressed', True):
            memory['retrievability'] *= self.suppression_factor
            memory['detail'] *= 0.7  # 细节模糊化
            
        # 去甲肾上腺素增强细节回忆
        memory['detail'] *= (1 + self.modulators['norepinephrine'] * 0.3)
        
    def regulate_trauma(self, memory: Dict, suppression: bool = True):
        """创伤记忆调控"""
        memory['suppressed'] = suppression
        if suppression:
            memory['retrievability'] *= self.suppression_factor
        else:
            memory['retrievability'] /= self.suppression_factor