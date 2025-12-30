"""
情绪处理模块

实现大脑的情绪处理功能，包括基本情绪状态、情绪调节和情绪对认知的影响。
"""

import numpy as np
from collections import deque


class BasicEmotion:
    """基础情绪类"""
    
    def __init__(self, params=None):
        """
        初始化基础情绪
        
        参数:
            params (dict): 配置参数，可包含：
                - intensity: 情绪强度，0-1之间
                - decay_rate: 衰减率，0-1之间
                - threshold: 激活阈值，0-1之间
        """
        self.params = params or {}
        self.intensity = self.params.get("intensity", 0.0)
        self.decay_rate = self.params.get("decay_rate", 0.05)
        self.threshold = self.params.get("threshold", 0.3)
        self.name = self.params.get("name", "emotion")
    
    def update(self, stimulus=0.0):
        """
        更新情绪状态
        
        参数:
            stimulus (float): 外部刺激，0-1之间
            
        返回:
            当前情绪强度
        """
        # 应用刺激
        self.intensity = min(1.0, max(0.0, self.intensity + stimulus))
        
        # 自然衰减
        self.intensity = max(0.0, self.intensity - self.decay_rate)
        
        return self.intensity
    
    def is_active(self):
        """检查情绪是否活跃"""
        return self.intensity >= self.threshold


class EmotionSystem:
    """情绪系统"""
    
    def __init__(self, params=None):
        """
        初始化情绪系统
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 基本情绪
        self.emotions = {
            "happiness": BasicEmotion({"name": "happiness", "decay_rate": 0.03}),
            "sadness": BasicEmotion({"name": "sadness", "decay_rate": 0.02}),
            "anger": BasicEmotion({"name": "anger", "decay_rate": 0.04}),
            "fear": BasicEmotion({"name": "fear", "decay_rate": 0.05}),
            "disgust": BasicEmotion({"name": "disgust", "decay_rate": 0.03}),
            "surprise": BasicEmotion({"name": "surprise", "decay_rate": 0.1})
        }
        
        # 情绪记忆
        self.memory_size = self.params.get("memory_size", 100)
        self.emotion_memory = deque(maxlen=self.memory_size)
        
        # 神经调质水平
        self.dopamine_level = 1.0  # 多巴胺 (奖励、动机)
        self.serotonin_level = 1.0  # 血清素 (情绪稳定)
        self.norepinephrine_level = 1.0  # 去甲肾上腺素 (警觉)
        
        # 情绪调节参数
        self.regulation_efficiency = self.params.get("regulation_efficiency", 0.7)
    
    def receive_stimulus(self, stimulus_type, intensity):
        """
        接收情绪刺激
        
        参数:
            stimulus_type (str): 刺激类型 (reward, threat, loss等)
            intensity (float): 刺激强度，0-1之间
        """
        # 根据刺激类型影响不同情绪
        if stimulus_type == "reward":
            self.emotions["happiness"].update(intensity * 1.5)
            self.emotions["surprise"].update(intensity * 0.5)
            self.dopamine_level = min(2.0, self.dopamine_level + intensity * 0.3)
        elif stimulus_type == "threat":
            self.emotions["fear"].update(intensity * 1.2)
            self.emotions["anger"].update(intensity * 0.8)
            self.norepinephrine_level = min(2.0, self.norepinephrine_level + intensity * 0.4)
        elif stimulus_type == "loss":
            self.emotions["sadness"].update(intensity * 1.3)
            self.emotions["anger"].update(intensity * 0.5)
            self.serotonin_level = max(0.3, self.serotonin_level - intensity * 0.2)
        
        # 记录情绪事件
        self.emotion_memory.append({
            "type": stimulus_type,
            "intensity": intensity,
            "time": len(self.emotion_memory)
        })
    
    def update_emotions(self):
        """更新所有情绪状态"""
        # 神经调质对情绪的影响
        happiness_stimulus = (self.dopamine_level - 1.0) * 0.2
        self.emotions["happiness"].update(happiness_stimulus)
        
        sadness_stimulus = (1.0 - self.serotonin_level) * 0.1
        self.emotions["sadness"].update(sadness_stimulus)
        
        fear_stimulus = (self.norepinephrine_level - 1.0) * 0.15
        self.emotions["fear"].update(fear_stimulus)
        
        # 更新所有情绪
        for emotion in self.emotions.values():
            emotion.update()
        
        # 神经调质自然回归
        self.dopamine_level = max(0.5, min(2.0, 
            self.dopamine_level * 0.95 + 0.05 * 1.0))
        self.serotonin_level = max(0.5, min(1.5, 
            self.serotonin_level * 0.97 + 0.03 * 1.0))
        self.norepinephrine_level = max(0.5, min(2.0, 
            self.norepinephrine_level * 0.93 + 0.07 * 1.0))
    
    def regulate_emotions(self, regulation_type="cognitive"):
        """
        情绪调节
        
        参数:
            regulation_type (str): 调节类型 (cognitive, behavioral, physiological)
        """
        if regulation_type == "cognitive":
            # 认知重评 - 降低负面情绪
            for name in ["anger", "fear", "sadness"]:
                reduction = self.emotions[name].intensity * self.regulation_efficiency * 0.5
                self.emotions[name].intensity = max(0.0, self.emotions[name].intensity - reduction)
            
            # 增强正面情绪
            self.emotions["happiness"].intensity = min(1.0, 
                self.emotions["happiness"].intensity + 0.1 * self.regulation_efficiency)
            
            # 提高血清素水平
            self.serotonin_level = min(1.5, self.serotonin_level + 0.1)
        
        elif regulation_type == "behavioral":
            # 行为调节 - 均匀降低所有高唤醒情绪
            for name in ["anger", "fear", "happiness"]:
                if self.emotions[name].intensity > 0.5:
                    reduction = self.emotions[name].intensity * self.regulation_efficiency * 0.3
                    self.emotions[name].intensity = max(0.0, self.emotions[name].intensity - reduction)
            
            # 降低去甲肾上腺素水平
            self.norepinephrine_level = max(0.7, self.norepinephrine_level - 0.15)
        
        elif regulation_type == "physiological":
            # 生理调节 - 降低所有情绪强度
            for emotion in self.emotions.values():
                emotion.intensity *= (1.0 - self.regulation_efficiency * 0.4)
            
            # 稳定神经调质水平
            self.dopamine_level = 0.8 * self.dopamine_level + 0.2 * 1.0
            self.serotonin_level = 0.8 * self.serotonin_level + 0.2 * 1.0
            self.norepinephrine_level = 0.7 * self.norepinephrine_level + 0.3 * 1.0
    
    def current_state(self):
        """
        获取当前情绪状态
        
        返回:
            包含当前情绪状态的字典
        """
        state = {
            "emotions": {name: emotion.intensity for name, emotion in self.emotions.items()},
            "neuromodulators": {
                "dopamine": self.dopamine_level,
                "serotonin": self.serotonin_level,
                "norepinephrine": self.norepinephrine_level
            },
            "dominant_emotion": self.get_dominant_emotion(),
            "valence": self.get_valence(),
            "arousal": self.get_arousal()
        }
        return state
    
    def get_dominant_emotion(self):
        """获取主导情绪"""
        active_emotions = [(name, e.intensity) for name, e in self.emotions.items() if e.is_active()]
        if not active_emotions:
            return "neutral"
        return max(active_emotions, key=lambda x: x[1])[0]
    
    def get_valence(self):
        """获取情绪效价 (负向-正向)"""
        positive = self.emotions["happiness"].intensity
        negative = max(
            self.emotions["sadness"].intensity,
            self.emotions["anger"].intensity,
            self.emotions["fear"].intensity
        )
        return positive - negative
    
    def get_arousal(self):
        """获取情绪唤醒度"""
        return max(
            self.emotions["anger"].intensity,
            self.emotions["fear"].intensity,
            self.emotions["happiness"].intensity,
            self.emotions["surprise"].intensity
        )


class EmotionInfluence:
    """情绪对认知功能的影响"""
    
    def __init__(self, params=None):
        """
        初始化情绪影响模块
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 情绪对注意力影响的参数
        self.attention_bias = {
            "fear": {"type": "threat", "strength": 0.7},
            "anger": {"type": "goal_related", "strength": 0.6},
            "happiness": {"type": "broad", "strength": 0.5}
        }
        
        # 情绪对记忆影响的参数
        self.memory_modulation = {
            "arousal_enhancement": 0.6,  # 高唤醒增强记忆编码
            "valence_bias": 0.4  # 正性情绪偏向记忆检索
        }
        
        # 情绪对决策影响的参数
        self.decision_bias = {
            "positive_risk_taking": 0.3,  # 正性情绪增加风险偏好
            "negative_caution": 0.5  # 负性情绪增加谨慎
        }
    
    def apply_attention_bias(self, attention_system, emotion_state):
        """
        应用情绪对注意力的影响
        
        参数:
            attention_system: 注意力系统实例
            emotion_state: 当前情绪状态
        """
        dominant_emotion = emotion_state.get("dominant_emotion")
        if dominant_emotion in self.attention_bias:
            bias = self.attention_bias[dominant_emotion]
            strength = bias["strength"] * emotion_state["emotions"][dominant_emotion]
            
            if bias["type"] == "threat":
                # 恐惧情绪导致对威胁信息的注意偏向
                attention_system.threat_sensitivity = min(1.0, 
                    attention_system.threat_sensitivity + strength * 0.5)
            
            elif bias["type"] == "goal_related":
                # 愤怒情绪导致对目标相关信息的注意偏向
                attention_system.goal_bias = min(1.0, 
                    attention_system.goal_bias + strength * 0.4)
            
            elif bias["type"] == "broad":
                # 快乐情绪导致注意范围扩大
                attention_system.attention_scope = min(1.5, 
                    attention_system.attention_scope + strength * 0.3)
    
    def apply_memory_bias(self, memory_system, emotion_state):
        """
        应用情绪对记忆的影响
        
        参数:
            memory_system: 记忆系统实例
            emotion_state: 当前情绪状态
        """
        # 唤醒度增强记忆编码
        arousal_effect = emotion_state["arousal"] * self.memory_modulation["arousal_enhancement"]
        if hasattr(memory_system, "encoding_strength"):
            memory_system.encoding_strength = min(1.5, 
                memory_system.encoding_strength + arousal_effect * 0.3)
        
        # 效价影响记忆检索
        valence_effect = emotion_state["valence"] * self.memory_modulation["valence_bias"]
        if hasattr(memory_system, "retrieval_bias"):
            memory_system.retrieval_bias = max(-0.5, min(0.5, 
                memory_system.retrieval_bias + valence_effect * 0.2))
    
    def apply_decision_bias(self, decision_system, emotion_state):
        """
        应用情绪对决策的影响
        
        参数:
            decision_system: 决策系统实例
            emotion_state: 当前情绪状态
        """
        valence = emotion_state["valence"]
        
        # 正性情绪增加风险偏好
        if valence > 0.3:
            risk_bias = valence * self.decision_bias["positive_risk_taking"]
            if hasattr(decision_system, "risk_aversion"):
                decision_system.risk_aversion = max(0.0, 
                    decision_system.risk_aversion - risk_bias * 0.4)
        
        # 负性情绪增加谨慎
        elif valence < -0.3:
            caution_bias = (-valence) * self.decision_bias["negative_caution"]
            if hasattr(decision_system, "risk_aversion"):
                decision_system.risk_aversion = min(1.0, 
                    decision_system.risk_aversion + caution_bias * 0.3)


class EmotionRegulationNetwork:
    """情绪调节网络"""
    
    def __init__(self, params=None):
        """
        初始化情绪调节网络
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 前额叶皮层调节能力
        self.pfc_strength = self.params.get("pfc_strength", 0.7)
        
        # 杏仁核反应性
        self.amygdala_reactivity = self.params.get("amygdala_reactivity", 0.8)
        
        # 前扣带回监控能力
        self.acc_sensitivity = self.params.get("acc_sensitivity", 0.6)
        
        # 当前调节策略
        self.current_strategy = None
        self.strategy_efficiency = 0.0
    
    def evaluate_situation(self, emotion_state):
        """
        评估情绪状态并选择调节策略
        
        参数:
            emotion_state: 当前情绪状态
            
        返回:
            选择的调节策略
        """
        arousal = emotion_state["arousal"]
        valence = emotion_state["valence"]
        dominant_emotion = emotion_state["dominant_emotion"]
        
        if arousal > 0.7:
            if valence < -0.5:
                # 高唤醒负性情绪 - 需要强烈调节
                self.current_strategy = "physiological"
                self.strategy_efficiency = self.pfc_strength * 0.9
            else:
                # 高唤醒情绪 - 行为调节
                self.current_strategy = "behavioral"
                self.strategy_efficiency = self.pfc_strength * 0.7
        else:
            # 低唤醒情绪 - 认知调节
            self.current_strategy = "cognitive"
            self.strategy_efficiency = self.pfc_strength * 0.8
        
        return self.current_strategy
    
    def apply_regulation(self, emotion_system, strategy=None):
        """
        应用情绪调节
        
        参数:
            emotion_system: 情绪系统实例
            strategy: 调节策略 (可选)
        """
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy == "cognitive":
            # 前额叶皮层主导的认知重评
            emotion_system.regulation_efficiency = min(1.0, 
                self.strategy_efficiency * self.pfc_strength)
            emotion_system.regulate_emotions("cognitive")
        
        elif strategy == "behavioral":
            # 行为调节
            emotion_system.regulation_efficiency = min(1.0, 
                self.strategy_efficiency * self.acc_sensitivity)
            emotion_system.regulate_emotions("behavioral")
        
        elif strategy == "physiological":
            # 生理调节
            emotion_system.regulation_efficiency = min(1.0, 
                self.strategy_efficiency * (1.0 - self.amygdala_reactivity * 0.5))
            emotion_system.regulate_emotions("physiological")