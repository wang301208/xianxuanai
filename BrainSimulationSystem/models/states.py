"""
认知状态模块
Cognitive States Module

定义了与认知过程相关的各种状态的数据类。
- NeuralOscillation: 神经振荡状态
- AttentionState: 注意力状态
- ConsciousnessState: 意识状态
"""
import time
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class NeuralOscillation:
    """神经振荡"""
    frequency: float
    amplitude: float
    phase: float
    coherence: float = 0.0
    power: float = 0.0
    
    def update(self, dt: float, coupling_strength: float = 0.0, 
               external_drive: float = 0.0) -> float:
        """更新振荡状态"""
        
        # 相位更新
        self.phase += 2 * np.pi * self.frequency * dt
        self.phase = self.phase % (2 * np.pi)
        
        # 幅度调节
        self.amplitude += external_drive * dt
        self.amplitude = np.clip(self.amplitude, 0.0, 10.0)
        
        # 计算当前值
        current_value = self.amplitude * np.sin(self.phase)
        
        # 更新功率
        self.power = self.amplitude ** 2
        
        return current_value

@dataclass
class AttentionState:
    """注意力状态"""
    focus_strength: float = 0.5
    focus_location: Tuple[float, float] = (0.0, 0.0)
    attention_span: float = 1.0
    distraction_level: float = 0.0
    
    # 注意力网络
    alerting_network: float = 0.5
    orienting_network: float = 0.5
    executive_network: float = 0.5
    
    def update_attention(self, sensory_input: Dict[str, float], 
                        top_down_control: float) -> Dict[str, float]:
        """更新注意力状态"""
        
        # 自下而上的注意力捕获
        bottom_up_saliency = max(sensory_input.values()) if sensory_input else 0.0
        
        # 自上而下的注意力控制
        top_down_bias = top_down_control * self.executive_network
        
        # 综合注意力强度
        total_attention = (bottom_up_saliency + top_down_bias) / 2.0
        self.focus_strength = np.clip(total_attention, 0.0, 1.0)
        
        # 注意力分配
        attention_weights = {}
        total_input = sum(sensory_input.values()) if sensory_input else 1.0
        
        for modality, intensity in sensory_input.items():
            # 基于显著性和注意力偏向的权重
            weight = (intensity / total_input) * self.focus_strength
            attention_weights[modality] = weight
        
        return attention_weights

@dataclass
class ConsciousnessState:
    """意识状态"""
    awareness_level: float = 0.7
    global_workspace_activity: float = 0.5
    integration_level: float = 0.5
    
    # 意识内容
    conscious_content: Dict[str, Any] = field(default_factory=dict)
    
    # 全局工作空间理论参数
    competition_threshold: float = 0.6
    coalition_strength: float = 0.0
    
    def update_consciousness(self, neural_activities: Dict[str, float],
                           attention_state: AttentionState) -> Dict[str, float]:
        """更新意识状态"""
        
        # 全局工作空间竞争
        competing_coalitions = {}
        for region, activity in neural_activities.items():
            if activity > self.competition_threshold:
                competing_coalitions[region] = activity
        
        # 选择获胜联盟
        if competing_coalitions:
            winner = max(competing_coalitions.items(), key=lambda x: x[1])
            self.coalition_strength = winner[1]
            
            # 全局广播
            broadcast_strength = self.coalition_strength * attention_state.focus_strength
            self.global_workspace_activity = broadcast_strength
            
            # 更新意识内容
            self.conscious_content[winner[0]] = {
                'activity': winner[1],
                'timestamp': time.time(),
                'attention_weight': attention_state.focus_strength
            }
        
        # 整合信息理论测量
        self.integration_level = self._calculate_phi(neural_activities)
        
        # 总体意识水平
        self.awareness_level = (self.global_workspace_activity + self.integration_level) / 2.0
        
        return {
            'awareness_level': self.awareness_level,
            'global_workspace_activity': self.global_workspace_activity,
            'integration_level': self.integration_level,
            'conscious_content': len(self.conscious_content)
        }
    
    def _calculate_phi(self, activities: Dict[str, float]) -> float:
        """计算整合信息Φ（简化版本）"""
        
        if len(activities) < 2:
            return 0.0

        # 简化的Φ计算：以“活动相似度”近似整合度量，避免对单样本向量使用相关矩阵。
        values = np.array(list(activities.values()), dtype=float)
        if values.size < 2:
            return 0.0

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        spread = float(np.max(values) - np.min(values))
        if spread < 1e-12:
            # 活动一致性高：整合度由总体激活强度决定
            return float(np.clip(np.mean(values), 0.0, 1.0))

        normalized = (values - float(np.min(values))) / spread
        diffs = np.abs(normalized[:, None] - normalized[None, :])
        similarity = 1.0 - diffs

        tri = similarity[np.triu_indices_from(similarity, k=1)]
        phi = float(np.mean(tri)) if tri.size else 0.0
        return float(np.clip(phi, 0.0, 1.0))
