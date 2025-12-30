"""
增强突触模型模块
实现具有多种可塑性机制的增强突触模型
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .neuron_base import SynapseBase, SynapseType
from .enhanced_configs import EnhancedSynapseConfig, PlasticityType, SynapseState

class EnhancedSynapse(SynapseBase):
    """增强突触模型"""
    
    def __init__(self, config: EnhancedSynapseConfig):
        self.config = config
        self.weight = config.weight
        self.delay = config.delay
        
        # 可塑性状态
        self.plasticity_states = {
            PlasticityType.STDP: 0.0,
            PlasticityType.HOMEOSTATIC: 0.0,
            PlasticityType.L_LTP: 0.0,
            PlasticityType.L_LTD: 0.0,
            PlasticityType.METAPLASTICITY: 0.0
        }
        
        # 突触状态
        self.state = SynapseState.ACTIVE
        self.tag_level = 0.0
        self.protein_synthesis_level = 0.0
        
        # 钙信号
        self.calcium_concentration = config.calcium_concentration
        self.camp_concentration = config.camp_concentration
        
        # 时间相关变量
        self.last_pre_spike_time = -float('inf')
        self.last_post_spike_time = -float('inf')
        self.activation_history = []
        
        # 神经调质影响
        self.neuromodulator_effects = {}
        
    def update(self, dt: float, pre_spike: bool, post_spike: bool, 
               neuromodulators: Optional[Dict[str, float]] = None) -> float:
        """更新突触状态"""
        
        # 更新钙浓度
        self._update_calcium(dt, pre_spike, post_spike)
        
        # 更新可塑性机制
        self._update_stdp(dt, pre_spike, post_spike)
        self._update_homeostatic_plasticity(dt)
        self._update_late_plasticity(dt)
        self._update_metaplasticity(dt)
        
        # 应用神经调质影响
        if neuromodulators:
            self._apply_neuromodulator_effects(neuromodulators, dt)
        
        # 更新突触状态
        self._update_synapse_state(dt)
        
        # 计算输出权重
        effective_weight = self._calculate_effective_weight()
        
        return effective_weight
    
    def _update_calcium(self, dt: float, pre_spike: bool, post_spike: bool):
        """更新钙浓度"""
        # 钙流入
        if pre_spike:
            self.calcium_concentration += 1.0  # 突触前钙流入
        
        if post_spike:
            self.calcium_concentration += 0.5  # 突触后钙流入
        
        # 钙衰减
        decay_rate = 0.01  # 1/ms
        self.calcium_concentration *= np.exp(-decay_rate * dt)
        
        # 钙浓度限制
        self.calcium_concentration = np.clip(self.calcium_concentration, 0.0, 10.0)
    
    def _update_stdp(self, dt: float, pre_spike: bool, post_spike: bool):
        """更新STDP可塑性"""
        if pre_spike:
            # 突触前发放，检查突触后发放时间
            time_diff = self.last_post_spike_time - self.last_pre_spike_time
            
            if time_diff > 0:  # 突触后先发放，LTD
                stdp_change = -0.01 * np.exp(-time_diff / 20.0)
            else:  # 突触前先发放，LTP
                stdp_change = 0.01 * np.exp(time_diff / 20.0)
            
            self.plasticity_states[PlasticityType.STDP] += stdp_change
            
            self.last_pre_spike_time = 0.0  # 重置时间
        
        if post_spike:
            self.last_post_spike_time = 0.0
    
    def _update_homeostatic_plasticity(self, dt: float):
        """更新稳态可塑性"""
        # 基于平均活动水平调整权重
        target_activity = 0.1  # 目标活动水平
        current_activity = np.mean(self.activation_history) if self.activation_history else 0.0
        
        activity_error = current_activity - target_activity
        homeostatic_change = -0.001 * activity_error * dt
        
        self.plasticity_states[PlasticityType.HOMEOSTATIC] += homeostatic_change
        
        # 更新活动历史
        self.activation_history.append(self.weight)
        if len(self.activation_history) > 1000:  # 保持最近1000次记录
            self.activation_history.pop(0)
    
    def _update_late_plasticity(self, dt: float):
        """更新晚期可塑性"""
        # 检查是否满足晚期LTP/LTD条件
        calcium_threshold_ltp = 2.0
        calcium_threshold_ltd = 1.5
        
        if self.calcium_concentration > calcium_threshold_ltp:
            # 晚期LTP
            self.plasticity_states[PlasticityType.L_LTP] += 0.0001 * dt
            self.tag_level += 0.001 * dt
            
        elif self.calcium_concentration > calcium_threshold_ltd:
            # 晚期LTD
            self.plasticity_states[PlasticityType.L_LTD] += 0.0001 * dt
            self.tag_level -= 0.001 * dt
        
        # 标签衰减
        tag_decay = np.exp(-dt / self.config.tag_decay_tau)
        self.tag_level *= tag_decay
        
        # 蛋白质合成
        if self.tag_level > self.config.protein_synthesis_threshold:
            self.protein_synthesis_level += 0.001 * dt
        else:
            self.protein_synthesis_level *= np.exp(-dt / 3600000.0)  # 1小时衰减
    
    def _update_metaplasticity(self, dt: float):
        """更新元可塑性"""
        # 基于历史活动调整可塑性阈值
        activity_variance = np.var(self.activation_history) if len(self.activation_history) > 10 else 0.0
        
        if activity_variance > 0.1:
            # 高活动方差，降低可塑性阈值
            metaplasticity_change = -0.0001 * dt
        else:
            # 低活动方差，提高可塑性阈值
            metaplasticity_change = 0.0001 * dt
        
        self.plasticity_states[PlasticityType.METAPLASTICITY] += metaplasticity_change
    
    def _apply_neuromodulator_effects(self, neuromodulators: Dict[str, float], dt: float):
        """应用神经调质影响"""
        for modulator, concentration in neuromodulators.items():
            if modulator == 'dopamine':
                # 多巴胺增强LTP，抑制LTD
                dopamine_effect = concentration * 0.1
                self.plasticity_states[PlasticityType.STDP] += dopamine_effect * dt
                self.plasticity_states[PlasticityType.L_LTP] += dopamine_effect * 0.5 * dt
            
            elif modulator == 'acetylcholine':
                # 乙酰胆碱促进学习
                ach_effect = concentration * 0.05
                self.plasticity_states[PlasticityType.STDP] += ach_effect * dt
            
            elif modulator == 'serotonin':
                # 血清素调节情绪相关可塑性
                serotonin_effect = concentration * 0.02
                self.plasticity_states[PlasticityType.HOMEOSTATIC] += serotonin_effect * dt
    
    def _update_synapse_state(self, dt: float):
        """更新突触状态"""
        # 基于可塑性水平更新状态
        total_plasticity = sum(self.plasticity_states.values())
        
        if total_plasticity > 0.5:
            self.state = SynapseState.POTENTIATED
        elif total_plasticity < -0.3:
            self.state = SynapseState.DEPRESSED
        elif abs(total_plasticity) < 0.1:
            self.state = SynapseState.SILENT
        else:
            self.state = SynapseState.ACTIVE
        
        # 检查标签状态
        if self.tag_level > 0.2:
            self.state = SynapseState.TAGGED
    
    def _calculate_effective_weight(self) -> float:
        """计算有效权重"""
        # 基础权重
        base_weight = self.weight
        
        # 可塑性影响
        plasticity_factor = 1.0 + sum(self.plasticity_states.values())
        
        # 蛋白质合成影响
        protein_factor = 1.0 + self.protein_synthesis_level * 0.1
        
        # 状态影响
        state_factor = 1.0
        if self.state == SynapseState.POTENTIATED:
            state_factor = 1.5
        elif self.state == SynapseState.DEPRESSED:
            state_factor = 0.7
        elif self.state == SynapseState.SILENT:
            state_factor = 0.3
        
        effective_weight = base_weight * plasticity_factor * protein_factor * state_factor
        
        # 权重限制
        return np.clip(effective_weight, 0.0, 5.0)
    
    def reset(self):
        """重置突触状态"""
        self.calcium_concentration = self.config.calcium_concentration
        self.camp_concentration = self.config.camp_concentration
        self.tag_level = 0.0
        self.protein_synthesis_level = 0.0
        self.activation_history.clear()
        
        # 重置可塑性状态
        for key in self.plasticity_states:
            self.plasticity_states[key] = 0.0
        
        self.state = SynapseState.ACTIVE
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取突触统计信息"""
        return {
            'weight': self.weight,
            'effective_weight': self._calculate_effective_weight(),
            'state': self.state.value,
            'calcium': self.calcium_concentration,
            'tag_level': self.tag_level,
            'protein_synthesis': self.protein_synthesis_level,
            'plasticity_states': {k.value: v for k, v in self.plasticity_states.items()}
        }