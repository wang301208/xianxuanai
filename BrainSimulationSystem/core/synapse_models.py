"""
统一突触模型实现
整合所有突触模型，消除重复实现
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .neuron_base import SynapseBase, SynapseType

class StaticSynapse(SynapseBase):
    """静态突触模型 - 固定权重"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        super().__init__(synapse_id, synapse_type, params)
        self._weight = params.get("weight", 1.0)
        self._delay = params.get("delay", 1.0)  # 传输延迟 (ms)
        self._current_time = 0.0
    
    def reset(self) -> None:
        """重置突触状态"""
        self._current_time = 0.0
        self._last_spike_time = -float('inf')
    
    def update(self, dt: float, pre_spike: bool = False) -> float:
        """更新突触状态"""
        self._current_time += dt
        
        if pre_spike:
            self._last_spike_time = self._current_time
        
        # 检查是否在延迟期内有脉冲到达
        if (self._current_time - self._last_spike_time >= self._delay and 
            self._current_time - self._last_spike_time < self._delay + dt):
            return self._weight
        
        return 0.0

class DynamicSynapse(SynapseBase):
    """动态突触模型 - 可塑性权重"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        # 先初始化 reset() 依赖的状态，避免基类构造期间调用 reset 时属性缺失
        self._current_time = 0.0
        self._pre_spike_times: List[float] = []
        self._post_spike_times: List[float] = []
        self._pending_pre_spikes: List[float] = []
        super().__init__(synapse_id, synapse_type, params)
        self._weight = params.get("initial_weight", 0.5)
        self._delay = params.get("delay", 1.0)
        self._tau_stdp = params.get("tau_stdp", 20.0)  # STDP时间常数
        self._A_plus = params.get("A_plus", 0.01)  # LTP幅度
        self._A_minus = params.get("A_minus", 0.015)  # LTD幅度
    
    def reset(self) -> None:
        """重置突触状态"""
        self._current_time = 0.0
        if not hasattr(self, "_pre_spike_times") or self._pre_spike_times is None:
            self._pre_spike_times = []
        else:
            self._pre_spike_times.clear()
        if not hasattr(self, "_post_spike_times") or self._post_spike_times is None:
            self._post_spike_times = []
        else:
            self._post_spike_times.clear()
        if not hasattr(self, "_pending_pre_spikes") or self._pending_pre_spikes is None:
            self._pending_pre_spikes = []
        else:
            self._pending_pre_spikes.clear()
    
    def _update_stdp(self, dt: float) -> None:
        """更新STDP可塑性"""
        if not self._pre_spike_times or not self._post_spike_times:
            return
        
        # 获取最近的脉冲时间
        latest_pre = max(self._pre_spike_times) if self._pre_spike_times else -float('inf')
        latest_post = max(self._post_spike_times) if self._post_spike_times else -float('inf')
        
        # 计算时间差
        if latest_pre > latest_post:  # 前脉冲在后脉冲之后
            delta_t = latest_pre - latest_post
            weight_change = self._A_plus * np.exp(-delta_t / self._tau_stdp)
        else:  # 前脉冲在后脉冲之前
            delta_t = latest_post - latest_pre
            weight_change = -self._A_minus * np.exp(-delta_t / self._tau_stdp)
        
        self._weight += weight_change * dt
        self._weight = max(0.0, min(1.0, self._weight))  # 限制权重范围
    
    def update(self, dt: float, pre_spike: bool = False, post_spike: bool = False) -> float:
        """更新动态突触状态"""
        self._current_time += dt
        
        if pre_spike:
            self._pre_spike_times.append(self._current_time)
            self._pending_pre_spikes.append(self._current_time)
        
        if post_spike:
            self._post_spike_times.append(self._current_time)
        
        # 更新STDP
        self._update_stdp(dt)
        
        delivered = 0.0
        if self._pending_pre_spikes:
            while self._pending_pre_spikes:
                t_pre = self._pending_pre_spikes[0]
                if self._current_time - t_pre >= self._delay:
                    delivered += float(self._weight)
                    self._pending_pre_spikes.pop(0)
                else:
                    break

        return delivered

class NMDAReceptorSynapse(SynapseBase):
    """NMDA受体突触模型"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        super().__init__(synapse_id, synapse_type, params)
        self._weight = params.get("weight", 0.3)
        self._delay = params.get("delay", 2.0)
        self._tau_rise = params.get("tau_rise", 2.0)  # 上升时间常数
        self._tau_decay = params.get("tau_decay", 150.0)  # 衰减时间常数
        self._mg_block = params.get("mg_block", True)  # 镁离子阻断
        self._mg_concentration = params.get("mg_concentration", 1.0)  # 镁离子浓度
        self._current_time = 0.0
        self._g = 0.0  # 电导
        self._last_spike_time = -float('inf')
    
    def _mg_block_factor(self, V: float) -> float:
        """计算镁离子阻断因子"""
        if not self._mg_block:
            return 1.0
        return 1.0 / (1.0 + np.exp(-0.062 * V) * (self._mg_concentration / 3.57))
    
    def reset(self) -> None:
        """重置突触状态"""
        self._current_time = 0.0
        self._g = 0.0
        self._last_spike_time = -float('inf')
    
    def update(self, dt: float, pre_spike: bool = False, post_voltage: float = -70.0) -> float:
        """更新NMDA突触状态"""
        self._current_time += dt
        
        if pre_spike:
            self._last_spike_time = self._current_time
        
        # 计算电导变化
        if self._current_time - self._last_spike_time < self._delay:
            dg = 0.0
        else:
            time_since_spike = self._current_time - self._last_spike_time - self._delay
            if time_since_spike >= 0:
                # 双指数函数模拟NMDA电流
                alpha = np.exp(-time_since_spike / self._tau_decay) - np.exp(-time_since_spike / self._tau_rise)
                dg = (alpha - self._g) / dt
            else:
                dg = -self._g / self._tau_decay
        
        self._g += dg * dt
        
        # 计算电流 (考虑镁离子阻断)
        mg_factor = self._mg_block_factor(post_voltage)
        current = self._weight * self._g * mg_factor * (post_voltage - 0.0)  # NMDA反转电位为0mV
        
        return current

class GABAReceptorSynapse(SynapseBase):
    """GABA受体突触模型 (抑制性)"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        super().__init__(synapse_id, synapse_type, params)
        self._weight = params.get("weight", -0.5)  # 负权重表示抑制
        self._delay = params.get("delay", 1.5)
        self._tau_rise = params.get("tau_rise", 1.0)
        self._tau_decay = params.get("tau_decay", 10.0)
        self._current_time = 0.0
        self._g = 0.0
        self._last_spike_time = -float('inf')
    
    def reset(self) -> None:
        """重置突触状态"""
        self._current_time = 0.0
        self._g = 0.0
        self._last_spike_time = -float('inf')
    
    def update(self, dt: float, pre_spike: bool = False, post_voltage: float = -70.0) -> float:
        """更新GABA突触状态"""
        self._current_time += dt
        
        if pre_spike:
            self._last_spike_time = self._current_time
        
        # 计算电导变化
        if self._current_time - self._last_spike_time < self._delay:
            dg = 0.0
        else:
            time_since_spike = self._current_time - self._last_spike_time - self._delay
            if time_since_spike >= 0:
                # 双指数函数
                alpha = np.exp(-time_since_spike / self._tau_decay) - np.exp(-time_since_spike / self._tau_rise)
                dg = (alpha - self._g) / dt
            else:
                dg = -self._g / self._tau_decay
        
        self._g += dg * dt
        
        # 计算电流 (GABA反转电位通常为-70mV)
        current = self._weight * self._g * (post_voltage - (-70.0))
        
        return current

class GapJunction(SynapseBase):
    """电突触 (间隙连接) 模型"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        super().__init__(synapse_id, synapse_type, params)
        self._conductance = params.get("conductance", 0.1)  # 电导 (nS)
        self._delay = params.get("delay", 0.1)  # 电突触延迟很短
    
    def reset(self) -> None:
        """重置突触状态"""
        pass  # 电突触无状态
    
    def update(self, dt: float, pre_voltage: float, post_voltage: float) -> float:
        """更新电突触状态"""
        # 欧姆定律: I = g * (V_pre - V_post)
        current = self._conductance * (pre_voltage - post_voltage)
        return current

class ModulatorySynapse(SynapseBase):
    """神经调质突触模型"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        super().__init__(synapse_id, synapse_type, params)
        self._modulator_type = params.get("modulator_type", "dopamine")
        self._release_probability = params.get("release_probability", 0.3)
        self._concentration = 0.0
        self._tau_clearance = params.get("tau_clearance", 1000.0)  # 清除时间常数
        self._current_time = 0.0
    
    def reset(self) -> None:
        """重置突触状态"""
        self._concentration = 0.0
        self._current_time = 0.0
    
    def update(self, dt: float, pre_spike: bool = False) -> Dict[str, Any]:
        """更新调质突触状态"""
        self._current_time += dt
        
        if pre_spike and np.random.random() < self._release_probability:
            self._concentration += 1.0  # 释放调质
        
        # 调质清除
        self._concentration -= self._concentration / self._tau_clearance * dt
        self._concentration = max(0.0, self._concentration)
        
        return {
            "modulator_type": self._modulator_type,
            "concentration": self._concentration,
            "modulation_effect": self._calculate_modulation_effect()
        }
    
    def _calculate_modulation_effect(self) -> Dict[str, float]:
        """计算调质对突触可塑性的影响"""
        effects = {
            "dopamine": {"stdp_amplitude": 1.0 + 0.5 * self._concentration, 
                         "weight_multiplier": 1.0 + 0.2 * self._concentration},
            "serotonin": {"stdp_amplitude": 1.0 - 0.3 * self._concentration,
                         "weight_multiplier": 1.0 + 0.1 * self._concentration},
            "acetylcholine": {"stdp_amplitude": 1.0 + 0.4 * self._concentration,
                            "weight_multiplier": 1.0 + 0.3 * self._concentration}
        }
        return effects.get(self._modulator_type, {"stdp_amplitude": 1.0, "weight_multiplier": 1.0})
