"""
具有短期可塑性（STP）的突触模型
Synapse with Short-Term Plasticity (STP)
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any

from .neuron_base import SynapseBase, SynapseType

@dataclass
class SynapticPlasticityState:
    """短期突触可塑性状态变量"""
    
    # 抑制变量
    depression: float = 1.0  # 可用囊泡部分
    tau_depression: float = 200.0  # 抑制时间常数 (ms)
    
    # 易化变量
    facilitation: float = 1.0  # 易化因子
    tau_facilitation: float = 50.0  # 易化时间常数 (ms)
    
    # 使用参数
    use_baseline: float = 0.2  # 基线释放概率
    use_facilitation: float = 0.1  # 易化增量
    
    # 状态追踪
    last_spike_time: float = -1000.0  # 上一个突触前脉冲时间
    
    def update(self, current_time: float, spike_occurred: bool) -> float:
        """更新可塑性状态并返回有效释放概率"""
        dt = current_time - self.last_spike_time
        
        if dt > 0:
            # 指数恢复
            self.depression += (1.0 - self.depression) * (1.0 - np.exp(-dt / self.tau_depression))
            self.facilitation *= np.exp(-dt / self.tau_facilitation)
        
        if spike_occurred:
            # 计算有效释放概率
            use_effective = self.use_baseline + self.facilitation * self.use_facilitation
            use_effective = min(use_effective, 1.0)
            
            # 应用抑制
            release_prob = use_effective * self.depression
            
            # 更新状态变量
            self.depression *= (1.0 - use_effective)
            self.facilitation += self.use_facilitation * (1.0 - self.facilitation)
            self.last_spike_time = current_time
            
            return release_prob
        
        return 0.0


class STPSynapse(SynapseBase):
    """具有短期可塑性（抑制和易化）的突触"""
    
    def __init__(self, pre_id: int, post_id: int, params: Dict[str, Any]):
        super().__init__(pre_id, post_id, params)
        self.stp_state = SynapticPlasticityState()
        
        # 如果提供，则覆盖STP参数
        if 'tau_depression' in params:
            self.stp_state.tau_depression = params['tau_depression']
        if 'tau_facilitation' in params:
            self.stp_state.tau_facilitation = params['tau_facilitation']
        if 'use_baseline' in params:
            self.stp_state.use_baseline = params['use_baseline']
        if 'use_facilitation' in params:
            self.stp_state.use_facilitation = params['use_facilitation']
    
    def transmit(self, pre_spike: bool, dt: float, current_time: float = None) -> float:
        """通过短期可塑性调制传递信号"""
        if current_time is None:
            current_time = time.time() * 1000  # 转换为毫秒
        
        # 更新STP状态并获取释放概率
        release_prob = self.stp_state.update(current_time, pre_spike)
        
        if pre_spike and release_prob > 0:
            # 应用延迟
            if not hasattr(self, '_delay_buffer'):
                self._delay_buffer = deque()
            self._delay_buffer.append((current_time + self.delay, self.weight * release_prob))
        
        # 处理延迟的脉冲
        current_input = 0.0
        if hasattr(self, '_delay_buffer'):
            while self._delay_buffer and self._delay_buffer[0][0] <= current_time:
                _, delayed_weight = self._delay_buffer.popleft()
                current_input += delayed_weight
        
        return current_input
    
    def reset(self):
        """重置突触状态"""
        super().reset()
        self.stp_state = SynapticPlasticityState()
        if hasattr(self, '_delay_buffer'):
            self._delay_buffer.clear()

__all__ = ["STPSynapse", "SynapticPlasticityState"]