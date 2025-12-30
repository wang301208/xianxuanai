"""
突触整合与后电位模型
"""

import numpy as np
from typing import Dict, List

class SynapticIntegration:
    """树突整合系统"""
    def __init__(self):
        # 突触类型参数 (mV/脉冲)
        self.psp_params = {
            'AMPA': {'amp': 0.5, 'tau_rise': 0.5, 'tau_decay': 2.0},
            'NMDA': {'amp': 1.2, 'tau_rise': 2.0, 'tau_decay': 50.0},
            'GABA_A': {'amp': -0.8, 'tau_rise': 0.3, 'tau_decay': 5.0}
        }
        self.history = []  # 脉冲历史记录
    
    def add_spikes(self, spikes: Dict[str, int]):
        """记录新到达的脉冲"""
        self.history.append({
            'time': len(self.history) * 0.1,  # 假设固定时间步
            'counts': spikes
        })
    
    def compute_current(self, t_now: float) -> float:
        """计算总突触后电流"""
        total = 0.0
        for event in self.history:
            t_diff = t_now - event['time']
            for syn_type, count in event['counts'].items():
                params = self.psp_params[syn_type]
                if t_diff > 0:
                    # 双指数模型
                    term = np.exp(-t_diff/params['tau_decay']) - np.exp(-t_diff/params['tau_rise'])
                    total += params['amp'] * count * term
        return total