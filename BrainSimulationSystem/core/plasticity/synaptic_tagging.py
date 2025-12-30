"""
突触标记分子动力学模块

实现蛋白合成依赖的突触可塑性标记系统
"""

import numpy as np
from ..events import EventBus

class SynapticTagging:
    def __init__(self):
        # 分子标记参数
        self.tag_lifetimes = {
            'CaMKII': 60,    # 分钟
            'AMPAR': 45,
            'mTOR': 30
        }
        
        # 突触标记状态
        self.tagged_synapses = {}  # {syn_id: {type: str, strength: float}}
        
        # 注册事件处理器
        EventBus.subscribe('synapse_activated', self.on_activation)
        EventBus.subscribe('sleep_spindle', self.on_spindle)
    
    def on_activation(self, synapse):
        """突触激活事件处理"""
        if synapse.id not in self.tagged_synapses:
            self.tagged_synapses[synapse.id] = {
                'type': 'late-LTP' if synapse.persistent else 'early-LTP',
                'strength': 0.0
            }
        
        # 根据活动强度更新标记
        self.tagged_synapses[synapse.id]['strength'] += (
            0.1 * synapse.activity_level)
            
        # 触发分子级事件
        if synapse.activity_level > 0.7:
            EventBus.publish('camkii_phosphorylation', 
                            synapse=synapse,
                            intensity=0.8)
    
    def on_spindle(self, spindle_power):
        """睡眠纺锤波事件处理"""
        # 增强被标记突触
        for syn_id, tag in list(self.tagged_synapses.items()):
            if tag['strength'] > 0.5:
                EventBus.publish('protein_synthesis',
                               synapse_id=syn_id,
                               amount=tag['strength'] * spindle_power)