"""
胶质系统集成层

安全地将胶质细胞和神经调质系统接入主模拟
"""

from ..glia import Astrocyte, MetabolicCoupling
from ..neuromodulators import ModulatorNetwork
from ..events import EventBus

class GlialIntegration:
    def __init__(self):
        self.astrocytes = []
        self.metabolic_system = MetabolicCoupling()
        self.modulator_network = ModulatorNetwork()
        
        # 注册事件处理器
        EventBus.subscribe('neuron_update', self.on_neuron_update)
        EventBus.subscribe('timestep_end', self.on_timestep_end)
        
    def on_neuron_update(self, neuron_data):
        """处理神经元活动事件"""
        for astrocyte in self.astrocytes:
            astrocyte.update(neuron_data.dt)
            
    def on_timestep_end(self, sim_state):
        """每个时间步结束时更新代谢和调质系统"""
        self.metabolic_system.update(sim_state.mean_activity, sim_state.dt)
        self.modulator_network.update(sim_state.dt)
        
    def add_astrocyte(self, position):
        """添加星形胶质细胞到模拟"""
        astro = Astrocyte()
        self.astrocytes.append(astro)
        return astro