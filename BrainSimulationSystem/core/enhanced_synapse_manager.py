"""
增强突触管理器模块
管理增强突触系统的整体运行和协调
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .enhanced_synapse import EnhancedSynapse
from .glia_system import GlialSystem
from .volume_transmission import VolumeTransmissionSystem
from .enhanced_configs import EnhancedSynapseConfig, GlialConfig, VolumeTransmissionConfig

class EnhancedSynapseManager:
    """增强突触管理器"""
    
    def __init__(self, volume: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)):
        self.volume = volume
        
        # 配置系统
        self.synapse_config = EnhancedSynapseConfig()
        self.glial_config = GlialConfig()
        self.volume_config = VolumeTransmissionConfig()
        
        # 子系统
        self.glial_system = GlialSystem(self.glial_config)
        self.volume_transmission = VolumeTransmissionSystem(self.volume_config)
        
        # 突触管理
        self.synapses = {}  # synapse_id -> EnhancedSynapse
        self.synapse_locations = {}  # synapse_id -> (x, y, z)
        self.neuron_connections = {}  # neuron_id -> [synapse_ids]
        
        # 初始化子系统
        self.glial_system.initialize_glial_cells(self.volume)
    
    def add_synapse(self, synapse_id: str, pre_neuron_id: str, post_neuron_id: str,
                   location: Tuple[float, float, float], config: Optional[EnhancedSynapseConfig] = None):
        """添加突触"""
        if config is None:
            config = self.synapse_config
        
        synapse = EnhancedSynapse(config)
        self.synapses[synapse_id] = synapse
        self.synapse_locations[synapse_id] = location
        
        # 更新神经元连接
        if pre_neuron_id not in self.neuron_connections:
            self.neuron_connections[pre_neuron_id] = []
        self.neuron_connections[pre_neuron_id].append(synapse_id)
        
        if post_neuron_id not in self.neuron_connections:
            self.neuron_connections[post_neuron_id] = []
        self.neuron_connections[post_neuron_id].append(synapse_id)
    
    def update_synapses(self, dt: float, neuron_spikes: Dict[str, bool],
                       neuromodulators: Optional[Dict[str, Dict[str, float]]] = None):
        """更新所有突触"""
        updated_weights = {}
        
        for synapse_id, synapse in self.synapses.items():
            # 获取突触位置
            location = self.synapse_locations.get(synapse_id, (0, 0, 0))
            
            # 获取神经调质浓度
            local_neuromodulators = {}
            if neuromodulators:
                for modulator, concentrations in neuromodulators.items():
                    if modulator in self.volume_transmission.diffusion_grid:
                        concentration = self.volume_transmission.get_concentration(
                            modulator, location)
                        local_neuromodulators[modulator] = concentration
            
            # 获取突触前和突触后神经元ID
            pre_neuron_id = None
            post_neuron_id = None
            
            for nid, connections in self.neuron_connections.items():
                if synapse_id in connections:
                    if pre_neuron_id is None:
                        pre_neuron_id = nid
                    else:
                        post_neuron_id = nid
            
            # 检查突触前和突触后发放
            pre_spike = neuron_spikes.get(pre_neuron_id, False) if pre_neuron_id else False
            post_spike = neuron_spikes.get(post_neuron_id, False) if post_neuron_id else False
            
            # 更新突触
            effective_weight = synapse.update(dt, pre_spike, post_spike, local_neuromodulators)
            updated_weights[synapse_id] = effective_weight
            
            # 如果突触前发放，释放神经递质
            if pre_spike:
                self._release_neurotransmitters(synapse_id, location, effective_weight)
        
        # 更新子系统
        self.glial_system.update_metabolism(dt)
        self.volume_transmission.update_diffusion(dt)
        
        return updated_weights
    
    def _release_neurotransmitters(self, synapse_id: str, location: Tuple[float, float, float],
                                 weight: float):
        """释放神经递质"""
        # 主要递质释放
        glutamate_amount = weight * 0.1
        self.volume_transmission.add_point_source('glutamate', location, glutamate_amount)
        
        # 根据突触类型可能释放其他递质
        synapse = self.synapses[synapse_id]
        synapse_stats = synapse.get_statistics()
        
        if synapse_stats['state'] == 'potentiated':
            # 强化的突触可能释放更多递质
            additional_glutamate = weight * 0.05
            self.volume_transmission.add_point_source('glutamate', location, additional_glutamate)
    
    def apply_glial_support(self, neuron_id: str, location: Tuple[float, float, float],
                           energy_demand: float) -> float:
        """应用胶质细胞支持"""
        return self.glial_system.metabolic_support(location, energy_demand)
    
    def apply_volume_transmission_effects(self, neuron_id: str, location: Tuple[float, float, float],
                                        receptor_types: List[str]) -> Dict[str, float]:
        """应用体积传导效应"""
        effects = {}
        
        for receptor in receptor_types:
            concentration = self.volume_transmission.get_concentration(receptor, location)
            
            # 简化的受体激活模型
            if concentration > 0.1:  # 阈值
                activation = 1.0 - np.exp(-concentration / 0.5)  # 饱和曲线
                effects[receptor] = activation
        
        return effects
    
    def trigger_calcium_wave(self, location: Tuple[float, float, float], intensity: float):
        """触发钙波"""
        self.glial_system.calcium_wave_propagation(location, intensity, 0.0)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'synapses': {
                'total': len(self.synapses),
                'active': sum(1 for s in self.synapses.values() 
                            if s.get_statistics()['state'] == 'active'),
                'potentiated': sum(1 for s in self.synapses.values() 
                                 if s.get_statistics()['state'] == 'potentiated'),
                'depressed': sum(1 for s in self.synapses.values() 
                               if s.get_statistics()['state'] == 'depressed')
            },
            'glial_system': self.glial_system.get_statistics(),
            'volume_transmission': self.volume_transmission.get_statistics()
        }
        
        return stats
    
    def reset_system(self):
        """重置整个系统"""
        for synapse in self.synapses.values():
            synapse.reset()
        
        # 重置子系统
        self.glial_system.initialize_glial_cells(self.volume)
        
        # 重置体积传导
        if self.volume_config.enabled:
            self.volume_transmission._initialize_diffusion_grid()