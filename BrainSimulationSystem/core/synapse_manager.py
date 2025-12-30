"""
突触管理器和工厂函数
Synapse Manager and Factory Functions
"""
import logging
import time
import numpy as np
from typing import Dict, Any, Optional

from .complete_synapse import CompleteSynapse

class SynapseManager:
    """
    管理网络中所有突触的创建、更新和交互。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synapses: Dict[int, CompleteSynapse] = {}
        self.synapse_types: Dict[int, str] = {}
        
        # 性能统计
        self.update_times: list[float] = []
        self.total_synapses = 0
        self.active_synapses = 0
        
        self.logger = logging.getLogger("SynapseManager")
    
    def create_synapse(self, pre_neuron_id: int, post_neuron_id: int,
                      synapse_config: Dict[str, Any]) -> int:
        """
        创建一个新的突触并添加到管理器中。

        Returns:
            int: 新创建的突触的ID。
        """
        synapse_id = len(self.synapses)
        
        synapse = CompleteSynapse(pre_neuron_id, post_neuron_id, synapse_config)
        self.synapses[synapse_id] = synapse
        
        # 记录突触类型以供分析
        synapse_type = f"{synapse.nt_type.value}_{pre_neuron_id}_{post_neuron_id}"
        self.synapse_types[synapse_id] = synapse_type
        
        self.total_synapses += 1
        self.logger.debug(f"创建突触 {synapse_id}: {pre_neuron_id} -> {post_neuron_id}")
        
        return synapse_id
    
    def process_spike(self, synapse_id: int, spike_time: float, spike_type: str = 'pre'):
        """处理来自特定突触的脉冲事件。"""
        if synapse_id not in self.synapses:
            return
            
        synapse = self.synapses[synapse_id]
        if spike_type == 'pre':
            synapse.process_presynaptic_spike(spike_time)
        elif spike_type == 'post':
            synapse.process_postsynaptic_spike(spike_time)
    
    def update_all_synapses(
        self,
        dt: float,
        current_time: float,
        neuron_voltages: Dict[int, float],
        astrocyte_activities: Dict[int, float] = None,
        neuromodulators: Optional[Dict[str, float]] = None,
    ) -> Dict[int, float]:
        """
        更新所有管理的突触，并返回每个突触后神经元的总电流。
        """
        start_time = time.time()
        
        if astrocyte_activities is None:
            astrocyte_activities = {}
        
        postsynaptic_currents: Dict[int, float] = {}
        active_count = 0
        
        for synapse_id, synapse in self.synapses.items():
            post_neuron_id = synapse.post_neuron_id
            post_voltage = neuron_voltages.get(post_neuron_id, -70.0)
            astrocyte_activity = astrocyte_activities.get(post_neuron_id, 0.0)
            
            # 更新突触并获取其产生的电流
            psc = synapse.update(dt, current_time, post_voltage, astrocyte_activity, neuromodulators=neuromodulators)
            
            # 累加到对应的突触后神经元
            if post_neuron_id not in postsynaptic_currents:
                postsynaptic_currents[post_neuron_id] = 0.0
            postsynaptic_currents[post_neuron_id] += psc

            if psc != 0.0:
                active_count += 1
        
        self.active_synapses = active_count
        self.update_times.append(time.time() - start_time)
        
        return postsynaptic_currents

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate synapse metrics for monitoring hooks."""

        neurotransmitters: Dict[str, int] = {}
        weights = []
        for synapse in self.synapses.values():
            nt = getattr(getattr(synapse, 'nt_type', None), 'value', None)
            if nt is None:
                nt = 'unknown'
            neurotransmitters[nt] = neurotransmitters.get(nt, 0) + 1
            weight = getattr(synapse, "current_weight", None)
            if isinstance(weight, (int, float, np.floating)):
                weights.append(float(weight))

        mean_update = float(np.mean(self.update_times)) if self.update_times else 0.0
        if weights:
            w = np.array(weights, dtype=float)
            weight_statistics = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "count": int(w.size),
            }
        else:
            weight_statistics = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            'total_synapses': int(self.total_synapses),
            'active_synapses': int(self.active_synapses),
            'mean_update_time': mean_update,
            'neurotransmitter_distribution': neurotransmitters,
            'weight_statistics': weight_statistics,
        }

    def apply_global_neuromodulation(self, modulator_type: str, concentration: float):
        """对所有突触应用全局神经调节。"""
        self.logger.info(f"应用 {modulator_type} 调节 (浓度={concentration}) 到 {len(self.synapses)} 个突触")
        for synapse in self.synapses.values():
            apply_fn = getattr(synapse, "apply_neuromodulation", None)
            if callable(apply_fn):
                try:
                    apply_fn(modulator_type, concentration)
                except Exception:
                    continue

# --- 工厂函数 ---

def create_glutamate_synapse_config(weight: float = 1.0, learning_enabled: bool = True) -> Dict[str, Any]:
    """创建兴奋性（谷氨酸）突触的配置字典。"""
    return {
        'weight': weight,
        'delay': np.random.uniform(1.0, 3.0),
        'neurotransmitter': 'glutamate',
        'receptors': {
            'ampa': 100.0,
            'nmda': 20.0,
        },
        'stp_enabled': True,
        'ltp_enabled': learning_enabled,
        'metaplasticity': True
    }

def create_gaba_synapse_config(weight: float = -1.0) -> Dict[str, Any]:
    """创建抑制性（GABA）突触的配置字典。"""
    return {
        'weight': weight,
        'delay': np.random.uniform(0.5, 2.0),
        'neurotransmitter': 'gaba',
        'receptors': {
            'gaba_a': 200.0,
        },
        'stp_enabled': True,
        'ltp_enabled': False,  # 抑制性突触通常不表现经典的STDP
    }

def create_synapse_manager(config: Dict[str, Any]) -> SynapseManager:
    """Factory helper that returns a configured ``SynapseManager`` instance."""
    return SynapseManager(config)

__all__ = [
    'SynapseManager',
    'create_synapse_manager',
    'create_glutamate_synapse_config',
    'create_gaba_synapse_config',
]
