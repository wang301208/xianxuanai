"""
统一网络模型实现
整合所有网络结构，消除重复实现
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .network import NeuralNetwork
from .neuron_models import LIFNeuron, IzhikevichNeuron, AdExNeuron, HodgkinHuxleyNeuron
from .synapse_models import StaticSynapse, DynamicSynapse, NMDAReceptorSynapse, GABAReceptorSynapse
from .neuron_base import NeuronBase, NeuronType, SynapseBase, SynapseType


class StructuredNetwork(NeuralNetwork):
    """基础神经网络模型，复用 :mod:`.network` 中定义的公共接口."""

    def __init__(self, network_id: int, params: Dict[str, Any]):
        config = dict(params)
        config["network_id"] = network_id
        super().__init__(config)
        self.network_id = network_id
        self.params = params
        # 精细模型仍使用特化的容器
        self.neurons: Dict[int, NeuronBase] = {}
        self.synapses: Dict[Tuple[int, int], SynapseBase] = {}  # (pre, post) -> synapse
        self._current_time = 0.0
        self._dt = params.get("dt", 0.1)  # 时间步长 (ms)

    def add_neuron(self, neuron_id: int, neuron_type: NeuronType, neuron_params: Dict[str, Any]) -> None:
        """添加神经元到网络"""
        neuron_class = self._get_neuron_class(neuron_type)
        params = dict(neuron_params) if isinstance(neuron_params, dict) else {}
        neuron = neuron_class(neuron_id, params)
        setattr(neuron, 'neuron_type', neuron_type)
        self.neurons[neuron_id] = neuron
    
    def add_synapse(self, pre_id: int, post_id: int, synapse_type: SynapseType, 
                   synapse_params: Dict[str, Any]) -> None:
        """添加突触到网络"""
        if pre_id not in self.neurons or post_id not in self.neurons:
            raise ValueError(f"无效的神经元ID: pre={pre_id}, post={post_id}")
        
        synapse_class = self._get_synapse_class(synapse_type)
        synapse_id = len(self.synapses)
        self.synapses[(pre_id, post_id)] = synapse_class(synapse_id, synapse_type, synapse_params)
    
    def _get_neuron_class(self, neuron_type: NeuronType):
        """根据类型获取神经元类"""
        neuron_classes = {
            NeuronType.LIF: LIFNeuron,
            NeuronType.IZHIKEVICH: IzhikevichNeuron,
            NeuronType.AD_EX: AdExNeuron,  # 使用修复后的枚举名称
            NeuronType.HODGKIN_HUXLEY: HodgkinHuxleyNeuron,
        }
        return neuron_classes.get(neuron_type, LIFNeuron)
    
    def _get_synapse_class(self, synapse_type: SynapseType):
        """根据类型获取突触类"""
        synapse_classes = {
            SynapseType.STATIC: StaticSynapse,
            SynapseType.DYNAMIC: DynamicSynapse,
            SynapseType.NMDA: NMDAReceptorSynapse,
            SynapseType.GABA: GABAReceptorSynapse,
        }
        return synapse_classes.get(synapse_type, StaticSynapse)
    
    def reset(self) -> None:
        """重置网络状态"""
        super().reset()
        self._current_time = 0.0
        for neuron in self.neurons.values():
            neuron.reset()
        for synapse in self.synapses.values():
            synapse.reset()

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """执行一个时间步的模拟"""
        if dt is not None:
            self._dt = dt
        elif self._dt is None:
            raise ValueError("时间步长未设置，调用 step() 时必须提供 dt")

        super().step(self._dt)

        # 收集前一个时间步的脉冲信息
        spike_info = {}
        for neuron_id, neuron in self.neurons.items():
            spike_info[neuron_id] = neuron.is_spiking

        # 更新突触
        synaptic_inputs = {}
        for (pre_id, post_id), synapse in self.synapses.items():
            pre_spike = spike_info.get(pre_id, False)
            post_neuron = self.neurons[post_id]
            
            # 根据突触类型调用不同的更新方法
            if isinstance(synapse, (NMDAReceptorSynapse, GABAReceptorSynapse)):
                current = synapse.update(self._dt, pre_spike, post_neuron.voltage)
            elif isinstance(synapse, DynamicSynapse):
                current = synapse.update(self._dt, pre_spike, spike_info.get(post_id, False))
            else:
                current = synapse.update(self._dt, pre_spike)
            
            synaptic_inputs[post_id] = synaptic_inputs.get(post_id, 0.0) + current
        
        # 更新神经元
        network_state = {}
        for neuron_id, neuron in self.neurons.items():
            input_current = synaptic_inputs.get(neuron_id, 0.0)
            # Use the compatibility ``step`` API so diverse neuron implementations
            # consistently receive synaptic input currents.
            network_state[neuron_id] = neuron.step(self._dt, input_current, current_time=self._current_time)

        self._current_time += self._dt

        return {
            "time": self._current_time,
            "network_state": network_state,
            "spike_times": self._get_spike_times()
        }
    
    def _get_spike_times(self) -> Dict[int, List[float]]:
        """获取所有神经元的脉冲时间"""
        spike_times = {}
        for neuron_id, neuron in self.neurons.items():
            spike_times[neuron_id] = neuron.spike_times.copy()
        return spike_times

class FeedForwardNetwork(StructuredNetwork):
    """前馈神经网络"""
    
    def __init__(self, network_id: int, params: Dict[str, Any]):
        super().__init__(network_id, params)
        self.layers: List[List[int]] = []  # 每层神经元ID列表
    
    def add_layer(self, neuron_ids: List[int]) -> None:
        """添加神经元层"""
        self.layers.append(neuron_ids)
    
    def connect_layers(self, pre_layer_idx: int, post_layer_idx: int, 
                      connection_prob: float = 0.5, synapse_params: Dict[str, Any] = None) -> None:
        """连接两个层"""
        if pre_layer_idx >= len(self.layers) or post_layer_idx >= len(self.layers):
            raise ValueError("层索引超出范围")
        
        pre_neurons = self.layers[pre_layer_idx]
        post_neurons = self.layers[post_layer_idx]
        
        synapse_params = synapse_params or {"weight": 1.0, "delay": 1.0}
        
        for pre_id in pre_neurons:
            for post_id in post_neurons:
                if np.random.random() < connection_prob:
                    self.add_synapse(pre_id, post_id, SynapseType.STATIC, synapse_params)

class RecurrentNetwork(StructuredNetwork):
    """递归神经网络"""
    
    def __init__(self, network_id: int, params: Dict[str, Any]):
        super().__init__(network_id, params)
        self.recurrent_connections: Dict[Tuple[int, int], float] = {}  # 递归连接权重
    
    def add_recurrent_connection(self, neuron_id: int, connection_strength: float = 0.1) -> None:
        """添加递归连接"""
        self.recurrent_connections[(neuron_id, neuron_id)] = connection_strength
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """重写step方法，包含递归连接"""
        result = super().step(dt)

        # 添加递归连接的影响
        for (neuron_id, _), strength in self.recurrent_connections.items():
            if neuron_id in self.neurons:
                neuron = self.neurons[neuron_id]
                # 递归连接影响下一次输入
                # 这里简化处理，实际应该在下一次step中体现
                pass
        
        return result

class ReservoirNetwork(StructuredNetwork):
    """储备池网络 (用于液体状态机)"""
    
    def __init__(self, network_id: int, params: Dict[str, Any]):
        super().__init__(network_id, params)
        self.reservoir_size = params.get("reservoir_size", 100)
        self.spectral_radius = params.get("spectral_radius", 1.2)
        self._initialize_reservoir()
    
    def _initialize_reservoir(self) -> None:
        """初始化储备池"""
        # 创建储备池神经元
        for i in range(self.reservoir_size):
            self.add_neuron(i, NeuronType.LIF, {
                "V_rest": -70.0,
                "threshold": -55.0,
                "tau_m": 10.0
            })
        
        # 创建随机连接
        connection_prob = 0.1
        for i in range(self.reservoir_size):
            for j in range(self.reservoir_size):
                if i != j and np.random.random() < connection_prob:
                    weight = np.random.normal(0, 0.5) * self.spectral_radius
                    self.add_synapse(i, j, SynapseType.STATIC, {"weight": weight})

class ModularNetwork(StructuredNetwork):
    """模块化网络"""
    
    def __init__(self, network_id: int, params: Dict[str, Any]):
        super().__init__(network_id, params)
        self.modules: Dict[str, List[int]] = {}  # 模块名称 -> 神经元ID列表
    
    def add_module(self, module_name: str, neuron_ids: List[int]) -> None:
        """添加模块"""
        self.modules[module_name] = neuron_ids
    
    def connect_modules(self, pre_module: str, post_module: str, 
                       connection_density: float = 0.3, synapse_params: Dict[str, Any] = None) -> None:
        """连接两个模块"""
        if pre_module not in self.modules or post_module not in self.modules:
            raise ValueError(f"模块不存在: {pre_module} -> {post_module}")
        
        pre_neurons = self.modules[pre_module]
        post_neurons = self.modules[post_module]
        
        synapse_params = synapse_params or {"weight": 1.0, "delay": 1.0}
        
        for pre_id in pre_neurons:
            for post_id in post_neurons:
                if np.random.random() < connection_density:
                    self.add_synapse(pre_id, post_id, SynapseType.STATIC, synapse_params)

class BrainInspiredNetwork(StructuredNetwork):
    """脑启发式网络 (包含多种神经元和突触类型)"""
    
    def __init__(self, network_id: int, params: Dict[str, Any]):
        super().__init__(network_id, params)
        self.brain_regions: Dict[str, Dict] = {}  # 脑区信息
    
    def add_brain_region(self, region_name: str, region_params: Dict[str, Any]) -> None:
        """添加脑区"""
        self.brain_regions[region_name] = {
            "neuron_type": region_params.get("neuron_type", NeuronType.LIF),
            "synapse_type": region_params.get("synapse_type", SynapseType.STATIC),
            "neuron_params": region_params.get("neuron_params", {}),
            "synapse_params": region_params.get("synapse_params", {})
        }
    
    def create_cortical_column(self, column_id: int, num_layers: int = 6, 
                              neurons_per_layer: int = 100) -> None:
        """创建皮层柱"""
        # 简化实现，实际应该更复杂
        for layer in range(num_layers):
            for i in range(neurons_per_layer):
                neuron_id = column_id * 1000 + layer * 100 + i
                self.add_neuron(neuron_id, NeuronType.LIF, {
                    "V_rest": -70.0,
                    "threshold": -55.0 + layer * 5.0  # 不同层有不同的阈值
                })


__all__ = [
    "NeuralNetwork",
    "StructuredNetwork",
    "FeedForwardNetwork",
    "RecurrentNetwork",
    "ReservoirNetwork",
    "ModularNetwork",
    "BrainInspiredNetwork",
]
