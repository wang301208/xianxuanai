"""
神经元模型统一基类
定义所有神经元模型的通用接口和抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import defaultdict
import numpy as np

class NeuronType(Enum):
    """神经元类型枚举"""
    # 基础模型
    LIF = "lif"
    IZHIKEVICH = "izhikevich"
    AD_EX = "adex"
    ADAPTIVE_EXPONENTIAL = "adex"  # 修复命名：ADAPTIVE_EXPONENTIAL -> AD_EX
    HODGKIN_HUXLEY = "hh"
    
    # 多室模型
    MULTI_COMPARTMENT = "multi_compartment"
    
    # 特化细胞类型
    PYRAMIDAL = "pyramidal"
    INTERNEURON = "interneuron"
    ASTROCYTE = "astrocyte"
    MICROGLIA = "microglia"

class SynapseType(Enum):
    """突触类型枚举"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    NMDA = "nmda"
    GABA = "gaba"
    GAP_JUNCTION = "gap"
    MODULATORY = "modulatory"

class NeuronBase(ABC):
    """神经元模型统一基类"""
    
    def __init__(self, neuron_id: int, neuron_type: NeuronType, params: Dict[str, Any]):
        """
        初始化神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            neuron_type: 神经元类型
            params: 神经元参数字典
        """
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.params = dict(params) if isinstance(params, dict) else {}

        # 默认的外部/突触电流占位，避免访问不存在属性
        self.I_ext = float(self.params.get('I_ext', 0.0))
        self.I_syn = float(self.params.get('I_syn', 0.0))

        # 将参数设置为实例属性
        for key, value in self.params.items():
            # 避免覆盖只读属性（例如 `id` 兼容属性）
            descriptor = getattr(type(self), key, None)
            if isinstance(descriptor, property) and descriptor.fset is None:
                continue
            if hasattr(self, key) and callable(getattr(self, key)):
                continue
            setattr(self, key, value)

        # 统一状态变量
        self.voltage = self.params.get('V_rest', -70.0)  # 膜电位 (mV)
        self.spike_times: List[float] = []  # 发放时间列表
        self.last_spike_time = -np.inf  # 最后发放时间
        self.is_spiking = False  # 当前是否发放

        # 神经调质(调制)状态
        self.neuromodulation: Dict[str, float] = defaultdict(float)
        
        # 初始化状态
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """重置神经元状态到初始值"""
        pass
    
    @abstractmethod
    def update(self, dt: float, input_current: float = 0.0) -> Dict[str, Any]:
        """
        统一更新接口
        
        Args:
            dt: 时间步长 (ms)
            input_current: 输入电流 (pA)
            
        Returns:
            Dict[str, Any]: 包含更新后的状态信息
        """
        pass
    
    @property
    def has_spiked(self) -> bool:
        """检查当前时间步是否发放了脉冲"""
        return self.is_spiking

    @property
    def id(self) -> int:
        """Compatibility alias for neuron identifier used by legacy components."""
        return int(self.neuron_id)

    @property
    def V(self) -> float:
        """膜电位别名(测试/模型兼容)。"""
        return float(self.voltage)

    @V.setter
    def V(self, value: float) -> None:
        self.voltage = float(value)

    @property
    def membrane_potential(self) -> float:
        """Alias used by some cortical/thalamic components."""
        return float(self.voltage)

    @membrane_potential.setter
    def membrane_potential(self, value: float) -> None:
        self.voltage = float(value)

    @property
    def external_current(self) -> float:
        """Alias for external input current."""
        return float(self.I_ext)

    @external_current.setter
    def external_current(self, value: float) -> None:
        try:
            self.I_ext = float(value)
        except (TypeError, ValueError):
            return

    @property
    def synaptic_current(self) -> float:
        """Alias for synaptic input current."""
        return float(self.I_syn)

    @synaptic_current.setter
    def synaptic_current(self, value: float) -> None:
        try:
            self.I_syn = float(value)
        except (TypeError, ValueError):
            return

    def add_input_current(self, current: float, *, replace: bool = False) -> None:
        """添加或覆盖外部输入电流(I_ext)。"""
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            return
        if replace:
            self.I_ext = current_value
        else:
            self.I_ext += current_value

    def apply_neuromodulation(self, modulator: str, concentration: float) -> None:
        """设置神经调质浓度。"""
        if not modulator:
            return
        try:
            value = float(concentration)
        except (TypeError, ValueError):
            return
        self.neuromodulation[modulator] = max(0.0, value)

    def step(self, dt: float, input_current: float = 0.0, current_time: float = 0.0) -> bool:
        """Compatibility step API used by higher-level systems.

        Many subsystems in this repo call ``neuron.step(dt, current)`` while
        neuron implementations expose ``update`` with varying signatures. This
        adapter sets ``I_ext`` and dispatches to ``update`` defensively.
        """

        try:
            current_value = float(input_current)
        except Exception:
            current_value = 0.0

        # Keep both legacy and biophysical-friendly fields in sync.
        self.external_current = current_value
        self.I_ext = current_value

        update_fn = getattr(self, "update", None)
        if update_fn is None:
            return False

        try:
            return bool(update_fn(float(dt), float(current_time)))
        except TypeError:
            try:
                return bool(update_fn(float(dt), input_current=float(input_current), current_time=float(current_time)))
            except TypeError:
                try:
                    return bool(update_fn(float(dt), float(input_current)))
                except Exception:
                    return False
        except Exception:
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前神经元状态"""
        return {
            'neuron_id': self.neuron_id,
            'neuron_type': self.neuron_type.value,
            'voltage': self.voltage,
            'spike_times': self.spike_times.copy(),
            'last_spike_time': self.last_spike_time,
            'is_spiking': self.is_spiking,
            'I_ext': self.I_ext,
            'I_syn': self.I_syn
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """动态更新参数"""
        self.params.update(params)
    
    @classmethod
    def create_neuron(cls, neuron_type: NeuronType, neuron_id: int, 
                     params: Optional[Dict[str, Any]] = None) -> 'NeuronBase':
        """工厂方法创建神经元实例"""
        from .neuron_models import (
            LIFNeuron, IzhikevichNeuron, AdExNeuron, 
            HodgkinHuxleyNeuron, MultiCompartmentNeuron
        )
        
        default_params = {
            'V_rest': -70.0,
            'threshold': -55.0,
            'refractory_period': 2.0
        }
        
        if params:
            default_params.update(params)
        
        neuron_classes = {
            NeuronType.LIF: LIFNeuron,
            NeuronType.IZHIKEVICH: IzhikevichNeuron,
            NeuronType.ADAPTIVE_EXPONENTIAL: AdExNeuron,
            NeuronType.HODGKIN_HUXLEY: HodgkinHuxleyNeuron,
            NeuronType.MULTI_COMPARTMENT: MultiCompartmentNeuron,
        }
        
        if neuron_type not in neuron_classes:
            raise ValueError(f"不支持的神经元类型: {neuron_type}")
        
        return neuron_classes[neuron_type](neuron_id, default_params)

class SynapseBase(ABC):
    """突触模型统一基类"""
    
    def __init__(self, synapse_id: int, synapse_type: SynapseType, params: Dict[str, Any]):
        """
        初始化突触
        
        Args:
            synapse_id: 突触唯一标识符
            synapse_type: 突触类型
            params: 突触参数字典
        """
        self.synapse_id = synapse_id
        self.synapse_type = synapse_type
        self.params = params
        
        # 统一状态变量
        self.weight = params.get('weight', 1.0)  # 突触权重
        self.delay = params.get('delay', 1.0)  # 突触延迟 (ms)
        self.last_activation_time = -np.inf  # 最后激活时间
        
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """重置突触状态到初始值"""
        pass
    
    @abstractmethod
    def update(self, dt: float, pre_spike: bool = False, post_voltage: float = None) -> float:
        """
        统一更新接口
        
        Args:
            dt: 时间步长 (ms)
            pre_spike: 前神经元是否发放
            post_voltage: 后神经元膜电位 (mV)
            
        Returns:
            float: 突触电流输出 (pA)
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前突触状态"""
        return {
            'synapse_id': self.synapse_id,
            'synapse_type': self.synapse_type.value,
            'weight': self.weight,
            'delay': self.delay,
            'last_activation_time': self.last_activation_time
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """动态更新参数"""
        self.params.update(params)
