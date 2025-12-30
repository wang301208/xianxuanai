"""
后端管理器
Backend Manager

统一管理软件框架和硬件接口
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import time
from dataclasses import dataclass

from .framework_integration import (
    FrameworkType, SimulationConfig, BaseFrameworkBackend,
    create_framework_backend, get_available_frameworks,
    DistributedSimulationManager, MultiGPUManager
)
from ..core.backends import (
    NeuromorphicPlatform,
    HardwareConfig,
    BaseNeuromorphicInterface,
    create_neuromorphic_interface,
    ModelHardwareTranslator,
    detect_available_hardware,
)

class BackendType(Enum):
    """后端类型"""
    SOFTWARE = "software"
    HARDWARE = "hardware"
    HYBRID = "hybrid"

@dataclass
class BackendCapabilities:
    """后端能力"""
    max_neurons: int
    max_synapses: int
    supports_learning: bool
    supports_real_time: bool
    supports_distributed: bool
    supports_gpu: bool
    power_efficient: bool
    precision: str

class UnifiedBackendManager:
    """统一后端管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger("UnifiedBackendManager")
        
        # 后端实例
        self.software_backends = {}
        self.hardware_interfaces = {}
        self.active_backend = None
        
        # 管理器
        self.distributed_manager = None
        self.gpu_manager = MultiGPUManager()
        self.translator = ModelHardwareTranslator()
        
        # 能力映射
        self.backend_capabilities = {}
        
        # 初始化
        self._initialize_capabilities()
        self._detect_available_backends()
    
    def _initialize_capabilities(self):
        """初始化后端能力映射"""
        
        # 软件框架能力
        self.backend_capabilities.update({
            FrameworkType.NEST: BackendCapabilities(
                max_neurons=10**7,
                max_synapses=10**10,
                supports_learning=True,
                supports_real_time=False,
                supports_distributed=True,
                supports_gpu=False,
                power_efficient=False,
                precision="float64"
            ),
            FrameworkType.CARLSIM: BackendCapabilities(
                max_neurons=10**6,
                max_synapses=10**9,
                supports_learning=True,
                supports_real_time=True,
                supports_distributed=False,
                supports_gpu=True,
                power_efficient=True,
                precision="float32"
            ),
            FrameworkType.GENN: BackendCapabilities(
                max_neurons=10**6,
                max_synapses=10**9,
                supports_learning=True,
                supports_real_time=True,
                supports_distributed=False,
                supports_gpu=True,
                power_efficient=True,
                precision="float32"
            ),
            FrameworkType.TVB: BackendCapabilities(
                max_neurons=10**5,
                max_synapses=10**7,
                supports_learning=False,
                supports_real_time=False,
                supports_distributed=True,
                supports_gpu=False,
                power_efficient=False,
                precision="float64"
            )
        })
        
        # 硬件平台能力
        self.backend_capabilities.update({
            NeuromorphicPlatform.SPINNAKER: BackendCapabilities(
                max_neurons=10**6,
                max_synapses=10**9,
                supports_learning=True,
                supports_real_time=True,
                supports_distributed=True,
                supports_gpu=False,
                power_efficient=True,
                precision="fixed16"
            ),
            NeuromorphicPlatform.LOIHI: BackendCapabilities(
                max_neurons=10**5,
                max_synapses=10**8,
                supports_learning=True,
                supports_real_time=True,
                supports_distributed=False,
                supports_gpu=False,
                power_efficient=True,
                precision="fixed8"
            ),
            NeuromorphicPlatform.TRUENORTH: BackendCapabilities(
                max_neurons=10**6,
                max_synapses=2.56*10**8,
                supports_learning=False,
                supports_real_time=True,
                supports_distributed=True,
                supports_gpu=False,
                power_efficient=True,
                precision="fixed1"
            )
        })
    
    def _detect_available_backends(self):
        """检测可用后端"""
        
        self.logger.info("检测可用后端...")
        
        # 检测软件框架
        available_frameworks = get_available_frameworks()
        self.logger.info(f"可用软件框架: {[f.value for f in available_frameworks]}")
        
        # 检测硬件平台
        available_hardware = detect_available_hardware()
        self.logger.info(f"可用硬件平台: {[h.value for h in available_hardware]}")
        
        # 检测GPU
        self.gpu_manager.detect_gpus()
    
    def select_optimal_backend(self, 
                              requirements: Dict[str, Any]) -> Union[FrameworkType, NeuromorphicPlatform]:
        """选择最优后端"""
        
        self.logger.info("选择最优后端...")
        
        # 提取需求
        n_neurons = requirements.get('n_neurons', 1000)
        n_synapses = requirements.get('n_synapses', 100000)
        real_time = requirements.get('real_time', False)
        learning = requirements.get('learning', False)
        power_budget = requirements.get('power_budget', float('inf'))
        precision = requirements.get('precision', 'float32')
        
        # 评分函数
        def score_backend(backend_type, capabilities):
            score = 0
            
            # 容量检查
            if n_neurons > capabilities.max_neurons:
                return -1
            if n_synapses > capabilities.max_synapses:
                return -1
            
            # 功能需求
            if real_time and not capabilities.supports_real_time:
                score -= 10
            if learning and not capabilities.supports_learning:
                score -= 10
            
            # 性能偏好
            if capabilities.supports_gpu:
                score += 5
            if capabilities.power_efficient and power_budget < 10:
                score += 8
            if capabilities.supports_distributed and n_neurons > 10**5:
                score += 3
            
            # 精度匹配
            if precision in capabilities.precision:
                score += 2
            
            return score
        
        # 评估所有后端
        best_backend = None
        best_score = -1
        
        for backend_type, capabilities in self.backend_capabilities.items():
            score = score_backend(backend_type, capabilities)
            
            if score > best_score:
                best_score = score
                best_backend = backend_type
        
        self.logger.info(f"选择后端: {best_backend}, 评分: {best_score}")
        return best_backend
    
    def create_backend(self, 
                      backend_type: Union[FrameworkType, NeuromorphicPlatform],
                      config: Dict[str, Any]) -> Union[BaseFrameworkBackend, BaseNeuromorphicInterface]:
        """创建后端实例"""
        
        if isinstance(backend_type, FrameworkType):
            # 创建软件框架后端
            sim_config = SimulationConfig(
                framework=backend_type,
                **config
            )
            backend = create_framework_backend(backend_type, sim_config)
            
            if backend.initialize():
                self.software_backends[backend_type] = backend
                self.active_backend = backend
                return backend
            else:
                raise RuntimeError(f"无法初始化框架: {backend_type}")
        
        elif isinstance(backend_type, NeuromorphicPlatform):
            # 创建硬件接口
            hw_config = HardwareConfig(
                platform=backend_type,
                **config
            )
            interface = create_neuromorphic_interface(backend_type, hw_config)
            
            if interface.connect():
                self.hardware_interfaces[backend_type] = interface
                self.active_backend = interface
                return interface
            else:
                raise RuntimeError(f"无法连接硬件: {backend_type}")
        
        else:
            raise ValueError(f"不支持的后端类型: {backend_type}")
    
    def setup_distributed_simulation(self, 
                                   node_configs: List[Dict[str, Any]]) -> bool:
        """设置分布式仿真"""
        
        self.logger.info("设置分布式仿真...")
        
        self.distributed_manager = DistributedSimulationManager({})
        return self.distributed_manager.initialize_cluster(node_configs)
    
    def translate_model(self, 
                       source_model: Dict[str, Any],
                       source_backend: Union[FrameworkType, NeuromorphicPlatform],
                       target_backend: Union[FrameworkType, NeuromorphicPlatform]) -> Dict[str, Any]:
        """在后端之间转换模型"""
        
        self.logger.info(f"转换模型: {source_backend} -> {target_backend}")
        
        # 软件到硬件
        if isinstance(source_backend, FrameworkType) and isinstance(target_backend, NeuromorphicPlatform):
            return self.translator.software_to_hardware(source_model, target_backend)
        
        # 硬件到软件
        elif isinstance(source_backend, NeuromorphicPlatform) and isinstance(target_backend, FrameworkType):
            return self.translator.hardware_to_software(source_model, target_backend.value)
        
        # 软件到软件或硬件到硬件
        else:
            # 需要实现更复杂的转换逻辑
            self.logger.warning("暂不支持该类型转换")
            return source_model
    
    def run_hybrid_simulation(self, 
                             model_parts: Dict[str, Dict[str, Any]],
                             backend_assignments: Dict[str, Union[FrameworkType, NeuromorphicPlatform]]) -> Dict[str, Any]:
        """运行混合仿真"""
        
        self.logger.info("运行混合仿真...")
        
        results = {}
        
        # 为每个模型部分分配后端
        for part_name, model_part in model_parts.items():
            backend_type = backend_assignments.get(part_name)
            
            if not backend_type:
                self.logger.warning(f"模型部分 {part_name} 未分配后端")
                continue
            
            # 创建或获取后端
            if backend_type in self.software_backends:
                backend = self.software_backends[backend_type]
            elif backend_type in self.hardware_interfaces:
                backend = self.hardware_interfaces[backend_type]
            else:
                # 创建新后端
                backend = self.create_backend(backend_type, {})
            
            # 运行仿真
            try:
                part_results = self._run_simulation_part(backend, model_part)
                results[part_name] = part_results
            except Exception as e:
                self.logger.error(f"模型部分 {part_name} 仿真失败: {e}")
                results[part_name] = {'error': str(e)}
        
        return results
    
    def _run_simulation_part(self, 
                           backend: Union[BaseFrameworkBackend, BaseNeuromorphicInterface],
                           model_part: Dict[str, Any]) -> Dict[str, Any]:
        """运行仿真部分"""
        
        # 根据后端类型调用相应方法
        if isinstance(backend, BaseFrameworkBackend):
            return self._run_software_simulation(backend, model_part)
        elif isinstance(backend, BaseNeuromorphicInterface):
            return self._run_hardware_simulation(backend, model_part)
        else:
            raise ValueError("未知后端类型")
    
    def _run_software_simulation(self, 
                               backend: BaseFrameworkBackend,
                               model_part: Dict[str, Any]) -> Dict[str, Any]:
        """运行软件仿真"""
        
        # 创建神经元群体
        populations = {}
        for pop_name, pop_data in model_part.get('populations', {}).items():
            pop_id = backend.create_neuron_population(
                pop_data['type'],
                pop_data['size'],
                pop_data.get('params', {})
            )
            populations[pop_name] = pop_id
        
        # 建立连接
        for conn_data in model_part.get('connections', []):
            source_pop = populations[conn_data['source']]
            target_pop = populations[conn_data['target']]
            backend.connect_populations(source_pop, target_pop, conn_data['params'])
        
        # 添加记录设备
        for pop_name, pop_id in populations.items():
            backend.add_recording_device(pop_id, 'spike_recorder', {})
        
        # 运行仿真
        sim_results = backend.run_simulation()
        data_results = backend.get_results()
        
        return {
            'simulation_info': sim_results,
            'data': data_results,
            'populations': populations
        }
    
    def _run_hardware_simulation(self, 
                               interface: BaseNeuromorphicInterface,
                               model_part: Dict[str, Any]) -> Dict[str, Any]:
        """运行硬件仿真"""
        
        # 映射神经元
        neuron_params = []
        for pop_data in model_part.get('populations', {}).values():
            for _ in range(pop_data['size']):
                neuron_params.append(pop_data.get('params', {}))
        
        neuron_mapping = interface.map_neurons(neuron_params)
        
        # 映射突触
        from .neuromorphic_hardware import WeightMapping
        connections = []
        for conn_data in model_part.get('connections', []):
            # 简化的连接映射
            weight_mapping = WeightMapping(
                source_neuron=0,  # 需要实际映射
                target_neuron=1,  # 需要实际映射
                weight=conn_data['params'].get('weight', 1.0),
                delay=conn_data['params'].get('delay', 1.0)
            )
            connections.append(weight_mapping)
        
        interface.map_synapses(connections)
        
        # 运行仿真
        duration = model_part.get('duration', 1000.0)
        interface.start_simulation(duration)
        
        # 等待仿真完成（简化）
        time.sleep(duration / 1000.0)
        
        # 获取结果
        spike_events = interface.receive_spike_events()
        interface.stop_simulation()
        
        return {
            'spike_events': spike_events,
            'neuron_mapping': neuron_mapping,
            'hardware_status': interface.get_hardware_status()
        }
    
    def get_backend_status(self) -> Dict[str, Any]:
        """获取后端状态"""
        
        status = {
            'active_backend': str(type(self.active_backend).__name__) if self.active_backend else None,
            'software_backends': list(self.software_backends.keys()),
            'hardware_interfaces': list(self.hardware_interfaces.keys()),
            'gpu_status': {
                'available_gpus': len(self.gpu_manager.available_gpus),
                'gpu_assignments': self.gpu_manager.gpu_assignments
            },
            'distributed_status': {
                'enabled': self.distributed_manager is not None,
                'nodes': len(self.distributed_manager.nodes) if self.distributed_manager else 0
            }
        }
        
        return status
    
    def cleanup(self):
        """清理所有后端资源"""
        
        self.logger.info("清理后端资源...")
        
        # 清理软件后端
        for backend in self.software_backends.values():
            try:
                backend.cleanup()
            except Exception as e:
                self.logger.error(f"清理软件后端失败: {e}")
        
        # 清理硬件接口
        for interface in self.hardware_interfaces.values():
            try:
                interface.disconnect()
            except Exception as e:
                self.logger.error(f"清理硬件接口失败: {e}")
        
        # 清理GPU资源
        for task_id in list(self.gpu_manager.gpu_assignments.keys()):
            self.gpu_manager.release_gpu(task_id)
        
        self.software_backends.clear()
        self.hardware_interfaces.clear()
        self.active_backend = None

def create_unified_backend_manager() -> UnifiedBackendManager:
    """创建统一后端管理器"""
    return UnifiedBackendManager()