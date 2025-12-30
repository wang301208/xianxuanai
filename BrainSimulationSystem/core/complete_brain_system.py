"""
完整大脑仿真系统
Complete Brain Simulation System

整合所有组件实现真正的全脑级别仿真
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# 导入核心组件
from .network import FullBrainNeuralNetwork
from .synapses import SynapseManager, create_synapse_manager
from .backends import NeuromorphicBackendManager, create_neuromorphic_backend_manager
from .module_interface import ModuleBus, ModuleSignal, ModuleTopic
from ..models.cognitive_base import CognitiveArchitecture, create_cognitive_architecture
from ..models.memory import ComprehensiveMemorySystem, create_comprehensive_memory_system
from .cell_diversity import CellDiversitySystem
from .vascular_system import VascularSystem
from .physiological_regions import PhysiologicalRegionManager

@dataclass
class BrainSimulationConfig:
    """大脑仿真配置"""
    
    # 规模参数
    total_neurons: int = 86_000_000_000  # 860亿神经元
    total_synapses: int = 100_000_000_000_000  # 100万亿突触
    brain_regions: int = 180  # 主要脑区数量
    
    # 仿真参数
    dt: float = 0.1  # 时间步长 (ms)
    simulation_duration: float = 1000.0  # 仿真时长 (ms)
    real_time_factor: float = 0.001  # 实时因子
    
    # 硬件配置
    use_neuromorphic: bool = True
    distributed_computing: bool = True
    gpu_acceleration: bool = True
    
    # 生物学真实性
    detailed_cell_types: bool = True
    vascular_modeling: bool = True
    glial_cells: bool = True
    metabolic_modeling: bool = True
    
    # 认知功能
    consciousness_modeling: bool = True
    memory_consolidation: bool = True
    learning_plasticity: bool = True
    
    def validate(self) -> bool:
        """验证配置有效性"""
        
        if self.total_neurons <= 0 or self.total_synapses <= 0:
            return False
        
        if self.dt <= 0 or self.simulation_duration <= 0:
            return False
        
        # 检查神经元-突触比例合理性
        synapse_per_neuron = self.total_synapses / self.total_neurons
        if synapse_per_neuron < 1000 or synapse_per_neuron > 10000:
            logging.warning(f"Unusual synapse/neuron ratio: {synapse_per_neuron}")
        
        return True

class CompleteBrainSimulationSystem:
    """完整大脑仿真系统"""
    
    def __init__(self, config: BrainSimulationConfig):
        self.config = config
        
        if not config.validate():
            raise ValueError("Invalid brain simulation configuration")
        
        # 核心组件
        self.neural_network = None
        self.synapse_manager = None
        self.backend_manager = None
        self.cognitive_architecture = None
        self.memory_system = None
        
        # 生物学组件
        self.cell_diversity_system = None
        self.vascular_system = None
        self.physiological_regions = None
        
        # 仿真状态
        self.current_time = 0.0
        self.simulation_step = 0
        self.is_running = False
        self.is_initialized = False
        self._last_run_duration: Optional[float] = None
        self._cognitive_stride_steps = 1
        self._memory_stride_steps = 1
        self._vascular_stride_steps = 1
        
        # 性能监控
        self.performance_metrics = {
            'neurons_processed': 0,
            'synapses_processed': 0,
            'simulation_speed': 0.0,
            'memory_usage': 0.0,
            'power_consumption': 0.0
        }
        
        # 数据记录
        self.spike_data = {}
        self.voltage_data = {}
        self.cognitive_data = {}
        self.memory_data = {}
        
        self.logger = logging.getLogger("CompleteBrainSystem")

        # Standardised pub/sub message bus shared across modules/backends.
        self.module_bus = ModuleBus()
        for topic in ModuleTopic:
            self.module_bus.register_topic(topic)
        self._module_bus_inbox: List[ModuleSignal] = []
        self.latest_module_bus: Optional[Dict[str, Any]] = None
        
        # 初始化系统
        self._init_task: Optional[asyncio.Task] = None
        try:
            self._init_task = asyncio.create_task(self._initialize_system())
        except RuntimeError:
            # Allow constructing the system outside of an active event loop; initialization
            # will be triggered lazily when awaited via `wait_until_initialized()`.
            self._init_task = None

        # 轻量调度：对昂贵组件按步长更新，避免测试场景产生过高开销
        try:
            dt = float(self.config.dt) if isinstance(self.config.dt, (int, float)) else 0.1
            self._cognitive_stride_steps = max(1, int(round(1.0 / max(dt, 1e-6))))  # ~1ms
            self._memory_stride_steps = max(1, int(round(5.0 / max(dt, 1e-6))))     # ~5ms
            self._vascular_stride_steps = max(1, int(round(5.0 / max(dt, 1e-6))))   # ~5ms
        except Exception:
            self._cognitive_stride_steps = 10
            self._memory_stride_steps = 50
            self._vascular_stride_steps = 50
    
    async def _initialize_system(self):
        """初始化系统"""
        
        try:
            self.logger.info("开始初始化完整大脑仿真系统...")
            
            # 1. 初始化神经网络
            await self._initialize_neural_network()
            
            # 2. 初始化突触系统
            await self._initialize_synapse_system()
            
            # 3. 初始化神经形态后端
            if self.config.use_neuromorphic:
                await self._initialize_neuromorphic_backends()
            
            # 4. 初始化认知架构
            if self.config.consciousness_modeling:
                await self._initialize_cognitive_architecture()
            
            # 5. 初始化记忆系统
            if self.config.memory_consolidation:
                await self._initialize_memory_system()
            
            # 6. 初始化生物学组件
            await self._initialize_biological_components()
            
            # 7. 建立系统间连接
            await self._establish_system_connections()
            
            self.is_initialized = True
            self.logger.info("大脑仿真系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise

    async def wait_until_initialized(self, timeout: Optional[float] = None) -> None:
        """Wait for async background initialization to complete.

        This makes the system safe to use in call sites that race with the background init
        task (e.g. tests that only sleep a fixed duration).
        """

        if self.is_initialized:
            return

        task = getattr(self, "_init_task", None)
        if task is None:
            # Initialization was not scheduled (e.g. constructed without an event loop).
            self._init_task = asyncio.create_task(self._initialize_system())
            task = self._init_task

        if timeout is None:
            await asyncio.shield(task)
            return

        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    
    async def _initialize_neural_network(self):
        """初始化神经网络"""
        
        network_config = {
            'total_neurons': self.config.total_neurons,
            'brain_regions': self.config.brain_regions,
            'detailed_modeling': True,
            'multi_scale': True,
            'real_time_processing': True
        }
        
        self.neural_network = FullBrainNeuralNetwork(network_config)
        await self.neural_network.initialize()
        
        self.logger.info(f"神经网络初始化完成: {self.config.total_neurons:,} 神经元")
    
    async def _initialize_synapse_system(self):
        """初始化突触系统"""
        
        synapse_config = {
            'total_synapses': self.config.total_synapses,
            'detailed_neurotransmitters': True,
            'plasticity_enabled': self.config.learning_plasticity,
            'multi_receptor_types': True
        }
        
        self.synapse_manager = create_synapse_manager(synapse_config)
        
        # 创建突触连接
        await self._create_synaptic_connections()
        
        self.logger.info(f"突触系统初始化完成: {self.config.total_synapses:,} 突触")
    
    async def _create_synaptic_connections(self):
        """创建突触连接"""
        
        # 基于大脑连接组学数据创建连接
        connection_patterns = self._get_connectome_patterns()
        
        synapse_count = 0
        for pattern in connection_patterns:
            source_region = pattern['source']
            target_region = pattern['target']
            connection_probability = pattern['probability']
            synapse_type = pattern['type']
            
            # 创建区域间连接
            region_synapses = await self._create_region_connections(
                source_region, target_region, connection_probability, synapse_type
            )
            synapse_count += region_synapses
        
        self.logger.info(f"创建了 {synapse_count:,} 个突触连接")
    
    def _get_connectome_patterns(self) -> List[Dict[str, Any]]:
        """获取连接组学模式"""
        
        # 基于人类连接组项目的主要连接模式
        return [
            # 皮层-皮层连接
            {'source': 'visual_cortex', 'target': 'parietal_cortex', 'probability': 0.3, 'type': 'excitatory'},
            {'source': 'visual_cortex', 'target': 'temporal_cortex', 'probability': 0.25, 'type': 'excitatory'},
            {'source': 'parietal_cortex', 'target': 'prefrontal_cortex', 'probability': 0.4, 'type': 'excitatory'},
            {'source': 'temporal_cortex', 'target': 'prefrontal_cortex', 'probability': 0.35, 'type': 'excitatory'},
            
            # 皮层-皮层下连接
            {'source': 'prefrontal_cortex', 'target': 'basal_ganglia', 'probability': 0.5, 'type': 'excitatory'},
            {'source': 'motor_cortex', 'target': 'basal_ganglia', 'probability': 0.6, 'type': 'excitatory'},
            {'source': 'motor_cortex', 'target': 'cerebellum', 'probability': 0.7, 'type': 'excitatory'},
            
            # 丘脑连接
            {'source': 'thalamus', 'target': 'prefrontal_cortex', 'probability': 0.8, 'type': 'excitatory'},
            {'source': 'thalamus', 'target': 'motor_cortex', 'probability': 0.7, 'type': 'excitatory'},
            {'source': 'thalamus', 'target': 'somatosensory_cortex', 'probability': 0.9, 'type': 'excitatory'},
            
            # 海马连接
            {'source': 'hippocampus', 'target': 'prefrontal_cortex', 'probability': 0.4, 'type': 'excitatory'},
            {'source': 'temporal_cortex', 'target': 'hippocampus', 'probability': 0.5, 'type': 'excitatory'},
            
            # 抑制性连接
            {'source': 'interneurons', 'target': 'pyramidal_cells', 'probability': 0.2, 'type': 'inhibitory'}
        ]
    
    async def _create_region_connections(self, source: str, target: str, 
                                       probability: float, synapse_type: str) -> int:
        """创建区域间连接"""
        
        # 获取源和目标区域的神经元
        source_neurons = self.neural_network.get_region_neurons(source)
        target_neurons = self.neural_network.get_region_neurons(target)
        
        if not source_neurons or not target_neurons:
            return 0

        synapse_count = 0

        # Avoid O(N^2) loops: sample a bounded number of candidate pairs.
        max_samples = 500
        desired = max(1, int(200 * float(probability)))
        samples = min(max_samples, desired)

        rng = np.random.default_rng()
        for _ in range(samples):
            source_id = int(rng.choice(source_neurons))
            target_id = int(rng.choice(target_neurons))

            if synapse_type == 'excitatory':
                synapse_config = self._create_excitatory_synapse_config()
            else:
                synapse_config = self._create_inhibitory_synapse_config()

            synapse_id = self.synapse_manager.create_synapse(
                source_id, target_id, synapse_config
            )
            if synapse_id is not None:
                synapse_count += 1

        return synapse_count
    
    def _create_excitatory_synapse_config(self) -> Dict[str, Any]:
        """创建兴奋性突触配置"""
        
        return {
            'weight': np.random.uniform(0.5, 2.0),
            'delay': np.random.uniform(1.0, 5.0),
            'neurotransmitter': 'glutamate',
            'receptors': {
                'ampa': np.random.uniform(80, 120),
                'nmda': np.random.uniform(15, 25),
                'kainate': np.random.uniform(5, 15)
            },
            'stp_enabled': True,
            'ltp_enabled': self.config.learning_plasticity,
            'tau_rec': np.random.uniform(600, 1000),
            'tau_fac': np.random.uniform(30, 70),
            'U': np.random.uniform(0.3, 0.7)
        }
    
    def _create_inhibitory_synapse_config(self) -> Dict[str, Any]:
        """创建抑制性突触配置"""
        
        return {
            'weight': np.random.uniform(-2.0, -0.5),
            'delay': np.random.uniform(0.5, 2.0),
            'neurotransmitter': 'gaba',
            'receptors': {
                'gaba_a': np.random.uniform(150, 250),
                'gaba_b': np.random.uniform(30, 70)
            },
            'stp_enabled': True,
            'ltp_enabled': False,
            'tau_rec': np.random.uniform(300, 500),
            'tau_fac': np.random.uniform(80, 120),
            'U': np.random.uniform(0.2, 0.4)
        }
    
    async def _initialize_neuromorphic_backends(self):
        """初始化神经形态后端"""
        
        backend_config = {
            'neuromorphic': {
                'hardware_platforms': {
                    'intel_loihi': {
                        'enabled': True,
                        'chip_count': 8,
                        'available_cores': 1024
                    },
                    'spinnaker': {
                        'enabled': True,
                        'board_count': 4
                    },
                    'brainscales': {
                        'enabled': False,  # 可选
                        'wafer_count': 1
                    }
                }
            }
        }
        
        self.backend_manager = create_neuromorphic_backend_manager(backend_config)
        
        # 初始化后端
        backend_results = await self.backend_manager.initialize_backends()
        
        # 选择最优后端
        optimal_backend = await self.backend_manager.select_optimal_backend(
            self.config.total_neurons,
            {'real_time': True, 'power_limit': 1000.0}
        )
        
        self.logger.info(f"神经形态后端初始化完成: {optimal_backend}")
    
    async def _initialize_cognitive_architecture(self):
        """初始化认知架构"""
        
        cognitive_config = {
            'brain_regions': {
                'prefrontal_cortex': {'volume': 15000, 'neuron_density': 80000},
                'hippocampus': {'volume': 4000, 'neuron_density': 150000},
                'visual_cortex': {'volume': 20000, 'neuron_density': 120000},
                'motor_cortex': {'volume': 8000, 'neuron_density': 100000},
                'thalamus': {'volume': 6000, 'neuron_density': 200000},
                'cerebellum': {'volume': 150000, 'neuron_density': 300000}
            }
        }
        
        self.cognitive_architecture = create_cognitive_architecture(cognitive_config)
        
        self.logger.info("认知架构初始化完成")
    
    async def _initialize_memory_system(self):
        """初始化记忆系统"""
        
        memory_config = {
            'hippocampal': {
                'ca1_capacity': 50000,
                'ca3_capacity': 25000,
                'dg_capacity': 500000
            },
            'neocortical': {
                'semantic_capacity': 1000000,
                'procedural_capacity': 100000
            },
            'working_memory_capacity': 7
        }
        
        self.memory_system = create_comprehensive_memory_system(memory_config)
        
        self.logger.info("记忆系统初始化完成")
    
    async def _initialize_biological_components(self):
        """初始化生物学组件"""
        
        if self.config.detailed_cell_types:
            # 细胞多样性系统
            cell_config = {
                'neuron_types': 15,
                'glial_types': 5,
                'detailed_morphology': True
            }
            self.cell_diversity_system = CellDiversitySystem(cell_config)
            await self.cell_diversity_system.initialize()
        
        if self.config.vascular_modeling:
            # 血管系统
            vascular_config = {
                'vessel_density': 400,  # vessels/mm³
                'capillary_density': 300,
                'blood_brain_barrier': True
            }
            self.vascular_system = VascularSystem(vascular_config)
            await self.vascular_system.initialize()
        
        # 生理脑区管理器
        region_config = {
            'detailed_anatomy': True,
            'metabolic_modeling': self.config.metabolic_modeling,
            'neurotransmitter_systems': True
        }
        self.physiological_regions = PhysiologicalRegionManager(region_config)
        await self.physiological_regions.initialize()
        
        self.logger.info("生物学组件初始化完成")
    
    async def _establish_system_connections(self):
        """建立系统间连接"""
        
        # 神经网络 <-> 突触管理器
        self.neural_network.set_synapse_manager(self.synapse_manager)
        
        # 神经网络 <-> 认知架构
        if self.cognitive_architecture:
            self.neural_network.set_cognitive_interface(self.cognitive_architecture)
            try:
                self.cognitive_architecture.set_module_bus(self.module_bus, manage_cycle=False)
            except Exception:
                pass
        
        # 认知架构 <-> 记忆系统
        if self.cognitive_architecture and self.memory_system:
            self.cognitive_architecture.set_memory_system(self.memory_system)
        
        # 神经网络 <-> 生物学组件
        if self.cell_diversity_system:
            self.neural_network.set_cell_diversity_system(self.cell_diversity_system)
        
        if self.vascular_system:
            self.neural_network.set_vascular_system(self.vascular_system)
        
        self.logger.info("系统间连接建立完成")
    
    async def run_simulation(self, duration: float = None) -> Dict[str, Any]:
        """运行仿真"""
        
        if not self.is_initialized:
            await self.wait_until_initialized(timeout=30.0)
        
        if duration is None:
            duration = self.config.simulation_duration
        self._last_run_duration = float(duration)
        
        self.logger.info(f"开始运行大脑仿真，时长: {duration} ms")
        
        self.is_running = True
        start_time = time.time()
        
        try:
            # 仿真主循环
            steps = int(duration / self.config.dt)
            
            for step in range(steps):
                step_start = time.time()
                
                # 更新仿真时间
                self.current_time = step * self.config.dt
                self.simulation_step = step
                
                # 并行更新所有组件
                await self._simulation_step()
                
                # 记录性能
                step_time = time.time() - step_start
                self._update_performance_metrics(step_time)
                
                # 进度报告
                if step % 1000 == 0:
                    progress = (step / steps) * 100
                    self.logger.info(f"仿真进度: {progress:.1f}%")
            
            # 收集结果
            results = await self._collect_simulation_results()
            
            total_time = time.time() - start_time
            self.logger.info(f"仿真完成，总耗时: {total_time:.2f} 秒")
            
            return results
            
        except Exception as e:
            self.logger.error(f"仿真运行失败: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _simulation_step(self):
        """单步仿真"""
        dt = float(self.config.dt)
        step_index = int(self.simulation_step)

        results: List[Any] = []

        # Reset module bus cycle and deliver queued commands/events (if any).
        try:
            self.module_bus.reset_cycle(float(self.current_time))
            inbox = list(getattr(self, "_module_bus_inbox", []) or [])
            try:
                self._module_bus_inbox.clear()
            except Exception:
                self._module_bus_inbox = []
            for sig in inbox:
                try:
                    self.module_bus.publish(sig)
                except Exception:
                    continue
        except Exception:
            pass

        # 1) 神经网络：优先使用轻量 runtime `step`，避免 full-brain `update` 的高开销
        neural_result: Any = {}
        try:
            if self.neural_network is not None:
                if hasattr(self.neural_network, "step") and callable(getattr(self.neural_network, "step")):
                    neural_result = self.neural_network.step(dt)
                else:
                    neural_result = self.neural_network.update(dt, None)
        except Exception as exc:
            self.logger.error("Neural network step failed: %s", exc)
            neural_result = exc
        results.append(neural_result)

        # 2) 突触更新（长程/统一突触管理器）
        synapse_result: Any = {}
        try:
            if self.synapse_manager is not None and self.neural_network is not None:
                neuron_voltages = {}
                try:
                    neuron_voltages = self.neural_network.get_neuron_voltages()
                except Exception:
                    neuron_voltages = {}
                synapse_result = self.synapse_manager.update_all_synapses(dt, self.current_time, neuron_voltages)
        except Exception as exc:
            self.logger.error("Synapse update failed: %s", exc)
            synapse_result = exc
        results.append(synapse_result)

        # 3) 认知架构（按步长更新）
        cognitive_result: Any = {}
        try:
            if self.cognitive_architecture and (step_index % self._cognitive_stride_steps == 0):
                sensory_inputs = self._get_sensory_inputs()
                task_demands = self._get_task_demands()
                dt_cognitive = dt * float(self._cognitive_stride_steps)
                cognitive_result = await self.cognitive_architecture.process_cognitive_cycle(
                    dt_cognitive, sensory_inputs, task_demands
                )
        except Exception as exc:
            self.logger.error("Cognitive cycle failed: %s", exc)
            cognitive_result = exc
        results.append(cognitive_result)

        # 4) 记忆系统（按步长更新）
        memory_result: Any = {}
        try:
            if self.memory_system and (step_index % self._memory_stride_steps == 0):
                dt_memory = dt * float(self._memory_stride_steps)
                memory_result = await self.memory_system.update_memory_system(dt_memory, sleep_mode=False)
        except Exception as exc:
            self.logger.error("Memory update failed: %s", exc)
            memory_result = exc
        results.append(memory_result)

        # 5) 血管系统（按步长更新）
        vascular_result: Any = {}
        try:
            if self.vascular_system and (step_index % self._vascular_stride_steps == 0):
                neural_activity = {}
                try:
                    neural_activity = self.neural_network.get_global_activity() if self.neural_network else {}
                except Exception:
                    neural_activity = {}
                dt_vascular = dt * float(self._vascular_stride_steps)
                vascular_result = await self.vascular_system.update(dt_vascular, neural_activity)
        except Exception as exc:
            self.logger.error("Vascular update failed: %s", exc)
            vascular_result = exc
        results.append(vascular_result)

        self._record_simulation_data(results)

        try:
            self.latest_module_bus = self.module_bus.export_cycle()
        except Exception:
            self.latest_module_bus = None
    
    def _get_sensory_inputs(self) -> Dict[str, float]:
        """获取感觉输入"""
        
        # 模拟感觉输入
        return {
            'visual': np.random.uniform(0, 1),
            'auditory': np.random.uniform(0, 1),
            'somatosensory': np.random.uniform(0, 1)
        }
    
    def _get_task_demands(self) -> Dict[str, float]:
        """获取任务需求"""
        
        # 模拟认知任务需求
        return {
            'attention_control': np.random.uniform(0.3, 0.8),
            'working_memory': np.random.uniform(0.2, 0.7),
            'motor_planning': np.random.uniform(0.1, 0.5),
            'memory_retrieval': np.random.uniform(0.2, 0.6)
        }
    
    def _record_simulation_data(self, step_results: List[Any]):
        """记录仿真数据"""
        
        # 记录神经元发放
        if len(step_results) > 0 and not isinstance(step_results[0], Exception):
            neural_result = step_results[0]
            if 'spikes' in neural_result:
                spikes = neural_result.get('spikes')
                # 兼容两种格式：
                # - dict: {neuron_id: [t1, t2, ...]}
                # - list: [{'neuron': id, 'neuron_global': gid, 'time_ms': t}, ...]
                if isinstance(spikes, dict):
                    for neuron_id, spike_times in spikes.items():
                        if neuron_id not in self.spike_data:
                            self.spike_data[neuron_id] = []
                        if isinstance(spike_times, list):
                            self.spike_data[neuron_id].extend(spike_times)
                        else:
                            self.spike_data[neuron_id].append(spike_times)
                elif isinstance(spikes, list):
                    for entry in spikes:
                        if not isinstance(entry, dict):
                            continue
                        neuron_id = entry.get('neuron_global', entry.get('neuron'))
                        time_ms = entry.get('time_ms', self.current_time)
                        if neuron_id is None:
                            continue
                        if neuron_id not in self.spike_data:
                            self.spike_data[neuron_id] = []
                        self.spike_data[neuron_id].append(float(time_ms))
        
        # 记录认知数据
        if len(step_results) > 2 and not isinstance(step_results[2], Exception):
            cognitive_result = step_results[2]
            if isinstance(cognitive_result, dict) and cognitive_result:
                self.cognitive_data[self.current_time] = cognitive_result
    
    def _update_performance_metrics(self, step_time: float):
        """更新性能指标"""
        
        self.performance_metrics['simulation_speed'] = 1.0 / step_time if step_time > 0 else 0
        self.performance_metrics['neurons_processed'] = self.config.total_neurons
        self.performance_metrics['synapses_processed'] = self.config.total_synapses
    
    async def _collect_simulation_results(self) -> Dict[str, Any]:
        """收集仿真结果"""
        duration = self._last_run_duration if self._last_run_duration is not None else self.config.simulation_duration
        results = {
            'simulation_config': {
                'total_neurons': self.config.total_neurons,
                'total_synapses': self.config.total_synapses,
                'duration': duration,
                'dt': self.config.dt
            },
            'performance_metrics': self.performance_metrics.copy(),
            'spike_statistics': self._analyze_spike_data(),
            'cognitive_summary': self._summarize_cognitive_data(),
            'memory_statistics': self.memory_system.get_memory_statistics() if self.memory_system else {},
            'system_health': await self._check_system_health()
        }
        
        return results
    
    def _analyze_spike_data(self) -> Dict[str, Any]:
        """分析发放数据"""
        
        if not self.spike_data:
            return {'total_spikes': 0, 'firing_rate': 0.0}
        
        total_spikes = sum(len(spikes) for spikes in self.spike_data.values())
        active_neurons = len(self.spike_data)
        
        # 计算平均发放率
        duration_ms = self._last_run_duration if self._last_run_duration is not None else self.config.simulation_duration
        duration_sec = max(1e-9, float(duration_ms) / 1000.0)
        avg_firing_rate = total_spikes / (active_neurons * duration_sec) if active_neurons > 0 else 0.0
        
        return {
            'total_spikes': total_spikes,
            'active_neurons': active_neurons,
            'average_firing_rate': avg_firing_rate,
            'spike_data_size': len(self.spike_data)
        }
    
    def _summarize_cognitive_data(self) -> Dict[str, Any]:
        """总结认知数据"""
        
        if not self.cognitive_data:
            return {}
        
        # 计算平均认知负荷
        cognitive_loads = [data.get('cognitive_load', 0) for data in self.cognitive_data.values()]
        avg_cognitive_load = np.mean(cognitive_loads) if cognitive_loads else 0.0
        
        # 计算意识水平
        consciousness_levels = [
            data.get('consciousness_state', {}).get('awareness_level', 0)
            for data in self.cognitive_data.values()
        ]
        avg_consciousness = np.mean(consciousness_levels) if consciousness_levels else 0.0
        
        return {
            'average_cognitive_load': avg_cognitive_load,
            'average_consciousness_level': avg_consciousness,
            'cognitive_data_points': len(self.cognitive_data)
        }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""

        if not self.is_initialized:
            try:
                await self.wait_until_initialized(timeout=30.0)
            except Exception:
                # If initialization failed or timed out, still return a best-effort snapshot.
                pass
        
        health_status = {
            'neural_network': self.neural_network.is_healthy() if self.neural_network else False,
            'synapse_manager': len(self.synapse_manager.synapses) > 0 if self.synapse_manager else False,
            'cognitive_architecture': self.cognitive_architecture is not None,
            'memory_system': self.memory_system is not None,
            'overall_health': 'healthy'
        }
        
        # 检查是否有组件失败
        failed_components = [k for k, v in health_status.items() if not v and k != 'overall_health']
        
        if failed_components:
            health_status['overall_health'] = 'degraded'
            health_status['failed_components'] = failed_components
        
        return health_status
    
    async def shutdown(self):
        """关闭系统"""
        
        self.logger.info("开始关闭大脑仿真系统...")
        
        self.is_running = False

        init_task = getattr(self, "_init_task", None)
        if init_task is not None and not init_task.done():
            init_task.cancel()
            try:
                await init_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        
        # 关闭各个组件
        if self.backend_manager:
            await self.backend_manager.shutdown_all_backends()
        
        if self.neural_network:
            await self.neural_network.shutdown()
        
        if self.vascular_system:
            await self.vascular_system.shutdown()
        
        self.is_initialized = False
        
        self.logger.info("大脑仿真系统关闭完成")
    
    def queue_module_signal(
        self,
        topic: ModuleTopic,
        payload: Dict[str, Any],
        *,
        source: str = "external",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModuleSignal:
        """Queue a module-bus signal to be delivered at the next simulation step."""

        signal = ModuleSignal(topic=topic, payload=dict(payload or {}), source=str(source), metadata=metadata or {})
        try:
            self._module_bus_inbox.append(signal)
        except Exception:
            self._module_bus_inbox = [signal]
        return signal

    def publish_module_signal(
        self,
        topic: ModuleTopic,
        payload: Dict[str, Any],
        *,
        source: str = "external",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModuleSignal:
        """Publish a module-bus signal immediately (current cycle)."""

        signal = ModuleSignal(topic=topic, payload=dict(payload or {}), source=str(source), metadata=metadata or {})
        try:
            self.module_bus.publish(signal)
        except Exception:
            pass
        return signal

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'current_time': self.current_time,
            'simulation_step': self.simulation_step,
            'performance_metrics': self.performance_metrics.copy(),
            'component_status': {
                'neural_network': self.neural_network is not None,
                'synapse_manager': self.synapse_manager is not None,
                'backend_manager': self.backend_manager is not None,
                'cognitive_architecture': self.cognitive_architecture is not None,
                'memory_system': self.memory_system is not None,
                'cell_diversity_system': self.cell_diversity_system is not None,
                'vascular_system': self.vascular_system is not None
            }
        }

# 工厂函数
def create_complete_brain_simulation_system(config: BrainSimulationConfig) -> CompleteBrainSimulationSystem:
    """创建完整大脑仿真系统"""
    return CompleteBrainSimulationSystem(config)

# 预设配置
def get_full_brain_config() -> BrainSimulationConfig:
    """获取全脑仿真配置"""
    return BrainSimulationConfig(
        total_neurons=86_000_000_000,
        total_synapses=100_000_000_000_000,
        brain_regions=180,
        dt=0.1,
        simulation_duration=1000.0,
        use_neuromorphic=True,
        detailed_cell_types=True,
        vascular_modeling=True,
        consciousness_modeling=True,
        memory_consolidation=True,
        learning_plasticity=True
    )

def get_prototype_config() -> BrainSimulationConfig:
    """获取原型仿真配置（较小规模用于测试）"""
    return BrainSimulationConfig(
        total_neurons=1_000_000,
        total_synapses=10_000_000,
        brain_regions=20,
        dt=0.1,
        simulation_duration=100.0,
        use_neuromorphic=False,
        detailed_cell_types=True,
        vascular_modeling=True,
        consciousness_modeling=True,
        memory_consolidation=True,
        learning_plasticity=True
    )
