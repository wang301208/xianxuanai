"""
增强的大脑仿真系统

整合所有增强组件：
- 详细的脑区配置（基于SCOPE_CONFIG）
- 扩展的网络与突触层
- 深化的认知与功能模块
- 后端与神经形态适配
- 计算与工程支持基础设施

实现完整的皮层柱+丘脑环路系统
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path

# 导入增强的组件
from .config.enhanced_brain_config import (
    BRAIN_REGIONS_CONFIG,
    CORTICAL_LAYERS_CONFIG,
    THALAMIC_NUCLEI_CONFIG,
    NEURON_TYPES_CONFIG,
    create_brain_region_config
)

from .core.enhanced_network import (
    EnhancedNetworkBuilder,
    RegionalNetworkManager,
    MultiTypeNeuronFactory,
    NetworkTopology
)

from .core.enhanced_synapse import NeuromodulatedSynapse
from .core.enhanced_synapse_manager import SynapticPlasticityManager
from .core.glia_system import GlialModulatedSynapse

from .models.enhanced_cognitive_system import (
    EnhancedCognitiveSystem,
    HippocampalCircuit,
    BasalGangliaCircuit,
    PrefrontalCortexCircuit
)

from .backends.enhanced_backend_system import (
    EnhancedBackendManager,
    BackendType,
    ExecutionMode,
    SimulationTask
)

from .infrastructure import (
    SimulationPipeline,
    TestExecutor,
    RecordingManager,
    DataVisualizer
)

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """系统模式"""
    DEVELOPMENT = "development"
    RESEARCH = "research"
    PRODUCTION = "production"
    BENCHMARK = "benchmark"

@dataclass
class SystemConfig:
    """系统配置"""
    mode: SystemMode = SystemMode.RESEARCH
    
    # 脑区配置
    enabled_regions: List[str] = field(default_factory=lambda: [
        'neocortex', 'hippocampus', 'thalamus', 'basal_ganglia'
    ])
    
    # 网络配置
    network_scale: str = "medium"  # "small", "medium", "large", "full"
    enable_plasticity: bool = True
    enable_neuromodulation: bool = True
    
    # 后端配置
    preferred_backend: BackendType = BackendType.NEST
    enable_gpu_acceleration: bool = True
    
    # 记录配置
    enable_recording: bool = True
    recording_types: List[str] = field(default_factory=lambda: [
        'spike_times', 'membrane_voltage', 'synaptic_weights'
    ])
    
    # 可视化配置
    enable_real_time_visualization: bool = False
    visualization_update_interval: float = 100.0  # ms

class EnhancedBrainSystem:
    """增强的大脑仿真系统"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("EnhancedBrainSystem")
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        self.simulation_time = 0.0
        
        # 核心组件
        self.network_manager = None
        self.cognitive_system = None
        self.backend_manager = None
        self.recording_manager = None
        self.visualizer = None
        
        # 脑区网络
        self.brain_regions = {}
        self.cortical_columns = {}
        self.thalamic_nuclei = {}
        
        # 连接映射
        self.inter_regional_connections = {}
        self.cortical_thalamic_loops = {}
        
        # 性能监控
        self.performance_metrics = {
            'initialization_time': 0.0,
            'simulation_speed': 0.0,
            'memory_usage': 0.0,
            'cpu_utilization': 0.0
        }
    
    async def initialize(self) -> bool:
        """初始化系统"""
        start_time = time.time()
        
        try:
            self.logger.info("开始初始化增强大脑仿真系统...")
            
            # 1. 初始化网络管理器
            await self._initialize_network_manager()
            
            # 2. 初始化认知系统
            await self._initialize_cognitive_system()
            
            # 3. 初始化后端管理器
            await self._initialize_backend_manager()
            
            # 4. 初始化记录管理器
            await self._initialize_recording_manager()
            
            # 5. 初始化可视化器
            await self._initialize_visualizer()
            
            # 6. 构建脑区网络
            await self._build_brain_regions()
            
            # 7. 建立区域间连接
            await self._establish_inter_regional_connections()
            
            # 8. 配置皮层-丘脑环路
            await self._configure_cortical_thalamic_loops()
            
            self.is_initialized = True
            initialization_time = time.time() - start_time
            self.performance_metrics['initialization_time'] = initialization_time
            
            self.logger.info(f"系统初始化完成，耗时: {initialization_time:.2f} 秒")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {str(e)}")
            return False
    
    async def _initialize_network_manager(self):
        """初始化网络管理器"""
        self.network_manager = RegionalNetworkManager()
        
        # 配置网络规模
        scale_configs = {
            "small": {"max_neurons_per_region": 1000, "connection_density": 0.1},
            "medium": {"max_neurons_per_region": 10000, "connection_density": 0.05},
            "large": {"max_neurons_per_region": 100000, "connection_density": 0.01},
            "full": {"max_neurons_per_region": 1000000, "connection_density": 0.001}
        }
        
        scale_config = scale_configs.get(self.config.network_scale, scale_configs["medium"])
        self.network_manager.configure_scale(scale_config)
        
        self.logger.info(f"网络管理器初始化完成，规模: {self.config.network_scale}")
    
    async def _initialize_cognitive_system(self):
        """初始化认知系统"""
        self.cognitive_system = EnhancedCognitiveSystem()
        
        # 配置认知模块
        cognitive_config = {
            'enable_hippocampal_memory': True,
            'enable_basal_ganglia_action': True,
            'enable_prefrontal_executive': True,
            'enable_cross_modal_integration': True
        }
        
        await self.cognitive_system.initialize(cognitive_config)
        
        self.logger.info("认知系统初始化完成")
    
    async def _initialize_backend_manager(self):
        """初始化后端管理器"""
        from .backends.enhanced_backend_system import create_enhanced_backend_manager
        
        self.backend_manager = create_enhanced_backend_manager()
        
        # 注册首选后端
        backend_configs = {
            BackendType.NEST: {
                "resolution": 0.1,
                "num_threads": 8,
                "use_mpi": False
            },
            BackendType.CARLSIM: {
                "gpu_device": 0,
                "precision": "float32"
            }
        }
        
        if self.config.preferred_backend in backend_configs:
            config = backend_configs[self.config.preferred_backend]
            success = self.backend_manager.register_backend(self.config.preferred_backend, config)
            
            if success:
                self.logger.info(f"后端 {self.config.preferred_backend.value} 注册成功")
            else:
                self.logger.warning(f"后端 {self.config.preferred_backend.value} 注册失败")
    
    async def _initialize_recording_manager(self):
        """初始化记录管理器"""
        if not self.config.enable_recording:
            return
        
        from .infrastructure.recording_visualization import (
            RecordingManager, RecordingConfig, RecordingType, StorageFormat
        )
        
        self.recording_manager = RecordingManager()
        
        # 创建记录配置
        recording_configs = []
        
        for recording_type_str in self.config.recording_types:
            if recording_type_str == 'spike_times':
                recording_type = RecordingType.SPIKE_TIMES
            elif recording_type_str == 'membrane_voltage':
                recording_type = RecordingType.MEMBRANE_VOLTAGE
            elif recording_type_str == 'synaptic_weights':
                recording_type = RecordingType.SYNAPTIC_WEIGHTS
            else:
                continue
            
            config = RecordingConfig(
                recording_id=f"main_{recording_type_str}",
                recording_type=recording_type,
                target_populations=self.config.enabled_regions,
                sampling_rate=1000.0,  # Hz
                storage_format=StorageFormat.HDF5,
                compression=True
            )
            recording_configs.append(config)
        
        # 创建记录会话
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.recording_session = self.recording_manager.create_recording_session(
            session_id, recording_configs
        )
        
        self.logger.info(f"记录管理器初始化完成，会话ID: {session_id}")
    
    async def _initialize_visualizer(self):
        """初始化可视化器"""
        from .infrastructure.recording_visualization import DataVisualizer, RealTimeVisualizer
        
        self.visualizer = DataVisualizer()
        
        if self.config.enable_real_time_visualization:
            self.real_time_visualizer = RealTimeVisualizer(
                update_interval=self.config.visualization_update_interval
            )
        
        self.logger.info("可视化器初始化完成")
    
    async def _build_brain_regions(self):
        """构建脑区网络"""
        for region_name in self.config.enabled_regions:
            if region_name not in BRAIN_REGIONS_CONFIG:
                self.logger.warning(f"未知脑区: {region_name}")
                continue
            
            region_config = BRAIN_REGIONS_CONFIG[region_name]
            
            if region_name == 'neocortex':
                await self._build_neocortex(region_config)
            elif region_name == 'hippocampus':
                await self._build_hippocampus(region_config)
            elif region_name == 'thalamus':
                await self._build_thalamus(region_config)
            elif region_name == 'basal_ganglia':
                await self._build_basal_ganglia(region_config)
            
            self.logger.info(f"脑区 {region_name} 构建完成")
    
    async def _build_neocortex(self, region_config: Dict[str, Any]):
        """构建新皮层"""
        # 创建皮层柱
        cortical_areas = region_config.get('subregions', {})
        
        for area_name, area_config in cortical_areas.items():
            # 为每个皮层区域创建多个皮层柱
            num_columns = area_config.get('num_columns', 10)
            
            area_columns = {}
            for col_idx in range(num_columns):
                column_id = f"{area_name}_column_{col_idx}"
                
                # 创建6层皮层柱
                column_layers = {}
                for layer_num in range(1, 7):
                    layer_config = CORTICAL_LAYERS_CONFIG[f'L{layer_num}']
                    
                    # 创建该层的神经元网络
                    layer_network = await self.network_manager.create_regional_network(
                        region_id=f"{column_id}_L{layer_num}",
                        neuron_types=layer_config['cell_types'],
                        network_size=layer_config['neuron_density'] // 10,  # 缩放
                        topology=NetworkTopology.LAYERED
                    )
                    
                    column_layers[f'L{layer_num}'] = layer_network
                
                # 建立层间连接
                await self._establish_columnar_connections(column_layers)
                
                area_columns[column_id] = column_layers
            
            self.cortical_columns[area_name] = area_columns
        
        self.brain_regions['neocortex'] = self.cortical_columns
    
    async def _build_hippocampus(self, region_config: Dict[str, Any]):
        """构建海马"""
        hippocampal_regions = {}
        
        # 构建海马各个子区域
        subregions = ['DG', 'CA3', 'CA1', 'subiculum']
        
        for subregion in subregions:
            if subregion in region_config.get('subregions', {}):
                subregion_config = region_config['subregions'][subregion]
                
                # 创建子区域网络
                network = await self.network_manager.create_regional_network(
                    region_id=f"hippocampus_{subregion}",
                    neuron_types=subregion_config.get('cell_types', ['pyramidal', 'interneuron']),
                    network_size=subregion_config.get('neuron_count', 1000),
                    topology=NetworkTopology.HIPPOCAMPAL
                )
                
                hippocampal_regions[subregion] = network
        
        # 建立海马三突触环路
        await self._establish_hippocampal_trisynaptic_pathway(hippocampal_regions)
        
        self.brain_regions['hippocampus'] = hippocampal_regions
    
    async def _build_thalamus(self, region_config: Dict[str, Any]):
        """构建丘脑"""
        thalamic_nuclei = {}
        
        # 构建各个丘脑核团
        for nucleus_name, nucleus_config in THALAMIC_NUCLEI_CONFIG.items():
            # 创建核团网络
            network = await self.network_manager.create_regional_network(
                region_id=f"thalamus_{nucleus_name}",
                neuron_types=nucleus_config.get('cell_types', ['relay', 'interneuron']),
                network_size=nucleus_config.get('neuron_count', 500),
                topology=NetworkTopology.THALAMIC
            )
            
            thalamic_nuclei[nucleus_name] = network
        
        self.thalamic_nuclei = thalamic_nuclei
        self.brain_regions['thalamus'] = thalamic_nuclei
    
    async def _build_basal_ganglia(self, region_config: Dict[str, Any]):
        """构建基底神经节"""
        basal_ganglia_regions = {}
        
        # 构建基底神经节各个核团
        nuclei = ['striatum', 'gpe', 'gpi', 'stn', 'snc', 'snr']
        
        for nucleus in nuclei:
            if nucleus in region_config.get('subregions', {}):
                nucleus_config = region_config['subregions'][nucleus]
                
                network = await self.network_manager.create_regional_network(
                    region_id=f"basal_ganglia_{nucleus}",
                    neuron_types=nucleus_config.get('cell_types', ['projection']),
                    network_size=nucleus_config.get('neuron_count', 800),
                    topology=NetworkTopology.BASAL_GANGLIA
                )
                
                basal_ganglia_regions[nucleus] = network
        
        # 建立基底神经节环路
        await self._establish_basal_ganglia_loops(basal_ganglia_regions)
        
        self.brain_regions['basal_ganglia'] = basal_ganglia_regions
    
    async def _establish_columnar_connections(self, column_layers: Dict[str, Any]):
        """建立皮层柱内连接"""
        # 典型的皮层层间连接模式
        layer_connections = [
            ('L4', 'L2/3', 0.3),  # 主要上行通路
            ('L4', 'L5', 0.2),
            ('L2/3', 'L5', 0.1),
            ('L5', 'L6', 0.2),
            ('L6', 'L4', 0.1),    # 反馈连接
            ('L2/3', 'L2/3', 0.2), # 水平连接
            ('L5', 'L5', 0.15)
        ]
        
        for source_layer, target_layer, connection_prob in layer_connections:
            if source_layer in column_layers and target_layer in column_layers:
                await self.network_manager.connect_regions(
                    source_region=column_layers[source_layer],
                    target_region=column_layers[target_layer],
                    connection_probability=connection_prob,
                    synapse_type='excitatory' if connection_prob > 0.15 else 'inhibitory'
                )
    
    async def _establish_hippocampal_trisynaptic_pathway(self, hippocampal_regions: Dict[str, Any]):
        """建立海马三突触通路"""
        # DG -> CA3 -> CA1 经典通路
        pathway_connections = [
            ('DG', 'CA3', 0.1),      # 苔藓纤维
            ('CA3', 'CA1', 0.2),     # Schaffer侧支
            ('CA3', 'CA3', 0.05),    # CA3递归连接
        ]
        
        for source, target, prob in pathway_connections:
            if source in hippocampal_regions and target in hippocampal_regions:
                await self.network_manager.connect_regions(
                    source_region=hippocampal_regions[source],
                    target_region=hippocampal_regions[target],
                    connection_probability=prob,
                    synapse_type='excitatory'
                )
    
    async def _establish_basal_ganglia_loops(self, basal_ganglia_regions: Dict[str, Any]):
        """建立基底神经节环路"""
        # 直接通路和间接通路
        bg_connections = [
            ('striatum', 'gpi', 0.3),    # 直接通路
            ('striatum', 'gpe', 0.3),    # 间接通路
            ('gpe', 'gpi', 0.4),
            ('gpe', 'stn', 0.2),
            ('stn', 'gpi', 0.5),
            ('snc', 'striatum', 0.1),    # 多巴胺调节
        ]
        
        for source, target, prob in bg_connections:
            if source in basal_ganglia_regions and target in basal_ganglia_regions:
                synapse_type = 'inhibitory' if source in ['striatum', 'gpe'] else 'excitatory'
                
                await self.network_manager.connect_regions(
                    source_region=basal_ganglia_regions[source],
                    target_region=basal_ganglia_regions[target],
                    connection_probability=prob,
                    synapse_type=synapse_type
                )
    
    async def _establish_inter_regional_connections(self):
        """建立区域间连接"""
        # 皮层-海马连接
        if 'neocortex' in self.brain_regions and 'hippocampus' in self.brain_regions:
            await self._connect_cortex_hippocampus()
        
        # 皮层-基底神经节连接
        if 'neocortex' in self.brain_regions and 'basal_ganglia' in self.brain_regions:
            await self._connect_cortex_basal_ganglia()
        
        # 其他区域间连接...
        
        self.logger.info("区域间连接建立完成")
    
    async def _connect_cortex_hippocampus(self):
        """连接皮层和海马"""
        # 内嗅皮层作为皮层-海马接口
        cortical_areas = self.brain_regions['neocortex']
        hippocampal_regions = self.brain_regions['hippocampus']
        
        # 简化的连接：皮层 -> DG, CA1 -> 皮层
        for area_name, area_columns in cortical_areas.items():
            for column_id, column_layers in area_columns.items():
                if 'L2/3' in column_layers and 'DG' in hippocampal_regions:
                    await self.network_manager.connect_regions(
                        source_region=column_layers['L2/3'],
                        target_region=hippocampal_regions['DG'],
                        connection_probability=0.01,
                        synapse_type='excitatory'
                    )
                
                if 'L5' in column_layers and 'CA1' in hippocampal_regions:
                    await self.network_manager.connect_regions(
                        source_region=hippocampal_regions['CA1'],
                        target_region=column_layers['L5'],
                        connection_probability=0.01,
                        synapse_type='excitatory'
                    )
    
    async def _connect_cortex_basal_ganglia(self):
        """连接皮层和基底神经节"""
        cortical_areas = self.brain_regions['neocortex']
        basal_ganglia_regions = self.brain_regions['basal_ganglia']
        
        # 皮层 -> 纹状体投射
        for area_name, area_columns in cortical_areas.items():
            for column_id, column_layers in area_columns.items():
                if 'L5' in column_layers and 'striatum' in basal_ganglia_regions:
                    await self.network_manager.connect_regions(
                        source_region=column_layers['L5'],
                        target_region=basal_ganglia_regions['striatum'],
                        connection_probability=0.05,
                        synapse_type='excitatory'
                    )
    
    async def _configure_cortical_thalamic_loops(self):
        """配置皮层-丘脑环路"""
        if 'neocortex' not in self.brain_regions or 'thalamus' not in self.brain_regions:
            return
        
        cortical_areas = self.brain_regions['neocortex']
        thalamic_nuclei = self.brain_regions['thalamus']
        
        # 建立皮层-丘脑-皮层环路
        cortical_thalamic_mappings = {
            'primary_visual': 'LGN',
            'primary_auditory': 'MGN',
            'primary_somatosensory': 'VPL',
            'prefrontal': 'MD'
        }
        
        for cortical_area, thalamic_nucleus in cortical_thalamic_mappings.items():
            if cortical_area in cortical_areas and thalamic_nucleus in thalamic_nuclei:
                
                # 为该皮层区域的所有皮层柱建立丘脑连接
                area_columns = cortical_areas[cortical_area]
                nucleus_network = thalamic_nuclei[thalamic_nucleus]
                
                for column_id, column_layers in area_columns.items():
                    # 丘脑 -> 皮层L4 (上行)
                    if 'L4' in column_layers:
                        await self.network_manager.connect_regions(
                            source_region=nucleus_network,
                            target_region=column_layers['L4'],
                            connection_probability=0.2,
                            synapse_type='excitatory'
                        )
                    
                    # 皮层L6 -> 丘脑 (下行反馈)
                    if 'L6' in column_layers:
                        await self.network_manager.connect_regions(
                            source_region=column_layers['L6'],
                            target_region=nucleus_network,
                            connection_probability=0.1,
                            synapse_type='excitatory'
                        )
                
                # 记录环路配置
                loop_id = f"{cortical_area}_{thalamic_nucleus}_loop"
                self.cortical_thalamic_loops[loop_id] = {
                    'cortical_area': cortical_area,
                    'thalamic_nucleus': thalamic_nucleus,
                    'columns': list(area_columns.keys())
                }
        
        self.logger.info(f"配置了 {len(self.cortical_thalamic_loops)} 个皮层-丘脑环路")
    
    async def run_simulation(self, duration: float, dt: float = 0.1) -> Dict[str, Any]:
        """运行仿真"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化")
        
        self.logger.info(f"开始仿真，时长: {duration} ms")
        
        # 开始记录
        if self.recording_manager:
            self.recording_manager.start_recording(self.recording_session.session_id)
        
        # 开始实时可视化
        if hasattr(self, 'real_time_visualizer'):
            from .infrastructure.recording_visualization import VisualizationType
            self.real_time_visualizer.start_real_time_visualization([
                VisualizationType.RASTER_PLOT,
                VisualizationType.FIRING_RATE
            ])
        
        self.is_running = True
        start_time = time.time()
        
        try:
            # 创建仿真任务
            network_config = self._generate_network_config()
            simulation_params = {
                'simulation_time': duration,
                'dt': dt,
                'stimuli': self._generate_stimuli()
            }
            
            task = SimulationTask(
                task_id=f"main_simulation_{int(time.time())}",
                network_config=network_config,
                simulation_params=simulation_params,
                backend_type=self.config.preferred_backend,
                execution_mode=ExecutionMode.PARALLEL
            )
            
            # 执行仿真
            result = self.backend_manager.execute_task(task)
            
            # 处理结果
            simulation_results = result.results if result.success else {}
            
            # 更新仿真时间
            self.simulation_time += duration
            
            # 计算性能指标
            execution_time = time.time() - start_time
            self.performance_metrics['simulation_speed'] = duration / (execution_time * 1000)
            
            self.logger.info(f"仿真完成，实际耗时: {execution_time:.2f} 秒")
            
            return {
                'success': result.success,
                'simulation_time': self.simulation_time,
                'execution_time': execution_time,
                'results': simulation_results,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"仿真执行失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'simulation_time': self.simulation_time
            }
        
        finally:
            self.is_running = False
            
            # 停止记录
            if self.recording_manager:
                self.recording_manager.stop_recording(self.recording_session.session_id)
            
            # 停止实时可视化
            if hasattr(self, 'real_time_visualizer'):
                self.real_time_visualizer.stop_real_time_visualization()
    
    def _generate_network_config(self) -> Dict[str, Any]:
        """生成网络配置"""
        # 统计网络信息
        total_neurons = 0
        total_connections = 0
        
        populations = {}
        connections = []
        
        # 收集所有脑区的网络信息
        for region_name, region_networks in self.brain_regions.items():
            if region_name == 'neocortex':
                # 处理皮层柱结构
                for area_name, area_columns in region_networks.items():
                    for column_id, column_layers in area_columns.items():
                        for layer_name, layer_network in column_layers.items():
                            pop_name = f"{area_name}_{column_id}_{layer_name}"
                            populations[pop_name] = {
                                'size': 100,  # 简化
                                'neuron_model': 'iaf_psc_alpha'
                            }
                            total_neurons += 100
            else:
                # 处理其他脑区
                for subregion_name, network in region_networks.items():
                    pop_name = f"{region_name}_{subregion_name}"
                    populations[pop_name] = {
                        'size': 100,  # 简化
                        'neuron_model': 'iaf_psc_alpha'
                    }
                    total_neurons += 100
        
        return {
            'populations': populations,
            'connections': connections,
            'total_neurons': total_neurons,
            'total_connections': total_connections
        }
    
    def _generate_stimuli(self) -> List[Dict[str, Any]]:
        """生成刺激"""
        stimuli = []
        
        # 为感觉皮层添加输入刺激
        sensory_areas = ['primary_visual', 'primary_auditory', 'primary_somatosensory']
        
        for area in sensory_areas:
            if area in self.cortical_columns:
                stimulus = {
                    'type': 'poisson_generator',
                    'parameters': {'rate': 20.0},
                    'target': f"{area}_column_0_L4",
                    'start_time': 100.0,
                    'duration': 500.0
                }
                stimuli.append(stimulus)
        
        return stimuli
    
    async def generate_visualization_report(self, output_directory: str = "./brain_simulation_report"):
        """生成可视化报告"""
        if not self.visualizer:
            return
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("生成可视化报告...")
        
        # 模拟数据用于可视化
        spike_trains = self._generate_mock_spike_data()
        
        # 1. 尖峰栅格图
        raster_fig = self.visualizer.create_raster_plot(
            spike_trains,
            title="大脑仿真系统 - 尖峰栅格图",
            save_path=str(output_path / "raster_plot.html")
        )
        
        # 2. 发放频率图
        firing_rate_fig = self.visualizer.create_firing_rate_plot(
            spike_trains,
            title="大脑仿真系统 - 发放频率",
            save_path=str(output_path / "firing_rate.html")
        )
        
        # 3. 网络连接图
        connectivity_matrix = self._generate_mock_connectivity_matrix()
        network_fig = self.visualizer.create_network_graph(
            connectivity_matrix,
            title="大脑仿真系统 - 网络拓扑",
            save_path=str(output_path / "network_topology.html")
        )
        
        # 4. 3D脑区可视化
        brain_regions_3d = self._generate_brain_regions_3d()
        brain_3d_fig = self.visualizer.create_3d_brain_visualization(
            brain_regions_3d,
            title="大脑仿真系统 - 3D脑区可视化",
            save_path=str(output_path / "brain_3d.html")
        )
        
        # 5. 相关性分析
        correlation_fig = self.visualizer.create_correlation_matrix(
            spike_trains,
            title="大脑仿真系统 - 神经元相关性",
            save_path=str(output_path / "correlation_analysis.html")
        )
        
        self.logger.info(f"可视化报告已生成到: {output_directory}")
    
    def _generate_mock_spike_data(self) -> Dict[int, np.ndarray]:
        """生成模拟尖峰数据"""
        spike_trains = {}
        
        for neuron_id in range(50):
            # 生成泊松尖峰序列
            rate = np.random.uniform(5, 30)  # Hz
            duration = 1000.0  # ms
            
            num_spikes = np.random.poisson(rate * duration / 1000.0)
            spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
            spike_trains[neuron_id] = spike_times
        
        return spike_trains
    
    def _generate_mock_connectivity_matrix(self) -> np.ndarray:
        """生成模拟连接矩阵"""
        size = 30
        connectivity = np.random.rand(size, size)
        connectivity = (connectivity > 0.8).astype(float)  # 稀疏连接
        np.fill_diagonal(connectivity, 0)  # 无自连接
        return connectivity
    
    def _generate_brain_regions_3d(self) -> Dict[str, Dict[str, Any]]:
        """生成3D脑区数据"""
        return {
            '新皮层': {'position': [0, 0, 2], 'size': 2.0},
            '丘脑': {'position': [0, 0, 0], 'size': 1.5},
            '海马': {'position': [-2, 0, 1], 'size': 1.0},
            '基底神经节': {'position': [2, 0, 1], 'size': 1.2}
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'simulation_time': self.simulation_time,
            'enabled_regions': self.config.enabled_regions,
            'network_scale': self.config.network_scale,
            'cortical_thalamic_loops': len(self.cortical_thalamic_loops),
            'performance_metrics': self.performance_metrics
        }
    
    async def cleanup(self):
        """清理系统资源"""
        self.logger.info("开始清理系统资源...")
        
        # 停止仿真
        self.is_running = False
        
        # 清理后端
        if self.backend_manager:
            self.backend_manager.cleanup_all_backends()
        
        # 关闭记录
        if self.recording_manager and hasattr(self, 'recording_session'):
            self.recording_manager.stop_recording(self.recording_session.session_id)
        
        # 清理网络
        if self.network_manager:
            await self.network_manager.cleanup()
        
        self.logger.info("系统资源清理完成")

async def create_enhanced_brain_system(config: Optional[SystemConfig] = None) -> EnhancedBrainSystem:
    """创建增强大脑仿真系统的便捷函数"""
    if config is None:
        config = SystemConfig()
    
    system = EnhancedBrainSystem(config)
    
    # 初始化系统
    success = await system.initialize()
    
    if not success:
        raise RuntimeError("系统初始化失败")
    
    return system

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')