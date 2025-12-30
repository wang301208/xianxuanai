"""
大规模网络管理器 - 860亿神经元级别的网络管理
Large Scale Network Manager - 86 Billion Neuron Level Network Management
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import psutil
import gc
from queue import Queue, Empty
from collections import defaultdict, deque
import pickle
import mmap
import h5py
from pathlib import Path

from .connectome_manager import ConnectomeManager, BrainRegion, NeuronPopulation
from .neuromorphic_hardware import HardwareManager, HardwareConfiguration, HardwareType, SpikeEvent

class SimulationMode(Enum):
    """仿真模式"""
    FULL_PRECISION = "full_precision"      # 全精度仿真
    REDUCED_PRECISION = "reduced_precision" # 降精度仿真
    HYBRID_HARDWARE = "hybrid_hardware"     # 混合硬件仿真
    DISTRIBUTED = "distributed"             # 分布式仿真
    REAL_TIME = "real_time"                # 实时仿真

class PartitionStrategy(Enum):
    """分区策略"""
    ANATOMICAL = "anatomical"      # 解剖分区
    FUNCTIONAL = "functional"      # 功能分区
    COMPUTATIONAL = "computational" # 计算负载分区
    HARDWARE_BASED = "hardware_based" # 硬件分区

@dataclass
class SimulationParameters:
    """仿真参数"""
    dt: float = 0.1  # 时间步长 (ms)
    simulation_time: float = 1000.0  # 仿真时间 (ms)
    
    # 数值方法
    integration_method: str = "euler"  # euler, runge_kutta, adaptive
    
    # 精度控制
    voltage_precision: int = 16  # 电压精度 (bits)
    weight_precision: int = 8    # 权重精度 (bits)
    
    # 内存管理
    max_memory_gb: float = 64.0  # 最大内存使用 (GB)
    use_memory_mapping: bool = True
    
    # 并行化
    num_threads: int = mp.cpu_count()
    use_gpu: bool = True
    
    # 输出控制
    record_spikes: bool = True
    record_voltages: bool = False
    output_interval: float = 1.0  # ms

@dataclass
class NetworkPartition:
    """网络分区"""
    partition_id: str
    populations: List[str]
    hardware_target: Optional[str] = None
    
    # 计算资源
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_memory_gb: float = 0.0
    
    # 通信接口
    input_buffers: Dict[str, Queue] = field(default_factory=dict)
    output_buffers: Dict[str, Queue] = field(default_factory=dict)
    
    # 性能监控
    computation_time: float = 0.0
    communication_time: float = 0.0
    memory_usage: float = 0.0

class LargeScaleNetworkManager:
    """大规模网络管理器"""
    
    def __init__(self, connectome: ConnectomeManager, 
                 simulation_params: SimulationParameters):
        self.connectome = connectome
        self.params = simulation_params
        self.logger = logging.getLogger("LargeScaleManager")
        
        # 硬件管理
        self.hardware_manager = HardwareManager()
        
        # 网络分区
        self.partitions: Dict[str, NetworkPartition] = {}
        self.partition_mapping: Dict[str, str] = {}  # population_id -> partition_id
        self.partition_to_hardware: Dict[str, str] = {}
        self.hardware_to_partition: Dict[str, str] = {}
        self.population_target_partitions: Dict[str, Set[str]] = {}
        self.partition_hardware_targets: Dict[str, Set[str]] = {}
        
        # 仿真状态
        self.is_initialized = False
        self.is_running = False
        self.current_time = 0.0
        
        # 数据存储
        self.spike_data: Dict[str, List[SpikeEvent]] = {}
        self.voltage_data: Dict[str, np.ndarray] = {}
        
        # 性能监控
        self.performance_metrics = {
            'total_computation_time': 0.0,
            'total_communication_time': 0.0,
            'memory_usage_peak': 0.0,
            'throughput_spikes_per_second': 0.0,
            'hardware_utilization': {}
        }
        
        # 内存映射文件
        self.memory_maps: Dict[str, mmap.mmap] = {}
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.params.num_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=min(8, mp.cpu_count()))
    
    async def initialize(self, partition_strategy: PartitionStrategy = PartitionStrategy.ANATOMICAL,
                        hardware_configs: Optional[List[HardwareConfiguration]] = None):
        """初始化大规模网络"""
        self.logger.info("初始化大规模神经网络仿真系统...")
        
        # 检查系统资源
        self._check_system_resources()
        
        # 创建网络分区
        await self._create_network_partitions(partition_strategy)
        
        # 初始化硬件
        if hardware_configs:
            await self._initialize_hardware(hardware_configs)

        # 构建路由表
        self._build_routing_tables()
        
        # 分配计算资源
        self._allocate_computational_resources()
        
        # 初始化数据结构
        self._initialize_data_structures()
        
        # 设置内存映射
        if self.params.use_memory_mapping:
            self._setup_memory_mapping()
        
        self.is_initialized = True
        self.logger.info(f"大规模网络初始化完成: {len(self.partitions)} 个分区")
    
    def _check_system_resources(self):
        """检查系统资源"""
        # 检查内存
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        required_memory = self._estimate_memory_requirements()
        
        if available_memory < required_memory:
            self.logger.warning(f"可用内存 {available_memory:.1f}GB 可能不足，估计需要 {required_memory:.1f}GB")
        
        # 检查CPU
        cpu_count = mp.cpu_count()
        self.logger.info(f"检测到 {cpu_count} 个CPU核心")
        
        # 检查GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    self.logger.info(f"GPU {i}: {gpu_memory:.1f}GB 显存")
        except ImportError:
            self.logger.info("未检测到CUDA支持")
    
    def _estimate_memory_requirements(self) -> float:
        """估计内存需求"""
        # 神经元状态: 每个神经元约32字节 (电压、电流、阈值等)
        neuron_memory = self.connectome.total_neurons * 32 / (1024**3)
        
        # 突触权重: 每个突触约4字节 (float32)
        synapse_memory = self.connectome.total_synapses * 4 / (1024**3)
        
        # 尖峰缓冲区: 估计1%神经元同时发放
        spike_buffer_memory = self.connectome.total_neurons * 0.01 * 16 / (1024**3)
        
        # 其他开销 (20%)
        total_memory = (neuron_memory + synapse_memory + spike_buffer_memory) * 1.2
        
        self.logger.info(f"估计内存需求: 神经元 {neuron_memory:.1f}GB, "
                        f"突触 {synapse_memory:.1f}GB, 总计 {total_memory:.1f}GB")
        
        return total_memory
    
    async def _create_network_partitions(self, strategy: PartitionStrategy):
        """创建网络分区"""
        self.logger.info(f"使用 {strategy.value} 策略创建网络分区")
        
        if strategy == PartitionStrategy.ANATOMICAL:
            await self._create_anatomical_partitions()
        elif strategy == PartitionStrategy.FUNCTIONAL:
            await self._create_functional_partitions()
        elif strategy == PartitionStrategy.COMPUTATIONAL:
            await self._create_computational_partitions()
        elif strategy == PartitionStrategy.HARDWARE_BASED:
            await self._create_hardware_partitions()
    
    async def _create_anatomical_partitions(self):
        """基于解剖结构创建分区"""
        # 每个主要脑区作为一个分区
        for region, population_ids in self.connectome.region_populations.items():
            partition_id = f"partition_{region.value}"
            
            # 估计该分区的计算需求
            total_neurons = sum(self.connectome.populations[pop_id].count 
                              for pop_id in population_ids)
            
            # 根据神经元数量分配资源
            memory_gb = max(1.0, total_neurons * 32 / (1024**3) * 1.5)  # 1.5倍安全系数
            cpu_cores = max(1, min(8, total_neurons // 1_000_000))  # 每百万神经元1核心
            
            partition = NetworkPartition(
                partition_id=partition_id,
                populations=population_ids,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb
            )
            
            self.partitions[partition_id] = partition
            
            # 更新映射
            for pop_id in population_ids:
                self.partition_mapping[pop_id] = partition_id
        
        self.logger.info(f"创建了 {len(self.partitions)} 个解剖分区")
    
    async def _create_functional_partitions(self):
        """基于功能创建分区"""
        # 功能分组
        functional_groups = {
            'sensory': [BrainRegion.PRIMARY_VISUAL, BrainRegion.PRIMARY_AUDITORY, 
                       BrainRegion.PRIMARY_SOMATOSENSORY],
            'motor': [BrainRegion.PRIMARY_MOTOR, BrainRegion.PREMOTOR_CORTEX],
            'cognitive': [BrainRegion.PREFRONTAL_CORTEX, BrainRegion.POSTERIOR_PARIETAL],
            'memory': [BrainRegion.HIPPOCAMPUS, BrainRegion.ENTORHINAL_CORTEX],
            'subcortical': [BrainRegion.THALAMUS, BrainRegion.BASAL_GANGLIA, BrainRegion.AMYGDALA],
            'cerebellar': [BrainRegion.CEREBELLUM]
        }
        
        for group_name, regions in functional_groups.items():
            partition_id = f"partition_{group_name}"
            population_ids = []
            
            for region in regions:
                if region in self.connectome.region_populations:
                    population_ids.extend(self.connectome.region_populations[region])
            
            if population_ids:
                total_neurons = sum(self.connectome.populations[pop_id].count 
                                  for pop_id in population_ids)
                
                memory_gb = max(2.0, total_neurons * 32 / (1024**3) * 1.5)
                cpu_cores = max(2, min(16, total_neurons // 500_000))
                
                partition = NetworkPartition(
                    partition_id=partition_id,
                    populations=population_ids,
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb
                )
                
                self.partitions[partition_id] = partition
                
                for pop_id in population_ids:
                    self.partition_mapping[pop_id] = partition_id
        
        self.logger.info(f"创建了 {len(self.partitions)} 个功能分区")
    
    async def _create_computational_partitions(self):
        """基于计算负载创建分区"""
        # 根据神经元数量和连接密度进行负载均衡分区
        target_partitions = min(32, max(4, mp.cpu_count()))  # 4-32个分区
        
        # 计算每个群体的计算负载
        population_loads = {}
        for pop_id, population in self.connectome.populations.items():
            # 负载 = 神经元数 + 输入连接数 * 权重
            input_connections = sum(1 for conn in self.connectome.connections.values()
                                  if conn.target_population == pop_id)
            load = population.count + input_connections * 0.1
            population_loads[pop_id] = load
        
        # 使用贪心算法进行负载均衡分区
        sorted_populations = sorted(population_loads.items(), key=lambda x: x[1], reverse=True)
        partition_loads = [0.0] * target_partitions
        partition_populations = [[] for _ in range(target_partitions)]
        
        for pop_id, load in sorted_populations:
            # 找到负载最小的分区
            min_partition = np.argmin(partition_loads)
            partition_populations[min_partition].append(pop_id)
            partition_loads[min_partition] += load
        
        # 创建分区
        for i, population_ids in enumerate(partition_populations):
            if population_ids:
                partition_id = f"partition_compute_{i}"
                
                total_neurons = sum(self.connectome.populations[pop_id].count 
                                  for pop_id in population_ids)
                
                memory_gb = max(1.0, total_neurons * 32 / (1024**3) * 1.2)
                cpu_cores = max(1, self.params.num_threads // target_partitions)
                
                partition = NetworkPartition(
                    partition_id=partition_id,
                    populations=population_ids,
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb
                )
                
                self.partitions[partition_id] = partition
                
                for pop_id in population_ids:
                    self.partition_mapping[pop_id] = partition_id
        
        self.logger.info(f"创建了 {len(self.partitions)} 个计算负载均衡分区")
    
    async def _create_hardware_partitions(self):
        """基于硬件创建分区"""
        # 根据可用硬件创建分区
        available_hardware = self.hardware_manager.available_hardware
        
        partition_count = 0
        
        # 为每种硬件类型创建分区
        if available_hardware.get('loihi', False):
            # Loihi适合稀疏连接的皮层网络
            cortical_regions = [BrainRegion.PREFRONTAL_CORTEX, BrainRegion.PRIMARY_VISUAL]
            for region in cortical_regions:
                if region in self.connectome.region_populations:
                    partition_id = f"partition_loihi_{partition_count}"
                    population_ids = self.connectome.region_populations[region]
                    
                    partition = NetworkPartition(
                        partition_id=partition_id,
                        populations=population_ids,
                        hardware_target="loihi",
                        cpu_cores=1,
                        memory_gb=2.0
                    )
                    
                    self.partitions[partition_id] = partition
                    partition_count += 1
        
        if available_hardware.get('spinnaker', False):
            # SpiNNaker适合大规模并行处理
            large_regions = [BrainRegion.CEREBELLUM, BrainRegion.THALAMUS]
            for region in large_regions:
                if region in self.connectome.region_populations:
                    partition_id = f"partition_spinnaker_{partition_count}"
                    population_ids = self.connectome.region_populations[region]
                    
                    partition = NetworkPartition(
                        partition_id=partition_id,
                        populations=population_ids,
                        hardware_target="spinnaker",
                        cpu_cores=2,
                        memory_gb=4.0
                    )
                    
                    self.partitions[partition_id] = partition
                    partition_count += 1
        
        # 其余使用CPU/GPU分区
        remaining_populations = []
        for pop_id in self.connectome.populations.keys():
            if pop_id not in self.partition_mapping:
                remaining_populations.append(pop_id)
        
        if remaining_populations:
            # 按负载分配到CPU/GPU分区
            num_cpu_partitions = max(1, self.params.num_threads // 4)
            
            for i in range(num_cpu_partitions):
                partition_id = f"partition_cpu_{i}"
                start_idx = i * len(remaining_populations) // num_cpu_partitions
                end_idx = (i + 1) * len(remaining_populations) // num_cpu_partitions
                
                partition_pops = remaining_populations[start_idx:end_idx]
                
                if partition_pops:
                    total_neurons = sum(self.connectome.populations[pop_id].count 
                                      for pop_id in partition_pops)
                    
                    partition = NetworkPartition(
                        partition_id=partition_id,
                        populations=partition_pops,
                        hardware_target="cpu",
                        cpu_cores=self.params.num_threads // num_cpu_partitions,
                        memory_gb=max(2.0, total_neurons * 32 / (1024**3) * 1.5),
                        gpu_memory_gb=2.0 if self.params.use_gpu else 0.0
                    )
                    
                    self.partitions[partition_id] = partition
        
        # 更新映射
        for partition in self.partitions.values():
            for pop_id in partition.populations:
                self.partition_mapping[pop_id] = partition.partition_id
        
        self.logger.info(f"创建了 {len(self.partitions)} 个硬件分区")
    
    async def _initialize_hardware(self, hardware_configs: List[HardwareConfiguration]):
        """初始化硬件"""
        self.logger.info("初始化神经形态硬件...")
        
        results = await self.hardware_manager.initialize_hardware(hardware_configs)
        
        # 将硬件分配给分区
        hardware_mapping = {}
        for partition_id, partition in self.partitions.items():
            if partition.hardware_target and partition.hardware_target != "cpu":
                # 寻找匹配的硬件
                for config in hardware_configs:
                    if config.hardware_type.value == partition.hardware_target:
                        hardware_mapping[partition_id] = config.device_id
                        break

        self.partition_to_hardware = hardware_mapping
        self.hardware_to_partition = {device: part for part, device in hardware_mapping.items()}
        
        if hardware_mapping:
            # 部署网络到硬件
            network_config = self._prepare_hardware_network_config()
            await self.hardware_manager.deploy_network_to_hardware(
                network_config, hardware_mapping
            )

    def _build_routing_tables(self) -> None:
        """构建硬件与软件分区之间的路由表"""

        self.population_target_partitions = {}
        self.partition_hardware_targets = {}

        for connection in self.connectome.connections.values():
            source_partition = self.partition_mapping.get(connection.source_population)
            target_partition = self.partition_mapping.get(connection.target_population)

            if not source_partition or not target_partition:
                continue

            target_set = self.population_target_partitions.setdefault(
                connection.source_population, set()
            )
            target_set.add(target_partition)

            hardware_id = self.partition_to_hardware.get(target_partition)
            if hardware_id:
                hardware_targets = self.partition_hardware_targets.setdefault(
                    source_partition, set()
                )
                hardware_targets.add(hardware_id)
    
    def _prepare_hardware_network_config(self) -> Dict[str, Any]:
        """准备硬件网络配置"""
        config = {}
        
        for partition_id, partition in self.partitions.items():
            if partition.hardware_target and partition.hardware_target != "cpu":
                # 提取该分区的网络配置
                neuron_groups = []
                connections = []
                
                for pop_id in partition.populations:
                    population = self.connectome.populations[pop_id]
                    
                    neuron_groups.append({
                        'id': pop_id,
                        'size': population.count,
                        'neuron_type': population.neuron_type,
                        'threshold': population.threshold_potential,
                        'reset': population.reset_potential,
                        'tau_m': 20.0,  # 默认膜时间常数
                        'refractory': population.refractory_period
                    })
                
                # 提取连接
                for conn_id, connection in self.connectome.connections.items():
                    if (connection.source_population in partition.populations and
                        connection.target_population in partition.populations):
                        
                        connections.append({
                            'source_group': connection.source_population,
                            'target_group': connection.target_population,
                            'type': connection.connection_type,
                            'probability': connection.connection_probability,
                            'weight_mean': connection.weight_mean,
                            'weight_std': connection.weight_std,
                            'delay': connection.delay_mean
                        })
                
                config[partition_id] = {
                    'neuron_groups': neuron_groups,
                    'connections': connections
                }
        
        return config
    
    def _allocate_computational_resources(self):
        """分配计算资源"""
        total_cpu_cores = sum(p.cpu_cores for p in self.partitions.values())
        total_memory = sum(p.memory_gb for p in self.partitions.values())
        
        self.logger.info(f"总计算资源需求: {total_cpu_cores} CPU核心, {total_memory:.1f}GB 内存")
        
        # 检查资源是否充足
        available_cores = mp.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if total_cpu_cores > available_cores:
            self.logger.warning(f"CPU核心需求 ({total_cpu_cores}) 超过可用核心 ({available_cores})")
        
        if total_memory > available_memory:
            self.logger.warning(f"内存需求 ({total_memory:.1f}GB) 超过可用内存 ({available_memory:.1f}GB)")
    
    def _initialize_data_structures(self):
        """初始化数据结构"""
        self.logger.info("初始化仿真数据结构...")

        # 初始化尖峰数据存储
        if self.params.record_spikes:
            for partition_id in self.partitions.keys():
                self.spike_data[partition_id] = []

        # 初始化电压数据存储
        if self.params.record_voltages:
            for partition_id, partition in self.partitions.items():
                total_neurons = sum(self.connectome.populations[pop_id].count
                                  for pop_id in partition.populations)

                # 估计记录时间点数
                num_timepoints = int(self.params.simulation_time / self.params.output_interval)

                # 使用内存映射减少内存占用
                if self.params.use_memory_mapping:
                    voltage_file = f"voltage_data_{partition_id}.dat"
                    self.voltage_data[partition_id] = np.memmap(
                        voltage_file, dtype=np.float32, mode='w+',
                        shape=(total_neurons, num_timepoints)
                    )
                else:
                    self.voltage_data[partition_id] = np.zeros(
                        (total_neurons, num_timepoints), dtype=np.float32
                    )

        # 初始化分区缓冲区
        for partition in self.partitions.values():
            partition.input_buffers = partition.input_buffers or {}
            partition.output_buffers = partition.output_buffers or {}
    
    def _setup_memory_mapping(self):
        """设置内存映射"""
        if not self.params.use_memory_mapping:
            return
        
        self.logger.info("设置内存映射文件...")
        
        # 为大型数据结构创建内存映射
        for partition_id, partition in self.partitions.items():
            total_neurons = sum(self.connectome.populations[pop_id].count 
                              for pop_id in partition.populations)
            
            # 神经元状态内存映射
            state_file = f"neuron_states_{partition_id}.dat"
            state_map = np.memmap(
                state_file, dtype=np.float32, mode='w+',
                shape=(total_neurons, 8)  # 电压、电流、阈值等8个状态变量
            )
            
            # 创建内存映射文件对象
            with open(state_file, 'r+b') as f:
                self.memory_maps[f"states_{partition_id}"] = mmap.mmap(f.fileno(), 0)
    
    async def run_simulation(self, mode: SimulationMode = SimulationMode.DISTRIBUTED):
        """运行仿真"""
        if not self.is_initialized:
            raise RuntimeError("网络未初始化，请先调用 initialize()")
        
        self.logger.info(f"开始 {mode.value} 模式仿真...")
        self.is_running = True
        self.current_time = 0.0
        
        start_time = time.time()
        
        try:
            if mode == SimulationMode.DISTRIBUTED:
                await self._run_distributed_simulation()
            elif mode == SimulationMode.HYBRID_HARDWARE:
                await self._run_hybrid_hardware_simulation()
            elif mode == SimulationMode.REAL_TIME:
                await self._run_real_time_simulation()
            else:
                await self._run_standard_simulation()
        
        except Exception as e:
            self.logger.error(f"仿真执行错误: {e}")
            raise
        
        finally:
            self.is_running = False
            
            # 计算性能指标
            total_time = time.time() - start_time
            self.performance_metrics['total_computation_time'] = total_time
            
            total_spikes = sum(len(spikes) for spikes in self.spike_data.values())
            self.performance_metrics['throughput_spikes_per_second'] = total_spikes / total_time
            
            self.logger.info(f"仿真完成: {total_time:.2f}秒, "
                           f"{total_spikes:,} 个尖峰, "
                           f"{self.performance_metrics['throughput_spikes_per_second']:.0f} 尖峰/秒")
    
    async def _run_distributed_simulation(self):
        """运行分布式仿真"""
        # 为每个分区创建独立的仿真任务
        tasks = []
        
        for partition_id, partition in self.partitions.items():
            task = asyncio.create_task(
                self._simulate_partition(partition_id, partition)
            )
            tasks.append(task)
        
        # 并行执行所有分区
        await asyncio.gather(*tasks)
    
    async def _simulate_partition(self, partition_id: str, partition: NetworkPartition):
        """仿真单个分区"""
        self.logger.debug(f"开始仿真分区 {partition_id}")
        
        # 初始化分区状态
        neuron_states = self._initialize_partition_states(partition)
        
        # 仿真循环
        num_steps = int(self.params.simulation_time / self.params.dt)
        
        for step in range(num_steps):
            current_time = step * self.params.dt
            
            # 更新神经元状态
            spikes = self._update_neuron_states(partition, neuron_states, current_time)
            
            # 记录尖峰
            if self.params.record_spikes and spikes:
                self.spike_data[partition_id].extend(spikes)
            
            # 记录电压
            if self.params.record_voltages and step % int(self.params.output_interval / self.params.dt) == 0:
                output_idx = step // int(self.params.output_interval / self.params.dt)
                if output_idx < self.voltage_data[partition_id].shape[1]:
                    self.voltage_data[partition_id][:, output_idx] = neuron_states[:, 0]  # 电压
            
            # 处理分区间通信
            await self._handle_inter_partition_communication(partition_id, spikes)
        
        self.logger.debug(f"分区 {partition_id} 仿真完成")
    
    def _initialize_partition_states(self, partition: NetworkPartition) -> np.ndarray:
        """初始化分区神经元状态"""
        total_neurons = sum(self.connectome.populations[pop_id].count 
                          for pop_id in partition.populations)
        
        # 状态变量: [电压, 电流, 阈值, 重置电位, 不应期计数器, ...]
        states = np.zeros((total_neurons, 8), dtype=np.float32)
        
        neuron_idx = 0
        for pop_id in partition.populations:
            population = self.connectome.populations[pop_id]
            
            # 初始化电压 (随机分布在静息电位附近)
            states[neuron_idx:neuron_idx+population.count, 0] = np.random.normal(
                population.resting_potential, 5.0, population.count
            )
            
            # 设置阈值
            states[neuron_idx:neuron_idx+population.count, 2] = population.threshold_potential
            
            # 设置重置电位
            states[neuron_idx:neuron_idx+population.count, 3] = population.reset_potential
            
            neuron_idx += population.count
        
        return states
    
    def _update_neuron_states(self, partition: NetworkPartition, 
                            states: np.ndarray, current_time: float) -> List[SpikeEvent]:
        """更新神经元状态"""
        spikes = []
        
        # 简化的LIF神经元模型更新
        dt = self.params.dt
        
        # 更新电压 (简化欧拉方法)
        # dV/dt = (V_rest - V + I) / tau_m
        tau_m = 20.0  # ms
        V_rest = -70.0  # mV
        
        # 获取输入电流 (从连接和外部输入)
        input_current = self._calculate_input_current(partition, states, current_time)
        
        # 更新电压
        dV = (V_rest - states[:, 0] + input_current) / tau_m * dt
        states[:, 0] += dV
        
        # 检测尖峰
        spike_mask = states[:, 0] >= states[:, 2]  # V >= V_thresh
        spike_indices = np.where(spike_mask)[0]
        
        # 处理尖峰
        if len(spike_indices) > 0:
            # 重置电压
            states[spike_indices, 0] = states[spike_indices, 3]  # V = V_reset
            
            # 设置不应期
            states[spike_indices, 4] = 2.0 / dt  # 2ms不应期
            
            # 创建尖峰事件
            for idx in spike_indices:
                spike = SpikeEvent(
                    neuron_id=int(idx),
                    timestamp=current_time * 1000,  # 转换为微秒
                    chip_id=0,
                    core_id=0
                )
                spikes.append(spike)
        
        # 更新不应期计数器
        refractory_mask = states[:, 4] > 0
        states[refractory_mask, 4] -= 1
        
        # 不应期内的神经元电压保持重置值
        states[refractory_mask, 0] = states[refractory_mask, 3]
        
        return spikes
    
    def _calculate_input_current(self, partition: NetworkPartition,
                               states: np.ndarray, current_time: float) -> np.ndarray:
        """计算输入电流"""
        total_neurons = states.shape[0]
        input_current = np.zeros(total_neurons, dtype=np.float32)

        if not hasattr(partition, "delayed_events"):
            partition.delayed_events = defaultdict(deque)

        # 简化实现：添加随机背景电流
        background_current = np.random.normal(0.0, 0.1, total_neurons)
        input_current += background_current

        # 处理延迟到达的事件
        for source_id, delayed_queue in list(partition.delayed_events.items()):
            ready_events = []
            remaining_events = deque()

            while delayed_queue:
                ready_time, connection_id, spike = delayed_queue.popleft()
                if ready_time <= current_time * 1000.0:
                    ready_events.append((connection_id, spike))
                else:
                    remaining_events.append((ready_time, connection_id, spike))

            partition.delayed_events[source_id] = remaining_events

            for connection_id, spike in ready_events:
                connection = self.connectome.connections.get(connection_id)
                if connection:
                    self._accumulate_synaptic_input(partition, connection, spike, input_current)

        # 处理来自其他分区或硬件的尖峰事件
        for source_id, buffer in list(partition.input_buffers.items()):
            while True:
                try:
                    spike = buffer.get_nowait()
                except Empty:
                    break

                if source_id in self.partitions:
                    source_pop = self._get_neuron_population(source_id, spike.neuron_id)
                else:
                    source_pop = None
                if not source_pop:
                    # 无法识别的来源，视为外部调制输入
                    input_current += 0.01
                    continue

                # 根据连接关系累积突触输入
                for connection_id, connection in self.connectome.connections.items():
                    if connection.source_population != source_pop:
                        continue
                    if connection.target_population not in partition.populations:
                        continue

                    synaptic_delay_us = max(connection.delay_mean, 0.0) * 1000.0
                    deliver_time = spike.timestamp + synaptic_delay_us

                    if deliver_time > current_time * 1000.0:
                        partition.delayed_events[source_id].append((deliver_time, connection_id, spike))
                        continue

                    self._accumulate_synaptic_input(partition, connection, spike, input_current)

        return input_current

    def _accumulate_synaptic_input(self, partition: NetworkPartition,
                                   connection, spike: SpikeEvent,
                                   input_current: np.ndarray) -> None:
        """根据连接关系将尖峰累积到目标神经元电流中。"""

        start_idx, end_idx = self._get_population_local_indices(
            partition, connection.target_population
        )
        if end_idx <= start_idx:
            return

        target_size = end_idx - start_idx
        kernel = np.full(target_size, getattr(spike, "amplitude", 1.0), dtype=np.float32)

        weight_matrix = getattr(connection, "weight_matrix", None)
        if weight_matrix is not None:
            weights = np.asarray(weight_matrix, dtype=np.float32)
            if weights.shape[0] != target_size:
                weights = np.resize(weights, target_size)
        else:
            weights = np.random.normal(
                connection.weight_mean, connection.weight_std, target_size
            ).astype(np.float32)

        connection_mask = (np.random.random(target_size) <
                           max(connection.connection_probability, 1e-4)).astype(np.float32)
        sign = 1.0 if connection.connection_type == "excitatory" else -1.0

        input_current[start_idx:end_idx] += sign * weights * kernel * connection_mask
    
    async def _handle_inter_partition_communication(self, partition_id: str,
                                                  spikes: List[SpikeEvent]):
        """处理分区间通信"""
        if not spikes:
            return
        
        # 将尖峰发送到目标分区
        for spike in spikes:
            # 查找该神经元的输出连接
            source_pop = self._get_neuron_population(partition_id, spike.neuron_id)
            
            for conn_id, connection in self.connectome.connections.items():
                if connection.source_population == source_pop:
                    target_partition = self.partition_mapping.get(connection.target_population)

                    if target_partition and target_partition != partition_id:
                        # 发送到目标分区的输入缓冲区
                        target_buffers = self.partitions[target_partition].input_buffers
                        if partition_id not in target_buffers:
                            target_buffers[partition_id] = Queue()

                        # 添加延迟
                        delayed_spike = SpikeEvent(
                            neuron_id=spike.neuron_id,
                            timestamp=spike.timestamp + connection.delay_mean * 1000,
                            chip_id=spike.chip_id,
                            core_id=spike.core_id
                        )

                        target_buffers[partition_id].put(delayed_spike)
                        # 记录输出缓冲区，供监控或硬件传输
                        if target_partition not in self.partitions[partition_id].output_buffers:
                            self.partitions[partition_id].output_buffers[target_partition] = Queue()
                        self.partitions[partition_id].output_buffers[target_partition].put(delayed_spike)
    
    def _get_neuron_population(self, partition_id: str, neuron_id: int) -> str:
        """获取神经元所属的群体"""
        partition = self.partitions[partition_id]

        current_idx = 0
        for pop_id in partition.populations:
            population = self.connectome.populations[pop_id]
            if current_idx <= neuron_id < current_idx + population.count:
                return pop_id
            current_idx += population.count

        return ""

    def _get_population_local_indices(self, partition: NetworkPartition, pop_id: str) -> Tuple[int, int]:
        """返回群体在分区状态数组中的局部索引范围"""
        current_idx = 0
        for pid in partition.populations:
            population = self.connectome.populations[pid]
            start_idx = current_idx
            end_idx = current_idx + population.count
            if pid == pop_id:
                return start_idx, end_idx
            current_idx = end_idx
        return 0, 0
    
    async def _run_hybrid_hardware_simulation(self):
        """运行混合硬件仿真"""
        # 启动硬件执行
        hardware_results = await self.hardware_manager.start_all_hardware()
        
        # 运行软件分区
        software_tasks = []
        for partition_id, partition in self.partitions.items():
            if not partition.hardware_target or partition.hardware_target == "cpu":
                task = asyncio.create_task(
                    self._simulate_partition(partition_id, partition)
                )
                software_tasks.append(task)
        
        # 处理硬件-软件通信
        communication_task = asyncio.create_task(
            self._handle_hardware_software_communication()
        )
        
        # 等待所有任务完成
        await asyncio.gather(*software_tasks, communication_task)
        
        # 停止硬件
        await self.hardware_manager.stop_all_hardware()
    
    async def _handle_hardware_software_communication(self):
        """处理硬件-软件通信"""
        while self.is_running:
            # 从硬件接收事件
            hardware_events = await self.hardware_manager.receive_events_from_hardware()

            # 将事件分发到软件分区
            for hardware_id, events in hardware_events.items():
                source_partition = self.hardware_to_partition.get(hardware_id)
                if not source_partition:
                    continue

                for event in events:
                    source_population = self._get_neuron_population(
                        source_partition, event.neuron_id
                    )
                    if not source_population:
                        continue

                    target_partitions = self.population_target_partitions.get(
                        source_population, set()
                    )

                    for partition_id in target_partitions:
                        partition = self.partitions.get(partition_id)
                        if not partition or (
                            partition.hardware_target
                            and partition.hardware_target != "cpu"
                        ):
                            continue

                        if hardware_id not in partition.input_buffers:
                            partition.input_buffers[hardware_id] = Queue()
                        partition.input_buffers[hardware_id].put(event)

            # 收集软件分区的输出事件发送到硬件
            software_events: Dict[str, List[SpikeEvent]] = {}
            for partition_id, partition in self.partitions.items():
                for target_partition_id, buffer in list(partition.output_buffers.items()):
                    hardware_id = self.partition_to_hardware.get(target_partition_id)
                    if not hardware_id:
                        continue

                    while True:
                        try:
                            event = buffer.get_nowait()
                            software_events.setdefault(hardware_id, []).append(event)
                        except Empty:
                            break

            if software_events:
                await self.hardware_manager.send_events_to_hardware(software_events)
            
            await asyncio.sleep(0.001)  # 1ms通信间隔
    
    async def _run_real_time_simulation(self):
        """运行实时仿真"""
        self.logger.info("启动实时仿真模式")
        
        # 实时仿真需要严格的时间控制
        real_time_factor = 1.0  # 1:1实时
        step_duration = self.params.dt / 1000.0  # 转换为秒
        
        num_steps = int(self.params.simulation_time / self.params.dt)
        
        for step in range(num_steps):
            step_start_time = time.time()
            
            # 并行执行所有分区的一个时间步
            tasks = []
            for partition_id, partition in self.partitions.items():
                task = asyncio.create_task(
                    self._simulate_partition_step(partition_id, partition, step)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # 时间同步
            elapsed_time = time.time() - step_start_time
            sleep_time = step_duration - elapsed_time
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            elif sleep_time < -step_duration * 0.1:  # 超时10%以上
                self.logger.warning(f"实时仿真延迟: 步骤 {step} 超时 {-sleep_time*1000:.1f}ms")
    
    async def _simulate_partition_step(self, partition_id: str,
                                     partition: NetworkPartition, step: int):
        """仿真分区的单个时间步"""
        # 这是 _simulate_partition 的单步版本
        if not hasattr(partition, "_runtime_states"):
            partition._runtime_states = self._initialize_partition_states(partition)

        neuron_states = partition._runtime_states
        current_time = step * self.params.dt

        spikes = self._update_neuron_states(partition, neuron_states, current_time)

        if self.params.record_spikes and spikes:
            self.spike_data.setdefault(partition_id, []).extend(spikes)

        if self.params.record_voltages and step % int(self.params.output_interval / self.params.dt) == 0:
            output_idx = step // int(self.params.output_interval / self.params.dt)
            if output_idx < self.voltage_data[partition_id].shape[1]:
                self.voltage_data[partition_id][:, output_idx] = neuron_states[:, 0]

        await self._handle_inter_partition_communication(partition_id, spikes)
    
    async def _run_standard_simulation(self):
        """运行标准仿真"""
        self.logger.info("启动标准仿真模式")

        num_steps = int(self.params.simulation_time / self.params.dt)
        step_duration = self.params.dt / 1000.0

        # 预初始化所有分区的运行时状态，避免在循环中重复创建
        for partition_id, partition in self.partitions.items():
            if not hasattr(partition, "_runtime_states"):
                partition._runtime_states = self._initialize_partition_states(partition)

        # 顺序迭代每个时间步，并在步内顺序处理每个分区
        for step in range(num_steps):
            step_start_time = time.time()

            for partition_id, partition in self.partitions.items():
                await self._simulate_partition_step(partition_id, partition, step)

            # 按照 dt 控制时间步长，保持与其他模式一致的时间基准
            elapsed_time = time.time() - step_start_time
            sleep_time = step_duration - elapsed_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """获取仿真结果"""
        return {
            'spike_data': self.spike_data,
            'voltage_data': self.voltage_data if self.params.record_voltages else None,
            'performance_metrics': self.performance_metrics,
            'simulation_parameters': {
                'dt': self.params.dt,
                'simulation_time': self.params.simulation_time,
                'total_neurons': self.connectome.total_neurons,
                'total_synapses': self.connectome.total_synapses
            }
        }
    
    def save_results(self, filepath: str):
        """保存仿真结果"""
        self.logger.info(f"保存仿真结果到 {filepath}")
        
        results = self.get_simulation_results()
        
        # 使用HDF5保存大型数据
        with h5py.File(filepath, 'w') as f:
            # 保存尖峰数据
            spike_group = f.create_group('spikes')
            for partition_id, spikes in self.spike_data.items():
                if spikes:
                    spike_array = np.array([(s.neuron_id, s.timestamp, s.chip_id, s.core_id) 
                                          for s in spikes])
                    spike_group.create_dataset(partition_id, data=spike_array)
            
            # 保存电压数据
            if self.params.record_voltages:
                voltage_group = f.create_group('voltages')
                for partition_id, voltages in self.voltage_data.items():
                    voltage_group.create_dataset(partition_id, data=voltages)
            
            # 保存元数据
            metadata = f.create_group('metadata')
            metadata.attrs['dt'] = self.params.dt
            metadata.attrs['simulation_time'] = self.params.simulation_time
            metadata.attrs['total_neurons'] = self.connectome.total_neurons
            metadata.attrs['total_synapses'] = self.connectome.total_synapses
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理仿真资源...")
        
        # 停止仿真
        self.is_running = False
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # 清理硬件连接
        await self.hardware_manager.cleanup()
        
        # 关闭内存映射
        for mmap_obj in self.memory_maps.values():
            mmap_obj.close()
        
        # 清理临时文件
        for partition_id in self.partitions.keys():
            temp_files = [
                f"voltage_data_{partition_id}.dat",
                f"neuron_states_{partition_id}.dat"
            ]
            
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
        
        # 强制垃圾回收
        gc.collect()
        
        self.logger.info("资源清理完成")

# 工厂函数
def create_large_scale_manager(connectome: ConnectomeManager, 
                             simulation_params: Optional[SimulationParameters] = None) -> LargeScaleNetworkManager:
    """创建大规模网络管理器"""
    if simulation_params is None:
        simulation_params = SimulationParameters()
    
    return LargeScaleNetworkManager(connectome, simulation_params)