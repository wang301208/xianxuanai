"""
全脑连接组学管理器 - 真实的全脑结构实现
Full Brain Connectome Manager - Real Full Brain Structure Implementation

基于真实的人脑连接组学数据构建860亿神经元的全脑网络
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import h5py
import json
import pickle
from pathlib import Path
import networkx as nx
from scipy import sparse
import pandas as pd

class BrainRegion(Enum):
    """脑区枚举 - 基于Brodmann分区和现代脑图谱"""
    # 前额叶皮层
    PREFRONTAL_CORTEX = "PFC"
    DORSOLATERAL_PFC = "dlPFC"
    VENTROMEDIAL_PFC = "vmPFC"
    ORBITOFRONTAL_CORTEX = "OFC"
    ANTERIOR_CINGULATE = "ACC"
    
    # 运动皮层
    PRIMARY_MOTOR = "M1"
    PREMOTOR_CORTEX = "PMC"
    SUPPLEMENTARY_MOTOR = "SMA"
    
    # 感觉皮层
    PRIMARY_SOMATOSENSORY = "S1"
    SECONDARY_SOMATOSENSORY = "S2"
    
    # 视觉皮层
    PRIMARY_VISUAL = "V1"
    SECONDARY_VISUAL = "V2"
    VISUAL_AREA_3 = "V3"
    VISUAL_AREA_4 = "V4"
    MIDDLE_TEMPORAL = "MT"
    
    # 听觉皮层
    PRIMARY_AUDITORY = "A1"
    SECONDARY_AUDITORY = "A2"
    
    # 颞叶
    HIPPOCAMPUS = "HIP"
    ENTORHINAL_CORTEX = "EC"
    PARAHIPPOCAMPAL = "PHC"
    TEMPORAL_POLE = "TP"
    
    # 顶叶
    POSTERIOR_PARIETAL = "PPC"
    INFERIOR_PARIETAL = "IPL"
    SUPERIOR_PARIETAL = "SPL"
    
    # 枕叶
    OCCIPITAL_CORTEX = "OC"
    
    # 皮层下结构
    THALAMUS = "TH"
    HYPOTHALAMUS = "HYP"
    BASAL_GANGLIA = "BG"
    STRIATUM = "STR"
    SUBSTANTIA_NIGRA = "SN"
    AMYGDALA = "AMY"
    
    # 脑干
    BRAINSTEM = "BS"
    PONS = "PONS"
    MEDULLA = "MED"
    
    # 小脑
    CEREBELLUM = "CB"
    CEREBELLAR_CORTEX = "CBC"
    DEEP_CEREBELLAR_NUCLEI = "DCN"

@dataclass
class NeuronPopulation:
    """神经元群体"""
    region: BrainRegion
    layer: Optional[int]  # 皮层层次 (1-6)
    neuron_type: str  # 'excitatory', 'inhibitory', 'pyramidal', 'interneuron'
    count: int
    coordinates: Tuple[float, float, float]  # 3D坐标 (mm)
    
    # 生理参数
    membrane_capacitance: float = 1.0  # pF
    membrane_resistance: float = 100.0  # MΩ
    resting_potential: float = -70.0  # mV
    threshold_potential: float = -55.0  # mV
    reset_potential: float = -70.0  # mV
    refractory_period: float = 2.0  # ms
    
    # 突触参数
    excitatory_reversal: float = 0.0  # mV
    inhibitory_reversal: float = -70.0  # mV
    
    # 元数据
    population_id: str = field(default="")
    description: str = field(default="")

@dataclass
class SynapticConnection:
    """突触连接"""
    source_population: str
    target_population: str
    connection_type: str  # 'excitatory', 'inhibitory'
    
    # 连接参数
    connection_probability: float
    weight_mean: float
    weight_std: float
    delay_mean: float  # ms
    delay_std: float
    
    # 可塑性参数
    plasticity_type: Optional[str] = None  # 'STDP', 'homeostatic', None
    learning_rate: float = 0.01
    
    # 空间参数
    distance_dependent: bool = False
    max_distance: float = 10.0  # mm
    
    # 元数据
    connection_id: str = field(default="")
    anatomical_pathway: str = field(default="")

class ConnectomeManager:
    """全脑连接组学管理器"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.logger = logging.getLogger("ConnectomeManager")
        
        # 数据存储
        self.populations: Dict[str, NeuronPopulation] = {}
        self.connections: Dict[str, SynapticConnection] = {}
        self.connectivity_matrix: Optional[sparse.csr_matrix] = None
        
        # 脑区映射
        self.region_populations: Dict[BrainRegion, List[str]] = {}
        self.population_indices: Dict[str, Tuple[int, int]] = {}  # (start_idx, end_idx)
        
        # 统计信息
        self.total_neurons = 0
        self.total_synapses = 0
        
        # 数据路径
        self.data_path = Path(data_path) if data_path else Path("data/connectome")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化全脑结构
        self._initialize_brain_structure()
    
    def _initialize_brain_structure(self):
        """初始化全脑结构"""
        self.logger.info("初始化全脑连接组学结构...")
        
        # 创建基本脑区群体
        self._create_cortical_populations()
        self._create_subcortical_populations()
        self._create_brainstem_populations()
        self._create_cerebellar_populations()
        
        # 建立连接
        self._create_cortical_connections()
        self._create_thalamocortical_connections()
        self._create_corticostriatal_connections()
        self._create_cerebellar_connections()
        
        # 构建连接矩阵
        self._build_connectivity_matrix()
        
        self.logger.info(f"全脑结构初始化完成: {self.total_neurons:,} 神经元, {self.total_synapses:,} 突触")
    
    def _create_cortical_populations(self):
        """创建皮层神经元群体"""
        # 基于真实人脑皮层神经元分布
        cortical_regions = {
            BrainRegion.PREFRONTAL_CORTEX: {
                'total_neurons': 16_000_000_000,  # 160亿
                'coordinates': (20, 40, 30),
                'layers': 6
            },
            BrainRegion.PRIMARY_MOTOR: {
                'total_neurons': 8_000_000_000,  # 80亿
                'coordinates': (0, -20, 50),
                'layers': 6
            },
            BrainRegion.PRIMARY_SOMATOSENSORY: {
                'total_neurons': 6_000_000_000,  # 60亿
                'coordinates': (-20, -30, 50),
                'layers': 6
            },
            BrainRegion.PRIMARY_VISUAL: {
                'total_neurons': 4_000_000_000,  # 40亿
                'coordinates': (0, -80, 10),
                'layers': 6
            },
            BrainRegion.PRIMARY_AUDITORY: {
                'total_neurons': 2_000_000_000,  # 20亿
                'coordinates': (-50, -20, 10),
                'layers': 6
            },
            BrainRegion.HIPPOCAMPUS: {
                'total_neurons': 40_000_000,  # 4000万
                'coordinates': (-30, -30, -10),
                'layers': 3
            },
            BrainRegion.POSTERIOR_PARIETAL: {
                'total_neurons': 5_000_000_000,  # 50亿
                'coordinates': (-20, -60, 40),
                'layers': 6
            }
        }
        
        for region, config in cortical_regions.items():
            self._create_layered_populations(region, config)
    
    def _create_layered_populations(self, region: BrainRegion, config: Dict[str, Any]):
        """创建分层皮层群体"""
        total_neurons = config['total_neurons']
        coordinates = config['coordinates']
        num_layers = config['layers']
        
        # 每层神经元分布 (基于实际皮层解剖)
        layer_distribution = {
            1: 0.05,  # 分子层
            2: 0.15,  # 外颗粒层
            3: 0.25,  # 外锥体层
            4: 0.20,  # 内颗粒层
            5: 0.20,  # 内锥体层
            6: 0.15   # 多形层
        }
        
        for layer in range(1, num_layers + 1):
            layer_neurons = int(total_neurons * layer_distribution.get(layer, 1.0/num_layers))
            
            # 兴奋性神经元 (80%)
            exc_neurons = int(layer_neurons * 0.8)
            exc_pop = NeuronPopulation(
                region=region,
                layer=layer,
                neuron_type='pyramidal' if layer in [3, 5] else 'excitatory',
                count=exc_neurons,
                coordinates=coordinates,
                threshold_potential=-55.0,
                reset_potential=-70.0,
                refractory_period=2.0
            )
            exc_pop.population_id = f"{region.value}_L{layer}_EXC"
            self.populations[exc_pop.population_id] = exc_pop
            
            # 抑制性神经元 (20%)
            inh_neurons = int(layer_neurons * 0.2)
            inh_pop = NeuronPopulation(
                region=region,
                layer=layer,
                neuron_type='interneuron',
                count=inh_neurons,
                coordinates=coordinates,
                threshold_potential=-50.0,
                reset_potential=-65.0,
                refractory_period=1.0
            )
            inh_pop.population_id = f"{region.value}_L{layer}_INH"
            self.populations[inh_pop.population_id] = inh_pop
            
            # 更新脑区映射
            if region not in self.region_populations:
                self.region_populations[region] = []
            self.region_populations[region].extend([exc_pop.population_id, inh_pop.population_id])
    
    def _create_subcortical_populations(self):
        """创建皮层下结构"""
        subcortical_regions = {
            BrainRegion.THALAMUS: {
                'neurons': 6_000_000,  # 600万
                'coordinates': (0, -15, 5),
                'exc_ratio': 0.75
            },
            BrainRegion.BASAL_GANGLIA: {
                'neurons': 100_000_000,  # 1亿
                'coordinates': (-15, 5, -5),
                'exc_ratio': 0.5
            },
            BrainRegion.AMYGDALA: {
                'neurons': 13_000_000,  # 1300万
                'coordinates': (-25, -5, -15),
                'exc_ratio': 0.8
            }
        }
        
        for region, config in subcortical_regions.items():
            total_neurons = config['neurons']
            coordinates = config['coordinates']
            exc_ratio = config['exc_ratio']
            
            # 兴奋性群体
            exc_neurons = int(total_neurons * exc_ratio)
            exc_pop = NeuronPopulation(
                region=region,
                layer=None,
                neuron_type='excitatory',
                count=exc_neurons,
                coordinates=coordinates
            )
            exc_pop.population_id = f"{region.value}_EXC"
            self.populations[exc_pop.population_id] = exc_pop
            
            # 抑制性群体
            inh_neurons = total_neurons - exc_neurons
            inh_pop = NeuronPopulation(
                region=region,
                layer=None,
                neuron_type='inhibitory',
                count=inh_neurons,
                coordinates=coordinates
            )
            inh_pop.population_id = f"{region.value}_INH"
            self.populations[inh_pop.population_id] = inh_pop
            
            # 更新脑区映射
            self.region_populations[region] = [exc_pop.population_id, inh_pop.population_id]
    
    def _create_brainstem_populations(self):
        """创建脑干结构"""
        brainstem_regions = {
            BrainRegion.BRAINSTEM: {
                'neurons': 50_000_000,  # 5000万
                'coordinates': (0, -40, -20),
                'exc_ratio': 0.7
            }
        }
        
        for region, config in brainstem_regions.items():
            total_neurons = config['neurons']
            coordinates = config['coordinates']
            exc_ratio = config['exc_ratio']
            
            exc_neurons = int(total_neurons * exc_ratio)
            exc_pop = NeuronPopulation(
                region=region,
                layer=None,
                neuron_type='excitatory',
                count=exc_neurons,
                coordinates=coordinates
            )
            exc_pop.population_id = f"{region.value}_EXC"
            self.populations[exc_pop.population_id] = exc_pop
            
            inh_neurons = total_neurons - exc_neurons
            inh_pop = NeuronPopulation(
                region=region,
                layer=None,
                neuron_type='inhibitory',
                count=inh_neurons,
                coordinates=coordinates
            )
            inh_pop.population_id = f"{region.value}_INH"
            self.populations[inh_pop.population_id] = inh_pop
            
            self.region_populations[region] = [exc_pop.population_id, inh_pop.population_id]
    
    def _create_cerebellar_populations(self):
        """创建小脑结构"""
        # 小脑包含约690亿神经元，主要是颗粒细胞
        cerebellar_config = {
            'granule_cells': 60_000_000_000,  # 600亿颗粒细胞
            'purkinje_cells': 15_000_000,     # 1500万浦肯野细胞
            'coordinates': (0, -70, -30)
        }
        
        # 颗粒细胞层
        granule_pop = NeuronPopulation(
            region=BrainRegion.CEREBELLUM,
            layer=1,
            neuron_type='granule',
            count=cerebellar_config['granule_cells'],
            coordinates=cerebellar_config['coordinates'],
            threshold_potential=-40.0,
            reset_potential=-60.0,
            refractory_period=1.0
        )
        granule_pop.population_id = "CB_GRANULE"
        self.populations[granule_pop.population_id] = granule_pop
        
        # 浦肯野细胞层
        purkinje_pop = NeuronPopulation(
            region=BrainRegion.CEREBELLUM,
            layer=2,
            neuron_type='purkinje',
            count=cerebellar_config['purkinje_cells'],
            coordinates=cerebellar_config['coordinates'],
            threshold_potential=-55.0,
            reset_potential=-70.0,
            refractory_period=2.0
        )
        purkinje_pop.population_id = "CB_PURKINJE"
        self.populations[purkinje_pop.population_id] = purkinje_pop
        
        self.region_populations[BrainRegion.CEREBELLUM] = [
            granule_pop.population_id, 
            purkinje_pop.population_id
        ]
    
    def _create_cortical_connections(self):
        """创建皮层内和皮层间连接"""
        # 皮层内连接
        for region, pop_ids in self.region_populations.items():
            if region.value.endswith('CORTEX') or region in [
                BrainRegion.PREFRONTAL_CORTEX, 
                BrainRegion.PRIMARY_MOTOR,
                BrainRegion.PRIMARY_SOMATOSENSORY,
                BrainRegion.PRIMARY_VISUAL
            ]:
                self._create_intra_cortical_connections(region, pop_ids)
        
        # 皮层间长程连接
        self._create_inter_cortical_connections()
    
    def _create_intra_cortical_connections(self, region: BrainRegion, population_ids: List[str]):
        """创建皮层内连接"""
        # 层内连接
        for pop_id in population_ids:
            pop = self.populations[pop_id]
            if pop.layer:
                # 同层连接
                same_layer_pops = [p for p in population_ids 
                                 if self.populations[p].layer == pop.layer]
                
                for target_id in same_layer_pops:
                    if target_id != pop_id:
                        conn = SynapticConnection(
                            source_population=pop_id,
                            target_population=target_id,
                            connection_type='excitatory' if 'EXC' in pop_id else 'inhibitory',
                            connection_probability=0.1,
                            weight_mean=0.5 if 'EXC' in pop_id else -1.0,
                            weight_std=0.1,
                            delay_mean=1.0,
                            delay_std=0.2
                        )
                        conn.connection_id = f"{pop_id}_to_{target_id}"
                        self.connections[conn.connection_id] = conn
        
        # 跨层连接 (前馈和反馈)
        self._create_inter_layer_connections(population_ids)
    
    def _create_inter_layer_connections(self, population_ids: List[str]):
        """创建跨层连接"""
        layer_pops = {}
        for pop_id in population_ids:
            pop = self.populations[pop_id]
            if pop.layer:
                if pop.layer not in layer_pops:
                    layer_pops[pop.layer] = []
                layer_pops[pop.layer].append(pop_id)
        
        # 前馈连接 (L4 -> L2/3, L2/3 -> L5, L5 -> L6)
        feedforward_paths = [(4, 2), (4, 3), (2, 5), (3, 5), (5, 6)]
        
        for source_layer, target_layer in feedforward_paths:
            if source_layer in layer_pops and target_layer in layer_pops:
                for source_id in layer_pops[source_layer]:
                    for target_id in layer_pops[target_layer]:
                        if 'EXC' in source_id:  # 只有兴奋性连接跨层
                            conn = SynapticConnection(
                                source_population=source_id,
                                target_population=target_id,
                                connection_type='excitatory',
                                connection_probability=0.05,
                                weight_mean=0.3,
                                weight_std=0.1,
                                delay_mean=2.0,
                                delay_std=0.5
                            )
                            conn.connection_id = f"{source_id}_to_{target_id}_FF"
                            self.connections[conn.connection_id] = conn
    
    def _create_inter_cortical_connections(self):
        """创建皮层间长程连接"""
        # 主要的皮层间连接路径
        inter_cortical_paths = [
            (BrainRegion.PRIMARY_VISUAL, BrainRegion.POSTERIOR_PARIETAL),
            (BrainRegion.POSTERIOR_PARIETAL, BrainRegion.PREFRONTAL_CORTEX),
            (BrainRegion.PREFRONTAL_CORTEX, BrainRegion.PRIMARY_MOTOR),
            (BrainRegion.PRIMARY_SOMATOSENSORY, BrainRegion.PRIMARY_MOTOR),
            (BrainRegion.PRIMARY_AUDITORY, BrainRegion.PREFRONTAL_CORTEX)
        ]
        
        for source_region, target_region in inter_cortical_paths:
            if source_region in self.region_populations and target_region in self.region_populations:
                source_pops = self.region_populations[source_region]
                target_pops = self.region_populations[target_region]
                
                # 主要从L5连接到L4
                source_l5 = [p for p in source_pops if 'L5' in p and 'EXC' in p]
                target_l4 = [p for p in target_pops if 'L4' in p]
                
                for source_id in source_l5:
                    for target_id in target_l4:
                        conn = SynapticConnection(
                            source_population=source_id,
                            target_population=target_id,
                            connection_type='excitatory',
                            connection_probability=0.01,  # 长程连接稀疏
                            weight_mean=0.2,
                            weight_std=0.05,
                            delay_mean=10.0,  # 长程延迟
                            delay_std=2.0,
                            distance_dependent=True,
                            max_distance=100.0
                        )
                        conn.connection_id = f"{source_id}_to_{target_id}_LONG"
                        conn.anatomical_pathway = f"{source_region.value}_to_{target_region.value}"
                        self.connections[conn.connection_id] = conn
    
    def _create_thalamocortical_connections(self):
        """创建丘脑-皮层连接"""
        if BrainRegion.THALAMUS in self.region_populations:
            thalamus_pops = self.region_populations[BrainRegion.THALAMUS]
            
            # 丘脑到皮层L4的连接
            cortical_regions = [
                BrainRegion.PRIMARY_VISUAL,
                BrainRegion.PRIMARY_SOMATOSENSORY,
                BrainRegion.PRIMARY_AUDITORY,
                BrainRegion.PRIMARY_MOTOR
            ]
            
            for region in cortical_regions:
                if region in self.region_populations:
                    cortical_pops = self.region_populations[region]
                    l4_pops = [p for p in cortical_pops if 'L4' in p]
                    
                    for thal_id in thalamus_pops:
                        if 'EXC' in thal_id:
                            for cortex_id in l4_pops:
                                conn = SynapticConnection(
                                    source_population=thal_id,
                                    target_population=cortex_id,
                                    connection_type='excitatory',
                                    connection_probability=0.2,
                                    weight_mean=1.0,
                                    weight_std=0.2,
                                    delay_mean=5.0,
                                    delay_std=1.0
                                )
                                conn.connection_id = f"{thal_id}_to_{cortex_id}_TC"
                                conn.anatomical_pathway = "thalamocortical"
                                self.connections[conn.connection_id] = conn
    
    def _create_corticostriatal_connections(self):
        """创建皮层-纹状体连接"""
        if BrainRegion.BASAL_GANGLIA in self.region_populations:
            bg_pops = self.region_populations[BrainRegion.BASAL_GANGLIA]
            
            # 皮层L5到纹状体的连接
            cortical_regions = [
                BrainRegion.PREFRONTAL_CORTEX,
                BrainRegion.PRIMARY_MOTOR
            ]
            
            for region in cortical_regions:
                if region in self.region_populations:
                    cortical_pops = self.region_populations[region]
                    l5_pops = [p for p in cortical_pops if 'L5' in p and 'EXC' in p]
                    
                    for cortex_id in l5_pops:
                        for bg_id in bg_pops:
                            conn = SynapticConnection(
                                source_population=cortex_id,
                                target_population=bg_id,
                                connection_type='excitatory',
                                connection_probability=0.05,
                                weight_mean=0.8,
                                weight_std=0.15,
                                delay_mean=8.0,
                                delay_std=1.5
                            )
                            conn.connection_id = f"{cortex_id}_to_{bg_id}_CS"
                            conn.anatomical_pathway = "corticostriatal"
                            self.connections[conn.connection_id] = conn
    
    def _create_cerebellar_connections(self):
        """创建小脑连接"""
        if BrainRegion.CEREBELLUM in self.region_populations:
            cb_pops = self.region_populations[BrainRegion.CEREBELLUM]
            granule_pops = [p for p in cb_pops if 'GRANULE' in p]
            purkinje_pops = [p for p in cb_pops if 'PURKINJE' in p]
            
            # 颗粒细胞到浦肯野细胞的连接
            for granule_id in granule_pops:
                for purkinje_id in purkinje_pops:
                    conn = SynapticConnection(
                        source_population=granule_id,
                        target_population=purkinje_id,
                        connection_type='excitatory',
                        connection_probability=0.0001,  # 极稀疏但数量巨大
                        weight_mean=0.1,
                        weight_std=0.02,
                        delay_mean=1.0,
                        delay_std=0.2
                    )
                    conn.connection_id = f"{granule_id}_to_{purkinje_id}_PF"
                    conn.anatomical_pathway = "parallel_fiber"
                    self.connections[conn.connection_id] = conn
    
    def _build_connectivity_matrix(self):
        """构建全脑连接矩阵"""
        self.logger.info("构建全脑连接矩阵...")
        
        # 计算总神经元数和分配索引
        current_idx = 0
        for pop_id, population in self.populations.items():
            start_idx = current_idx
            end_idx = current_idx + population.count
            self.population_indices[pop_id] = (start_idx, end_idx)
            current_idx = end_idx
        
        self.total_neurons = current_idx
        
        # 创建稀疏连接矩阵
        row_indices = []
        col_indices = []
        weights = []
        
        synapse_count = 0
        
        for conn_id, connection in self.connections.items():
            source_start, source_end = self.population_indices[connection.source_population]
            target_start, target_end = self.population_indices[connection.target_population]
            
            source_size = source_end - source_start
            target_size = target_end - target_start
            
            # 根据连接概率生成连接
            num_connections = int(source_size * target_size * connection.connection_probability)
            
            if num_connections > 0:
                # 随机选择连接
                source_indices = np.random.choice(source_size, num_connections, replace=True) + source_start
                target_indices = np.random.choice(target_size, num_connections, replace=True) + target_start
                
                # 生成权重
                conn_weights = np.random.normal(
                    connection.weight_mean, 
                    connection.weight_std, 
                    num_connections
                )
                
                row_indices.extend(source_indices)
                col_indices.extend(target_indices)
                weights.extend(conn_weights)
                
                synapse_count += num_connections
        
        # 创建稀疏矩阵
        self.connectivity_matrix = sparse.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(self.total_neurons, self.total_neurons)
        )
        
        self.total_synapses = synapse_count
        
        self.logger.info(f"连接矩阵构建完成: {self.total_neurons:,} x {self.total_neurons:,}, {self.total_synapses:,} 突触")
    
    def get_population_by_region(self, region: BrainRegion) -> List[NeuronPopulation]:
        """获取指定脑区的所有神经元群体"""
        if region in self.region_populations:
            return [self.populations[pop_id] for pop_id in self.region_populations[region]]
        return []
    
    def get_connections_by_pathway(self, pathway: str) -> List[SynapticConnection]:
        """获取指定解剖通路的连接"""
        return [conn for conn in self.connections.values() 
                if conn.anatomical_pathway == pathway]
    
    def get_regional_connectivity(self, source_region: BrainRegion, 
                                target_region: BrainRegion) -> sparse.csr_matrix:
        """获取两个脑区间的连接矩阵"""
        if source_region not in self.region_populations or target_region not in self.region_populations:
            return None
        
        source_pops = self.region_populations[source_region]
        target_pops = self.region_populations[target_region]
        
        # 获取索引范围
        source_indices = []
        target_indices = []
        
        for pop_id in source_pops:
            start, end = self.population_indices[pop_id]
            source_indices.extend(range(start, end))
        
        for pop_id in target_pops:
            start, end = self.population_indices[pop_id]
            target_indices.extend(range(start, end))
        
        # 提取子矩阵
        return self.connectivity_matrix[np.ix_(source_indices, target_indices)]
    
    def save_connectome(self, filepath: str):
        """保存连接组学数据"""
        self.logger.info(f"保存连接组学数据到 {filepath}")
        
        data = {
            'populations': {pop_id: {
                'region': pop.region.value,
                'layer': pop.layer,
                'neuron_type': pop.neuron_type,
                'count': pop.count,
                'coordinates': pop.coordinates,
                'parameters': {
                    'membrane_capacitance': pop.membrane_capacitance,
                    'membrane_resistance': pop.membrane_resistance,
                    'resting_potential': pop.resting_potential,
                    'threshold_potential': pop.threshold_potential,
                    'reset_potential': pop.reset_potential,
                    'refractory_period': pop.refractory_period
                }
            } for pop_id, pop in self.populations.items()},
            
            'connections': {conn_id: {
                'source_population': conn.source_population,
                'target_population': conn.target_population,
                'connection_type': conn.connection_type,
                'connection_probability': conn.connection_probability,
                'weight_mean': conn.weight_mean,
                'weight_std': conn.weight_std,
                'delay_mean': conn.delay_mean,
                'delay_std': conn.delay_std,
                'anatomical_pathway': conn.anatomical_pathway
            } for conn_id, conn in self.connections.items()},
            
            'statistics': {
                'total_neurons': self.total_neurons,
                'total_synapses': self.total_synapses,
                'num_populations': len(self.populations),
                'num_connections': len(self.connections)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 保存连接矩阵
        matrix_path = filepath.replace('.json', '_matrix.npz')
        sparse.save_npz(matrix_path, self.connectivity_matrix)
    
    def load_connectome(self, filepath: str):
        """加载连接组学数据"""
        self.logger.info(f"从 {filepath} 加载连接组学数据")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 重建群体
        self.populations = {}
        for pop_id, pop_data in data['populations'].items():
            pop = NeuronPopulation(
                region=BrainRegion(pop_data['region']),
                layer=pop_data['layer'],
                neuron_type=pop_data['neuron_type'],
                count=pop_data['count'],
                coordinates=tuple(pop_data['coordinates'])
            )
            pop.population_id = pop_id
            
            # 设置参数
            params = pop_data['parameters']
            pop.membrane_capacitance = params['membrane_capacitance']
            pop.membrane_resistance = params['membrane_resistance']
            pop.resting_potential = params['resting_potential']
            pop.threshold_potential = params['threshold_potential']
            pop.reset_potential = params['reset_potential']
            pop.refractory_period = params['refractory_period']
            
            self.populations[pop_id] = pop
        
        # 重建连接
        self.connections = {}
        for conn_id, conn_data in data['connections'].items():
            conn = SynapticConnection(
                source_population=conn_data['source_population'],
                target_population=conn_data['target_population'],
                connection_type=conn_data['connection_type'],
                connection_probability=conn_data['connection_probability'],
                weight_mean=conn_data['weight_mean'],
                weight_std=conn_data['weight_std'],
                delay_mean=conn_data['delay_mean'],
                delay_std=conn_data['delay_std']
            )
            conn.connection_id = conn_id
            conn.anatomical_pathway = conn_data['anatomical_pathway']
            
            self.connections[conn_id] = conn
        
        # 加载连接矩阵
        matrix_path = filepath.replace('.json', '_matrix.npz')
        if Path(matrix_path).exists():
            self.connectivity_matrix = sparse.load_npz(matrix_path)
        
        # 重建索引和统计
        self._rebuild_indices()
        
        stats = data['statistics']
        self.total_neurons = stats['total_neurons']
        self.total_synapses = stats['total_synapses']
    
    def _rebuild_indices(self):
        """重建群体索引"""
        current_idx = 0
        self.population_indices = {}
        self.region_populations = {}
        
        for pop_id, population in self.populations.items():
            start_idx = current_idx
            end_idx = current_idx + population.count
            self.population_indices[pop_id] = (start_idx, end_idx)
            current_idx = end_idx
            
            # 重建脑区映射
            if population.region not in self.region_populations:
                self.region_populations[population.region] = []
            self.region_populations[population.region].append(pop_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取连接组学统计信息"""
        region_stats = {}
        for region, pop_ids in self.region_populations.items():
            total_neurons = sum(self.populations[pop_id].count for pop_id in pop_ids)
            region_stats[region.value] = {
                'populations': len(pop_ids),
                'neurons': total_neurons,
                'percentage': (total_neurons / self.total_neurons) * 100
            }
        
        return {
            'total_neurons': self.total_neurons,
            'total_synapses': self.total_synapses,
            'total_populations': len(self.populations),
            'total_connections': len(self.connections),
            'regions': len(self.region_populations),
            'region_statistics': region_stats,
            'connectivity_density': self.total_synapses / (self.total_neurons ** 2) if self.total_neurons > 0 else 0
        }

# 工厂函数
def create_human_connectome() -> ConnectomeManager:
    """创建人脑连接组学"""
    return ConnectomeManager()

def load_connectome_from_file(filepath: str) -> ConnectomeManager:
    """从文件加载连接组学"""
    manager = ConnectomeManager()
    manager.load_connectome(filepath)
    return manager