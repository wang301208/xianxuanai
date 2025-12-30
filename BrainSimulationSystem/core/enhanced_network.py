"""
增强的网络系统

扩展原有 network.py，支持：
- 区域级配置（多层、长程连接、稀疏矩阵表示）
- 丘脑-皮层、海马环路、基底节等完整线路
- 多类型神经元（多分段 Hodgkin-Huxley、L23/5/6 pyramidal、interneuron 等）
- 图数据库或分层配置文件来生成大规模网络结构
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .enhanced_brain_config import EnhancedBrainConfig, BrainRegion, CellType

logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    """连接类型枚举"""
    FEEDFORWARD = "feedforward"
    FEEDBACK = "feedback"
    LATERAL = "lateral"
    RECURRENT = "recurrent"
    CROSS_MODAL = "cross_modal"
    THALAMOCORTICAL = "thalamocortical"
    CORTICOTHALAMIC = "corticothalamic"
    HIPPOCAMPAL_TRISYNAPTIC = "hippocampal_trisynaptic"
    BASAL_GANGLIA_DIRECT = "basal_ganglia_direct"
    BASAL_GANGLIA_INDIRECT = "basal_ganglia_indirect"
    CEREBELLAR_PARALLEL = "cerebellar_parallel"
    CEREBELLAR_CLIMBING = "cerebellar_climbing"

class NeuronModel(Enum):
    """神经元模型类型"""
    LEAKY_INTEGRATE_FIRE = "lif"
    ADAPTIVE_EXPONENTIAL = "adex"
    HODGKIN_HUXLEY = "hh"
    MULTI_COMPARTMENT_HH = "multi_hh"
    IZHIKEVICH = "izhikevich"
    CONDUCTANCE_BASED = "conductance"

@dataclass
class NeuronPopulation:
    """神经元群体"""
    id: str
    region: str
    subregion: str
    cell_type: CellType
    size: int
    model: NeuronModel
    parameters: Dict[str, Any]
    spatial_coordinates: Optional[np.ndarray] = None
    layer: Optional[str] = None
    
    def __post_init__(self):
        if self.spatial_coordinates is None:
            # 生成随机空间坐标
            self.spatial_coordinates = np.random.rand(self.size, 3) * 1000  # μm

@dataclass
class Connection:
    """连接定义"""
    source_population: str
    target_population: str
    connection_type: ConnectionType
    weight_matrix: sp.csr_matrix
    delay_matrix: Optional[np.ndarray] = None
    plasticity_rule: Optional[str] = None
    
    def get_connection_strength(self) -> float:
        """获取连接强度"""
        return float(np.mean(self.weight_matrix.data))
    
    def get_sparsity(self) -> float:
        """获取稀疏度"""
        total_possible = self.weight_matrix.shape[0] * self.weight_matrix.shape[1]
        actual_connections = self.weight_matrix.nnz
        return 1.0 - (actual_connections / total_possible)

class NetworkTopology:
    """网络拓扑结构管理"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.region_hierarchy = {}
        self.distance_matrix = None
    
    def add_region(self, region: str, parent: Optional[str] = None, 
                   coordinates: Optional[Tuple[float, float, float]] = None):
        """添加脑区"""
        self.graph.add_node(region, coordinates=coordinates, parent=parent)
        if parent:
            self.region_hierarchy[region] = parent
    
    def add_connection(self, source: str, target: str, 
                      connection_type: ConnectionType, strength: float):
        """添加区域间连接"""
        self.graph.add_edge(source, target, 
                           connection_type=connection_type, 
                           strength=strength)
    
    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """获取最短路径"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
    
    def compute_distance_matrix(self):
        """计算距离矩阵"""
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        self.distance_matrix = np.full((n_nodes, n_nodes), np.inf)
        
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i == j:
                    self.distance_matrix[i, j] = 0
                else:
                    try:
                        path_length = nx.shortest_path_length(self.graph, source, target)
                        self.distance_matrix[i, j] = path_length
                    except nx.NetworkXNoPath:
                        pass
    
    def get_hierarchical_structure(self) -> Dict[str, List[str]]:
        """获取层次结构"""
        hierarchy = {}
        for region, parent in self.region_hierarchy.items():
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(region)
        return hierarchy

class SparseConnectivityGenerator:
    """稀疏连接生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_sparse(self, source_size: int, target_size: int, 
                              connection_probability: float, 
                              weight_distribution: str = "normal",
                              weight_params: Dict[str, float] = None) -> sp.csr_matrix:
        """生成随机稀疏连接"""
        if weight_params is None:
            weight_params = {"mean": 0.5, "std": 0.1}
        
        # 生成连接掩码
        mask = np.random.rand(source_size, target_size) < connection_probability
        n_connections = np.sum(mask)
        
        if n_connections == 0:
            return sp.csr_matrix((source_size, target_size))
        
        # 生成权重
        if weight_distribution == "normal":
            weights = np.random.normal(weight_params["mean"], 
                                     weight_params["std"], 
                                     n_connections)
        elif weight_distribution == "uniform":
            weights = np.random.uniform(weight_params.get("min", 0), 
                                      weight_params.get("max", 1), 
                                      n_connections)
        elif weight_distribution == "exponential":
            weights = np.random.exponential(weight_params.get("scale", 1), 
                                          n_connections)
        else:
            weights = np.ones(n_connections) * weight_params.get("value", 0.5)
        
        # 确保权重为正
        weights = np.abs(weights)
        
        # 创建稀疏矩阵
        rows, cols = np.where(mask)
        return sp.csr_matrix((weights, (rows, cols)), 
                           shape=(source_size, target_size))
    
    def generate_distance_dependent(self, source_coords: np.ndarray, 
                                   target_coords: np.ndarray,
                                   connection_kernel: str = "exponential",
                                   kernel_params: Dict[str, float] = None) -> sp.csr_matrix:
        """生成距离依赖的连接"""
        if kernel_params is None:
            kernel_params = {"lambda": 100.0, "max_distance": 500.0}
        
        # 计算距离矩阵
        distances = np.linalg.norm(
            source_coords[:, np.newaxis, :] - target_coords[np.newaxis, :, :], 
            axis=2
        )
        
        # 应用连接核函数
        if connection_kernel == "exponential":
            probabilities = np.exp(-distances / kernel_params["lambda"])
        elif connection_kernel == "gaussian":
            sigma = kernel_params.get("sigma", 50.0)
            probabilities = np.exp(-distances**2 / (2 * sigma**2))
        elif connection_kernel == "power_law":
            alpha = kernel_params.get("alpha", 2.0)
            probabilities = 1.0 / (1.0 + (distances / kernel_params["lambda"])**alpha)
        else:
            probabilities = np.ones_like(distances)
        
        # 应用最大距离限制
        max_dist = kernel_params.get("max_distance", np.inf)
        probabilities[distances > max_dist] = 0
        
        # 生成连接
        mask = np.random.rand(*distances.shape) < probabilities
        weights = probabilities[mask]
        
        rows, cols = np.where(mask)
        return sp.csr_matrix((weights, (rows, cols)), shape=distances.shape)
    
    def generate_small_world(self, size: int, k: int, p: float, 
                           weight_scale: float = 1.0) -> sp.csr_matrix:
        """生成小世界网络连接"""
        # 使用NetworkX生成小世界图
        G = nx.watts_strogatz_graph(size, k, p)
        
        # 转换为稀疏矩阵
        adj_matrix = nx.adjacency_matrix(G)
        
        # 添加随机权重
        weights = np.random.exponential(weight_scale, adj_matrix.nnz)
        adj_matrix.data = weights
        
        return adj_matrix.tocsr()
    
    def generate_scale_free(self, size: int, m: int, 
                           weight_scale: float = 1.0) -> sp.csr_matrix:
        """生成无标度网络连接"""
        # 使用NetworkX生成无标度图
        G = nx.barabasi_albert_graph(size, m)
        
        # 转换为稀疏矩阵
        adj_matrix = nx.adjacency_matrix(G)
        
        # 添加随机权重
        weights = np.random.exponential(weight_scale, adj_matrix.nnz)
        adj_matrix.data = weights
        
        return adj_matrix.tocsr()

class MultiCompartmentNeuron:
    """多分段神经元模型"""
    
    def __init__(self, compartments: List[str], 
                 morphology: Dict[str, Any],
                 parameters: Dict[str, Dict[str, float]]):
        self.compartments = compartments
        self.morphology = morphology
        self.parameters = parameters
        self.n_compartments = len(compartments)
        
        # 初始化状态变量
        self.voltage = np.full(self.n_compartments, -70.0)  # mV
        self.sodium_m = np.zeros(self.n_compartments)
        self.sodium_h = np.ones(self.n_compartments)
        self.potassium_n = np.zeros(self.n_compartments)
        
        # 构建分段间连接矩阵
        self.coupling_matrix = self._build_coupling_matrix()
    
    def _build_coupling_matrix(self) -> np.ndarray:
        """构建分段间耦合矩阵"""
        coupling = np.zeros((self.n_compartments, self.n_compartments))
        
        # 基于形态学信息构建耦合
        for i, comp1 in enumerate(self.compartments):
            for j, comp2 in enumerate(self.compartments):
                if i != j:
                    # 检查是否相邻
                    if self._are_adjacent(comp1, comp2):
                        # 计算耦合强度
                        g_coupling = self._compute_coupling_conductance(comp1, comp2)
                        coupling[i, j] = g_coupling
        
        return coupling
    
    def _are_adjacent(self, comp1: str, comp2: str) -> bool:
        """检查两个分段是否相邻"""
        # 简化的相邻性检查
        adjacency_rules = {
            "soma": ["dendrite", "axon"],
            "dendrite": ["soma", "dendrite"],
            "axon": ["soma", "axon_hillock"],
            "axon_hillock": ["soma", "axon"]
        }
        return comp2 in adjacency_rules.get(comp1, [])
    
    def _compute_coupling_conductance(self, comp1: str, comp2: str) -> float:
        """计算分段间耦合电导"""
        # 基于分段直径和长度计算
        diameter1 = self.morphology.get(f"{comp1}_diameter", 1.0)  # μm
        diameter2 = self.morphology.get(f"{comp2}_diameter", 1.0)  # μm
        length = self.morphology.get(f"{comp1}_{comp2}_length", 10.0)  # μm
        
        # 轴向电阻 (Ω·cm)
        Ra = 100.0
        
        # 计算耦合电导 (μS)
        area = np.pi * (diameter1 + diameter2) / 4  # μm²
        g_coupling = area / (Ra * length * 1e-4)  # 转换单位
        
        return g_coupling
    
    def update(self, dt: float, input_current: np.ndarray) -> np.ndarray:
        """更新神经元状态"""
        # Hodgkin-Huxley方程组
        for i in range(self.n_compartments):
            params = self.parameters[self.compartments[i]]
            
            # 离子通道动力学
            alpha_m = self._alpha_m(self.voltage[i])
            beta_m = self._beta_m(self.voltage[i])
            alpha_h = self._alpha_h(self.voltage[i])
            beta_h = self._beta_h(self.voltage[i])
            alpha_n = self._alpha_n(self.voltage[i])
            beta_n = self._beta_n(self.voltage[i])
            
            # 更新门控变量
            self.sodium_m[i] += dt * (alpha_m * (1 - self.sodium_m[i]) - 
                                     beta_m * self.sodium_m[i])
            self.sodium_h[i] += dt * (alpha_h * (1 - self.sodium_h[i]) - 
                                     beta_h * self.sodium_h[i])
            self.potassium_n[i] += dt * (alpha_n * (1 - self.potassium_n[i]) - 
                                        beta_n * self.potassium_n[i])
            
            # 计算离子电流
            g_Na = params["g_Na_max"] * self.sodium_m[i]**3 * self.sodium_h[i]
            g_K = params["g_K_max"] * self.potassium_n[i]**4
            g_L = params["g_L"]
            
            I_Na = g_Na * (self.voltage[i] - params["E_Na"])
            I_K = g_K * (self.voltage[i] - params["E_K"])
            I_L = g_L * (self.voltage[i] - params["E_L"])
            
            # 分段间耦合电流
            I_coupling = np.sum(self.coupling_matrix[i, :] * 
                               (self.voltage[i] - self.voltage))
            
            # 更新电压
            I_total = input_current[i] - I_Na - I_K - I_L - I_coupling
            self.voltage[i] += dt * I_total / params["C_m"]
        
        return self.voltage.copy()
    
    def _alpha_m(self, V: float) -> float:
        """钠通道激活门控变量α函数"""
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    
    def _beta_m(self, V: float) -> float:
        """钠通道激活门控变量β函数"""
        return 4 * np.exp(-(V + 65) / 18)
    
    def _alpha_h(self, V: float) -> float:
        """钠通道失活门控变量α函数"""
        return 0.07 * np.exp(-(V + 65) / 20)
    
    def _beta_h(self, V: float) -> float:
        """钠通道失活门控变量β函数"""
        return 1 / (1 + np.exp(-(V + 35) / 10))
    
    def _alpha_n(self, V: float) -> float:
        """钾通道门控变量α函数"""
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
    def _beta_n(self, V: float) -> float:
        """钾通道门控变量β函数"""
        return 0.125 * np.exp(-(V + 65) / 80)

class EnhancedNetworkBuilder:
    """增强的网络构建器"""
    
    def __init__(self, config: EnhancedBrainConfig):
        self.config = config
        self.populations = {}
        self.connections = {}
        self.topology = NetworkTopology()
        self.connectivity_generator = SparseConnectivityGenerator()
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def build_network(self, selected_regions: List[BrainRegion] = None) -> Dict[str, Any]:
        """构建完整网络"""
        if selected_regions is None:
            selected_regions = list(self.config.regions.keys())
        
        self.logger.info(f"开始构建网络，包含 {len(selected_regions)} 个脑区")
        
        # 1. 创建神经元群体
        self._create_populations(selected_regions)
        
        # 2. 构建拓扑结构
        self._build_topology(selected_regions)
        
        # 3. 生成连接
        self._generate_connections(selected_regions)
        
        # 4. 优化网络结构
        self._optimize_network()
        
        network_info = {
            'populations': self.populations,
            'connections': self.connections,
            'topology': self.topology,
            'statistics': self._compute_network_statistics()
        }
        
        self.logger.info("网络构建完成")
        return network_info
    
    def _create_populations(self, regions: List[BrainRegion]):
        """创建神经元群体"""
        population_id = 0
        
        for region_enum in regions:
            region_config = self.config.get_region_config(region_enum)
            if not region_config or not region_config.enabled:
                continue
            
            for subregion_name, subregion in region_config.subregions.items():
                for cell_type, cell_config in subregion.cell_types.items():
                    # 计算该细胞类型的数量
                    cell_count = int(subregion.neuron_count * cell_config.proportion)
                    
                    if cell_count > 0:
                        # 选择神经元模型
                        model = self._select_neuron_model(cell_type)
                        
                        # 获取模型参数
                        parameters = self._get_model_parameters(cell_type, model)
                        
                        # 创建群体
                        pop_id = f"{region_config.name}_{subregion_name}_{cell_type.value}_{population_id}"
                        population = NeuronPopulation(
                            id=pop_id,
                            region=region_config.name,
                            subregion=subregion_name,
                            cell_type=cell_type,
                            size=cell_count,
                            model=model,
                            parameters=parameters
                        )
                        
                        self.populations[pop_id] = population
                        population_id += 1
        
        self.logger.info(f"创建了 {len(self.populations)} 个神经元群体")
    
    def _select_neuron_model(self, cell_type: CellType) -> NeuronModel:
        """选择合适的神经元模型"""
        # 基于细胞类型选择模型
        model_map = {
            CellType.L5_PYRAMIDAL: NeuronModel.MULTI_COMPARTMENT_HH,
            CellType.L23_PYRAMIDAL: NeuronModel.ADAPTIVE_EXPONENTIAL,
            CellType.PV_INTERNEURON: NeuronModel.LEAKY_INTEGRATE_FIRE,
            CellType.SST_INTERNEURON: NeuronModel.ADAPTIVE_EXPONENTIAL,
            CellType.CA3_PYRAMIDAL: NeuronModel.HODGKIN_HUXLEY,
            CellType.CA1_PYRAMIDAL: NeuronModel.HODGKIN_HUXLEY,
            CellType.PURKINJE_CELLS: NeuronModel.MULTI_COMPARTMENT_HH,
            CellType.GRANULE_CELLS: NeuronModel.LEAKY_INTEGRATE_FIRE
        }
        
        return model_map.get(cell_type, NeuronModel.LEAKY_INTEGRATE_FIRE)
    
    def _get_model_parameters(self, cell_type: CellType, model: NeuronModel) -> Dict[str, Any]:
        """获取模型参数"""
        # 基础参数
        base_params = {
            "C_m": 200.0,  # pF
            "g_L": 10.0,   # nS
            "E_L": -70.0,  # mV
            "V_th": -50.0, # mV
            "V_reset": -60.0, # mV
            "tau_ref": 2.0 # ms
        }
        
        # 根据细胞类型调整参数
        if cell_type == CellType.L5_PYRAMIDAL:
            base_params.update({
                "C_m": 340.0,
                "g_L": 8.0,
                "V_th": -45.0
            })
        elif cell_type == CellType.PV_INTERNEURON:
            base_params.update({
                "C_m": 115.0,
                "g_L": 15.0,
                "tau_ref": 1.0
            })
        
        # 根据模型类型添加特定参数
        if model == NeuronModel.HODGKIN_HUXLEY:
            base_params.update({
                "g_Na_max": 120.0,  # mS/cm²
                "g_K_max": 36.0,    # mS/cm²
                "E_Na": 50.0,       # mV
                "E_K": -77.0        # mV
            })
        elif model == NeuronModel.ADAPTIVE_EXPONENTIAL:
            base_params.update({
                "Delta_T": 2.0,     # mV
                "a": 2.0,           # nS
                "b": 60.0,          # pA
                "tau_w": 300.0      # ms
            })
        elif model == NeuronModel.MULTI_COMPARTMENT_HH:
            # 多分段参数
            base_params.update({
                "compartments": ["soma", "dendrite", "axon"],
                "morphology": {
                    "soma_diameter": 20.0,      # μm
                    "dendrite_diameter": 2.0,   # μm
                    "axon_diameter": 1.0,       # μm
                    "dendrite_length": 200.0,   # μm
                    "axon_length": 500.0        # μm
                }
            })
        
        return base_params
    
    def _build_topology(self, regions: List[BrainRegion]):
        """构建网络拓扑"""
        # 添加脑区节点
        for region_enum in regions:
            region_config = self.config.get_region_config(region_enum)
            if region_config and region_config.enabled:
                # 生成随机坐标（实际应用中应使用真实解剖坐标）
                coords = tuple(np.random.rand(3) * 100)
                self.topology.add_region(region_config.name, coordinates=coords)
                
                # 添加子区域
                for subregion_name in region_config.subregions.keys():
                    subregion_coords = tuple(coords[i] + np.random.rand() * 10 for i in range(3))
                    self.topology.add_region(f"{region_config.name}_{subregion_name}", 
                                           parent=region_config.name,
                                           coordinates=subregion_coords)
        
        # 添加区域间连接
        self._add_interregion_connections(regions)
        
        # 计算距离矩阵
        self.topology.compute_distance_matrix()
    
    def _add_interregion_connections(self, regions: List[BrainRegion]):
        """添加区域间连接"""
        # 定义典型的脑区连接模式
        connection_patterns = {
            # 丘脑-皮层连接
            (BrainRegion.THALAMUS, BrainRegion.NEOCORTEX): {
                "type": ConnectionType.THALAMOCORTICAL,
                "strength": 0.8
            },
            (BrainRegion.NEOCORTEX, BrainRegion.THALAMUS): {
                "type": ConnectionType.CORTICOTHALAMIC,
                "strength": 0.6
            },
            
            # 海马-皮层连接
            (BrainRegion.HIPPOCAMPUS, BrainRegion.NEOCORTEX): {
                "type": ConnectionType.FEEDFORWARD,
                "strength": 0.5
            },
            (BrainRegion.NEOCORTEX, BrainRegion.HIPPOCAMPUS): {
                "type": ConnectionType.FEEDBACK,
                "strength": 0.4
            },
            
            # 基底节-皮层连接
            (BrainRegion.BASAL_GANGLIA, BrainRegion.NEOCORTEX): {
                "type": ConnectionType.BASAL_GANGLIA_DIRECT,
                "strength": 0.7
            },
            
            # 小脑-皮层连接
            (BrainRegion.CEREBELLUM, BrainRegion.NEOCORTEX): {
                "type": ConnectionType.CEREBELLAR_PARALLEL,
                "strength": 0.6
            }
        }
        
        for (source_region, target_region), conn_info in connection_patterns.items():
            if source_region in regions and target_region in regions:
                source_config = self.config.get_region_config(source_region)
                target_config = self.config.get_region_config(target_region)
                
                if (source_config and source_config.enabled and 
                    target_config and target_config.enabled):
                    self.topology.add_connection(
                        source_config.name,
                        target_config.name,
                        conn_info["type"],
                        conn_info["strength"]
                    )
    
    def _generate_connections(self, regions: List[BrainRegion]):
        """生成详细连接"""
        connection_id = 0
        
        # 区域内连接
        for region_enum in regions:
            region_config = self.config.get_region_config(region_enum)
            if not region_config or not region_config.enabled:
                continue
            
            self._generate_intraregion_connections(region_config, connection_id)
        
        # 区域间连接
        self._generate_interregion_connections(regions)
    
    def _generate_intraregion_connections(self, region_config, connection_id: int):
        """生成区域内连接"""
        region_populations = [pop for pop in self.populations.values() 
                            if pop.region == region_config.name]
        
        for source_pop in region_populations:
            for target_pop in region_populations:
                if source_pop.id != target_pop.id:
                    # 确定连接类型和强度
                    conn_type, strength = self._determine_connection_properties(
                        source_pop, target_pop
                    )
                    
                    if strength > 0:
                        # 生成连接矩阵
                        weight_matrix = self._generate_connection_matrix(
                            source_pop, target_pop, strength
                        )
                        
                        if weight_matrix.nnz > 0:
                            # 生成延迟矩阵
                            delay_matrix = self._generate_delay_matrix(
                                source_pop, target_pop
                            )
                            
                            # 创建连接对象
                            connection = Connection(
                                source_population=source_pop.id,
                                target_population=target_pop.id,
                                connection_type=conn_type,
                                weight_matrix=weight_matrix,
                                delay_matrix=delay_matrix,
                                plasticity_rule=self._get_plasticity_rule(source_pop, target_pop)
                            )
                            
                            self.connections[f"conn_{connection_id}"] = connection
                            connection_id += 1
    
    def _generate_interregion_connections(self, regions: List[BrainRegion]):
        """生成区域间连接"""
        # 实现区域间的长程连接
        for source_region in regions:
            for target_region in regions:
                if source_region != target_region:
                    self._create_long_range_connections(source_region, target_region)
    
    def _determine_connection_properties(self, source_pop: NeuronPopulation, 
                                       target_pop: NeuronPopulation) -> Tuple[ConnectionType, float]:
        """确定连接属性"""
        # 基于细胞类型和层次确定连接
        if source_pop.subregion == target_pop.subregion:
            # 同一子区域内的连接
            if source_pop.layer == target_pop.layer:
                return ConnectionType.LATERAL, 0.1
            else:
                return ConnectionType.FEEDFORWARD, 0.15
        else:
            # 不同子区域间的连接
            return ConnectionType.CROSS_MODAL, 0.05
    
    def _generate_connection_matrix(self, source_pop: NeuronPopulation, 
                                   target_pop: NeuronPopulation, 
                                   strength: float) -> sp.csr_matrix:
        """生成连接矩阵"""
        # 基于距离和细胞类型生成连接
        if hasattr(source_pop, 'spatial_coordinates') and hasattr(target_pop, 'spatial_coordinates'):
            # 使用距离依赖连接
            return self.connectivity_generator.generate_distance_dependent(
                source_pop.spatial_coordinates,
                target_pop.spatial_coordinates,
                kernel_params={"lambda": 100.0 * strength, "max_distance": 500.0}
            )
        else:
            # 使用随机连接
            connection_prob = min(0.2 * strength, 0.1)  # 限制连接概率
            return self.connectivity_generator.generate_random_sparse(
                source_pop.size,
                target_pop.size,
                connection_prob,
                weight_params={"mean": strength, "std": strength * 0.2}
            )
    
    def _generate_delay_matrix(self, source_pop: NeuronPopulation, 
                              target_pop: NeuronPopulation) -> np.ndarray:
        """生成延迟矩阵"""
        # 基于距离计算传导延迟
        if hasattr(source_pop, 'spatial_coordinates') and hasattr(target_pop, 'spatial_coordinates'):
            distances = np.linalg.norm(
                source_pop.spatial_coordinates[:, np.newaxis, :] - 
                target_pop.spatial_coordinates[np.newaxis, :, :], 
                axis=2
            )
            
            # 传导速度 (m/s)
            conduction_velocity = 1.0  # 1 m/s for unmyelinated axons
            
            # 计算延迟 (ms)
            delays = distances * 1e-6 / conduction_velocity * 1000
            
            # 添加突触延迟
            synaptic_delay = 0.5  # ms
            delays += synaptic_delay
            
            return delays
        else:
            # 默认延迟
            return np.full((source_pop.size, target_pop.size), 1.5)
    
    def _get_plasticity_rule(self, source_pop: NeuronPopulation, 
                           target_pop: NeuronPopulation) -> Optional[str]:
        """获取可塑性规则"""
        # 基于连接类型确定可塑性规则
        if (source_pop.cell_type in [CellType.L23_PYRAMIDAL, CellType.L5_PYRAMIDAL] and
            target_pop.cell_type in [CellType.L23_PYRAMIDAL, CellType.L5_PYRAMIDAL]):
            return "stdp"  # 兴奋性-兴奋性连接使用STDP
        elif source_pop.cell_type == CellType.PV_INTERNEURON:
            return "homeostatic"  # 抑制性连接使用稳态可塑性
        else:
            return None
    
    def _create_long_range_connections(self, source_region: BrainRegion, 
                                     target_region: BrainRegion):
        """创建长程连接"""
        # 获取区域配置
        source_config = self.config.get_region_config(source_region)
        target_config = self.config.get_region_config(target_region)
        
        if not (source_config and target_config and 
                source_config.enabled and target_config.enabled):
            return
        
        # 获取相关的神经元群体
        source_populations = [pop for pop in self.populations.values() 
                            if pop.region == source_config.name]
        target_populations = [pop for pop in self.populations.values() 
                            if pop.region == target_config.name]
        
        # 创建选择性长程连接
        for source_pop in source_populations:
            for target_pop in target_populations:
                # 只在特定细胞类型间建立长程连接
                if self._should_create_long_range_connection(source_pop, target_pop):
                    strength = self._get_long_range_strength(source_region, target_region)
                    
                    # 生成稀疏长程连接
                    weight_matrix = self.connectivity_generator.generate_random_sparse(
                        source_pop.size,
                        target_pop.size,
                        0.01,  # 低连接概率
                        weight_params={"mean": strength, "std": strength * 0.3}
                    )
                    
                    if weight_matrix.nnz > 0:
                        # 长程连接有更大的延迟
                        delay_matrix = np.full((source_pop.size, target_pop.size), 
                                             5.0 + np.random.exponential(2.0))
                        
                        connection = Connection(
                            source_population=source_pop.id,
                            target_population=target_pop.id,
                            connection_type=ConnectionType.FEEDFORWARD,
                            weight_matrix=weight_matrix,
                            delay_matrix=delay_matrix,
                            plasticity_rule="stdp"
                        )
                        
                        conn_id = f"long_range_{source_pop.id}_{target_pop.id}"
                        self.connections[conn_id] = connection
    
    def _should_create_long_range_connection(self, source_pop: NeuronPopulation, 
                                           target_pop: NeuronPopulation) -> bool:
        """判断是否应该创建长程连接"""
        # 主要在锥体细胞间建立长程连接
        pyramidal_types = [CellType.L23_PYRAMIDAL, CellType.L5_PYRAMIDAL, CellType.L6_PYRAMIDAL]
        return (source_pop.cell_type in pyramidal_types and 
                target_pop.cell_type in pyramidal_types)
    
    def _get_long_range_strength(self, source_region: BrainRegion, 
                               target_region: BrainRegion) -> float:
        """获取长程连接强度"""
        # 基于解剖学知识的连接强度
        strength_map = {
            (BrainRegion.NEOCORTEX, BrainRegion.HIPPOCAMPUS): 0.3,
            (BrainRegion.HIPPOCAMPUS, BrainRegion.NEOCORTEX): 0.2,
            (BrainRegion.THALAMUS, BrainRegion.NEOCORTEX): 0.5,
            (BrainRegion.NEOCORTEX, BrainRegion.THALAMUS): 0.4,
            (BrainRegion.BASAL_GANGLIA, BrainRegion.NEOCORTEX): 0.4,
            (BrainRegion.CEREBELLUM, BrainRegion.NEOCORTEX): 0.3
        }
        
        return strength_map.get((source_region, target_region), 0.1)
    
    def _optimize_network(self):
        """优化网络结构"""
        self.logger.info("开始网络优化")
        
        # 1. 移除弱连接
        self._remove_weak_connections(threshold=0.01)
        
        # 2. 平衡兴奋抑制比例
        self._balance_excitation_inhibition()
        
        # 3. 优化稀疏性
        self._optimize_sparsity()
        
        self.logger.info("网络优化完成")
    
    def _remove_weak_connections(self, threshold: float):
        """移除弱连接"""
        connections_to_remove = []
        
        for conn_id, connection in self.connections.items():
            if connection.get_connection_strength() < threshold:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        self.logger.info(f"移除了 {len(connections_to_remove)} 个弱连接")
    
    def _balance_excitation_inhibition(self):
        """平衡兴奋抑制比例"""
        # 计算每个群体的兴奋抑制输入比例
        for pop_id, population in self.populations.items():
            excitatory_input = 0
            inhibitory_input = 0
            
            for connection in self.connections.values():
                if connection.target_population == pop_id:
                    source_pop = self.populations[connection.source_population]
                    strength = connection.get_connection_strength()
                    
                    if self._is_excitatory_cell_type(source_pop.cell_type):
                        excitatory_input += strength
                    else:
                        inhibitory_input += strength
            
            # 调整抑制性连接强度以维持平衡
            target_ratio = 0.2  # 抑制/兴奋 = 0.2
            if excitatory_input > 0:
                desired_inhibitory = excitatory_input * target_ratio
                if inhibitory_input > 0:
                    scaling_factor = desired_inhibitory / inhibitory_input
                    self._scale_inhibitory_connections(pop_id, scaling_factor)
    
    def _is_excitatory_cell_type(self, cell_type: CellType) -> bool:
        """判断是否为兴奋性细胞类型"""
        excitatory_types = [
            CellType.L23_PYRAMIDAL, CellType.L4_SPINY_STELLATE,
            CellType.L5_PYRAMIDAL, CellType.L6_PYRAMIDAL,
            CellType.CA3_PYRAMIDAL, CellType.CA1_PYRAMIDAL,
            CellType.GRANULE_CELLS, CellType.GRANULE_CELLS_CB
        ]
        return cell_type in excitatory_types
    
    def _scale_inhibitory_connections(self, target_pop_id: str, scaling_factor: float):
        """缩放抑制性连接"""
        for connection in self.connections.values():
            if connection.target_population == target_pop_id:
                source_pop = self.populations[connection.source_population]
                if not self._is_excitatory_cell_type(source_pop.cell_type):
                    connection.weight_matrix.data *= scaling_factor
    
    def _optimize_sparsity(self):
        """优化稀疏性"""
        for connection in self.connections.values():
            # 移除极小的权重
            small_weights = np.abs(connection.weight_matrix.data) < 0.001
            connection.weight_matrix.data[small_weights] = 0
            connection.weight_matrix.eliminate_zeros()
    
    def _compute_network_statistics(self) -> Dict[str, Any]:
        """计算网络统计信息"""
        total_neurons = sum(pop.size for pop in self.populations.values())
        total_connections = sum(conn.weight_matrix.nnz for conn in self.connections.values())
        
        # 计算平均连接度
        avg_degree = total_connections / total_neurons if total_neurons > 0 else 0
        
        # 计算稀疏度
        total_possible = sum(
            pop1.size * pop2.size 
            for pop1 in self.populations.values() 
            for pop2 in self.populations.values()
        )
        sparsity = 1.0 - (total_connections / total_possible) if total_possible > 0 else 1.0
        
        # 计算兴奋抑制比例
        excitatory_neurons = sum(
            pop.size for pop in self.populations.values() 
            if self._is_excitatory_cell_type(pop.cell_type)
        )
        inhibitory_neurons = total_neurons - excitatory_neurons
        ei_ratio = excitatory_neurons / inhibitory_neurons if inhibitory_neurons > 0 else float('inf')
        
        return {
            'total_neurons': total_neurons,
            'total_connections': total_connections,
            'average_degree': avg_degree,
            'sparsity': sparsity,
            'excitatory_neurons': excitatory_neurons,
            'inhibitory_neurons': inhibitory_neurons,
            'ei_ratio': ei_ratio,
            'num_populations': len(self.populations),
            'num_connection_types': len(set(conn.connection_type for conn in self.connections.values()))
        }
    
    def export_network(self, filepath: str, format: str = "json"):
        """导出网络结构"""
        if format == "json":
            self._export_json(filepath)
        elif format == "graphml":
            self._export_graphml(filepath)
        elif format == "hdf5":
            self._export_hdf5(filepath)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_json(self, filepath: str):
        """导出为JSON格式"""
        network_data = {
            'populations': {},
            'connections': {},
            'statistics': self._compute_network_statistics()
        }
        
        # 导出群体信息
        for pop_id, population in self.populations.items():
            network_data['populations'][pop_id] = {
                'region': population.region,
                'subregion': population.subregion,
                'cell_type': population.cell_type.value,
                'size': population.size,
                'model': population.model.value,
                'parameters': population.parameters
            }
        
        # 导出连接信息
        for conn_id, connection in self.connections.items():
            network_data['connections'][conn_id] = {
                'source': connection.source_population,
                'target': connection.target_population,
                'type': connection.connection_type.value,
                'strength': connection.get_connection_strength(),
                'sparsity': connection.get_sparsity(),
                'num_synapses': connection.weight_matrix.nnz
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2, ensure_ascii=False)
    
    def _export_graphml(self, filepath: str):
        """导出为GraphML格式"""
        G = nx.DiGraph()
        
        # 添加节点（神经元群体）
        for pop_id, population in self.populations.items():
            G.add_node(pop_id, 
                      region=population.region,
                      cell_type=population.cell_type.value,
                      size=population.size)
        
        # 添加边（连接）
        for connection in self.connections.values():
            G.add_edge(connection.source_population,
                      connection.target_population,
                      weight=connection.get_connection_strength(),
                      type=connection.connection_type.value)
        
        nx.write_graphml(G, filepath)
    
    def _export_hdf5(self, filepath: str):
        """导出为HDF5格式"""
        try:
            import h5py
        except ImportError:
            raise ImportError("需要安装 h5py 库来支持 HDF5 导出")
        
        with h5py.File(filepath, 'w') as f:
            # 保存群体信息
            pop_group = f.create_group('populations')
            for pop_id, population in self.populations.items():
                pop_subgroup = pop_group.create_group(pop_id)
                pop_subgroup.attrs['region'] = population.region
                pop_subgroup.attrs['cell_type'] = population.cell_type.value
                pop_subgroup.attrs['size'] = population.size
                
                if population.spatial_coordinates is not None:
                    pop_subgroup.create_dataset('coordinates', 
                                              data=population.spatial_coordinates)
            
            # 保存连接信息
            conn_group = f.create_group('connections')
            for conn_id, connection in self.connections.items():
                conn_subgroup = conn_group.create_group(conn_id)
                conn_subgroup.attrs['source'] = connection.source_population
                conn_subgroup.attrs['target'] = connection.target_population
                conn_subgroup.attrs['type'] = connection.connection_type.value
                
                # 保存稀疏矩阵
                conn_subgroup.create_dataset('weights', data=connection.weight_matrix.data)
                conn_subgroup.create_dataset('indices', data=connection.weight_matrix.indices)
                conn_subgroup.create_dataset('indptr', data=connection.weight_matrix.indptr)
                conn_subgroup.attrs['shape'] = connection.weight_matrix.shape
                
                if connection.delay_matrix is not None:
                    conn_subgroup.create_dataset('delays', data=connection.delay_matrix)

def create_enhanced_network(config_file: Optional[str] = None, 
                          selected_regions: Optional[List[str]] = None) -> Dict[str, Any]:
    """创建增强网络的便捷函数"""
    
    # 加载配置
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        # 这里需要从字典重建配置对象
        config = EnhancedBrainConfig()
    else:
        config = EnhancedBrainConfig()
    
    # 转换区域名称为枚举
    if selected_regions:
        region_enums = []
        for region_name in selected_regions:
            try:
                region_enum = BrainRegion(region_name)
                region_enums.append(region_enum)
            except ValueError:
                logger.warning(f"未知的脑区名称: {region_name}")
    else:
        region_enums = None
    
    # 构建网络
    builder = EnhancedNetworkBuilder(config)
    network = builder.build_network(region_enums)
    
    return network

if __name__ == "__main__":
    # 测试网络构建
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = EnhancedBrainConfig()
    
    # 选择要构建的脑区
    selected_regions = [
        BrainRegion.NEOCORTEX,
        BrainRegion.THALAMUS,
        BrainRegion.HIPPOCAMPUS
    ]
    
    # 构建网络
    builder = EnhancedNetworkBuilder(config)
    network = builder.build_network(selected_regions)
    
    # 打印统计信息
    stats = network['statistics']
    print(f"网络构建完成:")
    print(f"  总神经元数: {stats['total_neurons']:,}")
    print(f"  总连接数: {stats['total_connections']:,}")
    print(f"  平均连接度: {stats['average_degree']:.2f}")
    print(f"  稀疏度: {stats['sparsity']:.4f}")
    print(f"  兴奋/抑制比例: {stats['ei_ratio']:.2f}")
    
    # 导出网络
    builder.export_network("enhanced_brain_network.json", "json")