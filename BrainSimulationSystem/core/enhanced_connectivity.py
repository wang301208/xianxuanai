"""
增强连接层实现
Enhanced Connectivity Layer Implementation

扩展 core/network.py 支持：
- 区域级连接矩阵
- 长程轴突
- 延迟和概率分布
- 稀疏矩阵和图数据库存储
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import pickle
import json

# 可选依赖
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    neo4j = None

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

class ConnectionType(Enum):
    """连接类型枚举"""
    LOCAL = "local"                    # 局部连接
    INTERLAYER = "interlayer"         # 层间连接
    INTERCOLUMN = "intercolumn"       # 柱间连接
    INTERREGION = "interregion"       # 区域间连接
    LONG_RANGE = "long_range"         # 长程连接
    FEEDBACK = "feedback"             # 反馈连接
    FEEDFORWARD = "feedforward"       # 前馈连接

class DelayType(Enum):
    """延迟类型枚举"""
    FIXED = "fixed"                   # 固定延迟
    GAUSSIAN = "gaussian"             # 高斯分布
    EXPONENTIAL = "exponential"       # 指数分布
    DISTANCE_DEPENDENT = "distance"   # 距离依赖

@dataclass
class ConnectionParameters:
    """连接参数"""
    connection_type: ConnectionType
    probability: float = 0.1
    weight_mean: float = 1.0
    weight_std: float = 0.2
    delay_mean: float = 1.0
    delay_std: float = 0.2
    delay_type: DelayType = DelayType.GAUSSIAN
    
    # 距离依赖参数
    distance_decay: float = 0.1
    max_distance: float = 1000.0  # μm
    
    # 可塑性参数
    plasticity_enabled: bool = True
    learning_rate: float = 0.01
    
    # 轴突参数
    conduction_velocity: float = 1.0  # m/s
    myelination: bool = False

@dataclass
class AxonParameters:
    """轴突参数"""
    length: float = 1000.0           # μm
    diameter: float = 1.0            # μm
    conduction_velocity: float = 1.0  # m/s
    myelination: bool = False
    nodes_of_ranvier: int = 10
    
    # 分支参数
    branching_factor: int = 1
    branch_lengths: List[float] = field(default_factory=list)
    
    # 生理参数
    resistance: float = 100.0        # Ω
    capacitance: float = 1.0         # pF

class EnhancedConnectivityManager:
    """增强连接管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("EnhancedConnectivityManager")
        
        # 初始化组件
        self.sparse_matrix = None
        self.regional_matrix = None
        self.graph_db = None
        
        # 神经元位置信息
        self.neuron_positions = {}
        
        # 区域信息
        self.neuron_regions = {}
        self.regions = []
        
        # 连接统计
        self.connection_stats = {
            'total_connections': 0,
            'connections_by_type': {},
            'average_delay': 0.0,
            'average_weight': 0.0
        }
    
    def initialize(self, n_neurons: int, regions: List[str]):
        """初始化连接管理器"""
        
        from .sparse_matrix import SparseConnectionMatrix
        from .regional_connectivity import RegionalConnectivityMatrix
        from .graph_database import GraphDatabase
        
        self.sparse_matrix = SparseConnectionMatrix(n_neurons)
        self.regional_matrix = RegionalConnectivityMatrix(regions)
        self.regions = regions
        
        # 初始化图数据库（如果配置启用）
        graph_config = self.config.get('graph_database', {})
        if graph_config.get('enabled', False):
            self.graph_db = GraphDatabase(graph_config)
        
        self.logger.info(f"初始化连接管理器: {n_neurons} 个神经元, {len(regions)} 个区域")
    
    def set_neuron_position(self, neuron_id: int, position: Tuple[float, float, float]):
        """设置神经元位置"""
        self.neuron_positions[neuron_id] = position
    
    def set_neuron_region(self, neuron_id: int, region: str):
        """设置神经元所属区域"""
        self.neuron_regions[neuron_id] = region
    
    def add_local_connections(self, neurons: List[int], connection_params: ConnectionParameters):
        """添加局部连接"""
        
        from .probabilistic_connector import ProbabilisticConnector
        
        connector = ProbabilisticConnector(self.config.get('connector', {}))
        connections = connector.generate_connections(neurons, neurons, connection_params)
        
        for source_id, target_id, weight, delay in connections:
            self.sparse_matrix.add_connection(
                source_id, target_id, weight, delay, ConnectionType.LOCAL
            )
            
            # 添加到图数据库
            if self.graph_db:
                self.graph_db.create_connection_edge(
                    source_id, target_id,
                    {'weight': weight, 'delay': delay, 'type': 'local'}
                )
        
        self._update_statistics(connections, ConnectionType.LOCAL)
    
    def add_regional_connections(self, source_region: str, target_region: str,
                               connection_params: ConnectionParameters):
        """添加区域间连接"""
        
        from .probabilistic_connector import ProbabilisticConnector
        
        # 获取区域内的神经元
        source_neurons = [nid for nid, region in self.neuron_regions.items() 
                         if region == source_region]
        target_neurons = [nid for nid, region in self.neuron_regions.items() 
                         if region == target_region]
        
        if not source_neurons or not target_neurons:
            self.logger.warning(f"区域 {source_region} 或 {target_region} 没有神经元")
            return
        
        # 生成连接
        connector = ProbabilisticConnector(self.config.get('connector', {}))
        connections = connector.generate_connections(
            source_neurons, target_neurons, connection_params
        )
        
        # 添加长程轴突和连接
        for source_id, target_id, weight, delay in connections:
            # 创建长程轴突
            distance = self._calculate_inter_regional_distance(source_region, target_region)
            axon_params = AxonParameters(
                length=distance,
                conduction_velocity=connection_params.conduction_velocity,
                myelination=distance > 500.0  # 长距离轴突通常有髓鞘
            )
            
            self.regional_matrix.add_long_range_axon(
                source_id, source_region, target_region, axon_params
            )
            
            # 添加连接
            self.sparse_matrix.add_connection(
                source_id, target_id, weight, delay, ConnectionType.INTERREGION
            )
        
        self._update_statistics(connections, ConnectionType.INTERREGION)
    
    def _calculate_inter_regional_distance(self, source_region: str, target_region: str) -> float:
        """计算区域间距离"""
        # 简化的区域间距离计算
        region_distances = {
            ('cortex', 'thalamus'): 15000.0,  # 15mm
            ('cortex', 'hippocampus'): 20000.0,  # 20mm
            ('thalamus', 'brainstem'): 25000.0,  # 25mm
        }
        
        key = (source_region, target_region)
        reverse_key = (target_region, source_region)
        
        return region_distances.get(key, region_distances.get(reverse_key, 10000.0))
    
    def _update_statistics(self, connections: List[Tuple[int, int, float, float]], 
                          connection_type: ConnectionType):
        """更新连接统计"""
        
        if not connections:
            return
        
        self.connection_stats['total_connections'] += len(connections)
        
        type_name = connection_type.value
        self.connection_stats['connections_by_type'][type_name] = \
            self.connection_stats['connections_by_type'].get(type_name, 0) + len(connections)
        
        # 计算平均权重和延迟
        weights = [conn[2] for conn in connections]
        delays = [conn[3] for conn in connections]
        
        total_conns = self.connection_stats['total_connections']
        prev_total = total_conns - len(connections)
        
        if prev_total > 0:
            # 更新移动平均
            self.connection_stats['average_weight'] = (
                self.connection_stats['average_weight'] * prev_total + sum(weights)
            ) / total_conns
            
            self.connection_stats['average_delay'] = (
                self.connection_stats['average_delay'] * prev_total + sum(delays)
            ) / total_conns
        else:
            self.connection_stats['average_weight'] = np.mean(weights)
            self.connection_stats['average_delay'] = np.mean(delays)
    
    def get_connection_matrix(self) -> sp.csr_matrix:
        """获取连接矩阵"""
        return self.sparse_matrix.weight_matrix if self.sparse_matrix else None
    
    def get_delay_matrix(self) -> sp.csr_matrix:
        """获取延迟矩阵"""
        return self.sparse_matrix.delay_matrix if self.sparse_matrix else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取连接统计"""
        
        stats = self.connection_stats.copy()
        
        if self.sparse_matrix:
            stats.update(self.sparse_matrix.get_statistics())
        
        if self.regional_matrix:
            regional_stats = self.regional_matrix.get_regional_statistics()
            # 避免覆盖神经元级连接统计字段
            if 'total_connections' in regional_stats:
                regional_stats['regional_total_connections'] = regional_stats.pop('total_connections')
            if 'connection_density' in regional_stats:
                regional_stats['regional_connection_density'] = regional_stats.pop('connection_density')
            stats.update(regional_stats)
        
        return stats
    
    def save_connectivity(self, filepath: str):
        """保存连接数据"""
        
        if self.sparse_matrix:
            matrix_path = filepath.replace('.pkl', '_matrix.h5')
            self.sparse_matrix.save(matrix_path)
        
        # 保存其他数据
        data = {
            'neuron_positions': self.neuron_positions,
            'neuron_regions': self.neuron_regions,
            'regions': self.regions,
            'connection_stats': self.connection_stats,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"连接数据已保存到: {filepath}")
    
    def load_connectivity(self, filepath: str):
        """加载连接数据"""
        
        # 加载基本数据
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.neuron_positions = data['neuron_positions']
        self.neuron_regions = data['neuron_regions']
        self.regions = data['regions']
        self.connection_stats = data['connection_stats']
        
        # 加载矩阵数据
        if self.sparse_matrix:
            matrix_path = filepath.replace('.pkl', '_matrix.h5')
            self.sparse_matrix.load(matrix_path)
        
        self.logger.info(f"连接数据已从 {filepath} 加载")
    
    def close(self):
        """关闭连接管理器"""
        if self.graph_db:
            self.graph_db.close()

def create_enhanced_connectivity_manager(config: Dict[str, Any]) -> EnhancedConnectivityManager:
    """创建增强连接管理器的工厂函数"""
    return EnhancedConnectivityManager(config)
