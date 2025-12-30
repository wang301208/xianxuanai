"""
区域级连接矩阵实现
Regional Connectivity Matrix Implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .enhanced_connectivity import AxonParameters

class LongRangeAxon:
    """长程轴突模型"""
    
    def __init__(self, source_id: int, target_region: str, params: AxonParameters):
        self.source_id = source_id
        self.target_region = target_region
        self.params = params
        
        # 轴突路径
        self.path_points = []
        self.total_length = 0.0
        
        # 传导延迟
        self.conduction_delay = 0.0
        
        # 分支
        self.branches = []
        
        # 计算传导延迟
        self._calculate_conduction_delay()
        
        self.logger = logging.getLogger(f"Axon_{source_id}")
    
    def _calculate_conduction_delay(self):
        """计算传导延迟"""
        
        # 基础延迟：长度 / 传导速度
        base_delay = self.params.length / (self.params.conduction_velocity * 1000)  # ms
        
        # 髓鞘化影响
        if self.params.myelination:
            # 有髓轴突传导更快
            myelination_factor = 0.1
        else:
            # 无髓轴突传导较慢
            myelination_factor = 1.0
        
        # 直径影响
        diameter_factor = 1.0 / np.sqrt(self.params.diameter)
        
        self.conduction_delay = base_delay * myelination_factor * diameter_factor
    
    def add_branch(self, target_id: int, branch_length: float):
        """添加轴突分支"""
        
        branch_params = AxonParameters(
            length=branch_length,
            diameter=self.params.diameter * 0.8,  # 分支直径减小
            conduction_velocity=self.params.conduction_velocity,
            myelination=self.params.myelination
        )
        
        branch = LongRangeAxon(self.source_id, f"branch_{target_id}", branch_params)
        self.branches.append((target_id, branch))
    
    def get_total_delay(self, target_id: int) -> float:
        """获取到特定目标的总延迟"""
        
        # 主轴突延迟
        total_delay = self.conduction_delay
        
        # 查找分支延迟
        for branch_target_id, branch in self.branches:
            if branch_target_id == target_id:
                total_delay += branch.conduction_delay
                break
        
        return total_delay

class RegionalConnectivityMatrix:
    """区域级连接矩阵"""
    
    def __init__(self, regions: List[str]):
        self.regions = regions
        self.n_regions = len(regions)
        self.region_to_index = {region: i for i, region in enumerate(regions)}
        
        # 区域间连接强度矩阵
        self.connection_strength = np.zeros((self.n_regions, self.n_regions))
        
        # 区域间连接概率矩阵
        self.connection_probability = np.zeros((self.n_regions, self.n_regions))
        
        # 区域间延迟矩阵
        self.connection_delay = np.zeros((self.n_regions, self.n_regions))
        
        # 长程轴突
        self.long_range_axons = {}
        
        self.logger = logging.getLogger("RegionalConnectivity")
    
    def set_connection(self, source_region: str, target_region: str,
                      strength: float, probability: float, delay: float):
        """设置区域间连接"""
        
        if source_region not in self.region_to_index or target_region not in self.region_to_index:
            raise ValueError(f"未知区域: {source_region} 或 {target_region}")
        
        source_idx = self.region_to_index[source_region]
        target_idx = self.region_to_index[target_region]
        
        self.connection_strength[source_idx, target_idx] = strength
        self.connection_probability[source_idx, target_idx] = probability
        self.connection_delay[source_idx, target_idx] = delay
        
        self.logger.debug(f"设置连接: {source_region} -> {target_region}, "
                         f"强度: {strength:.3f}, 概率: {probability:.3f}, 延迟: {delay:.3f}")
    
    def add_long_range_axon(self, source_neuron_id: int, source_region: str,
                           target_region: str, axon_params: AxonParameters):
        """添加长程轴突"""
        
        axon = LongRangeAxon(source_neuron_id, target_region, axon_params)
        key = (source_neuron_id, source_region, target_region)
        self.long_range_axons[key] = axon
    
    def get_connection_parameters(self, source_region: str, target_region: str) -> Tuple[float, float, float]:
        """获取区域间连接参数"""
        
        source_idx = self.region_to_index[source_region]
        target_idx = self.region_to_index[target_region]
        
        strength = self.connection_strength[source_idx, target_idx]
        probability = self.connection_probability[source_idx, target_idx]
        delay = self.connection_delay[source_idx, target_idx]
        
        return strength, probability, delay
    
    def get_regional_statistics(self) -> Dict[str, Any]:
        """获取区域连接统计"""
        
        total_connections = np.sum(self.connection_strength > 0)
        mean_strength = np.mean(self.connection_strength[self.connection_strength > 0])
        mean_delay = np.mean(self.connection_delay[self.connection_delay > 0])
        
        return {
            'total_regions': self.n_regions,
            'total_connections': int(total_connections),
            'connection_density': total_connections / (self.n_regions ** 2),
            'mean_strength': float(mean_strength) if not np.isnan(mean_strength) else 0.0,
            'mean_delay': float(mean_delay) if not np.isnan(mean_delay) else 0.0,
            'long_range_axons': len(self.long_range_axons)
        }
