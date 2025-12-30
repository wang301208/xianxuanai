"""
概率连接器实现
Probabilistic Connector Implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .enhanced_connectivity import ConnectionParameters, DelayType

class ProbabilisticConnector:
    """概率连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = np.random.RandomState(config.get('seed', 42))
        self.logger = logging.getLogger("ProbabilisticConnector")
    
    def generate_connections(self, source_neurons: List[int], target_neurons: List[int],
                           connection_params: ConnectionParameters) -> List[Tuple[int, int, float, float]]:
        """生成概率连接"""
        
        connections = []
        
        for source_id in source_neurons:
            for target_id in target_neurons:
                if source_id == target_id:
                    continue
                
                # 连接概率判断
                if self.rng.random() < connection_params.probability:
                    # 生成权重
                    weight = self._sample_weight(connection_params)
                    
                    # 生成延迟
                    delay = self._sample_delay(connection_params, source_id, target_id)
                    
                    connections.append((source_id, target_id, weight, delay))
        
        self.logger.info(f"生成了 {len(connections)} 个连接")
        return connections
    
    def _sample_weight(self, params: ConnectionParameters) -> float:
        """采样连接权重"""
        
        weight = self.rng.normal(params.weight_mean, params.weight_std)
        
        # 确保权重为正（对于兴奋性连接）
        if params.weight_mean > 0:
            weight = max(0.01, weight)
        else:
            weight = min(-0.01, weight)
        
        return weight
    
    def _sample_delay(self, params: ConnectionParameters, source_id: int, target_id: int) -> float:
        """采样连接延迟"""
        
        if params.delay_type == DelayType.FIXED:
            return params.delay_mean
        
        elif params.delay_type == DelayType.GAUSSIAN:
            delay = self.rng.normal(params.delay_mean, params.delay_std)
            return max(0.1, delay)  # 最小延迟0.1ms
        
        elif params.delay_type == DelayType.EXPONENTIAL:
            delay = self.rng.exponential(params.delay_mean)
            return max(0.1, delay)
        
        elif params.delay_type == DelayType.DISTANCE_DEPENDENT:
            # 需要神经元位置信息
            distance = self._calculate_distance(source_id, target_id)
            base_delay = distance / (params.conduction_velocity * 1000)  # ms
            noise = self.rng.normal(0, params.delay_std)
            return max(0.1, base_delay + noise)
        
        else:
            return params.delay_mean
    
    def _calculate_distance(self, source_id: int, target_id: int) -> float:
        """计算神经元间距离（需要位置信息）"""
        # 这里需要从神经元位置数据库获取位置信息
        # 暂时返回随机距离
        return self.rng.uniform(10.0, 1000.0)  # μm