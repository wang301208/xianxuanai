"""
皮层柱管理器

这个模块提供了一个高级的皮层柱管理系统，包括：
- 多皮层柱的协调和同步
- 层间和柱间连接管理
- 功能性皮层区域的组织
- 皮层柱间的信息传递
- 可塑性和学习的协调
- 皮层振荡的同步化
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import time

from .enhanced_cortical_column import EnhancedCorticalColumnWithLoop
from .cortical_column import CorticalColumn


class CorticalAreaType(Enum):
    """皮层区域类型"""
    PRIMARY_VISUAL = "V1"
    SECONDARY_VISUAL = "V2"
    PRIMARY_AUDITORY = "A1"
    PRIMARY_SOMATOSENSORY = "S1"
    PRIMARY_MOTOR = "M1"
    PREFRONTAL = "PFC"
    PARIETAL = "PPC"
    TEMPORAL = "IT"
    CINGULATE = "ACC"


class ConnectionType(Enum):
    """连接类型"""
    FEEDFORWARD = "feedforward"
    FEEDBACK = "feedback"
    LATERAL = "lateral"
    CROSS_MODAL = "cross_modal"


@dataclass
class CorticalAreaConfig:
    """皮层区域配置"""
    area_type: CorticalAreaType
    column_ids: List[int] = field(default_factory=list)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (1.0, 1.0, 2.0)  # mm
    
    # 功能特性
    primary_modality: Optional[str] = None
    processing_hierarchy: int = 1  # 1=初级, 2=次级, 3=高级
    
    # 连接特性
    feedforward_targets: List[CorticalAreaType] = field(default_factory=list)
    feedback_sources: List[CorticalAreaType] = field(default_factory=list)
    lateral_connections: List[CorticalAreaType] = field(default_factory=list)


@dataclass
class InterColumnConnection:
    """皮层柱间连接"""
    source_column: int
    target_column: int
    source_layer: str
    target_layer: str
    connection_type: ConnectionType
    strength: float = 0.5
    delay: float = 2.0  # ms
    plasticity_enabled: bool = True
    
    # 连接统计
    synapse_count: int = 0
    active_synapses: int = 0
    last_update_time: float = 0.0


class CorticalColumnManager:
    """皮层柱管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("CorticalColumnManager")
        
        # 皮层柱管理
        self.columns: Dict[int, EnhancedCorticalColumnWithLoop] = {}
        self.column_positions: Dict[int, Tuple[float, float, float]] = {}
        
        # 皮层区域管理
        self.cortical_areas: Dict[CorticalAreaType, CorticalAreaConfig] = {}
        self.area_column_mapping: Dict[CorticalAreaType, Set[int]] = defaultdict(set)
        
        # 连接管理
        self.inter_column_connections: List[InterColumnConnection] = []
        self.connection_matrix: Dict[Tuple[int, int], InterColumnConnection] = {}
        
        # 同步和协调
        self.synchronization_groups: Dict[str, Set[int]] = {}
        self.oscillation_coordinators: Dict[str, Any] = {}
        
        # 活动监控
        self.activity_history: Dict[int, deque] = {}
        self.synchrony_metrics: Dict[str, float] = {}
        
        # 学习协调
        self.learning_coordinator: Optional[Any] = None
        self.plasticity_rules: Dict[str, Any] = {}
        
        # 性能监控
        self.performance_stats: Dict[str, Any] = {}
        
        self.logger.info("皮层柱管理器初始化完成")
    
    def add_column(self, column_id: int, column: EnhancedCorticalColumnWithLoop, 
                  position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """添加皮层柱"""
        self.columns[column_id] = column
        self.column_positions[column_id] = position
        
        # 初始化活动历史
        self.activity_history[column_id] = deque(maxlen=1000)
        
        self.logger.info(f"添加皮层柱 {column_id} 在位置 {position}")
    
    def remove_column(self, column_id: int):
        """移除皮层柱"""
        if column_id in self.columns:
            # 移除相关连接
            self._remove_column_connections(column_id)
            
            # 从区域中移除
            for area_type in self.area_column_mapping:
                self.area_column_mapping[area_type].discard(column_id)
            
            # 移除皮层柱
            del self.columns[column_id]
            del self.column_positions[column_id]
            
            if column_id in self.activity_history:
                del self.activity_history[column_id]
            
            self.logger.info(f"移除皮层柱 {column_id}")
    
    def create_cortical_area(self, area_config: CorticalAreaConfig):
        """创建皮层区域"""
        area_type = area_config.area_type
        self.cortical_areas[area_type] = area_config
        
        # 添加皮层柱到区域
        for column_id in area_config.column_ids:
            if column_id in self.columns:
                self.area_column_mapping[area_type].add(column_id)
        
        # 建立区域内连接
        self._create_intra_area_connections(area_config)
        
        self.logger.info(f"创建皮层区域 {area_type.value} 包含 {len(area_config.column_ids)} 个皮层柱")
    
    def _create_intra_area_connections(self, area_config: CorticalAreaConfig):
        """创建区域内连接"""
        column_ids = area_config.column_ids
        
        # 创建局部连接网络
        for i, col1_id in enumerate(column_ids):
            for j, col2_id in enumerate(column_ids):
                if i != j:
                    # 计算距离
                    pos1 = self.column_positions.get(col1_id, (0, 0, 0))
                    pos2 = self.column_positions.get(col2_id, (0, 0, 0))
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
                    
                    # 距离依赖的连接概率
                    if distance < 2.0:  # 2mm内的局部连接
                        self._create_lateral_connection(col1_id, col2_id, distance)
    
    def _create_lateral_connection(self, col1_id: int, col2_id: int, distance: float):
        """创建横向连接"""
        # L2/3 到 L2/3 的横向连接
        connection = InterColumnConnection(
            source_column=col1_id,
            target_column=col2_id,
            source_layer="L2/3",
            target_layer="L2/3",
            connection_type=ConnectionType.LATERAL,
            strength=0.3 * np.exp(-distance / 1.0),  # 距离衰减
            delay=1.0 + distance * 0.5,  # 距离依赖延迟
            plasticity_enabled=True
        )
        
        self.inter_column_connections.append(connection)
        self.connection_matrix[(col1_id, col2_id)] = connection
        
        # 实际建立突触连接
        self._establish_synaptic_connection(connection)
    
    def create_inter_area_connection(self, source_area: CorticalAreaType, 
                                   target_area: CorticalAreaType, 
                                   connection_type: ConnectionType):
        """创建区域间连接"""
        if source_area not in self.cortical_areas or target_area not in self.cortical_areas:
            self.logger.warning(f"区域 {source_area} 或 {target_area} 不存在")
            return
        
        source_columns = list(self.area_column_mapping[source_area])
        target_columns = list(self.area_column_mapping[target_area])
        
        # 根据连接类型确定层间连接模式
        layer_mapping = self._get_layer_mapping(connection_type)
        
        for source_col in source_columns:
            for target_col in target_columns:
                for source_layer, target_layer in layer_mapping:
                    connection = InterColumnConnection(
                        source_column=source_col,
                        target_column=target_col,
                        source_layer=source_layer,
                        target_layer=target_layer,
                        connection_type=connection_type,
                        strength=self._get_connection_strength(connection_type),
                        delay=self._get_connection_delay(connection_type),
                        plasticity_enabled=True
                    )
                    
                    self.inter_column_connections.append(connection)
                    self._establish_synaptic_connection(connection)
        
        self.logger.info(f"创建 {source_area.value} -> {target_area.value} {connection_type.value} 连接")
    
    def _get_layer_mapping(self, connection_type: ConnectionType) -> List[Tuple[str, str]]:
        """获取层间连接映射"""
        if connection_type == ConnectionType.FEEDFORWARD:
            return [
                ("L2/3", "L4"),    # 前馈到L4
                ("L5", "L2/3"),    # L5到高级区域L2/3
                ("L6", "L6")       # L6到L6
            ]
        elif connection_type == ConnectionType.FEEDBACK:
            return [
                ("L2/3", "L1"),    # 反馈到L1
                ("L5", "L5"),      # L5到L5
                ("L6", "L6")       # L6到L6
            ]
        elif connection_type == ConnectionType.LATERAL:
            return [
                ("L2/3", "L2/3"),  # 横向L2/3连接
                ("L5", "L5")       # 横向L5连接
            ]
        else:  # CROSS_MODAL
            return [
                ("L2/3", "L2/3"),
                ("L5", "L2/3")
            ]
    
    def _get_connection_strength(self, connection_type: ConnectionType) -> float:
        """获取连接强度"""
        strength_mapping = {
            ConnectionType.FEEDFORWARD: 0.8,
            ConnectionType.FEEDBACK: 0.6,
            ConnectionType.LATERAL: 0.4,
            ConnectionType.CROSS_MODAL: 0.3
        }
        return strength_mapping.get(connection_type, 0.5)
    
    def _get_connection_delay(self, connection_type: ConnectionType) -> float:
        """获取连接延迟"""
        delay_mapping = {
            ConnectionType.FEEDFORWARD: 2.0,
            ConnectionType.FEEDBACK: 5.0,
            ConnectionType.LATERAL: 1.5,
            ConnectionType.CROSS_MODAL: 8.0
        }
        return delay_mapping.get(connection_type, 3.0)
    
    def _establish_synaptic_connection(self, connection: InterColumnConnection):
        """建立实际的突触连接"""
        source_col = self.columns.get(connection.source_column)
        target_col = self.columns.get(connection.target_column)
        
        if not source_col or not target_col:
            return
        
        # 获取源层和目标层
        source_layer = source_col.layers.get(connection.source_layer + "_exc")
        target_layer = target_col.layers.get(connection.target_layer + "_exc")
        
        if not source_layer or not target_layer:
            return
        
        # 建立稀疏连接
        connection_prob = min(0.1, connection.strength)
        synapse_count = 0
        
        for source_neuron in source_layer.neurons[:50]:  # 限制连接数量
            for target_neuron in target_layer.neurons[:50]:
                if np.random.random() < connection_prob:
                    # 创建突触
                    synapse_params = {
                        'weight': connection.strength * np.random.uniform(0.5, 1.5),
                        'delay': connection.delay + np.random.uniform(-0.5, 0.5),
                        'learning_rate': 0.001 if connection.plasticity_enabled else 0.0
                    }
                    
                    # 添加突触到源皮层柱
                    source_col.add_synapse(
                        source_neuron.id, 
                        target_neuron.id, 
                        'stdp' if connection.plasticity_enabled else 'static',
                        synapse_params
                    )
                    
                    synapse_count += 1
        
        connection.synapse_count = synapse_count
        connection.last_update_time = time.time()
        
        self.logger.debug(f"建立连接 {connection.source_column}->{connection.target_column}: {synapse_count} 个突触")
    
    def _remove_column_connections(self, column_id: int):
        """移除与指定皮层柱相关的所有连接"""
        # 移除连接列表中的相关连接
        self.inter_column_connections = [
            conn for conn in self.inter_column_connections
            if conn.source_column != column_id and conn.target_column != column_id
        ]
        
        # 移除连接矩阵中的相关连接
        keys_to_remove = [
            key for key in self.connection_matrix.keys()
            if key[0] == column_id or key[1] == column_id
        ]
        
        for key in keys_to_remove:
            del self.connection_matrix[key]
    
    def create_synchronization_group(self, group_name: str, column_ids: List[int]):
        """创建同步组"""
        self.synchronization_groups[group_name] = set(column_ids)
        
        # 创建振荡协调器
        coordinator_config = {
            'target_frequency': 40.0,  # 40Hz gamma
            'synchrony_strength': 0.8,
            'phase_coupling': True
        }
        
        self.oscillation_coordinators[group_name] = OscillationCoordinator(
            column_ids, coordinator_config
        )
        
        self.logger.info(f"创建同步组 '{group_name}' 包含 {len(column_ids)} 个皮层柱")
    
    def update_synchronization(self, dt: float):
        """更新同步化"""
        for group_name, coordinator in self.oscillation_coordinators.items():
            column_ids = list(self.synchronization_groups[group_name])
            columns = [self.columns[cid] for cid in column_ids if cid in self.columns]
            
            if columns:
                coordinator.update(columns, dt)
    
    def calculate_synchrony_metrics(self) -> Dict[str, float]:
        """计算同步化指标"""
        metrics = {}
        
        # 计算每个同步组的同步度
        for group_name, column_ids in self.synchronization_groups.items():
            phases = []
            
            for column_id in column_ids:
                if column_id in self.columns:
                    column = self.columns[column_id]
                    if hasattr(column, 'oscillation_state'):
                        phases.append(column.oscillation_state.gamma_phase)
            
            if len(phases) > 1:
                # 计算相位同步指数
                phase_diffs = []
                for i in range(len(phases)):
                    for j in range(i + 1, len(phases)):
                        diff = abs(phases[i] - phases[j])
                        diff = min(diff, 2 * np.pi - diff)
                        phase_diffs.append(1.0 - diff / np.pi)
                
                metrics[f"{group_name}_synchrony"] = np.mean(phase_diffs)
        
        # 计算全局同步度
        all_phases = []
        for column in self.columns.values():
            if hasattr(column, 'oscillation_state'):
                all_phases.append(column.oscillation_state.gamma_phase)
        
        if len(all_phases) > 1:
            global_phase_diffs = []
            for i in range(len(all_phases)):
                for j in range(i + 1, len(all_phases)):
                    diff = abs(all_phases[i] - all_phases[j])
                    diff = min(diff, 2 * np.pi - diff)
                    global_phase_diffs.append(1.0 - diff / np.pi)
            
            metrics['global_synchrony'] = np.mean(global_phase_diffs)
        
        self.synchrony_metrics.update(metrics)
        return metrics
    
    def propagate_activity(self, source_column_id: int, activity_pattern: np.ndarray):
        """传播活动模式"""
        if source_column_id not in self.columns:
            return
        
        # 找到所有从该皮层柱出发的连接
        outgoing_connections = [
            conn for conn in self.inter_column_connections
            if conn.source_column == source_column_id
        ]
        
        for connection in outgoing_connections:
            target_column = self.columns.get(connection.target_column)
            if target_column:
                # 应用连接强度和延迟
                modulated_activity = activity_pattern * connection.strength
                
                # 发送到目标皮层柱（简化实现）
                target_column.receive_external_input(
                    connection.target_layer, 
                    modulated_activity
                )
    
    def step(self, dt: float) -> Dict[str, Any]:
        """管理器步进"""
        results = {
            'timestamp': time.time(),
            'column_results': {},
            'synchrony_metrics': {},
            'connection_stats': {},
            'area_activities': {}
        }
        
        try:
            # 1. 更新所有皮层柱
            for column_id, column in self.columns.items():
                column_result = column.step(dt)
                results['column_results'][column_id] = column_result
                
                # 记录活动历史
                if 'mean_activity' in column_result:
                    self.activity_history[column_id].append(column_result['mean_activity'])
            
            # 2. 更新同步化
            self.update_synchronization(dt)
            
            # 3. 计算同步指标
            synchrony_metrics = self.calculate_synchrony_metrics()
            results['synchrony_metrics'] = synchrony_metrics
            
            # 4. 计算区域活动
            for area_type, column_ids in self.area_column_mapping.items():
                area_activity = []
                for column_id in column_ids:
                    if column_id in results['column_results']:
                        activity = results['column_results'][column_id].get('mean_activity', 0.0)
                        area_activity.append(activity)
                
                if area_activity:
                    results['area_activities'][area_type.value] = {
                        'mean': np.mean(area_activity),
                        'std': np.std(area_activity),
                        'max': np.max(area_activity),
                        'min': np.min(area_activity)
                    }
            
            # 5. 更新连接统计
            self._update_connection_stats(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"管理器步进失败: {e}")
            return results
    
    def _update_connection_stats(self, results: Dict[str, Any]):
        """更新连接统计"""
        connection_stats = {}
        
        # 按连接类型统计
        for conn_type in ConnectionType:
            type_connections = [
                conn for conn in self.inter_column_connections
                if conn.connection_type == conn_type
            ]
            
            if type_connections:
                strengths = [conn.strength for conn in type_connections]
                delays = [conn.delay for conn in type_connections]
                
                connection_stats[conn_type.value] = {
                    'count': len(type_connections),
                    'mean_strength': np.mean(strengths),
                    'mean_delay': np.mean(delays),
                    'total_synapses': sum(conn.synapse_count for conn in type_connections)
                }
        
        results['connection_stats'] = connection_stats
    
    def get_area_connectivity_matrix(self) -> np.ndarray:
        """获取区域连接矩阵"""
        area_types = list(self.cortical_areas.keys())
        n_areas = len(area_types)
        
        connectivity_matrix = np.zeros((n_areas, n_areas))
        
        for i, source_area in enumerate(area_types):
            for j, target_area in enumerate(area_types):
                # 计算区域间连接强度
                source_columns = self.area_column_mapping[source_area]
                target_columns = self.area_column_mapping[target_area]
                
                total_strength = 0.0
                connection_count = 0
                
                for source_col in source_columns:
                    for target_col in target_columns:
                        if (source_col, target_col) in self.connection_matrix:
                            conn = self.connection_matrix[(source_col, target_col)]
                            total_strength += conn.strength
                            connection_count += 1
                
                if connection_count > 0:
                    connectivity_matrix[i, j] = total_strength / connection_count
        
        return connectivity_matrix
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        state = {
            'columns': {
                cid: {
                    'position': self.column_positions[cid],
                    'neuron_count': sum(len(layer.neurons) for layer in col.layers.values())
                }
                for cid, col in self.columns.items()
            },
            'cortical_areas': {
                area_type.value: {
                    'column_count': len(column_ids),
                    'position': config.position,
                    'size': config.size
                }
                for area_type, config in self.cortical_areas.items()
                for column_ids in [self.area_column_mapping[area_type]]
            },
            'connections': {
                'total_connections': len(self.inter_column_connections),
                'by_type': {
                    conn_type.value: len([
                        conn for conn in self.inter_column_connections
                        if conn.connection_type == conn_type
                    ])
                    for conn_type in ConnectionType
                }
            },
            'synchronization_groups': {
                name: list(column_ids)
                for name, column_ids in self.synchronization_groups.items()
            },
            'synchrony_metrics': self.synchrony_metrics.copy()
        }
        
        return state


class OscillationCoordinator:
    """振荡协调器"""
    
    def __init__(self, column_ids: List[int], config: Dict[str, Any]):
        self.column_ids = column_ids
        self.config = config
        
        self.target_frequency = config.get('target_frequency', 40.0)
        self.synchrony_strength = config.get('synchrony_strength', 0.8)
        self.phase_coupling = config.get('phase_coupling', True)
        
        # 协调状态
        self.master_phase = 0.0
        self.master_frequency = self.target_frequency
        self.coupling_strength = 0.1
    
    def update(self, columns: List[EnhancedCorticalColumnWithLoop], dt: float):
        """更新振荡协调"""
        if not self.phase_coupling or len(columns) < 2:
            return
        
        # 计算平均相位
        phases = []
        for column in columns:
            if hasattr(column, 'oscillation_state'):
                phases.append(column.oscillation_state.gamma_phase)
        
        if not phases:
            return
        
        # 计算相位中心
        mean_phase = np.angle(np.mean(np.exp(1j * np.array(phases))))
        
        # 更新主相位
        self.master_phase = mean_phase
        
        # 应用相位耦合
        for column in columns:
            if hasattr(column, 'oscillation_state'):
                current_phase = column.oscillation_state.gamma_phase
                phase_diff = self.master_phase - current_phase
                
                # 相位差归一化到[-π, π]
                phase_diff = np.angle(np.exp(1j * phase_diff))
                
                # 应用耦合力
                coupling_force = self.coupling_strength * self.synchrony_strength * phase_diff
                
                # 更新皮层柱的振荡参数
                column.oscillation_state.gamma_phase += coupling_force * dt
                column.oscillation_state.gamma_frequency = (
                    0.9 * column.oscillation_state.gamma_frequency + 
                    0.1 * self.target_frequency
                )


# 工厂函数
def create_cortical_column_manager(config: Optional[Dict[str, Any]] = None) -> CorticalColumnManager:
    """创建皮层柱管理器"""
    return CorticalColumnManager(config)


def create_standard_cortical_areas(manager: CorticalColumnManager, 
                                 column_ids: List[int]) -> Dict[CorticalAreaType, CorticalAreaConfig]:
    """创建标准皮层区域"""
    areas = {}
    
    # 分配皮层柱到不同区域
    n_columns = len(column_ids)
    
    if n_columns >= 4:
        # V1 - 初级视觉皮层
        v1_config = CorticalAreaConfig(
            area_type=CorticalAreaType.PRIMARY_VISUAL,
            column_ids=column_ids[:2],
            position=(0.0, 0.0, 0.0),
            primary_modality="visual",
            processing_hierarchy=1
        )
        areas[CorticalAreaType.PRIMARY_VISUAL] = v1_config
        manager.create_cortical_area(v1_config)
        
        # PFC - 前额叶皮层
        pfc_config = CorticalAreaConfig(
            area_type=CorticalAreaType.PREFRONTAL,
            column_ids=column_ids[2:4],
            position=(3.0, 0.0, 0.0),
            processing_hierarchy=3
        )
        areas[CorticalAreaType.PREFRONTAL] = pfc_config
        manager.create_cortical_area(pfc_config)
        
        # 创建前馈连接 V1 -> PFC
        manager.create_inter_area_connection(
            CorticalAreaType.PRIMARY_VISUAL,
            CorticalAreaType.PREFRONTAL,
            ConnectionType.FEEDFORWARD
        )
        
        # 创建反馈连接 PFC -> V1
        manager.create_inter_area_connection(
            CorticalAreaType.PREFRONTAL,
            CorticalAreaType.PRIMARY_VISUAL,
            ConnectionType.FEEDBACK
        )
    
    return areas