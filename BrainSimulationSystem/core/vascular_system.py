"""
Vascular System Implementation

实现完整的脑血管系统，包括：
- 血管网络结构（动脉、毛细血管、静脉）
- 血流动力学
- 血脑屏障
- 代谢物质交换
- 血管调节机制
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    class _FallbackDiGraph:
        def __init__(self):
            self.nodes = set()
            self.edges = set()

        def add_node(self, node):
            self.nodes.add(node)

        def add_edge(self, source, target):
            self.nodes.add(source)
            self.nodes.add(target)
            self.edges.add((source, target))

    nx = type("nx", (), {"DiGraph": _FallbackDiGraph})()

class VesselType(Enum):
    """血管类型"""
    ARTERIOLE = "arteriole"
    CAPILLARY = "capillary"
    VENULE = "venule"
    ARTERY = "artery"
    VEIN = "vein"

@dataclass
class VesselSegment:
    """血管段数据结构"""
    segment_id: int
    vessel_type: VesselType
    start_position: Tuple[float, float, float]
    end_position: Tuple[float, float, float]
    diameter: float  # μm
    length: float    # μm
    
    # 血流参数
    pressure_start: float  # mmHg
    pressure_end: float    # mmHg
    flow_rate: float       # ml/min
    resistance: float      # mmHg·min/ml
    
    # 血管壁参数
    wall_thickness: float  # μm
    permeability: float    # cm/s
    
    # 连接信息
    upstream_segments: List[int]
    downstream_segments: List[int]

class BloodBrainBarrier:
    """血脑屏障模型"""
    
    def __init__(self):
        # 不同物质的通透性系数 (cm/s)
        self.permeability_coefficients = {
            'oxygen': 1e-3,
            'carbon_dioxide': 1e-3,
            'glucose': 1e-6,
            'lactate': 1e-7,
            'glutamate': 1e-9,
            'dopamine': 1e-10,
            'serotonin': 1e-10,
            'water': 1e-4,
            'sodium': 1e-8,
            'potassium': 1e-8,
            'calcium': 1e-9
        }
        
        # 载体介导的转运
        self.transporter_kinetics = {
            'glucose': {'vmax': 1e-5, 'km': 5.0},  # mmol/L
            'lactate': {'vmax': 5e-6, 'km': 2.0},
            'glutamate': {'vmax': 1e-7, 'km': 0.1}
        }
    
    def calculate_transport_rate(self, substance: str, blood_concentration: float, 
                               brain_concentration: float, surface_area: float) -> float:
        """计算物质跨血脑屏障的转运速率"""
        
        if substance in self.transporter_kinetics:
            # 载体介导的转运（Michaelis-Menten动力学）
            kinetics = self.transporter_kinetics[substance]
            vmax = kinetics['vmax']
            km = kinetics['km']
            
            # 净转运速率
            forward_rate = (vmax * blood_concentration) / (km + blood_concentration)
            backward_rate = (vmax * brain_concentration) / (km + brain_concentration)
            net_rate = (forward_rate - backward_rate) * surface_area
            
        else:
            # 被动扩散
            permeability = self.permeability_coefficients.get(substance, 1e-10)
            concentration_gradient = blood_concentration - brain_concentration
            net_rate = permeability * concentration_gradient * surface_area
        
        return net_rate

class VascularNetwork:
    """血管网络"""
    
    def __init__(self, tissue_dimensions: Tuple[float, float, float]):
        self.tissue_dimensions = tissue_dimensions
        self.vessels: Dict[int, VesselSegment] = {}
        self.network_graph = nx.DiGraph()
        self.blood_brain_barrier = BloodBrainBarrier()
        
        # 血液成分浓度 (mmol/L)
        self.blood_concentrations = {
            'oxygen': 8.0,
            'carbon_dioxide': 1.2,
            'glucose': 5.0,
            'lactate': 1.0,
            'glutamate': 0.05,
            'sodium': 140.0,
            'potassium': 4.0,
            'calcium': 2.5
        }
        
        # 脑组织浓度
        self.brain_concentrations = {
            'oxygen': 2.0,
            'carbon_dioxide': 1.5,
            'glucose': 2.5,
            'lactate': 2.0,
            'glutamate': 10.0,  # μmol/L in extracellular space
            'sodium': 145.0,
            'potassium': 3.5,
            'calcium': 1.2
        }
        
        self.segment_counter = 0
    
    def generate_vascular_tree(self, branching_levels: int = 6):
        """生成分支血管树"""
        
        # 主要供血动脉（入口）
        main_artery = self._create_vessel_segment(
            vessel_type=VesselType.ARTERY,
            start_pos=(0, self.tissue_dimensions[1]/2, self.tissue_dimensions[2]/2),
            end_pos=(self.tissue_dimensions[0]*0.2, self.tissue_dimensions[1]/2, self.tissue_dimensions[2]/2),
            diameter=100.0,
            pressure_start=80.0  # mmHg
        )
        
        # 主要引流静脉（出口）
        main_vein = self._create_vessel_segment(
            vessel_type=VesselType.VEIN,
            start_pos=(self.tissue_dimensions[0]*0.8, self.tissue_dimensions[1]/2, self.tissue_dimensions[2]/2),
            end_pos=(self.tissue_dimensions[0], self.tissue_dimensions[1]/2, self.tissue_dimensions[2]/2),
            diameter=120.0,
            pressure_start=5.0  # mmHg
        )
        
        # 递归生成分支网络
        arterial_terminals = self._generate_arterial_tree(main_artery, branching_levels)
        venous_origins = self._generate_venous_tree(main_vein, branching_levels)
        
        # 连接动脉末端和静脉起始（毛细血管网络）
        self._generate_capillary_network(arterial_terminals, venous_origins)
        
        # 计算血流阻力
        self._calculate_vascular_resistance()
    
    def _create_vessel_segment(self, vessel_type: VesselType, start_pos: Tuple[float, float, float],
                             end_pos: Tuple[float, float, float], diameter: float, 
                             pressure_start: float = 0.0) -> VesselSegment:
        """创建血管段"""
        
        segment_id = self.segment_counter
        self.segment_counter += 1
        
        length = np.sqrt(sum((end_pos[i] - start_pos[i])**2 for i in range(3)))
        
        # 根据血管类型设置壁厚和通透性
        if vessel_type == VesselType.ARTERY:
            wall_thickness = diameter * 0.2
            permeability = 1e-8
        elif vessel_type == VesselType.ARTERIOLE:
            wall_thickness = diameter * 0.15
            permeability = 1e-7
        elif vessel_type == VesselType.CAPILLARY:
            wall_thickness = 0.5
            permeability = 1e-6
        elif vessel_type == VesselType.VENULE:
            wall_thickness = diameter * 0.1
            permeability = 1e-7
        else:  # VEIN
            wall_thickness = diameter * 0.15
            permeability = 1e-8
        
        segment = VesselSegment(
            segment_id=segment_id,
            vessel_type=vessel_type,
            start_position=start_pos,
            end_position=end_pos,
            diameter=diameter,
            length=length,
            pressure_start=pressure_start,
            pressure_end=0.0,
            flow_rate=0.0,
            resistance=0.0,
            wall_thickness=wall_thickness,
            permeability=permeability,
            upstream_segments=[],
            downstream_segments=[]
        )
        
        self.vessels[segment_id] = segment
        self.network_graph.add_node(segment_id)
        
        return segment
    
    def _generate_arterial_tree(self, parent_segment: VesselSegment, levels_remaining: int) -> List[VesselSegment]:
        """递归生成动脉分支树"""
        
        if levels_remaining <= 0:
            return [parent_segment]
        
        terminals = []
        
        # 分支数量（2-4个分支）
        num_branches = np.random.randint(2, 5)
        
        for i in range(num_branches):
            # 分支直径（Murray定律：d^3 = sum(d_i^3)）
            diameter_ratio = np.random.uniform(0.6, 0.8)
            branch_diameter = parent_segment.diameter * diameter_ratio
            
            # 分支方向
            branch_angle = np.random.uniform(-np.pi/3, np.pi/3)  # ±60度
            branch_length = parent_segment.length * np.random.uniform(0.8, 1.2)
            
            # 计算分支终点
            direction = np.array(parent_segment.end_position) - np.array(parent_segment.start_position)
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm == 0:
                direction = np.random.normal(0.0, 1.0, 3)
                norm = np.linalg.norm(direction) or 1.0
            direction = direction / norm
            
            # 旋转方向向量
            rotation_matrix = self._get_rotation_matrix(branch_angle)
            new_direction = rotation_matrix @ direction
            
            branch_end = np.array(parent_segment.end_position) + new_direction * branch_length
            
            # 确保在组织边界内
            branch_end = np.clip(branch_end, [0, 0, 0], self.tissue_dimensions)
            
            # 确定血管类型
            if levels_remaining > 3:
                vessel_type = VesselType.ARTERY
            elif levels_remaining > 1:
                vessel_type = VesselType.ARTERIOLE
            else:
                vessel_type = VesselType.CAPILLARY
            
            # 创建分支
            branch = self._create_vessel_segment(
                vessel_type=vessel_type,
                start_pos=tuple(parent_segment.end_position),
                end_pos=tuple(branch_end),
                diameter=branch_diameter
            )
            
            # 建立连接
            parent_segment.downstream_segments.append(branch.segment_id)
            branch.upstream_segments.append(parent_segment.segment_id)
            self.network_graph.add_edge(parent_segment.segment_id, branch.segment_id)
            
            # 递归生成子分支
            branch_terminals = self._generate_arterial_tree(branch, levels_remaining - 1)
            terminals.extend(branch_terminals)
        
        return terminals
    
    def _generate_venous_tree(self, parent_segment: VesselSegment, levels_remaining: int) -> List[VesselSegment]:
        """递归生成静脉汇合树（反向）"""
        
        if levels_remaining <= 0:
            return [parent_segment]
        
        origins = []
        
        # 汇合分支数量
        num_branches = np.random.randint(2, 4)
        
        for i in range(num_branches):
            # 分支直径
            diameter_ratio = np.random.uniform(0.6, 0.8)
            branch_diameter = parent_segment.diameter * diameter_ratio
            
            # 分支方向（向后）
            branch_angle = np.random.uniform(-np.pi/4, np.pi/4)
            branch_length = parent_segment.length * np.random.uniform(0.8, 1.2)
            
            # 计算分支起点
            direction = np.array(parent_segment.start_position) - np.array(parent_segment.end_position)
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm == 0:
                direction = np.random.normal(0.0, 1.0, 3)
                norm = np.linalg.norm(direction) or 1.0
            direction = direction / norm
            
            rotation_matrix = self._get_rotation_matrix(branch_angle)
            new_direction = rotation_matrix @ direction
            
            branch_start = np.array(parent_segment.start_position) + new_direction * branch_length
            branch_start = np.clip(branch_start, [0, 0, 0], self.tissue_dimensions)
            
            # 确定血管类型
            if levels_remaining > 3:
                vessel_type = VesselType.VEIN
            elif levels_remaining > 1:
                vessel_type = VesselType.VENULE
            else:
                vessel_type = VesselType.CAPILLARY
            
            # 创建分支
            branch = self._create_vessel_segment(
                vessel_type=vessel_type,
                start_pos=tuple(branch_start),
                end_pos=tuple(parent_segment.start_position),
                diameter=branch_diameter
            )
            
            # 建立连接
            branch.downstream_segments.append(parent_segment.segment_id)
            parent_segment.upstream_segments.append(branch.segment_id)
            self.network_graph.add_edge(branch.segment_id, parent_segment.segment_id)
            
            # 递归生成子分支
            branch_origins = self._generate_venous_tree(branch, levels_remaining - 1)
            origins.extend(branch_origins)
        
        return origins
    
    def _generate_capillary_network(self, arterial_terminals: List[VesselSegment], 
                                  venous_origins: List[VesselSegment]):
        """生成毛细血管网络连接动脉末端和静脉起始"""
        
        # 为每个动脉末端找到最近的静脉起始点
        for arterial_terminal in arterial_terminals:
            if arterial_terminal.vessel_type != VesselType.CAPILLARY:
                continue
                
            # 找到最近的静脉毛细血管
            min_distance = float('inf')
            closest_venous = None
            
            for venous_origin in venous_origins:
                if venous_origin.vessel_type != VesselType.CAPILLARY:
                    continue
                    
                distance = np.sqrt(sum((arterial_terminal.end_position[i] - venous_origin.start_position[i])**2 
                                     for i in range(3)))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_venous = venous_origin
            
            # 创建连接毛细血管
            if closest_venous and min_distance < 200.0:  # 200μm最大连接距离
                capillary = self._create_vessel_segment(
                    vessel_type=VesselType.CAPILLARY,
                    start_pos=arterial_terminal.end_position,
                    end_pos=closest_venous.start_position,
                    diameter=5.0  # 5μm毛细血管直径
                )
                
                # 建立连接
                arterial_terminal.downstream_segments.append(capillary.segment_id)
                capillary.upstream_segments.append(arterial_terminal.segment_id)
                capillary.downstream_segments.append(closest_venous.segment_id)
                closest_venous.upstream_segments.append(capillary.segment_id)
                
                self.network_graph.add_edge(arterial_terminal.segment_id, capillary.segment_id)
                self.network_graph.add_edge(capillary.segment_id, closest_venous.segment_id)
    
    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """获取绕z轴的旋转矩阵"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    
    def _calculate_vascular_resistance(self):
        """计算血管阻力（Poiseuille定律）"""
        
        # 血液粘度 (Pa·s)
        blood_viscosity = 3.5e-3
        
        for vessel in self.vessels.values():
            # Poiseuille定律：R = (8 * η * L) / (π * r^4)
            radius = vessel.diameter / 2 * 1e-6  # 转换为米
            length = vessel.length * 1e-6  # 转换为米
            
            resistance = (8 * blood_viscosity * length) / (np.pi * radius**4)
            
            # 转换为mmHg·min/ml
            vessel.resistance = resistance * 7.5e-9
    
    def solve_blood_flow(self):
        """求解血管网络中的血流分布"""
        
        # 使用节点分析法求解血流
        # 构建系数矩阵和常数向量
        
        vessel_ids = list(self.vessels.keys())
        n_vessels = len(vessel_ids)
        
        if n_vessels == 0:
            return
        
        # 压力边界条件
        inlet_pressure = 80.0  # mmHg
        outlet_pressure = 5.0   # mmHg
        
        # 简化求解：假设线性压力降
        for vessel_id, vessel in self.vessels.items():
            if vessel.vessel_type == VesselType.ARTERY:
                vessel.pressure_start = inlet_pressure
                vessel.pressure_end = inlet_pressure * 0.9
            elif vessel.vessel_type == VesselType.ARTERIOLE:
                vessel.pressure_start = inlet_pressure * 0.9
                vessel.pressure_end = inlet_pressure * 0.6
            elif vessel.vessel_type == VesselType.CAPILLARY:
                vessel.pressure_start = inlet_pressure * 0.6
                vessel.pressure_end = inlet_pressure * 0.3
            elif vessel.vessel_type == VesselType.VENULE:
                vessel.pressure_start = inlet_pressure * 0.3
                vessel.pressure_end = outlet_pressure * 2
            else:  # VEIN
                vessel.pressure_start = outlet_pressure * 2
                vessel.pressure_end = outlet_pressure
            
            # 计算流量：Q = ΔP / R
            pressure_drop = vessel.pressure_start - vessel.pressure_end
            vessel.flow_rate = pressure_drop / vessel.resistance if vessel.resistance > 0 else 0.0

    def calculate_blood_flow_dynamics(self, dt: float) -> None:
        """更新血流动力学（简化版本）"""

        # 当前实现为稳态近似：每个时间步重新求解一次血流分布。
        # 该接口由 CompleteBrainSystem 的 VascularSystem 包装器调用。
        _ = dt  # dt 预留给未来的动态模型
        self.solve_blood_flow()
    
    def calculate_metabolite_exchange(self, dt: float, tissue_demands: Dict[str, float]) -> Dict[str, float]:
        """计算代谢物质交换"""
        
        total_exchange = {substance: 0.0 for substance in self.blood_concentrations.keys()}
        
        for vessel in self.vessels.values():
            if vessel.vessel_type != VesselType.CAPILLARY:
                continue
            
            # 计算毛细血管表面积
            radius = vessel.diameter / 2 * 1e-4  # cm
            length = vessel.length * 1e-4  # cm
            surface_area = 2 * np.pi * radius * length
            
            # 计算各种物质的交换
            for substance in self.blood_concentrations.keys():
                blood_conc = self.blood_concentrations[substance]
                brain_conc = self.brain_concentrations[substance]
                
                # 考虑组织需求
                if substance in tissue_demands:
                    brain_conc *= (1 + tissue_demands[substance])
                
                # 计算跨血脑屏障转运
                transport_rate = self.blood_brain_barrier.calculate_transport_rate(
                    substance, blood_conc, brain_conc, surface_area
                )
                
                total_exchange[substance] += transport_rate * dt
        
        # 更新脑组织浓度
        tissue_volume = np.prod(self.tissue_dimensions) * 1e-12  # L
        for substance, exchange in total_exchange.items():
            concentration_change = exchange / tissue_volume
            self.brain_concentrations[substance] += concentration_change
            
            # 防止负浓度
            self.brain_concentrations[substance] = max(0, self.brain_concentrations[substance])
        
        return total_exchange
    
    def get_vascular_statistics(self) -> Dict[str, Any]:
        """获取血管系统统计信息"""
        
        stats = {}
        
        # 血管类型统计
        type_counts = {}
        total_length = 0.0
        total_volume = 0.0
        
        for vessel in self.vessels.values():
            vessel_type = vessel.vessel_type.value
            type_counts[vessel_type] = type_counts.get(vessel_type, 0) + 1
            
            total_length += vessel.length
            radius = vessel.diameter / 2
            volume = np.pi * radius**2 * vessel.length
            total_volume += volume
        
        stats['vessel_type_counts'] = type_counts
        stats['total_vessels'] = len(self.vessels)
        stats['total_length_um'] = total_length
        stats['total_volume_um3'] = total_volume
        
        # 血流统计
        flow_rates = [vessel.flow_rate for vessel in self.vessels.values()]
        stats['mean_flow_rate'] = np.mean(flow_rates)
        stats['total_flow_rate'] = sum(flow_rates)
        
        # 压力统计
        pressures = []
        for vessel in self.vessels.values():
            pressures.extend([vessel.pressure_start, vessel.pressure_end])
        
        stats['mean_pressure'] = np.mean(pressures)
        stats['pressure_range'] = [min(pressures), max(pressures)]
        
        # 代谢物浓度
        stats['brain_concentrations'] = self.brain_concentrations.copy()
        
        return stats
    
    def visualize_network(self) -> Dict[str, Any]:
        """生成血管网络可视化数据"""
        
        visualization_data = {
            'vessels': [],
            'connections': []
        }
        
        for vessel in self.vessels.values():
            vessel_data = {
                'id': vessel.segment_id,
                'type': vessel.vessel_type.value,
                'start': vessel.start_position,
                'end': vessel.end_position,
                'diameter': vessel.diameter,
                'flow_rate': vessel.flow_rate,
                'pressure_start': vessel.pressure_start,
                'pressure_end': vessel.pressure_end
            }
            visualization_data['vessels'].append(vessel_data)
            
            # 连接信息
            for downstream_id in vessel.downstream_segments:
                visualization_data['connections'].append({
                    'from': vessel.segment_id,
                    'to': downstream_id
                })
        
        return visualization_data


class VascularSystem:
    """High-level vascular system wrapper used by the complete brain system.

    The underlying implementation lives in :class:`VascularNetwork`. This class
    provides a stable async lifecycle API and lightweight update hooks suitable
    for unit tests and higher-level orchestration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.network: Optional[VascularNetwork] = None
        self.is_initialized = False
        self.last_exchange: Dict[str, float] = {}
        self.last_demands: Dict[str, float] = {}

    async def initialize(self) -> None:
        tissue_dimensions = self.config.get("tissue_dimensions") or (1.0, 1.0, 1.0)
        self.network = VascularNetwork(tuple(tissue_dimensions))
        branching_levels = int(self.config.get("branching_levels", 3))
        branching_levels = max(1, min(branching_levels, 6))
        self.network.generate_vascular_tree(branching_levels=branching_levels)
        self.is_initialized = True

    async def update(self, dt: float, neural_activity: Any = None) -> Dict[str, Any]:
        if not self.is_initialized or self.network is None:
            return {"initialized": False, "exchange": {}, "demands": {}}

        activity_level = 0.0
        try:
            if neural_activity is None:
                activity_level = 0.0
            elif isinstance(neural_activity, dict):
                values = list(neural_activity.values())
                activity_level = float(np.mean(values)) if values else 0.0
            else:
                activity_level = float(np.mean(neural_activity))
        except Exception:
            activity_level = 0.0

        activity_level = float(np.clip(activity_level, 0.0, 1.0))
        self.last_demands = {
            "oxygen": 0.15 * activity_level,
            "glucose": 0.10 * activity_level,
        }

        self.network.calculate_blood_flow_dynamics(dt)
        self.last_exchange = self.network.calculate_metabolite_exchange(dt, self.last_demands)

        return {
            "initialized": True,
            "exchange": dict(self.last_exchange),
            "demands": dict(self.last_demands),
        }

    async def shutdown(self) -> None:
        self.is_initialized = False
        self.network = None
        self.last_exchange = {}
        self.last_demands = {}

    def get_system_state(self) -> Dict[str, Any]:
        if not self.is_initialized or self.network is None:
            return {"initialized": False}
        state = self.network.get_vascular_statistics()
        state.update(
            {
                "initialized": True,
                "last_exchange": dict(self.last_exchange),
                "last_demands": dict(self.last_demands),
            }
        )
        return state
