"""
Enhanced Cell Type Diversity System

实现真实的细胞类型多样性，包括：
- 多种神经元亚型（锥体细胞、中间神经元等）
- 胶质细胞（星形胶质细胞、少突胶质细胞、小胶质细胞）
- 血管细胞（内皮细胞、周细胞、平滑肌细胞）
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from .gpu_acceleration import get_gpu_accelerator

class CellType(Enum):
    """细胞类型枚举"""
    # 神经元类型
    PYRAMIDAL_L23 = "pyramidal_L2/3"
    PYRAMIDAL_L5A = "pyramidal_L5A"
    PYRAMIDAL_L5B = "pyramidal_L5B"
    PYRAMIDAL_L6 = "pyramidal_L6"
    
    # 中间神经元类型
    PV_INTERNEURON = "parvalbumin_interneuron"
    SST_INTERNEURON = "somatostatin_interneuron"
    VIP_INTERNEURON = "vasoactive_intestinal_peptide_interneuron"
    LAMP5_INTERNEURON = "lamp5_interneuron"
    
    # 胶质细胞类型
    ASTROCYTE_PROTOPLASMIC = "astrocyte_protoplasmic"
    ASTROCYTE_FIBROUS = "astrocyte_fibrous"
    OLIGODENDROCYTE = "oligodendrocyte"
    MICROGLIA_RAMIFIED = "microglia_ramified"
    MICROGLIA_ACTIVATED = "microglia_activated"
    
    # 血管细胞类型
    ENDOTHELIAL_CELL = "endothelial_cell"
    PERICYTE = "pericyte"
    SMOOTH_MUSCLE_CELL = "smooth_muscle_cell"


@dataclass
class CellParameters:
    """细胞参数数据类"""
    # 基本参数
    cell_type: CellType
    diameter: float  # 细胞直径 (μm)
    membrane_capacitance: float  # 膜电容 (pF)
    resting_potential: float  # 静息电位 (mV)
    
    # 电生理参数
    threshold: float  # 阈值电位 (mV)
    reset_potential: float  # 复位电位 (mV)
    refractory_period: float  # 不应期 (ms)
    
    # 离子通道参数
    na_conductance: float  # 钠通道电导 (nS)
    k_conductance: float  # 钾通道电导 (nS)
    ca_conductance: float  # 钙通道电导 (nS)
    
    # 形态学参数
    dendritic_length: float  # 树突总长度 (μm)
    axonal_length: float  # 轴突长度 (μm)
    spine_density: float  # 树突棘密度 (spines/μm)
    
    # 代谢参数
    glucose_consumption: float  # 葡萄糖消耗率 (μmol/min/g)
    oxygen_consumption: float  # 氧气消耗率 (μmol/min/g)

    # 高级单细胞动力学参数（默认值用于非神经元细胞，神经元可覆写）
    leak_conductance: float = 0.3  # 泄露通道电导 (nS)
    leak_reversal: float = -54.4  # 泄露平衡电位 (mV)
    axial_conductance: float = 5.0  # 体段耦合电导 (nS)
    dendritic_compartment_capacitance: float = 150.0  # 树突舱膜电容 (pF)
    dendritic_time_constant: float = 25.0  # 树突舱时间常数 (ms)
    neuron_model: str = "hodgkin_huxley"  # 神经元动力学模型类型
    model_parameters: Dict[str, float] = field(default_factory=dict)  # 模型特定可调参


class CellTypeDatabase:
    """细胞类型数据库，存储各种细胞的生理参数"""
    
    def __init__(self):
        self.cell_parameters = self._initialize_cell_database()
    
    def _initialize_cell_database(self) -> Dict[CellType, CellParameters]:
        """初始化细胞类型数据库"""
        
        return {
            # L2/3锥体细胞
            CellType.PYRAMIDAL_L23: CellParameters(
                cell_type=CellType.PYRAMIDAL_L23,
                diameter=15.0,
                membrane_capacitance=150.0,
                resting_potential=-70.0,
                threshold=-50.0,
                reset_potential=-65.0,
                refractory_period=2.0,
                na_conductance=120.0,
                k_conductance=36.0,
                ca_conductance=0.5,
                dendritic_length=3000.0,
                axonal_length=8000.0,
                spine_density=1.2,
                glucose_consumption=0.8,
                oxygen_consumption=2.4
            ),
            
            # L5A锥体细胞
            CellType.PYRAMIDAL_L5A: CellParameters(
                cell_type=CellType.PYRAMIDAL_L5A,
                diameter=18.0,
                membrane_capacitance=200.0,
                resting_potential=-68.0,
                threshold=-48.0,
                reset_potential=-63.0,
                refractory_period=2.5,
                na_conductance=140.0,
                k_conductance=42.0,
                ca_conductance=0.8,
                dendritic_length=4500.0,
                axonal_length=12000.0,
                spine_density=1.5,
                glucose_consumption=1.2,
                oxygen_consumption=3.6
            ),
            
            # L5B锥体细胞（厚束锥体细胞）
            CellType.PYRAMIDAL_L5B: CellParameters(
                cell_type=CellType.PYRAMIDAL_L5B,
                diameter=22.0,
                membrane_capacitance=280.0,
                resting_potential=-65.0,
                threshold=-45.0,
                reset_potential=-60.0,
                refractory_period=3.0,
                na_conductance=180.0,
                k_conductance=50.0,
                ca_conductance=1.2,
                dendritic_length=6000.0,
                axonal_length=20000.0,
                spine_density=1.8,
                glucose_consumption=1.8,
                oxygen_consumption=5.4
            ),
            
            # PV中间神经元（快速抑制）
            CellType.PV_INTERNEURON: CellParameters(
                cell_type=CellType.PV_INTERNEURON,
                diameter=12.0,
                membrane_capacitance=80.0,
                resting_potential=-75.0,
                threshold=-52.0,
                reset_potential=-70.0,
                refractory_period=1.0,
                na_conductance=100.0,
                k_conductance=80.0,
                ca_conductance=0.2,
                dendritic_length=1500.0,
                axonal_length=3000.0,
                spine_density=0.3,
                glucose_consumption=1.5,
                oxygen_consumption=4.5
            ),
            
            # SST中间神经元（树突抑制）
            CellType.SST_INTERNEURON: CellParameters(
                cell_type=CellType.SST_INTERNEURON,
                diameter=10.0,
                membrane_capacitance=60.0,
                resting_potential=-72.0,
                threshold=-55.0,
                reset_potential=-67.0,
                refractory_period=1.5,
                na_conductance=80.0,
                k_conductance=60.0,
                ca_conductance=0.3,
                dendritic_length=2000.0,
                axonal_length=4000.0,
                spine_density=0.5,
                glucose_consumption=1.0,
                oxygen_consumption=3.0
            ),
            
            # 原生质型星形胶质细胞
            CellType.ASTROCYTE_PROTOPLASMIC: CellParameters(
                cell_type=CellType.ASTROCYTE_PROTOPLASMIC,
                diameter=25.0,
                membrane_capacitance=50.0,
                resting_potential=-85.0,
                threshold=0.0,  # 不产生动作电位
                reset_potential=-85.0,
                refractory_period=0.0,
                na_conductance=0.0,
                k_conductance=15.0,
                ca_conductance=2.0,
                dendritic_length=0.0,
                axonal_length=0.0,
                spine_density=0.0,
                glucose_consumption=0.3,
                oxygen_consumption=0.9
            ),
            
            # 少突胶质细胞
            CellType.OLIGODENDROCYTE: CellParameters(
                cell_type=CellType.OLIGODENDROCYTE,
                diameter=8.0,
                membrane_capacitance=30.0,
                resting_potential=-80.0,
                threshold=0.0,
                reset_potential=-80.0,
                refractory_period=0.0,
                na_conductance=0.0,
                k_conductance=10.0,
                ca_conductance=0.5,
                dendritic_length=0.0,
                axonal_length=0.0,
                spine_density=0.0,
                glucose_consumption=0.2,
                oxygen_consumption=0.6
            ),
            
            # 分支型小胶质细胞
            CellType.MICROGLIA_RAMIFIED: CellParameters(
                cell_type=CellType.MICROGLIA_RAMIFIED,
                diameter=6.0,
                membrane_capacitance=20.0,
                resting_potential=-60.0,
                threshold=0.0,
                reset_potential=-60.0,
                refractory_period=0.0,
                na_conductance=0.0,
                k_conductance=5.0,
                ca_conductance=1.0,
                dendritic_length=0.0,
                axonal_length=0.0,
                spine_density=0.0,
                glucose_consumption=0.1,
                oxygen_consumption=0.3
            ),
            
            # 血管内皮细胞
            CellType.ENDOTHELIAL_CELL: CellParameters(
                cell_type=CellType.ENDOTHELIAL_CELL,
                diameter=20.0,
                membrane_capacitance=40.0,
                resting_potential=-50.0,
                threshold=0.0,
                reset_potential=-50.0,
                refractory_period=0.0,
                na_conductance=0.0,
                k_conductance=8.0,
                ca_conductance=3.0,
                dendritic_length=0.0,
                axonal_length=0.0,
                spine_density=0.0,
                glucose_consumption=0.4,
                oxygen_consumption=1.2
            ),
            
            # 周细胞
            CellType.PERICYTE: CellParameters(
                cell_type=CellType.PERICYTE,
                diameter=15.0,
                membrane_capacitance=35.0,
                resting_potential=-45.0,
                threshold=0.0,
                reset_potential=-45.0,
                refractory_period=0.0,
                na_conductance=0.0,
                k_conductance=6.0,
                ca_conductance=4.0,
                dendritic_length=0.0,
                axonal_length=0.0,
                spine_density=0.0,
                glucose_consumption=0.3,
                oxygen_consumption=0.9
            )
        }
    
    def get_cell_parameters(self, cell_type: CellType) -> CellParameters:
        """获取指定细胞类型的参数"""
        return self.cell_parameters[cell_type]
    
    def get_all_neuron_types(self) -> List[CellType]:
        """获取所有神经元类型"""
        return [ct for ct in CellType if 'PYRAMIDAL' in ct.value or 'INTERNEURON' in ct.value]
    
    def get_all_glial_types(self) -> List[CellType]:
        """获取所有胶质细胞类型"""
        return [ct for ct in CellType if any(x in ct.value for x in ['ASTROCYTE', 'OLIGODENDROCYTE', 'MICROGLIA'])]
    
    def get_all_vascular_types(self) -> List[CellType]:
        """获取所有血管细胞类型"""
        return [ct for ct in CellType if any(x in ct.value for x in ['ENDOTHELIAL', 'PERICYTE', 'SMOOTH_MUSCLE'])]


class EnhancedCell(ABC):
    """增强型细胞基类"""
    
    def __init__(self, cell_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.position = position  # (x, y, z) 坐标
        
        # 从数据库获取参数
        db = CellTypeDatabase()
        self.parameters = db.get_cell_parameters(cell_type)
        
        # 状态变量
        self.membrane_potential = self.parameters.resting_potential
        self.calcium_concentration = 0.1  # μM
        self.metabolic_state = 1.0  # 代谢活跃度
        
        # 连接信息
        self.connections: List[int] = []  # 连接的细胞ID列表
        
    @abstractmethod
    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """更新细胞状态"""
        # 更新膜电位
        input_current = sum(inputs.values())
        
        # 简化的膜电位更新
        if hasattr(self, 'membrane_potential'):
            self.membrane_potential += dt * (-self.membrane_potential + input_current) / 10.0
        else:
            self.membrane_potential = input_current * dt
        
        # 更新代谢状态
        self.metabolic_state = max(0.1, min(1.0, self.metabolic_state + dt * 0.001))
        
        # 检查是否发放动作电位
        spike_occurred = False
        if hasattr(self, 'threshold') and self.membrane_potential > self.threshold:
            spike_occurred = True
            self.membrane_potential = getattr(self, 'reset_potential', -70.0)
        
        return {
            'membrane_potential': getattr(self, 'membrane_potential', 0.0),
            'metabolic_state': self.metabolic_state,
            'spike': spike_occurred,
            'cell_type': self.cell_type.value if hasattr(self.cell_type, 'value') else str(self.cell_type)
        }
    
    def get_distance_to(self, other_cell: 'EnhancedCell') -> float:
        """计算到另一个细胞的距离"""
        dx = self.position[0] - other_cell.position[0]
        dy = self.position[1] - other_cell.position[1]
        dz = self.position[2] - other_cell.position[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)


class GenericCell(EnhancedCell):
    """通用细胞，占位实现用于未专门建模的细胞类型"""

    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """调用基类更新逻辑"""
        numeric_inputs = {k: v for k, v in inputs.items() if isinstance(v, (int, float))}
        return super().update(dt, numeric_inputs)


class EnhancedNeuron(EnhancedCell):
    """增强型神经元：支持霍奇金-赫胥黎 / Izhikevich 模型及多段电缆结构"""

    def __init__(self, cell_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        super().__init__(cell_id, cell_type, position)

        # 神经元发放状态
        self.spike_times: List[float] = []
        self.last_spike_time = -np.inf
        self.refractory_timer = 0.0
        self._prev_voltage = self.membrane_potential

        # 模型选择（缺省为 Hodgkin-Huxley，可覆写为 Izhikevich）
        self.model_type = getattr(self.parameters, 'neuron_model', 'hodgkin_huxley').lower()
        self.model_config = dict(getattr(self.parameters, 'model_parameters', {}))

        # 多段电缆/树突舱基础参数
        self.dendritic_potential = self.membrane_potential - 1.5
        self.axial_conductance = getattr(self.parameters, 'axial_conductance', 5.0)
        self.leak_conductance = getattr(self.parameters, 'leak_conductance', 0.3)
        self.leak_reversal = getattr(self.parameters, 'leak_reversal', self.parameters.resting_potential + 10.0)
        self.dendritic_capacitance = max(1e-3, getattr(self.parameters, 'dendritic_compartment_capacitance', 150.0))
        self.dendritic_time_constant = max(1e-3, getattr(self.parameters, 'dendritic_time_constant', 25.0))

        default_profiles: Dict[CellType, Tuple[str, Dict[str, float]]] = {
            CellType.PV_INTERNEURON: ('izhikevich', {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0, 'v_peak': 25.0, 'soma_synaptic_share': 0.85}),
            CellType.VIP_INTERNEURON: ('izhikevich', {'a': 0.02, 'b': 0.2, 'c': -50.0, 'd': 2.0, 'v_peak': 35.0, 'soma_synaptic_share': 0.75}),
            CellType.SST_INTERNEURON: ('izhikevich', {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.0, 'v_peak': 30.0, 'soma_synaptic_share': 0.8}),
            CellType.LAMP5_INTERNEURON: ('izhikevich', {'a': 0.02, 'b': 0.3, 'c': -60.0, 'd': 4.0, 'v_peak': 30.0, 'soma_synaptic_share': 0.7}),
            CellType.PYRAMIDAL_L23: ('izhikevich', {'a': 0.02, 'b': 0.2, 'c': -60.0, 'd': 6.0, 'v_peak': 30.0, 'soma_synaptic_share': 0.9, 'dendritic_synaptic_share': 0.1, 'axial_conductance': 0.2}),
            CellType.PYRAMIDAL_L5A: ('izhikevich', {'a': 0.02, 'b': 0.2, 'c': -55.0, 'd': 4.0, 'v_peak': 32.0, 'soma_synaptic_share': 0.9, 'dendritic_synaptic_share': 0.1, 'axial_conductance': 0.25}),
            CellType.PYRAMIDAL_L5B: ('izhikevich', {'a': 0.02, 'b': 0.2, 'c': -52.0, 'd': 4.0, 'v_peak': 32.0, 'soma_synaptic_share': 0.9, 'dendritic_synaptic_share': 0.1, 'axial_conductance': 0.25})
        }
        profile = default_profiles.get(cell_type)
        if profile and not getattr(self.parameters, 'model_parameters', {}):
            default_model, preset_params = profile
            self.model_type = default_model
            self.model_config.update(preset_params)

        override_targets = {
            'leak_conductance': ('leak_conductance', 1e-6),
            'leak_reversal': ('leak_reversal', None),
            'axial_conductance': ('axial_conductance', 1e-6),
            'dendritic_time_constant': ('dendritic_time_constant', 1e-6),
            'dendritic_compartment_capacitance': ('dendritic_capacitance', 1e-6)
        }
        for key, (attr, minimum) in override_targets.items():
            if key in self.model_config:
                value = float(self.model_config[key])
                if minimum is not None:
                    value = max(minimum, value)
                setattr(self, attr, value)

        # 默认离子门变量与适应状态
        self.adaptation_current = 0.0
        self.na_activation = 0.0
        self.na_inactivation = 1.0
        self.k_activation = 0.0
        self.ca_activation = 0.1
        self.izhikevich_params: Optional[Dict[str, float]] = None

        if self.model_type == 'izhikevich':
            default_izh = {
                'a': 0.02,
                'b': 0.2,
                'c': -65.0,
                'd': 6.0,
                'v_peak': 30.0
            }
            default_izh.update({k: float(v) for k, v in self.model_config.items() if isinstance(v, (int, float))})
            self.izhikevich_params = default_izh
            self.adaptation_current = self.izhikevich_params['b'] * self.membrane_potential
        else:
            self.model_type = 'hodgkin_huxley'
            self.na_activation, self.na_inactivation, self.k_activation = self._initialize_hodgkin_huxley_gates(self.membrane_potential)

    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """更新神经元状态，输出包括多段电缆与离子动力学信息"""
        current_time = float(inputs.get('time', 0.0))
        synaptic_current = float(inputs.get('synaptic_current', 0.0))
        external_current = float(inputs.get('external_current', 0.0))
        dendritic_drive = float(inputs.get('dendritic_current', 0.0))
        noise_current = float(inputs.get('noise', 0.0))

        self.refractory_timer = max(0.0, self.refractory_timer - dt)

        if self.model_type == 'izhikevich':
            spike_occurred = self._izhikevich_step(
                dt,
                current_time,
                synaptic_current + external_current + noise_current,
                synaptic_current + noise_current,
                dendritic_drive
            )
        else:
            spike_occurred = self._hodgkin_huxley_step(
                dt,
                current_time,
                synaptic_current + noise_current,
                external_current,
                dendritic_drive
            )

        output = {
            'spike': spike_occurred,
            'voltage': self.membrane_potential,
            'dendrite_voltage': self.dendritic_potential,
            'calcium': self.calcium_concentration,
            'adaptation': self.adaptation_current,
            'model': self.model_type
        }
        if hasattr(self, '_last_effective_current'):
            output['effective_current'] = self._last_effective_current
        if hasattr(self, '_last_stimulus_current'):
            output['stimulus_current'] = self._last_stimulus_current
        if hasattr(self, '_last_ionic_current'):
            output['ionic_current'] = self._last_ionic_current
        if hasattr(self, '_last_coupling_current'):
            output['dendritic_coupling_current'] = self._last_coupling_current
        return output

    def _initialize_hodgkin_huxley_gates(self, voltage: float) -> Tuple[float, float, float]:
        """根据静息电位初始化 m/h/n 门变量"""
        alpha_m = self._alpha_m(voltage)
        beta_m = self._beta_m(voltage)
        alpha_h = self._alpha_h(voltage)
        beta_h = self._beta_h(voltage)
        alpha_n = self._alpha_n(voltage)
        beta_n = self._beta_n(voltage)
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        n_inf = alpha_n / (alpha_n + beta_n)
        return m_inf, h_inf, n_inf

    def _hodgkin_huxley_step(
        self,
        dt: float,
        current_time: float,
        synaptic_current: float,
        external_current: float,
        dendritic_drive: float
    ) -> bool:
        """霍奇金-赫胥黎 + 双段电缆模型积分"""
        prev_voltage = self._prev_voltage
        current_voltage = self.membrane_potential

        coupling_current = self._update_dendritic_compartment(dt, current_voltage, synaptic_current, dendritic_drive)

        alpha_m = self._alpha_m(current_voltage)
        beta_m = self._beta_m(current_voltage)
        alpha_h = self._alpha_h(current_voltage)
        beta_h = self._beta_h(current_voltage)
        alpha_n = self._alpha_n(current_voltage)
        beta_n = self._beta_n(current_voltage)

        dm_dt = alpha_m * (1.0 - self.na_activation) - beta_m * self.na_activation
        dh_dt = alpha_h * (1.0 - self.na_inactivation) - beta_h * self.na_inactivation
        dn_dt = alpha_n * (1.0 - self.k_activation) - beta_n * self.k_activation

        self.na_activation += dm_dt * dt
        self.na_inactivation += dh_dt * dt
        self.k_activation += dn_dt * dt

        self.na_activation = float(np.clip(self.na_activation, 0.0, 1.0))
        self.na_inactivation = float(np.clip(self.na_inactivation, 0.0, 1.0))
        self.k_activation = float(np.clip(self.k_activation, 0.0, 1.0))

        ca_inf = 1.0 / (1.0 + np.exp(-(current_voltage + 35.0) / 6.0))
        tau_ca = self.model_config.get('ca_tau', 80.0)
        self.ca_activation += dt * (ca_inf - self.ca_activation) / max(tau_ca, 1.0)
        self.ca_activation = float(np.clip(self.ca_activation, 0.0, 1.0))

        e_na = 50.0
        e_k = -77.0
        e_ca = 132.0

        g_na = self.parameters.na_conductance * self.model_config.get('na_conductance_scale', 1.0)
        g_k = self.parameters.k_conductance * self.model_config.get('k_conductance_scale', 1.0)
        g_ca = self.parameters.ca_conductance * self.model_config.get('ca_conductance_scale', 1.0)
        g_leak = self.leak_conductance

        i_na = g_na * (self.na_activation ** 3) * self.na_inactivation * (current_voltage - e_na)
        i_k = g_k * (self.k_activation ** 4) * (current_voltage - e_k)
        i_ca = g_ca * (self.ca_activation ** 2) * (current_voltage - e_ca)
        i_leak = g_leak * (current_voltage - self.leak_reversal)

        ionic_current = i_na + i_k + i_ca + i_leak
        self._last_ionic_current = ionic_current

        adaptation_tau = self.model_config.get('adaptation_tau', 120.0)
        self.adaptation_current *= np.exp(-dt / max(adaptation_tau, 1.0))
        
        soma_share = float(np.clip(self.model_config.get('soma_synaptic_share', 0.5), 0.0, 1.0))
        stimulus_current = external_current + synaptic_current * soma_share + coupling_current - self.adaptation_current
        self._last_stimulus_current = stimulus_current
        total_current = stimulus_current - ionic_current

        dv_dt = total_current / max(self.parameters.membrane_capacitance, 1e-3)
        self.membrane_potential += dv_dt * dt

        self._update_calcium(dt, inward_current=i_ca)

        spike_occurred = False
        if self.refractory_timer <= 0.0 and prev_voltage < self.parameters.threshold <= self.membrane_potential:
            spike_occurred = True
            self.spike_times.append(current_time)
            self.last_spike_time = current_time
            self.refractory_timer = self.parameters.refractory_period
            self.adaptation_current += self.model_config.get('adaptation_increment', 2.5)

        self.membrane_potential = float(np.clip(self.membrane_potential, -120.0, 80.0))
        self._prev_voltage = self.membrane_potential
        return spike_occurred

    def _izhikevich_step(
        self,
        dt: float,
        current_time: float,
        total_input_current: float,
        synaptic_component: float,
        dendritic_drive: float
    ) -> bool:
        """Izhikevich 模型积分，同时考虑树突舱耦合"""
        params = self.izhikevich_params or {}

        coupling_current = self._update_dendritic_compartment(dt, self.membrane_potential, synaptic_component, dendritic_drive)
        soma_share = float(np.clip(self.model_config.get('soma_synaptic_share', 0.5), 0.0, 1.0))
        external_component = total_input_current - synaptic_component
        effective_current = external_component + synaptic_component * soma_share + coupling_current
        self._last_coupling_current = coupling_current
        self._last_effective_current = effective_current

        sub_steps = max(1, int(np.ceil(dt / 0.5)))
        sub_dt = dt / sub_steps
        v = self.membrane_potential
        u = self.adaptation_current
        spike_occurred = False

        for _ in range(sub_steps):
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + effective_current
            du = params.get('a', 0.02) * (params.get('b', 0.2) * v - u)
            v += dv * sub_dt
            u += du * sub_dt
            if v >= params.get('v_peak', 30.0):
                v = params.get('c', -65.0)
                u += params.get('d', 6.0)
                spike_occurred = True
                self.spike_times.append(current_time)
                self.last_spike_time = current_time
                self.refractory_timer = self.parameters.refractory_period

        self.membrane_potential = float(np.clip(v, -120.0, 80.0))
        self.adaptation_current = u
        self._prev_voltage = self.membrane_potential
        self._update_calcium(dt, spike=spike_occurred)
        return spike_occurred

    def _update_dendritic_compartment(
        self,
        dt: float,
        soma_voltage: float,
        synaptic_current: float,
        dendritic_drive: float
    ) -> float:
        """树突舱动力学，返回与胞体耦合产生的电流"""
        soma_share = float(np.clip(self.model_config.get('soma_synaptic_share', 0.5), 0.0, 1.0))
        dendritic_share = float(np.clip(self.model_config.get('dendritic_synaptic_share', 1.0 - soma_share), 0.0, 1.0))
        dendritic_input = dendritic_share * synaptic_current + dendritic_drive
        relaxation = -(self.dendritic_potential - self.parameters.resting_potential) / max(self.dendritic_time_constant, 1e-3)
        drive = dendritic_input / max(self.dendritic_capacitance, 1e-3)
        self.dendritic_potential += (relaxation + drive) * dt
        coupling = self.axial_conductance * (self.dendritic_potential - soma_voltage)
        return coupling

    def _update_calcium(self, dt: float, spike: bool = False, inward_current: float = 0.0) -> None:
        """更新胞内钙浓度，结合尖峰驱动与钙电流"""
        influx = 0.0
        if spike:
            influx += 0.8
        if inward_current < 0.0:
            influx += (-inward_current) * 0.001
        decay_tau = self.model_config.get('calcium_tau', 60.0)
        self.calcium_concentration += dt * (influx - self.calcium_concentration / max(decay_tau, 1.0))
        self.calcium_concentration = max(0.0, self.calcium_concentration)

    @staticmethod
    def _alpha_m(voltage: float) -> float:
        delta = voltage + 40.0
        if abs(delta) < 1e-5:
            return 1.0
        return 0.1 * delta / (1.0 - np.exp(-delta / 10.0))

    @staticmethod
    def _beta_m(voltage: float) -> float:
        return 4.0 * np.exp(-(voltage + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(voltage: float) -> float:
        return 0.07 * np.exp(-(voltage + 65.0) / 20.0)

    @staticmethod
    def _beta_h(voltage: float) -> float:
        return 1.0 / (1.0 + np.exp(-(voltage + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(voltage: float) -> float:
        delta = voltage + 55.0
        if abs(delta) < 1e-5:
            return 0.1
        return 0.01 * delta / (1.0 - np.exp(-delta / 10.0))

    @staticmethod
    def _beta_n(voltage: float) -> float:
        return 0.125 * np.exp(-(voltage + 65.0) / 80.0)

class Astrocyte(EnhancedCell):
    """星形胶质细胞"""
    
    def __init__(self, cell_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        super().__init__(cell_id, cell_type, position)
        
        # 星形胶质细胞特有状态
        self.glutamate_uptake_rate = 0.0
        self.potassium_buffering = 0.0
        self.glycogen_stores = 100.0  # μmol/g
        self.lactate_production = 0.0
        
        # 血管连接
        self.vascular_contacts: List[int] = []
        
    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """更新星形胶质细胞状态"""
        
        # 谷氨酸摄取
        extracellular_glutamate = inputs.get('glutamate', 0.0)
        self.glutamate_uptake_rate = 0.8 * extracellular_glutamate  # 摄取80%
        
        # 钾离子缓冲
        extracellular_k = inputs.get('potassium', 3.5)  # mM
        if extracellular_k > 3.5:
            self.potassium_buffering = 0.6 * (extracellular_k - 3.5)
        
        # 糖原代谢和乳酸生产
        neural_activity = inputs.get('neural_activity', 0.0)
        if neural_activity > 0.5:
            glycogen_consumption = 2.0 * neural_activity * dt
            self.glycogen_stores = max(0, self.glycogen_stores - glycogen_consumption)
            self.lactate_production = glycogen_consumption * 2.0  # 产生乳酸
        
        # 糖原补充（较慢）
        self.glycogen_stores = min(100.0, self.glycogen_stores + 0.1 * dt)
        
        return {
            'glutamate_uptake': self.glutamate_uptake_rate,
            'k_buffering': self.potassium_buffering,
            'glycogen': self.glycogen_stores,
            'lactate': self.lactate_production
        }


class Microglia(EnhancedCell):
    """小胶质细胞"""
    
    def __init__(self, cell_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        super().__init__(cell_id, cell_type, position)
        
        # 小胶质细胞状态
        self.activation_level = 0.0  # 0=静息，1=完全激活
        self.phagocytic_activity = 0.0
        self.cytokine_release = 0.0
        self.surveillance_radius = 50.0  # μm
        
    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """更新小胶质细胞状态"""
        
        # 检测损伤信号
        damage_signals = inputs.get('damage_signals', 0.0)
        inflammatory_signals = inputs.get('inflammatory_signals', 0.0)
        
        # 激活响应
        activation_stimulus = damage_signals + inflammatory_signals
        if activation_stimulus > 0.3:
            self.activation_level = min(1.0, self.activation_level + 0.1 * dt)
        else:
            self.activation_level = max(0.0, self.activation_level - 0.05 * dt)
        
        # 吞噬活动
        self.phagocytic_activity = self.activation_level * 0.8
        
        # 细胞因子释放
        if self.activation_level > 0.5:
            self.cytokine_release = self.activation_level * 0.6
        else:
            self.cytokine_release = 0.0
        
        return {
            'activation': self.activation_level,
            'phagocytosis': self.phagocytic_activity,
            'cytokines': self.cytokine_release
        }


class VascularCell(EnhancedCell):
    """血管细胞基类"""
    
    def __init__(self, cell_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        super().__init__(cell_id, cell_type, position)
        
        # 血管细胞共同特征
        self.vessel_diameter = 10.0  # μm
        self.blood_flow_rate = 0.0  # ml/min/g
        self.oxygen_delivery = 0.0
        self.glucose_delivery = 0.0
        
    def update(self, dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """更新血管细胞状态"""
        
        # 血流调节
        metabolic_demand = inputs.get('metabolic_demand', 1.0)
        self.blood_flow_rate = 0.5 * metabolic_demand  # 基础血流调节
        
        # 氧气和葡萄糖输送
        self.oxygen_delivery = self.blood_flow_rate * 0.2  # μmol/min/g
        self.glucose_delivery = self.blood_flow_rate * 0.5  # μmol/min/g
        
        return {
            'blood_flow': self.blood_flow_rate,
            'oxygen_delivery': self.oxygen_delivery,
            'glucose_delivery': self.glucose_delivery
        }


class CellPopulationManager:
    """细胞群体管理器"""
    
    def __init__(self, tissue_volume: Tuple[float, float, float]):
        self.tissue_volume = tissue_volume  # (width, height, depth) in μm
        self.cells: Dict[int, EnhancedCell] = {}
        self.cell_counter = 0
        self._gpu_accelerator = get_gpu_accelerator()
        
        # 细胞密度（每立方毫米）
        self.cell_densities = {
            CellType.PYRAMIDAL_L23: 15000,
            CellType.PYRAMIDAL_L5A: 8000,
            CellType.PYRAMIDAL_L5B: 5000,
            CellType.PV_INTERNEURON: 2000,
            CellType.SST_INTERNEURON: 1500,
            CellType.ASTROCYTE_PROTOPLASMIC: 3000,
            CellType.OLIGODENDROCYTE: 1000,
            CellType.MICROGLIA_RAMIFIED: 500,
            CellType.ENDOTHELIAL_CELL: 800,
            CellType.PERICYTE: 200
        }
    
    def populate_tissue(self, layer_boundaries: Dict[str, Tuple[float, float]]):
        """根据层边界填充组织"""
        
        volume_mm3 = (self.tissue_volume[0] * self.tissue_volume[1] * self.tissue_volume[2]) / (1000**3)
        
        for cell_type, density in self.cell_densities.items():
            # 计算该类型细胞数量
            if 'L23' in cell_type.value:
                layer_fraction = self._get_layer_fraction('L2/3', layer_boundaries)
            elif 'L5' in cell_type.value:
                layer_fraction = self._get_layer_fraction('L5', layer_boundaries)
            else:
                layer_fraction = 1.0  # 胶质细胞和血管细胞分布在所有层
            
            cell_count = int(density * volume_mm3 * layer_fraction)
            
            # 创建细胞
            for _ in range(cell_count):
                position = self._generate_cell_position(cell_type, layer_boundaries)
                cell = self._create_cell(cell_type, position)
                self.cells[cell.cell_id] = cell
    
    def _get_layer_fraction(self, layer_name: str, layer_boundaries: Dict[str, Tuple[float, float]]) -> float:
        """计算层在总体积中的比例"""
        total_thickness = float(self.tissue_volume[2])
        if total_thickness <= 0:
            return 0.0

        def _span(bounds: Tuple[float, float]) -> float:
            return abs(bounds[1] - bounds[0])

        if layer_name in layer_boundaries:
            return _span(layer_boundaries[layer_name]) / total_thickness

        if layer_name == 'L2/3':
            spans = [_span(layer_boundaries[name]) for name in ('L2', 'L3', 'L2/3') if name in layer_boundaries]
            return sum(spans) / total_thickness if spans else 0.0
        if layer_name == 'L5':
            spans = [_span(layer_boundaries[name]) for name in ('L5', 'L5A', 'L5B') if name in layer_boundaries]
            return sum(spans) / total_thickness if spans else 0.0

        return 0.0
    
    def _generate_cell_position(self, cell_type: CellType, layer_boundaries: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float]:
        """生成细胞位置"""
        
        x = np.random.uniform(0, self.tissue_volume[0])
        y = np.random.uniform(0, self.tissue_volume[1])

        def choose_depth(candidate_layers: Tuple[str, ...]) -> float:
            segments = [(layer_boundaries[name][0], layer_boundaries[name][1]) for name in candidate_layers if name in layer_boundaries]
            if not segments:
                return np.random.uniform(0, self.tissue_volume[2])
            lengths = [abs(seg[1] - seg[0]) for seg in segments]
            total = sum(lengths)
            if total <= 0:
                return np.random.uniform(segments[0][0], segments[0][1])
            r = np.random.uniform(0, total)
            accum = 0.0
            for (z_min, z_max), length in zip(segments, lengths):
                if r <= accum + length:
                    return np.random.uniform(z_min, z_max)
                accum += length
            return np.random.uniform(segments[-1][0], segments[-1][1])
        
        # 根据细胞类型确定z坐标
        if 'L23' in cell_type.value:
            z = choose_depth(('L2', 'L3', 'L2/3'))
        elif 'L4' in cell_type.value or 'stellate' in cell_type.value.lower():
            z = choose_depth(('L4',))
        elif 'L5' in cell_type.value:
            z = choose_depth(('L5', 'L5A', 'L5B'))
        elif 'L6' in cell_type.value:
            z = choose_depth(('L6',))
        else:
            z = np.random.uniform(0, self.tissue_volume[2])
        
        return (x, y, z)
    
    def _create_cell(self, cell_type: CellType, position: Tuple[float, float, float]) -> EnhancedCell:
        """创建指定类型的细胞"""
        
        cell_id = self.cell_counter
        self.cell_counter += 1
        
        if any(x in cell_type.value for x in ['PYRAMIDAL', 'INTERNEURON']):
            return EnhancedNeuron(cell_id, cell_type, position)
        elif 'ASTROCYTE' in cell_type.value:
            return Astrocyte(cell_id, cell_type, position)
        elif 'MICROGLIA' in cell_type.value:
            return Microglia(cell_id, cell_type, position)
        elif any(x in cell_type.value for x in ['ENDOTHELIAL', 'PERICYTE']):
            return VascularCell(cell_id, cell_type, position)
        else:
            return GenericCell(cell_id, cell_type, position)
    
    def get_cells_by_type(self, cell_type: CellType) -> List[EnhancedCell]:
        """获取指定类型的所有细胞"""
        return [cell for cell in self.cells.values() if cell.cell_type == cell_type]
    
    def get_cells_in_radius(self, center: Tuple[float, float, float], radius: float) -> List[EnhancedCell]:
        """获取指定半径内的所有细胞"""
        result = []
        for cell in self.cells.values():
            distance = np.sqrt(sum((cell.position[i] - center[i])**2 for i in range(3)))
            if distance <= radius:
                result.append(cell)
        return result
    
    def update_all_cells(self, dt: float, global_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新所有细胞状态，优先尝试使用GPU批量加速。"""
        
        results: Dict[int, Dict[str, Any]] = {}
        gpu_entries: List[Tuple[int, EnhancedCell, Dict[str, Any]]] = []
        accelerator = self._gpu_accelerator or get_gpu_accelerator()
        self._gpu_accelerator = accelerator
        
        for cell_id, cell in self.cells.items():
            # 为每个细胞准备输入
            cell_inputs = global_inputs.copy()
            
            # 添加局部环境信息
            local_cells = self.get_cells_in_radius(cell.position, 50.0)  # 50μm半径
            cell_inputs['local_cell_count'] = len(local_cells)
            
            if accelerator and isinstance(cell, EnhancedCell) and accelerator.can_accelerate(cell, cell_inputs):
                gpu_entries.append((cell_id, cell, cell_inputs))
            else:
                results[cell_id] = cell.update(dt, cell_inputs)
        
        if accelerator and gpu_entries:
            gpu_results = accelerator.run(gpu_entries, dt)
            results.update(gpu_results)
        
        return results
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """获取细胞群体统计信息"""
        
        stats = {}
        
        # 按类型统计细胞数量
        type_counts = {}
        for cell in self.cells.values():
            cell_type = cell.cell_type
            type_counts[cell_type.value] = type_counts.get(cell_type.value, 0) + 1
        
        stats['cell_type_counts'] = type_counts
        stats['total_cells'] = len(self.cells)
        
        # 神经元统计
        neurons = [cell for cell in self.cells.values() if isinstance(cell, EnhancedNeuron)]
        if neurons:
            spike_rates = []
            for neuron in neurons:
                if len(neuron.spike_times) > 0:
                    # 计算最近1秒的发放率
                    recent_spikes = [t for t in neuron.spike_times if t > (max(neuron.spike_times) - 1000)]
                    spike_rates.append(len(recent_spikes))
                else:
                    spike_rates.append(0)
            
            stats['mean_spike_rate'] = np.mean(spike_rates)
            stats['active_neurons'] = sum(1 for rate in spike_rates if rate > 0)
        
        # 胶质细胞统计
        astrocytes = [cell for cell in self.cells.values() if isinstance(cell, Astrocyte)]
        if astrocytes:
            stats['mean_glycogen'] = np.mean([ast.glycogen_stores for ast in astrocytes])
            stats['mean_lactate_production'] = np.mean([ast.lactate_production for ast in astrocytes])
        
        return stats

class CellDiversitySystem:
    """Facade providing high level access to cell diversity resources."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.database = CellTypeDatabase()
        self.population_manager = CellPopulationManager(self.database)

    async def initialize(self) -> None:
        """Async lifecycle hook used by the complete brain system."""
        return None

    def register_cell_type(self, cell_type: CellType, params: Dict[str, float]) -> None:
        self.database.register_cell_type(cell_type, params)

    def create_population(self, name: str, cell_type: CellType, size: int) -> Dict[str, Any]:
        return self.population_manager.create_population(name, cell_type, size)

    def get_population_stats(self, name: str) -> Dict[str, Any]:
        return self.population_manager.get_population_stats(name)

    def list_available_cell_types(self) -> Dict[str, CellParameters]:
        return self.database.cell_types

    def get_cell_type_distribution(self) -> Dict[str, int]:
        """Return a lightweight distribution snapshot for tests/telemetry."""
        params = getattr(self.database, "cell_parameters", {}) or {}
        return {cell_type.value: 1 for cell_type in params.keys()}
__all__ = ['CellType', 'CellParameters', 'CellTypeDatabase', 'CellPopulationManager', 'CellDiversitySystem']
