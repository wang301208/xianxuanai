"""
增强突触系统的配置类
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

class SynapseState(Enum):
    """突触状态枚举"""
    ACTIVE = "active"
    TAGGED = "tagged"
    POTENTIATED = "potentiated"
    DEPRESSED = "depressed"
    SILENT = "silent"

class PlasticityType(Enum):
    """可塑性类型枚举"""
    STDP = "stdp"
    HOMEOSTATIC = "homeostatic"
    L_LTP = "l_ltp"
    L_LTD = "l_ltd"
    METAPLASTICITY = "metaplasticity"

class NeuromodulatorType(Enum):
    """神经调质类型枚举"""
    DOPAMINE = "dopamine"
    ACETYLCHOLINE = "acetylcholine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    GLUTAMATE = "glutamate"
    GABA = "gaba"

@dataclass
class GlialConfig:
    """胶质细胞系统配置"""
    astrocyte_density: float = 50000.0  # cells/mm³
    microglia_density: float = 5000.0
    oligodendrocyte_density: float = 10000.0
    glutamate_uptake_rate: float = 100.0
    k_buffering_capacity: float = 0.8
    calcium_wave_speed: float = 20.0  # μm/ms
    atp_release_threshold: float = 1.0
    glucose_consumption: float = 0.1
    lactate_production: float = 0.05
    oxygen_consumption: float = 0.02
    molecular_weight_dependence: bool = True

@dataclass
class VolumeTransmissionConfig:
    """体积传导系统配置"""
    enabled: bool = True
    diffusion_space_fraction: float = 0.2
    tortuosity: float = 1.6
    clearance_rate: float = 0.1
    molecular_weight_dependence: bool = True

@dataclass
class NeuromodulatorConfig:
    """神经调质配置"""
    type: NeuromodulatorType
    baseline_concentration: float = 0.1  # μM
    release_rate: float = 0.5
    decay_tau: float = 1000.0  # ms
    diffusion_coefficient: float = 0.76e-6  # cm²/s
    receptor_affinity: float = 0.1
    max_effect: float = 2.0
    modulation_targets: List[str] = None
    dose_response_curve: str = "sigmoid"
    cooperativity: float = 2.0
    
    def __post_init__(self):
        if self.modulation_targets is None:
            self.modulation_targets = ['release_probability', 'synaptic_strength']

@dataclass
class EnhancedSynapseConfig:
    """增强突触配置"""
    weight: float = 1.0
    delay: float = 1.0  # ms
    baseline_release_probability: float = 0.5
    vesicle_pool_size: int = 100
    vesicle_refill_rate: float = 10.0  # vesicles/s
    tag_decay_tau: float = 3600000.0  # 1小时
    protein_synthesis_threshold: float = 0.5
    calcium_concentration: float = 0.1  # μM
    camp_concentration: float = 1.0  # μM
    
    def __post_init__(self):
        self.protein_levels = {
            'CaMKII': 1.0,
            'PKA': 1.0,
            'CREB': 1.0,
            'Arc': 0.0,
            'Homer': 1.0
        }