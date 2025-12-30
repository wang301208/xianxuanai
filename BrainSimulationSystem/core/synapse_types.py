"""
突触系统相关的枚举和数据类
Enums and data classes related to the synapse system.
"""
from enum import Enum
from dataclasses import dataclass, field

class NeurotransmitterType(Enum):
    """神经递质类型"""
    GLUTAMATE = "glutamate"
    GABA = "gaba"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"
    HISTAMINE = "histamine"
    GLYCINE = "glycine"

class ReceptorType(Enum):
    """受体类型"""
    # 谷氨酸受体
    AMPA = "ampa"
    NMDA = "nmda"
    KAINATE = "kainate"
    MGLUR1 = "mglur1"
    MGLUR2 = "mglur2"
    MGLUR5 = "mglur5"
    
    # GABA受体
    GABA_A = "gaba_a"
    GABA_B = "gaba_b"
    
    # 多巴胺受体
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"
    D4 = "d4"
    D5 = "d5"
    
    # 血清素受体
    HTR1A = "5ht1a"
    HTR2A = "5ht2a"
    HTR3 = "5ht3"
    
    # 胆碱受体
    NICOTINIC = "nicotinic"
    MUSCARINIC_M1 = "muscarinic_m1"
    MUSCARINIC_M2 = "muscarinic_m2"

@dataclass
class VesiclePool:
    """囊泡池模型"""
    readily_releasable: int = 10      # 即刻可释放池
    recycling: int = 100              # 循环池
    reserve: int = 1000               # 储备池
    
    # 动力学参数
    refill_rate: float = 0.1          # 补充速率 (1/ms)
    mobilization_rate: float = 0.05   # 动员速率 (1/ms)
    
    # 当前状态
    current_rr: int = field(init=False)
    current_recycling: int = field(init=False)
    current_reserve: int = field(init=False)
    
    def __post_init__(self):
        self.current_rr = self.readily_releasable
        self.current_recycling = self.recycling
        self.current_reserve = self.reserve

@dataclass
class ReceptorKinetics:
    """受体动力学参数"""
    receptor_type: ReceptorType
    
    # 结合动力学
    kon: float                        # 结合速率常数 (1/mM/ms)
    koff: float                       # 解离速率常数 (1/ms)
    
    # 门控动力学
    alpha: float                      # 开放速率 (1/ms)
    beta: float                       # 关闭速率 (1/ms)
    
    # 失敏动力学
    desensitization_rate: float       # 失敏速率 (1/ms)
    recovery_rate: float              # 恢复速率 (1/ms)
    
    # 电导参数
    single_channel_conductance: float # 单通道电导 (nS)
    reversal_potential: float         # 反转电位 (mV)
    
    # 调节参数
    mg_block: bool = False            # 镁离子阻断（NMDA）
    voltage_dependence: bool = False   # 电压依赖性