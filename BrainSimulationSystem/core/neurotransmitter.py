"""
神经递质动力学模型
Neurotransmitter Dynamics Model
"""
import logging
import numpy as np
from .synapse_types import NeurotransmitterType

class NeurotransmitterDynamics:
    """
    模拟神经递质在突触间隙中的释放、扩散和清除过程。
    """
    
    def __init__(self, nt_type: NeurotransmitterType):
        """
        初始化神经递质动力学模型。

        Args:
            nt_type (NeurotransmitterType): 神经递质的类型。
        """
        self.nt_type = nt_type
        self.logger = logging.getLogger(f"NT_{nt_type.value}")
        
        # 从预定义数据库中获取该递质的参数
        params = self._get_nt_params()
        self.release_probability: float = params['release_probability']
        self.quantal_size: int = params['quantal_size']
        self.diffusion_coefficient: float = params['diffusion_coefficient']
        self.uptake_rate: float = params['uptake_rate']
        self.degradation_rate: float = params['degradation_rate']
        
        # 突触间隙的几何参数
        self.synaptic_cleft_width: float = 0.02  # μm
        
        # 状态变量
        self.concentration: float = 0.0  # 当前浓度 (mM)
        self.peak_concentration: float = 0.0

    def _get_nt_params(self) -> dict:
        """
        提供一个包含各种神经递质参数的数据库。
        在实际应用中，这些数据可能从外部文件或数据库加载。
        """
        params_db = {
            NeurotransmitterType.GLUTAMATE: {'release_probability': 0.3, 'quantal_size': 3000, 'diffusion_coefficient': 0.33, 'uptake_rate': 0.1, 'degradation_rate': 0.01},
            NeurotransmitterType.GABA: {'release_probability': 0.5, 'quantal_size': 2000, 'diffusion_coefficient': 0.25, 'uptake_rate': 0.05, 'degradation_rate': 0.01},
            NeurotransmitterType.DOPAMINE: {'release_probability': 0.1, 'quantal_size': 1000, 'diffusion_coefficient': 0.20, 'uptake_rate': 0.02, 'degradation_rate': 0.005},
            NeurotransmitterType.SEROTONIN: {'release_probability': 0.1, 'quantal_size': 1500, 'diffusion_coefficient': 0.18, 'uptake_rate': 0.03, 'degradation_rate': 0.005},
            NeurotransmitterType.ACETYLCHOLINE: {'release_probability': 0.4, 'quantal_size': 5000, 'diffusion_coefficient': 0.40, 'uptake_rate': 0.5, 'degradation_rate': 0.2},
            NeurotransmitterType.NOREPINEPHRINE: {'release_probability': 0.2, 'quantal_size': 800, 'diffusion_coefficient': 0.15, 'uptake_rate': 0.02, 'degradation_rate': 0.01},
        }
        # 如果找不到特定类型，则返回谷氨酸的默认值
        return params_db.get(self.nt_type, params_db[NeurotransmitterType.GLUTAMATE])

    def release_vesicles(self, num_vesicles: int) -> float:
        """
        模拟囊泡的释放，并计算导致的神经递质浓度增加。

        Args:
            num_vesicles (int): 释放的囊泡数量。

        Returns:
            float: 释放后的当前神经递质浓度 (mM)。
        """
        if num_vesicles <= 0:
            return self.concentration

        # 计算释放的总分子数
        total_molecules = num_vesicles * self.quantal_size
        
        # 将分子数转换为浓度 (mM)
        # 假设突触间隙为圆柱体，体积 V = π * r^2 * h
        synaptic_radius = 0.5  # μm
        synaptic_volume_um3 = np.pi * (synaptic_radius**2) * self.synaptic_cleft_width # μm³
        synaptic_volume_L = synaptic_volume_um3 * 1e-15  # 转换为升 (L)
        
        avogadro_constant = 6.022e23
        moles = total_molecules / avogadro_constant
        concentration_M = moles / synaptic_volume_L  # 摩尔浓度 (M)
        concentration_increase_mM = concentration_M * 1000  # 转换为毫摩尔 (mM)
        
        self.concentration += concentration_increase_mM
        self.peak_concentration = max(self.peak_concentration, self.concentration)
        
        self.logger.debug(f"释放 {num_vesicles} 个囊泡, 浓度增加 {concentration_increase_mM:.4f} mM")
        return self.concentration

    def update(self, dt: float, astrocyte_uptake_rate: float = 0.0) -> float:
        """
        更新神经递质浓度，考虑扩散、再摄取和降解。

        Args:
            dt (float): 时间步长 (ms)。
            astrocyte_uptake_rate (float): 由星形胶质细胞介导的额外摄取速率。

        Returns:
            float: 更新后的当前神经递-质浓度 (mM)。
        """
        if self.concentration <= 0:
            return 0.0

        # 1. 扩散导致的损失 (简化为一阶衰减)
        diffusion_loss = self.concentration * self.diffusion_coefficient * dt / (self.synaptic_cleft_width**2)
        
        # 2. 神经末梢和胶质细胞的再摄取
        total_uptake_rate = self.uptake_rate + astrocyte_uptake_rate
        uptake_loss = self.concentration * total_uptake_rate * dt
        
        # 3. 酶降解
        degradation_loss = self.concentration * self.degradation_rate * dt
        
        # 计算总损失并更新浓度
        total_loss = diffusion_loss + uptake_loss + degradation_loss
        self.concentration = max(0.0, self.concentration - total_loss)
        
        return self.concentration