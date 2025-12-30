"""
突触后密度蛋白动态模拟模块

实现突触后密度(PSD)蛋白的动态变化模拟，包括:
1. 主要PSD蛋白的表达和降解
2. 蛋白相互作用网络
3. 受体内化和外化过程
4. 突触可塑性相关的PSD重构
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from enum import Enum
from dataclasses import dataclass
import random


class ReceptorType(Enum):
    """受体类型枚举"""
    AMPA = 0  # α-氨基-3-羟基-5-甲基-4-异恶唑丙酸受体
    NMDA = 1  # N-甲基-D-天冬氨酸受体
    GABA_A = 2  # γ-氨基丁酸A型受体
    GABA_B = 3  # γ-氨基丁酸B型受体
    mGluR = 4  # 代谢型谷氨酸受体


class PSDProteinType(Enum):
    """突触后密度蛋白类型枚举"""
    PSD95 = 0    # 突触后密度蛋白95
    SHANK = 1    # SH3和多个锚蛋白重复结构域蛋白
    HOMER = 2    # Homer蛋白
    GKAP = 3     # 鸟苷酸激酶相关蛋白
    CAMKII = 4   # 钙调蛋白依赖性蛋白激酶II
    AKAP = 5     # A-激酶锚定蛋白
    GRIP = 6     # 谷氨酸受体相互作用蛋白
    PICK1 = 7    # 蛋白相互作用C激酶1
    NEUROLIGIN = 8  # 神经连接蛋白
    SYNGAP = 9   # 突触GTPase激活蛋白


@dataclass
class Receptor:
    """受体模型"""
    type: ReceptorType  # 受体类型
    surface_count: int  # 表面受体数量
    internal_count: int  # 内部受体数量
    exocytosis_rate: float  # 外化率
    endocytosis_rate: float  # 内化率
    recycling_rate: float  # 循环率
    degradation_rate: float  # 降解率
    synthesis_rate: float  # 合成率
    
    def update(self, dt: float, activity: float = 0.0) -> None:
        """
        更新受体状态
        
        Args:
            dt: 时间步长 (ms)
            activity: 突触活动水平 (0-1)
        """
        # 将时间步长转换为秒
        dt_sec = dt / 1000.0
        
        # 活动依赖的内化率调节
        activity_factor = 1.0 + activity * 2.0  # 活动越高，内化越快
        
        # 计算各过程的受体数量变化
        exocytosis = self.internal_count * self.exocytosis_rate * dt_sec
        endocytosis = self.surface_count * self.endocytosis_rate * activity_factor * dt_sec
        degradation = self.internal_count * self.degradation_rate * dt_sec
        synthesis = self.synthesis_rate * dt_sec
        
        # 更新受体数量
        self.surface_count += int(exocytosis - endocytosis)
        self.internal_count += int(synthesis - exocytosis + endocytosis - degradation)
        
        # 确保数量非负
        self.surface_count = max(0, self.surface_count)
        self.internal_count = max(0, self.internal_count)


@dataclass
class PSDProtein:
    """突触后密度蛋白模型"""
    type: PSDProteinType  # 蛋白类型
    count: int  # 蛋白数量
    half_life: float  # 半衰期 (小时)
    synthesis_rate: float  # 合成率
    phosphorylation: float  # 磷酸化水平 (0-1)
    
    def update(self, dt: float, activity: float = 0.0) -> None:
        """
        更新蛋白状态
        
        Args:
            dt: 时间步长 (ms)
            activity: 突触活动水平 (0-1)
        """
        # 将时间步长转换为小时
        dt_hour = dt / (1000.0 * 60.0 * 60.0)
        
        # 计算降解率
        degradation_rate = np.log(2) / self.half_life  # 半衰期转换为降解率
        
        # 活动依赖的合成率调节
        activity_factor = 1.0 + activity  # 活动越高，合成越快
        
        # 计算蛋白数量变化
        degradation = self.count * degradation_rate * dt_hour
        synthesis = self.synthesis_rate * activity_factor * dt_hour
        
        # 更新蛋白数量
        self.count += int(synthesis - degradation)
        
        # 确保数量非负
        self.count = max(0, self.count)
        
        # 更新磷酸化水平
        # 活动增加磷酸化，静息去磷酸化
        phosphorylation_change = (activity - self.phosphorylation) * 0.1 * dt_hour
        self.phosphorylation += phosphorylation_change
        self.phosphorylation = max(0.0, min(1.0, self.phosphorylation))


class ProteinInteraction:
    """蛋白相互作用模型"""
    
    def __init__(self, protein1: PSDProtein, protein2: PSDProtein, strength: float):
        """
        初始化蛋白相互作用
        
        Args:
            protein1: 第一个蛋白
            protein2: 第二个蛋白
            strength: 相互作用强度 (0-1)
        """
        self.protein1 = protein1
        self.protein2 = protein2
        self.strength = strength
    
    def update(self, dt: float) -> None:
        """
        更新相互作用状态
        
        Args:
            dt: 时间步长 (ms)
        """
        # 将时间步长转换为小时
        dt_hour = dt / (1000.0 * 60.0 * 60.0)
        
        # 磷酸化水平影响相互作用强度
        phospho_effect = (self.protein1.phosphorylation + self.protein2.phosphorylation) / 2.0
        
        # 相互作用强度变化
        strength_change = (phospho_effect - self.strength) * 0.05 * dt_hour
        self.strength += strength_change
        self.strength = max(0.0, min(1.0, self.strength))


class PostsynapticDensity:
    """突触后密度模型"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化突触后密度
        
        Args:
            params: 参数字典
        """
        params = params or {}
        
        # 初始化受体
        self.receptors = {
            ReceptorType.AMPA: Receptor(
                type=ReceptorType.AMPA,
                surface_count=params.get("ampa_surface", 50),
                internal_count=params.get("ampa_internal", 100),
                exocytosis_rate=0.1,  # 每秒10%的内部受体外化
                endocytosis_rate=0.05,  # 每秒5%的表面受体内化
                recycling_rate=0.2,  # 每秒20%的内化受体循环
                degradation_rate=0.01,  # 每秒1%的内部受体降解
                synthesis_rate=5.0  # 每秒合成5个新受体
            ),
            ReceptorType.NMDA: Receptor(
                type=ReceptorType.NMDA,
                surface_count=params.get("nmda_surface", 20),
                internal_count=params.get("nmda_internal", 40),
                exocytosis_rate=0.05,  # 每秒5%的内部受体外化
                endocytosis_rate=0.02,  # 每秒2%的表面受体内化
                recycling_rate=0.1,  # 每秒10%的内化受体循环
                degradation_rate=0.005,  # 每秒0.5%的内部受体降解
                synthesis_rate=2.0  # 每秒合成2个新受体
            ),
            ReceptorType.GABA_A: Receptor(
                type=ReceptorType.GABA_A,
                surface_count=params.get("gaba_a_surface", 30),
                internal_count=params.get("gaba_a_internal", 60),
                exocytosis_rate=0.08,  # 每秒8%的内部受体外化
                endocytosis_rate=0.04,  # 每秒4%的表面受体内化
                recycling_rate=0.15,  # 每秒15%的内化受体循环
                degradation_rate=0.008,  # 每秒0.8%的内部受体降解
                synthesis_rate=3.0  # 每秒合成3个新受体
            )
        }
        
        # 初始化PSD蛋白
        self.proteins = {
            PSDProteinType.PSD95: PSDProtein(
                type=PSDProteinType.PSD95,
                count=params.get("psd95_count", 300),
                half_life=24.0,  # 24小时半衰期
                synthesis_rate=10.0,  # 每小时合成10个
                phosphorylation=0.2  # 初始磷酸化水平
            ),
            PSDProteinType.SHANK: PSDProtein(
                type=PSDProteinType.SHANK,
                count=params.get("shank_count", 150),
                half_life=36.0,  # 36小时半衰期
                synthesis_rate=5.0,  # 每小时合成5个
                phosphorylation=0.3  # 初始磷酸化水平
            ),
            PSDProteinType.CAMKII: PSDProtein(
                type=PSDProteinType.CAMKII,
                count=params.get("camkii_count", 500),
                half_life=12.0,  # 12小时半衰期
                synthesis_rate=30.0,  # 每小时合成30个
                phosphorylation=0.1  # 初始磷酸化水平
            )
        }
        
        # 初始化蛋白相互作用
        self.interactions = []
        self._initialize_interactions()
        
        # PSD大小和形状
        self.size = params.get("size", 0.1)  # μm²
        self.shape_factor = params.get("shape_factor", 1.0)  # 1.0表示圆形
        
        # 活动历史
        self.activity_history = []
        self.max_history = 1000  # 最大历史记录数
    
    def _initialize_interactions(self) -> None:
        """初始化蛋白相互作用"""
        # PSD95-SHANK相互作用
        if PSDProteinType.PSD95 in self.proteins and PSDProteinType.SHANK in self.proteins:
            self.interactions.append(ProteinInteraction(
                self.proteins[PSDProteinType.PSD95],
                self.proteins[PSDProteinType.SHANK],
                0.7  # 强相互作用
            ))
        
        # PSD95-CAMKII相互作用
        if PSDProteinType.PSD95 in self.proteins and PSDProteinType.CAMKII in self.proteins:
            self.interactions.append(ProteinInteraction(
                self.proteins[PSDProteinType.PSD95],
                self.proteins[PSDProteinType.CAMKII],
                0.4  # 中等相互作用
            ))
        
        # SHANK-CAMKII相互作用
        if PSDProteinType.SHANK in self.proteins and PSDProteinType.CAMKII in self.proteins:
            self.interactions.append(ProteinInteraction(
                self.proteins[PSDProteinType.SHANK],
                self.proteins[PSDProteinType.CAMKII],
                0.3  # 弱相互作用
            ))
    
    def update(self, dt: float, activity: float) -> None:
        """
        更新突触后密度状态
        
        Args:
            dt: 时间步长 (ms)
            activity: 突触活动水平 (0-1)
        """
        # 记录活动历史
        self.activity_history.append(activity)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        
        # 更新受体
        for receptor in self.receptors.values():
            receptor.update(dt, activity)
        
        # 更新蛋白
        for protein in self.proteins.values():
            protein.update(dt, activity)
        
        # 更新相互作用
        for interaction in self.interactions:
            interaction.update(dt)
        
        # 更新PSD大小
        self._update_psd_size(dt, activity)
    
    def _update_psd_size(self, dt: float, activity: float) -> None:
        """
        更新PSD大小
        
        Args:
            dt: 时间步长 (ms)
            activity: 突触活动水平 (0-1)
        """
        # 将时间步长转换为小时
        dt_hour = dt / (1000.0 * 60.0 * 60.0)
        
        # 计算平均蛋白数量
        total_proteins = sum(protein.count for protein in self.proteins.values())
        
        # 计算目标大小
        target_size = 0.05 + 0.0001 * total_proteins  # 基础大小 + 蛋白依赖大小
        
        # 活动依赖的大小调节
        if len(self.activity_history) > 0:
            recent_activity = np.mean(self.activity_history[-100:]) if len(self.activity_history) >= 100 else np.mean(self.activity_history)
            
            # 高活动增大PSD，低活动减小PSD
            activity_effect = (recent_activity - 0.5) * 0.2
            target_size *= (1.0 + activity_effect)
        
        # 逐渐调整大小
        size_change = (target_size - self.size) * 0.1 * dt_hour
        self.size += size_change
        
        # 确保大小合理
        self.size = max(0.05, min(0.5, self.size))  # 0.05-0.5 μm²
    
    def get_receptor_conductance(self, receptor_type: ReceptorType) -> float:
        """
        获取受体电导
        
        Args:
            receptor_type: 受体类型
            
        Returns:
            受体电导 (nS)
        """
        if receptor_type not in self.receptors:
            return 0.0
        
        receptor = self.receptors[receptor_type]
        
        # 单个受体电导 (nS)
        single_conductance = {
            ReceptorType.AMPA: 10.0,  # pS
            ReceptorType.NMDA: 50.0,  # pS
            ReceptorType.GABA_A: 30.0,  # pS
            ReceptorType.GABA_B: 5.0,  # pS
            ReceptorType.mGluR: 0.0,  # 代谢型受体无直接电导
        }.get(receptor_type, 0.0) / 1000.0  # 转换为nS
        
        # 总电导 = 表面受体数量 * 单个受体电导
        return receptor.surface_count * single_conductance
    
    def get_total_ampa_nmda_ratio(self) -> float:
        """
        获取AMPA/NMDA比例
        
        Returns:
            AMPA/NMDA比例
        """
        ampa_conductance = self.get_receptor_conductance(ReceptorType.AMPA)
        nmda_conductance = self.get_receptor_conductance(ReceptorType.NMDA)
        
        if nmda_conductance == 0:
            return float('inf')
        
        return ampa_conductance / nmda_conductance
    
    def apply_ltp(self, strength: float = 1.0) -> None:
        """
        应用长时程增强(LTP)
        
        Args:
            strength: LTP强度 (0-1)
        """
        # 增加AMPA受体外化
        if ReceptorType.AMPA in self.receptors:
            ampa = self.receptors[ReceptorType.AMPA]
            # 将一部分内部受体移动到表面
            move_count = int(ampa.internal_count * 0.2 * strength)
            ampa.internal_count -= move_count
            ampa.surface_count += move_count
        
        # 增加CaMKII磷酸化
        if PSDProteinType.CAMKII in self.proteins:
            camkii = self.proteins[PSDProteinType.CAMKII]
            camkii.phosphorylation = min(1.0, camkii.phosphorylation + 0.3 * strength)
    
    def apply_ltd(self, strength: float = 1.0) -> None:
        """
        应用长时程抑制(LTD)
        
        Args:
            strength: LTD强度 (0-1)
        """
        # 增加AMPA受体内化
        if ReceptorType.AMPA in self.receptors:
            ampa = self.receptors[ReceptorType.AMPA]
            # 将一部分表面受体移动到内部
            move_count = int(ampa.surface_count * 0.2 * strength)
            ampa.surface_count -= move_count
            ampa.internal_count += move_count
        
        # 减少CaMKII磷酸化
        if PSDProteinType.CAMKII in self.proteins:
            camkii = self.proteins[PSDProteinType.CAMKII]
            camkii.phosphorylation = max(0.0, camkii.phosphorylation - 0.3 * strength)


class SynapticScaffold:
    """突触支架模型，模拟突触后密度的结构支架"""
    
    def __init__(self, psd: PostsynapticDensity):
        """
        初始化突触支架
        
        Args:
            psd: 突触后密度
        """
        self.psd = psd
        self.slots = {}  # 受体槽位
        self.scaffold_strength = 1.0  # 支架强度
        
        # 初始化受体槽位
        self._initialize_slots()
    
    def _initialize_slots(self) -> None:
        """初始化受体槽位"""
        # 根据PSD95数量确定AMPA受体槽位
        if PSDProteinType.PSD95 in self.psd.proteins:
            psd95_count = self.psd.proteins[PSDProteinType.PSD95].count
            self.slots[ReceptorType.AMPA] = int(psd95_count * 2)  # 每个PSD95可以结合2个AMPA受体
        else:
            self.slots[ReceptorType.AMPA] = 100  # 默认值
        
        # 根据PSD95和SHANK数量确定NMDA受体槽位
        if PSDProteinType.PSD95 in self.psd.proteins and PSDProteinType.SHANK in self.psd.proteins:
            psd95_count = self.psd.proteins[PSDProteinType.PSD95].count
            shank_count = self.psd.proteins[PSDProteinType.SHANK].count
            self.slots[ReceptorType.NMDA] = int(psd95_count * 0.5 + shank_count * 0.5)
        else:
            self.slots[ReceptorType.NMDA] = 40  # 默认值
        
        # GABA受体槽位
        self.slots[ReceptorType.GABA_A] = 60
    
    def update(self, dt: float) -> None:
        """
        更新突触支架状态
        
        Args:
            dt: 时间步长 (ms)
        """
        # 更新受体槽位
        self._update_slots()
        
        # 限制表面受体数量不超过槽位数量
        for receptor_type, receptor in self.psd.receptors.items():
            if receptor_type in self.slots:
                max_surface = self.slots[receptor_type]
                if receptor.surface_count > max_surface:
                    # 将多余的受体内化
                    excess = receptor.surface_count - max_surface
                    receptor.surface_count = max_surface
                    receptor.internal_count += excess
    
    def _update_slots(self) -> None:
        """更新受体槽位"""
        # 更新AMPA受体槽位
        if PSDProteinType.PSD95 in self.psd.proteins:
            psd95_count = self.psd.proteins[PSDProteinType.PSD95].count
            psd95_phospho = self.psd.proteins[PSDProteinType.PSD95].phosphorylation
            
            # 磷酸化增加槽位效率
            phospho_factor = 1.0 + psd95_phospho
            
            self.slots[ReceptorType.AMPA] = int(psd95_count * 2 * phospho_factor)
        
        # 更新NMDA受体槽位
        if PSDProteinType.PSD95 in self.psd.proteins and PSDProteinType.SHANK in self.psd.proteins:
            psd95_count = self.psd.proteins[PSDProteinType.PSD95].count
            shank_count = self.psd.proteins[PSDProteinType.SHANK].count
            
            # 找到PSD95-SHANK相互作用
            interaction_strength = 1.0
            for interaction in self.psd.interactions:
                if (interaction.protein1.type == PSDProteinType.PSD95 and 
                    interaction.protein2.type == PSDProteinType.SHANK) or \
                   (interaction.protein1.type == PSDProteinType.SHANK and 
                    interaction.protein2.type == PSDProteinType.PSD95):
                    interaction_strength = interaction.strength
                    break
            
            # 相互作用强度影响槽位数量
            self.slots[ReceptorType.NMDA] = int((psd95_count * 0.5 + shank_count * 0.5) * interaction_strength)


def create_postsynaptic_density(synapse_type: str, params: Dict[str, Any] = None) -> PostsynapticDensity:
    """
    创建突触后密度
    
    Args:
        synapse_type: 突触类型 ('excitatory', 'inhibitory', 'modulatory')
        params: 参数字典
        
    Returns:
        突触后密度实例
    """
    params = params or {}
    
    if synapse_type == 'excitatory':
        # 兴奋性突触，AMPA和NMDA受体为主
        return PostsynapticDensity({
            "ampa_surface": params.get("ampa_surface", 50),
            "ampa_internal": params.get("ampa_internal", 100),
            "nmda_surface": params.get("nmda_surface", 20),
            "nmda_internal": params.get("nmda_internal", 40),
            "gaba_a_surface": 0,
            "gaba_a_internal": 0,
            "psd95_count": params.get("psd95_count", 300),
            "shank_count": params.get("shank_count", 150),
            "camkii_count": params.get("camkii_count", 500),
            "size": params.get("size", 0.1)
        })
    
    elif synapse_type == 'inhibitory':
        # 抑制性突触，GABA受体为主
        return PostsynapticDensity({
            "ampa_surface": 0,
            "ampa_internal": 0,
            "nmda_surface": 0,
            "nmda_internal": 0,
            "gaba_a_surface": params.get("gaba_a_surface", 30),
            "gaba_a_internal": params.get("gaba_a_internal", 60),
            "psd95_count": params.get("psd95_count", 100),
            "shank_count": params.get("shank_count", 50),
            "camkii_count": params.get("camkii_count", 200),
            "size": params.get("size", 0.08)
        })
    
    elif synapse_type == 'modulatory':
        # 调节性突触，代谢型受体为主
        return PostsynapticDensity({
            "ampa_surface": params.get("ampa_surface", 10),
            "ampa_internal": params.get("ampa_internal", 20),
            "nmda_surface": params.get("nmda_surface", 5),
            "nmda_internal": params.get("nmda_internal", 10),
            "gaba_a_surface": params.get("gaba_a_surface", 5),
            "gaba_a_internal": params.get("gaba_a_internal", 10),
            "psd95_count": params.get("psd95_count", 150),
            "shank_count": params.get("shank_count", 100),
            "camkii_count": params.get("camkii_count", 300),
            "size": params.get("size", 0.07)
        })
    
    else:
        # 默认为兴奋性突触
        return PostsynapticDensity(params)