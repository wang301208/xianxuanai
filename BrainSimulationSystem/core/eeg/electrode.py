"""
EEG电极定义模块

定义EEG电极位置和相关数据结构
"""

from enum import Enum
from typing import Dict, List, Tuple
from BrainSimulationSystem.core.connectome import BrainRegion


class EEGElectrode(Enum):
    """EEG电极位置枚举(国际10-20系统)"""
    # 前额区
    FP1 = 0
    FP2 = 1
    F7 = 2
    F3 = 3
    FZ = 4
    F4 = 5
    F8 = 6
    
    # 中央区
    T3 = 7
    C3 = 8
    CZ = 9
    C4 = 10
    T4 = 11
    
    # 顶区
    T5 = 12
    P3 = 13
    PZ = 14
    P4 = 15
    T6 = 16
    
    # 枕区
    O1 = 17
    OZ = 18
    O2 = 19


class EEGFrequencyBand(Enum):
    """EEG频段枚举"""
    DELTA = (0.5, 4.0)    # δ波: 0.5-4 Hz
    THETA = (4.0, 8.0)    # θ波: 4-8 Hz
    ALPHA = (8.0, 13.0)   # α波: 8-13 Hz
    BETA = (13.0, 30.0)   # β波: 13-30 Hz
    GAMMA = (30.0, 100.0) # γ波: 30-100 Hz
    
    def __init__(self, low_freq, high_freq):
        self.low_freq = low_freq
        self.high_freq = high_freq


class ElectrodeManager:
    """电极管理器，提供电极位置和脑区映射"""
    
    def __init__(self):
        """初始化电极管理器"""
        # 电极位置(x, y, z)坐标，基于标准化头皮坐标
        self.electrode_positions = {
            EEGElectrode.FP1: (-0.3, 0.9, 0.0),
            EEGElectrode.FP2: (0.3, 0.9, 0.0),
            EEGElectrode.F7: (-0.7, 0.5, 0.0),
            EEGElectrode.F3: (-0.4, 0.5, 0.0),
            EEGElectrode.FZ: (0.0, 0.5, 0.0),
            EEGElectrode.F4: (0.4, 0.5, 0.0),
            EEGElectrode.F8: (0.7, 0.5, 0.0),
            EEGElectrode.T3: (-0.9, 0.0, 0.0),
            EEGElectrode.C3: (-0.5, 0.0, 0.0),
            EEGElectrode.CZ: (0.0, 0.0, 0.0),
            EEGElectrode.C4: (0.5, 0.0, 0.0),
            EEGElectrode.T4: (0.9, 0.0, 0.0),
            EEGElectrode.T5: (-0.7, -0.5, 0.0),
            EEGElectrode.P3: (-0.4, -0.5, 0.0),
            EEGElectrode.PZ: (0.0, -0.5, 0.0),
            EEGElectrode.P4: (0.4, -0.5, 0.0),
            EEGElectrode.T6: (0.7, -0.5, 0.0),
            EEGElectrode.O1: (-0.3, -0.9, 0.0),
            EEGElectrode.OZ: (0.0, -0.9, 0.0),
            EEGElectrode.O2: (0.3, -0.9, 0.0)
        }
        
        # 电极到脑区的映射(简化模型)
        self.electrode_to_region_map = {
            EEGElectrode.FP1: [BrainRegion.DLPFC, BrainRegion.OFC],
            EEGElectrode.FP2: [BrainRegion.DLPFC, BrainRegion.OFC],
            EEGElectrode.F7: [BrainRegion.DLPFC],
            EEGElectrode.F3: [BrainRegion.DLPFC, BrainRegion.ACC],
            EEGElectrode.FZ: [BrainRegion.ACC, BrainRegion.VMPFC],
            EEGElectrode.F4: [BrainRegion.DLPFC, BrainRegion.ACC],
            EEGElectrode.F8: [BrainRegion.DLPFC],
            EEGElectrode.T3: [BrainRegion.STG, BrainRegion.MTG],
            EEGElectrode.C3: [BrainRegion.PPC],
            EEGElectrode.CZ: [BrainRegion.PPC],
            EEGElectrode.C4: [BrainRegion.PPC],
            EEGElectrode.T4: [BrainRegion.STG, BrainRegion.MTG],
            EEGElectrode.T5: [BrainRegion.MTG, BrainRegion.ITG],
            EEGElectrode.P3: [BrainRegion.PPC, BrainRegion.IPL],
            EEGElectrode.PZ: [BrainRegion.PPC],
            EEGElectrode.P4: [BrainRegion.PPC, BrainRegion.IPL],
            EEGElectrode.T6: [BrainRegion.MTG, BrainRegion.ITG],
            EEGElectrode.O1: [BrainRegion.V1, BrainRegion.V2],
            EEGElectrode.OZ: [BrainRegion.V1],
            EEGElectrode.O2: [BrainRegion.V1, BrainRegion.V2]
        }
    
    def get_position(self, electrode: EEGElectrode) -> Tuple[float, float, float]:
        """
        获取电极位置坐标
        
        Args:
            electrode: 电极
            
        Returns:
            电极位置坐标 (x, y, z)
        """
        return self.electrode_positions.get(electrode, (0.0, 0.0, 0.0))
    
    def get_related_regions(self, electrode: EEGElectrode) -> List[BrainRegion]:
        """
        获取与电极相关的脑区
        
        Args:
            electrode: 电极
            
        Returns:
            相关脑区列表
        """
        return self.electrode_to_region_map.get(electrode, [])
    
    def get_all_electrodes(self) -> List[EEGElectrode]:
        """
        获取所有电极
        
        Returns:
            所有电极列表
        """
        return list(EEGElectrode)