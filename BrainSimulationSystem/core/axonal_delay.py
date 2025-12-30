"""
轴突传导延迟模块

实现精确的轴突传导延迟模型，包括:
1. 基于轴突长度和髓鞘化程度的延迟计算
2. 可变传导速度模型
3. 轴突分支点延迟累积
4. 传导可靠性模拟
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
import random
from enum import Enum
from dataclasses import dataclass


class AxonType(Enum):
    """轴突类型枚举"""
    UNMYELINATED = 0  # 无髓鞘
    LIGHTLY_MYELINATED = 1  # 轻度髓鞘化
    HEAVILY_MYELINATED = 2  # 重度髓鞘化


class AxonalDelayCalculator:
    """轴突传导延迟计算器"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化轴突延迟计算器
        
        Args:
            params: 参数字典，包含:
                - base_velocity: 基础传导速度 (μm/ms)
                - max_velocity: 最大传导速度 (μm/ms)
                - temperature: 温度 (摄氏度)
                - branch_delay: 每个分支点的延迟 (ms)
        """
        params = params or {}
        
        # 传导速度参数
        self.base_velocity = params.get("base_velocity", 1.0)  # μm/ms (无髓鞘)
        self.max_velocity = params.get("max_velocity", 100.0)  # μm/ms (完全髓鞘化)
        
        # 温度参数 (Q10 = 2，温度每升高10度，速度翻倍)
        self.reference_temp = 37.0  # 参考温度 (摄氏度)
        self.temperature = params.get("temperature", 37.0)  # 当前温度 (摄氏度)
        
        # 分支点延迟
        self.branch_delay = params.get("branch_delay", 0.3)  # ms/分支点
    
    def calculate_conduction_velocity(self, 
                                     axon_type: AxonType, 
                                     myelination: float = 0.7, 
                                     diameter: float = 1.0) -> float:
        """
        计算轴突传导速度
        
        Args:
            axon_type: 轴突类型
            myelination: 髓鞘化程度 (0-1)
            diameter: 轴突直径 (μm)
            
        Returns:
            传导速度 (μm/ms)
        """
        # 基于轴突类型的基础速度
        if axon_type == AxonType.UNMYELINATED:
            base_velocity = self.base_velocity
            myelination_factor = 0.0
        elif axon_type == AxonType.LIGHTLY_MYELINATED:
            base_velocity = self.base_velocity * 2.0
            myelination_factor = myelination * 0.5
        else:  # HEAVILY_MYELINATED
            base_velocity = self.base_velocity * 5.0
            myelination_factor = myelination
        
        # 髓鞘化对速度的影响 (非线性)
        myelination_effect = myelination_factor ** 2  # 平方关系使得髓鞘化效果更显著
        
        # 计算基础速度
        velocity = base_velocity + (self.max_velocity - base_velocity) * myelination_effect
        
        # 轴突直径对速度的影响 (线性关系)
        # 直径越大，速度越快
        diameter_factor = np.sqrt(diameter)  # 直径的平方根关系
        
        # 温度对速度的影响
        temp_factor = 2.0 ** ((self.temperature - self.reference_temp) / 10.0)  # Q10 = 2
        
        return velocity * diameter_factor * temp_factor
    
    def calculate_delay(self, 
                       length: float, 
                       axon_type: AxonType, 
                       myelination: float = 0.7,
                       diameter: float = 1.0, 
                       branch_points: int = 0) -> float:
        """
        计算轴突传导延迟
        
        Args:
            length: 轴突长度 (μm)
            axon_type: 轴突类型
            myelination: 髓鞘化程度 (0-1)
            diameter: 轴突直径 (μm)
            branch_points: 分支点数量
            
        Returns:
            传导延迟 (ms)
        """
        # 计算传导速度
        velocity = self.calculate_conduction_velocity(axon_type, myelination, diameter)
        
        # 基础延迟 = 长度 / 速度
        base_delay = length / velocity
        
        # 分支点引起的额外延迟
        branch_delay = branch_points * self.branch_delay
        
        # 总延迟
        total_delay = base_delay + branch_delay
        
        return max(0.1, total_delay)  # 最小延迟为0.1ms
    
    def add_jitter(self, delay: float, jitter_factor: float = 0.1) -> float:
        """
        添加随机抖动到延迟中
        
        Args:
            delay: 基础延迟 (ms)
            jitter_factor: 抖动因子 (0-1)
            
        Returns:
            添加抖动后的延迟 (ms)
        """
        jitter = random.uniform(1.0 - jitter_factor, 1.0 + jitter_factor)
        return delay * jitter


class AxonalDelayQueue:
    """轴突延迟队列，用于模拟轴突传导延迟"""
    
    def __init__(self, reliability: float = 0.98):
        """
        初始化轴突延迟队列
        
        Args:
            reliability: 传导可靠性 (0-1)
        """
        self.spike_times = []  # 存储脉冲到达时间
        self.current_time = 0.0  # 当前时间
        self.reliability = reliability  # 传导可靠性
    
    def add_spike(self, delay: float) -> None:
        """
        添加脉冲到队列
        
        Args:
            delay: 脉冲延迟 (ms)
        """
        # 根据可靠性决定是否传导
        if random.random() < self.reliability:
            # 计算脉冲到达时间
            arrival_time = self.current_time + delay
            self.spike_times.append(arrival_time)
    
    def update(self, dt: float) -> bool:
        """
        更新队列状态
        
        Args:
            dt: 时间步长 (ms)
            
        Returns:
            是否有脉冲到达
        """
        # 更新当前时间
        self.current_time += dt
        
        # 检查是否有脉冲到达
        spike_arrived = False
        remaining_spikes = []
        
        for spike_time in self.spike_times:
            if spike_time <= self.current_time:
                # 脉冲已到达
                spike_arrived = True
            else:
                # 脉冲尚未到达
                remaining_spikes.append(spike_time)
        
        # 更新队列
        self.spike_times = remaining_spikes
        
        return spike_arrived
    
    def reset(self) -> None:
        """重置队列状态"""
        self.spike_times = []
        self.current_time = 0.0


class AxonalPathway:
    """轴突通路，包含多个轴突段和分支点"""
    
    @dataclass
    class AxonSegment:
        """轴突段"""
        length: float  # 长度 (μm)
        diameter: float  # 直径 (μm)
        myelination: float  # 髓鞘化程度 (0-1)
        axon_type: AxonType  # 轴突类型
    
    def __init__(self, calculator: AxonalDelayCalculator = None):
        """
        初始化轴突通路
        
        Args:
            calculator: 轴突延迟计算器
        """
        self.segments = []  # 轴突段列表
        self.calculator = calculator or AxonalDelayCalculator()
        self.branch_points = 0  # 分支点数量
    
    def add_segment(self, 
                   length: float, 
                   diameter: float = 1.0, 
                   myelination: float = 0.7,
                   axon_type: AxonType = AxonType.LIGHTLY_MYELINATED) -> None:
        """
        添加轴突段
        
        Args:
            length: 长度 (μm)
            diameter: 直径 (μm)
            myelination: 髓鞘化程度 (0-1)
            axon_type: 轴突类型
        """
        segment = self.AxonSegment(length, diameter, myelination, axon_type)
        self.segments.append(segment)
    
    def add_branch_point(self, count: int = 1) -> None:
        """
        添加分支点
        
        Args:
            count: 分支点数量
        """
        self.branch_points += count
    
    def calculate_total_delay(self, add_jitter: bool = True) -> float:
        """
        计算总传导延迟
        
        Args:
            add_jitter: 是否添加随机抖动
            
        Returns:
            总传导延迟 (ms)
        """
        total_delay = 0.0
        
        # 计算每个轴突段的延迟
        for segment in self.segments:
            segment_delay = self.calculator.calculate_delay(
                segment.length,
                segment.axon_type,
                segment.myelination,
                segment.diameter,
                0  # 分支点在单独计算
            )
            total_delay += segment_delay
        
        # 添加分支点延迟
        branch_delay = self.branch_points * self.calculator.branch_delay
        total_delay += branch_delay
        
        # 添加随机抖动
        if add_jitter:
            total_delay = self.calculator.add_jitter(total_delay)
        
        return total_delay


def create_axonal_pathway(pre_pos: List[float], 
                         post_pos: List[float], 
                         params: Dict[str, Any] = None) -> AxonalPathway:
    """
    根据神经元位置创建轴突通路
    
    Args:
        pre_pos: 前神经元位置 [x, y, z]
        post_pos: 后神经元位置 [x, y, z]
        params: 参数字典
        
    Returns:
        轴突通路
    """
    params = params or {}
    
    # 计算距离
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pre_pos, post_pos)))
    
    # 创建轴突通路
    pathway = AxonalPathway()
    
    # 添加轴突段
    axon_type = params.get("axon_type", AxonType.LIGHTLY_MYELINATED)
    myelination = params.get("myelination", 0.7)
    diameter = params.get("diameter", 1.0)
    
    # 如果距离较长，分成多段
    if distance > 1000:  # 超过1mm
        # 初始段 (轴突丘)
        pathway.add_segment(
            length=50.0,
            diameter=diameter * 1.5,  # 轴突丘直径较大
            myelination=0.0,  # 轴突丘无髓鞘
            axon_type=AxonType.UNMYELINATED
        )
        
        # 中间段 (主轴突)
        pathway.add_segment(
            length=distance - 100.0,  # 减去初始段和末端段
            diameter=diameter,
            myelination=myelination,
            axon_type=axon_type
        )
        
        # 末端段 (轴突终末)
        pathway.add_segment(
            length=50.0,
            diameter=diameter * 0.7,  # 轴突终末直径较小
            myelination=0.0,  # 轴突终末无髓鞘
            axon_type=AxonType.UNMYELINATED
        )
        
        # 添加分支点
        branch_points = params.get("branch_points", int(distance / 500))
        pathway.add_branch_point(branch_points)
    else:
        # 短距离，单段轴突
        pathway.add_segment(
            length=distance,
            diameter=diameter,
            myelination=myelination,
            axon_type=axon_type
        )
        
        # 添加分支点
        branch_points = params.get("branch_points", 0)
        pathway.add_branch_point(branch_points)
    
    return pathway