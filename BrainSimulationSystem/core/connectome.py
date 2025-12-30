"""
脑区连接组数据模块

实现基于人类脑连接组数据的大尺度脑区连接模型，包括:
1. 主要脑区定义
2. 区域间连接强度
3. 连接概率和拓扑结构
4. 基于DTI/fMRI数据的连接矩阵
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from enum import Enum
import os
import json


class BrainRegion(Enum):
    """脑区枚举"""
    # 前额叶区域
    DLPFC = 0      # 背外侧前额叶皮层
    VMPFC = 1      # 腹内侧前额叶皮层
    OFC = 2        # 眶额皮层
    ACC = 3        # 前扣带回皮层
    
    # 顶叶区域
    PPC = 4        # 后顶叶皮层
    IPL = 5        # 下顶叶小叶
    SPL = 6        # 上顶叶小叶
    
    # 颞叶区域
    HPC = 7        # 海马
    AMY = 8        # 杏仁核
    STG = 9        # 上颞回
    MTG = 10       # 中颞回
    ITG = 11       # 下颞回
    
    # 枕叶区域
    V1 = 12        # 初级视觉皮层
    V2 = 13        # 次级视觉皮层
    V4 = 14        # 视觉区V4
    MT = 15        # 中颞区(V5)
    
    # 基底神经节
    STR = 16       # 纹状体
    GPe = 17       # 苍白球外侧部
    GPi = 18       # 苍白球内侧部
    STN = 19       # 丘脑底核
    SN = 20        # 黑质
    
    # 丘脑
    TH = 21        # 丘脑
    
    # 脑干区域
    VTA = 22       # 腹侧被盖区
    LC = 23        # 蓝斑
    RAPHE = 24     # 中缝核
    
    # 小脑
    CB = 25        # 小脑


class ConnectomeData:
    """脑连接组数据类"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化脑连接组数据
        
        Args:
            data_path: 连接组数据文件路径，如果为None则使用默认数据
        """
        # 脑区数量
        self.num_regions = len(BrainRegion)
        
        # 连接矩阵: 区域间连接强度 (0-1)
        self.connectivity_matrix = np.zeros((self.num_regions, self.num_regions))
        
        # 距离矩阵: 区域间距离 (mm)
        self.distance_matrix = np.zeros((self.num_regions, self.num_regions))
        
        # 连接概率矩阵
        self.probability_matrix = np.zeros((self.num_regions, self.num_regions))
        
        # 脑区坐标 (MNI空间, mm)
        self.region_coordinates = {}
        
        # 脑区体积 (mm³)
        self.region_volumes = {}
        
        # 脑区神经元数量
        self.region_neuron_counts = {}
        
        # 加载数据
        if data_path and os.path.exists(data_path):
            self._load_data(data_path)
        else:
            self._initialize_default_data()
    
    def _load_data(self, data_path: str) -> None:
        """
        从文件加载连接组数据
        
        Args:
            data_path: 数据文件路径
        """
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # 加载连接矩阵
            if 'connectivity_matrix' in data:
                self.connectivity_matrix = np.array(data['connectivity_matrix'])
            
            # 加载距离矩阵
            if 'distance_matrix' in data:
                self.distance_matrix = np.array(data['distance_matrix'])
            
            # 加载概率矩阵
            if 'probability_matrix' in data:
                self.probability_matrix = np.array(data['probability_matrix'])
            
            # 加载脑区坐标
            if 'region_coordinates' in data:
                self.region_coordinates = {BrainRegion[k]: v for k, v in data['region_coordinates'].items()}
            
            # 加载脑区体积
            if 'region_volumes' in data:
                self.region_volumes = {BrainRegion[k]: v for k, v in data['region_volumes'].items()}
            
            # 加载脑区神经元数量
            if 'region_neuron_counts' in data:
                self.region_neuron_counts = {BrainRegion[k]: v for k, v in data['region_neuron_counts'].items()}
        
        except Exception as e:
            print(f"加载连接组数据失败: {e}")
            self._initialize_default_data()
    
    def _initialize_default_data(self) -> None:
        """初始化默认连接组数据"""
        # 初始化脑区坐标 (MNI空间, mm)
        self.region_coordinates = {
            BrainRegion.DLPFC: (-40, 40, 30),
            BrainRegion.VMPFC: (0, 50, -5),
            BrainRegion.OFC: (0, 30, -15),
            BrainRegion.ACC: (0, 30, 20),
            BrainRegion.PPC: (-35, -50, 50),
            BrainRegion.IPL: (-45, -45, 45),
            BrainRegion.SPL: (-25, -55, 60),
            BrainRegion.HPC: (-30, -25, -15),
            BrainRegion.AMY: (-25, -5, -20),
            BrainRegion.STG: (-55, -20, 5),
            BrainRegion.MTG: (-55, -30, -5),
            BrainRegion.ITG: (-55, -40, -15),
            BrainRegion.V1: (-10, -85, 5),
            BrainRegion.V2: (-15, -95, 10),
            BrainRegion.V4: (-30, -80, -15),
            BrainRegion.MT: (-45, -70, 5),
            BrainRegion.STR: (-15, 10, 5),
            BrainRegion.GPe: (-20, 0, 0),
            BrainRegion.GPi: (-18, -3, 0),
            BrainRegion.STN: (-12, -10, -5),
            BrainRegion.SN: (-10, -15, -10),
            BrainRegion.TH: (-10, -15, 10),
            BrainRegion.VTA: (0, -15, -15),
            BrainRegion.LC: (5, -30, -25),
            BrainRegion.RAPHE: (0, -25, -25),
            BrainRegion.CB: (0, -55, -25)
        }
        
        # 初始化脑区体积 (mm³)
        self.region_volumes = {
            BrainRegion.DLPFC: 15000,
            BrainRegion.VMPFC: 10000,
            BrainRegion.OFC: 12000,
            BrainRegion.ACC: 8000,
            BrainRegion.PPC: 14000,
            BrainRegion.IPL: 12000,
            BrainRegion.SPL: 10000,
            BrainRegion.HPC: 6000,
            BrainRegion.AMY: 3000,
            BrainRegion.STG: 9000,
            BrainRegion.MTG: 11000,
            BrainRegion.ITG: 10000,
            BrainRegion.V1: 15000,
            BrainRegion.V2: 14000,
            BrainRegion.V4: 8000,
            BrainRegion.MT: 5000,
            BrainRegion.STR: 10000,
            BrainRegion.GPe: 1000,
            BrainRegion.GPi: 800,
            BrainRegion.STN: 300,
            BrainRegion.SN: 500,
            BrainRegion.TH: 7000,
            BrainRegion.VTA: 200,
            BrainRegion.LC: 100,
            BrainRegion.RAPHE: 150,
            BrainRegion.CB: 120000
        }
        
        # 初始化脑区神经元数量 (百万)
        self.region_neuron_counts = {
            BrainRegion.DLPFC: 50,
            BrainRegion.VMPFC: 40,
            BrainRegion.OFC: 45,
            BrainRegion.ACC: 30,
            BrainRegion.PPC: 55,
            BrainRegion.IPL: 45,
            BrainRegion.SPL: 40,
            BrainRegion.HPC: 20,
            BrainRegion.AMY: 12,
            BrainRegion.STG: 35,
            BrainRegion.MTG: 40,
            BrainRegion.ITG: 38,
            BrainRegion.V1: 140,
            BrainRegion.V2: 120,
            BrainRegion.V4: 60,
            BrainRegion.MT: 30,
            BrainRegion.STR: 70,
            BrainRegion.GPe: 5,
            BrainRegion.GPi: 3,
            BrainRegion.STN: 0.5,
            BrainRegion.SN: 1.2,
            BrainRegion.TH: 30,
            BrainRegion.VTA: 0.4,
            BrainRegion.LC: 0.05,
            BrainRegion.RAPHE: 0.2,
            BrainRegion.CB: 70000  # 小脑含有大量颗粒细胞
        }
        
        # 计算距离矩阵
        self._calculate_distance_matrix()
        
        # 初始化连接矩阵
        self._initialize_connectivity_matrix()
        
        # 初始化概率矩阵
        self._initialize_probability_matrix()
    
    def _calculate_distance_matrix(self) -> None:
        """计算脑区间距离矩阵"""
        for i, region_i in enumerate(BrainRegion):
            for j, region_j in enumerate(BrainRegion):
                if i == j:
                    self.distance_matrix[i, j] = 0.0
                else:
                    # 计算欧氏距离
                    coord_i = self.region_coordinates[region_i]
                    coord_j = self.region_coordinates[region_j]
                    self.distance_matrix[i, j] = np.sqrt(
                        (coord_i[0] - coord_j[0])**2 +
                        (coord_i[1] - coord_j[1])**2 +
                        (coord_i[2] - coord_j[2])**2
                    )
    
    def _initialize_connectivity_matrix(self) -> None:
        """初始化连接矩阵"""
        # 初始化为低连接强度
        self.connectivity_matrix = np.random.uniform(0.0, 0.1, (self.num_regions, self.num_regions))
        
        # 设置自连接为0
        np.fill_diagonal(self.connectivity_matrix, 0.0)
        
        # 添加已知的强连接
        # 视觉通路
        self.set_connection(BrainRegion.V1, BrainRegion.V2, 0.8)
        self.set_connection(BrainRegion.V2, BrainRegion.V4, 0.7)
        self.set_connection(BrainRegion.V4, BrainRegion.MT, 0.6)
        self.set_connection(BrainRegion.V4, BrainRegion.ITG, 0.5)
        
        # 前额叶连接
        self.set_connection(BrainRegion.DLPFC, BrainRegion.VMPFC, 0.6)
        self.set_connection(BrainRegion.DLPFC, BrainRegion.OFC, 0.5)
        self.set_connection(BrainRegion.DLPFC, BrainRegion.ACC, 0.7)
        
        # 基底神经节环路
        self.set_connection(BrainRegion.STR, BrainRegion.GPe, 0.7)
        self.set_connection(BrainRegion.STR, BrainRegion.GPi, 0.7)
        self.set_connection(BrainRegion.GPe, BrainRegion.STN, 0.6)
        self.set_connection(BrainRegion.STN, BrainRegion.GPi, 0.7)
        self.set_connection(BrainRegion.GPi, BrainRegion.TH, 0.6)
        self.set_connection(BrainRegion.SN, BrainRegion.STR, 0.7)
        
        # 皮层-丘脑环路
        self.set_connection(BrainRegion.TH, BrainRegion.DLPFC, 0.6)
        self.set_connection(BrainRegion.TH, BrainRegion.PPC, 0.6)
        self.set_connection(BrainRegion.TH, BrainRegion.V1, 0.5)
        
        # 海马连接
        self.set_connection(BrainRegion.HPC, BrainRegion.DLPFC, 0.5)
        self.set_connection(BrainRegion.HPC, BrainRegion.VMPFC, 0.6)
        
        # 杏仁核连接
        self.set_connection(BrainRegion.AMY, BrainRegion.VMPFC, 0.6)
        self.set_connection(BrainRegion.AMY, BrainRegion.OFC, 0.5)
        self.set_connection(BrainRegion.AMY, BrainRegion.ACC, 0.5)
        
        # 神经调质系统
        self.set_connection(BrainRegion.VTA, BrainRegion.STR, 0.7)  # 多巴胺
        self.set_connection(BrainRegion.VTA, BrainRegion.DLPFC, 0.5)
        self.set_connection(BrainRegion.LC, BrainRegion.DLPFC, 0.5)  # 去甲肾上腺素
        self.set_connection(BrainRegion.LC, BrainRegion.PPC, 0.5)
        self.set_connection(BrainRegion.RAPHE, BrainRegion.VMPFC, 0.5)  # 血清素
        self.set_connection(BrainRegion.RAPHE, BrainRegion.AMY, 0.5)
        
        # 小脑连接
        self.set_connection(BrainRegion.CB, BrainRegion.TH, 0.6)
        self.set_connection(BrainRegion.CB, BrainRegion.PPC, 0.4)
        
        # 使矩阵对称 (简化模型，实际连接是有向的)
        # self.connectivity_matrix = (self.connectivity_matrix + self.connectivity_matrix.T) / 2
    
    def _initialize_probability_matrix(self) -> None:
        """初始化连接概率矩阵"""
        # 基于距离的概率衰减
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i == j:
                    self.probability_matrix[i, j] = 0.0
                else:
                    # 距离越远，连接概率越低
                    distance = self.distance_matrix[i, j]
                    self.probability_matrix[i, j] = np.exp(-0.01 * distance)
        
        # 调整概率以反映连接强度
        self.probability_matrix = self.probability_matrix * (self.connectivity_matrix > 0.2)
    
    def set_connection(self, region_from: BrainRegion, region_to: BrainRegion, strength: float) -> None:
        """
        设置区域间连接强度
        
        Args:
            region_from: 源脑区
            region_to: 目标脑区
            strength: 连接强度 (0-1)
        """
        self.connectivity_matrix[region_from.value, region_to.value] = strength
    
    def get_connection(self, region_from: BrainRegion, region_to: BrainRegion) -> float:
        """
        获取区域间连接强度
        
        Args:
            region_from: 源脑区
            region_to: 目标脑区
            
        Returns:
            连接强度 (0-1)
        """
        return self.connectivity_matrix[region_from.value, region_to.value]
    
    def get_distance(self, region_from: BrainRegion, region_to: BrainRegion) -> float:
        """
        获取区域间距离
        
        Args:
            region_from: 源脑区
            region_to: 目标脑区
            
        Returns:
            距离 (mm)
        """
        return self.distance_matrix[region_from.value, region_to.value]
    
    def get_connection_probability(self, region_from: BrainRegion, region_to: BrainRegion) -> float:
        """
        获取区域间连接概率
        
        Args:
            region_from: 源脑区
            region_to: 目标脑区
            
        Returns:
            连接概率 (0-1)
        """
        return self.probability_matrix[region_from.value, region_to.value]
    
    def get_region_neuron_count(self, region: BrainRegion) -> float:
        """
        获取脑区神经元数量
        
        Args:
            region: 脑区
            
        Returns:
            神经元数量 (百万)
        """
        return self.region_neuron_counts.get(region, 0.0)
    
    def get_region_volume(self, region: BrainRegion) -> float:
        """
        获取脑区体积
        
        Args:
            region: 脑区
            
        Returns:
            体积 (mm³)
        """
        return self.region_volumes.get(region, 0.0)
    
    def get_region_coordinate(self, region: BrainRegion) -> Tuple[float, float, float]:
        """
        获取脑区坐标
        
        Args:
            region: 脑区
            
        Returns:
            坐标 (x, y, z) in MNI空间
        """
        return self.region_coordinates.get(region, (0.0, 0.0, 0.0))
    
    def save_data(self, data_path: str) -> None:
        """
        保存连接组数据到文件
        
        Args:
            data_path: 数据文件路径
        """
        data = {
            'connectivity_matrix': self.connectivity_matrix.tolist(),
            'distance_matrix': self.distance_matrix.tolist(),
            'probability_matrix': self.probability_matrix.tolist(),
            'region_coordinates': {region.name: coord for region, coord in self.region_coordinates.items()},
            'region_volumes': {region.name: volume for region, volume in self.region_volumes.items()},
            'region_neuron_counts': {region.name: count for region, count in self.region_neuron_counts.items()}
        }
        
        try:
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"连接组数据已保存到 {data_path}")
        except Exception as e:
            print(f"保存连接组数据失败: {e}")


class DefaultModeNetwork:
    """默认模式网络模型"""
    
    def __init__(self, connectome: ConnectomeData):
        """
        初始化默认模式网络
        
        Args:
            connectome: 脑连接组数据
        """
        self.connectome = connectome
        
        # DMN核心区域
        self.dmn_regions = [
            BrainRegion.VMPFC,  # 腹内侧前额叶皮层
            BrainRegion.PPC,    # 后顶叶皮层
            BrainRegion.ACC,    # 前扣带回皮层
            BrainRegion.HPC     # 海马
        ]
        
        # DMN活动水平 (0-1)
        self.activity_level = 0.5
        
        # DMN区域活动
        self.region_activity = {region: 0.5 for region in self.dmn_regions}
        
        # 活动历史
        self.activity_history = []
        self.max_history = 1000
    
    def update(self, dt: float, external_input: Dict[BrainRegion, float] = None) -> None:
        """
        更新默认模式网络状态
        
        Args:
            dt: 时间步长 (ms)
            external_input: 外部输入，区域到活动水平的映射
        """
        external_input = external_input or {}
        
        # 计算DMN区域间相互作用
        new_activity = {}
        for region in self.dmn_regions:
            # 初始活动为当前活动
            activity = self.region_activity[region]
            
            # 添加来自其他DMN区域的输入
            for other_region in self.dmn_regions:
                if other_region != region:
                    connection_strength = self.connectome.get_connection(other_region, region)
                    other_activity = self.region_activity[other_region]
                    activity += connection_strength * other_activity * 0.01 * dt
            
            # 添加外部输入
            if region in external_input:
                activity += external_input[region] * 0.01 * dt
            
            # 活动水平衰减
            activity -= 0.001 * dt
            
            # 确保活动水平在合理范围内
            activity = max(0.0, min(1.0, activity))
            
            new_activity[region] = activity
        
        # 更新区域活动
        self.region_activity = new_activity
        
        # 计算整体DMN活动水平
        self.activity_level = sum(self.region_activity.values()) / len(self.dmn_regions)
        
        # 记录活动历史
        self.activity_history.append(self.activity_level)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
    
    def get_activity(self, region: BrainRegion = None) -> float:
        """
        获取DMN活动水平
        
        Args:
            region: 指定脑区，如果为None则返回整体活动水平
            
        Returns:
            活动水平 (0-1)
        """
        if region is None:
            return self.activity_level
        elif region in self.region_activity:
            return self.region_activity[region]
        else:
            return 0.0
    
    def is_active(self, threshold: float = 0.6) -> bool:
        """
        判断DMN是否处于活跃状态
        
        Args:
            threshold: 活动阈值
            
        Returns:
            是否活跃
        """
        return self.activity_level >= threshold


class BrainNetworkFactory:
    """脑网络工厂类，用于创建不同类型的脑网络"""
    
    @staticmethod
    def create_default_mode_network(connectome: Optional[ConnectomeData] = None) -> DefaultModeNetwork:
        """
        创建默认模式网络
        
        Args:
            connectome: 脑连接组数据，如果为None则使用默认数据
            
        Returns:
            默认模式网络实例
        """
        if connectome is None:
            connectome = ConnectomeData()
        
        return DefaultModeNetwork(connectome)
    
    @staticmethod
    def create_attention_network(connectome: Optional[ConnectomeData] = None) -> Dict:
        """
        创建注意力网络
        
        Args:
            connectome: 脑连接组数据，如果为None则使用默认数据
            
        Returns:
            注意力网络配置
        """
        if connectome is None:
            connectome = ConnectomeData()
        
        # 注意力网络核心区域
        attention_regions = [
            BrainRegion.DLPFC,  # 背外侧前额叶皮层
            BrainRegion.PPC,    # 后顶叶皮层
            BrainRegion.ACC,    # 前扣带回皮层
            BrainRegion.TH      # 丘脑
        ]
        
        # 提取注意力网络连接
        attention_connectivity = np.zeros((len(attention_regions), len(attention_regions)))
        for i, region_i in enumerate(attention_regions):
            for j, region_j in enumerate(attention_regions):
                attention_connectivity[i, j] = connectome.get_connection(region_i, region_j)
        
        return {
            'regions': attention_regions,
            'connectivity': attention_connectivity,
            'region_coordinates': {region: connectome.get_region_coordinate(region) for region in attention_regions}
        }
    
    @staticmethod
    def create_language_network(connectome: Optional[ConnectomeData] = None) -> Dict:
        """
        创建语言网络
        
        Args:
            connectome: 脑连接组数据，如果为None则使用默认数据
            
        Returns:
            语言网络配置
        """
        if connectome is None:
            connectome = ConnectomeData()
        
        # 语言网络核心区域
        language_regions = [
            BrainRegion.STG,    # 上颞回
            BrainRegion.MTG,    # 中颞回
            BrainRegion.ITG,    # 下颞回
            BrainRegion.DLPFC   # 背外侧前额叶皮层
        ]
        
        # 提取语言网络连接
        language_connectivity = np.zeros((len(language_regions), len(language_regions)))
        for i, region_i in enumerate(language_regions):
            for j, region_j in enumerate(language_regions):
                language_connectivity[i, j] = connectome.get_connection(region_i, region_j)
        
        return {
            'regions': language_regions,
            'connectivity': language_connectivity,
            'region_coordinates': {region: connectome.get_region_coordinate(region) for region in language_regions}
        }


if __name__ == "__main__":
    # 创建连接组数据
    connectome = ConnectomeData()
    
    # 创建默认模式网络
    dmn = BrainNetworkFactory.create_default_mode_network(connectome)
    
    # 模拟DMN活动
    for _ in range(100):
        dmn.update(10.0)
    
    # 获取DMN活动水平
    print(f"DMN活动水平: {dmn.get_activity():.2f}")
    
    # 检查DMN是否活跃
    print(f"DMN是否活跃: {dmn.is_active()}")