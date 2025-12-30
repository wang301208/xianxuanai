"""
EEG模拟器主模块

整合电极、信号生成、伪影和分析功能，提供完整的EEG模拟功能
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode, EEGFrequencyBand, ElectrodeManager
from BrainSimulationSystem.core.eeg.signal_generator import SignalGenerator
from BrainSimulationSystem.core.eeg.artifacts import ArtifactGenerator
from BrainSimulationSystem.core.eeg.analyzer import EEGAnalyzer
from BrainSimulationSystem.core.eeg.visualizer import EEGVisualizer
from BrainSimulationSystem.core.connectome import BrainRegion


class EEGSimulator:
    """EEG信号模拟器"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        初始化EEG模拟器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        
        # 初始化组件
        self.electrode_manager = ElectrodeManager()
        self.signal_generator = SignalGenerator(sampling_rate)
        self.artifact_generator = ArtifactGenerator(sampling_rate)
        self.analyzer = EEGAnalyzer(sampling_rate)
        self.visualizer = EEGVisualizer()
        
        # 当前模拟的EEG数据
        self.current_eeg_data = {}
        self.time_points = []
        self.duration = 0.0
    
    def simulate(self, duration: float, region_activities: Dict[BrainRegion, float]) -> Dict[EEGElectrode, np.ndarray]:
        """
        模拟EEG信号
        
        Args:
            duration: 持续时间(秒)
            region_activities: 脑区活动水平字典
            
        Returns:
            电极到EEG信号的映射
        """
        self.duration = duration
        self.time_points = np.linspace(0, duration, int(duration * self.sampling_rate), endpoint=False)
        
        # 为每个电极生成EEG信号
        eeg_signals = {}
        for electrode in EEGElectrode:
            # 获取与电极相关的脑区
            related_regions = self.electrode_manager.get_related_regions(electrode)
            
            # 生成信号
            eeg_signals[electrode] = self.signal_generator.generate_eeg_signal(
                duration, region_activities, related_regions
            )
        
        # 添加伪影
        eeg_signals = self.artifact_generator.add_artifacts(eeg_signals, duration)
        
        # 保存当前数据
        self.current_eeg_data = eeg_signals
        
        return eeg_signals
    
    def get_current_data(self) -> Dict[EEGElectrode, np.ndarray]:
        """
        获取当前EEG数据
        
        Returns:
            电极到EEG信号的映射
        """
        return self.current_eeg_data
    
    def get_time_points(self) -> np.ndarray:
        """
        获取时间点
        
        Returns:
            时间点数组
        """
        return self.time_points
    
    def calculate_power_spectrum(self, electrode: EEGElectrode) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算特定电极的功率谱
        
        Args:
            electrode: 电极
            
        Returns:
            频率和功率谱密度
        """
        if electrode not in self.current_eeg_data:
            return np.array([]), np.array([])
        
        return self.analyzer.calculate_power_spectrum(self.current_eeg_data[electrode])
    
    def calculate_band_power(self, electrode: EEGElectrode) -> Dict[EEGFrequencyBand, float]:
        """
        计算特定电极的频段功率
        
        Args:
            electrode: 电极
            
        Returns:
            频段到功率的映射
        """
        if electrode not in self.current_eeg_data:
            return {band: 0.0 for band in EEGFrequencyBand}
        
        return self.analyzer.calculate_band_power(self.current_eeg_data[electrode])
    
    def calculate_coherence(self, electrode1: EEGElectrode, electrode2: EEGElectrode) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算两个电极之间的相干性
        
        Args:
            electrode1: 第一个电极
            electrode2: 第二个电极
            
        Returns:
            频率和相干性
        """
        if electrode1 not in self.current_eeg_data or electrode2 not in self.current_eeg_data:
            return np.array([]), np.array([])
        
        return self.analyzer.calculate_coherence(
            self.current_eeg_data[electrode1], self.current_eeg_data[electrode2]
        )
    
    def plot_eeg(self, electrodes: List[EEGElectrode] = None, time_range: Tuple[float, float] = None,
                title: str = "EEG Signal Simulation") -> plt.Figure:
        """
        绘制EEG信号
        
        Args:
            electrodes: 要绘制的电极列表，如果为None则绘制所有电极
            time_range: 时间范围(秒)，如果为None则绘制全部时间
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        return self.visualizer.plot_eeg(
            self.current_eeg_data, self.sampling_rate, electrodes, time_range, title
        )
    
    def plot_power_spectrum(self, electrode: EEGElectrode, title: str = None) -> plt.Figure:
        """
        绘制功率谱
        
        Args:
            electrode: 电极
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        if electrode not in self.current_eeg_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No spectrum data available", ha='center', va='center')
            return fig
        
        if title is None:
            title = f"Power Spectrum - {electrode.name}"
        
        return self.analyzer.plot_power_spectrum(self.current_eeg_data[electrode], title)
    
    def plot_topographic_map(self, band: EEGFrequencyBand = None, time_point: float = None,
                            title: str = None) -> plt.Figure:
        """
        绘制地形图
        
        Args:
            band: 频段，如果为None则绘制原始信号
            time_point: 时间点(秒)，如果为None则使用平均功率
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        if not self.current_eeg_data:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No EEG data available", ha='center', va='center')
            return fig
        
        values = {}
        
        for electrode in self.current_eeg_data.keys():
            if band is not None:
                # 使用频段功率
                band_powers = self.calculate_band_power(electrode)
                values[electrode] = band_powers[band]
            elif time_point is not None:
                # 使用特定时间点的值
                time_idx = min(int(time_point * self.sampling_rate), len(self.current_eeg_data[electrode]) - 1)
                values[electrode] = self.current_eeg_data[electrode][time_idx]
            else:
                # 使用平均值
                values[electrode] = np.mean(self.current_eeg_data[electrode])
        
        # 设置标题
        if title is None:
            if band is not None:
                title = f"Topographic Map - {band.name} Band"
            elif time_point is not None:
                title = f"Topographic Map at t={time_point:.2f}s"
            else:
                title = "Topographic Map - Mean Amplitude"
        
        return self.visualizer.plot_topographic_map(values, title)
    
    def plot_coherence_matrix(self, band: EEGFrequencyBand, title: str = None) -> plt.Figure:
        """
        绘制相干性矩阵
        
        Args:
            band: 频段
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        return self.visualizer.plot_coherence_matrix(
            self.current_eeg_data, band, self.sampling_rate, title
        )
    
    def save_eeg_data(self, file_path: str) -> bool:
        """
        保存EEG数据到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            # 将数据转换为可序列化格式
            data = {
                'sampling_rate': self.sampling_rate,
                'duration': self.duration,
                'eeg_data': {electrode.name: signal.tolist() for electrode, signal in self.current_eeg_data.items()}
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            print(f"保存EEG数据失败: {e}")
            return False
    
    def load_eeg_data(self, file_path: str) -> bool:
        """
        从文件加载EEG数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功加载
        """
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.sampling_rate = data['sampling_rate']
            self.duration = data['duration']
            self.time_points = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
            
            # 转换数据
            self.current_eeg_data = {
                EEGElectrode[electrode_name]: np.array(signal)
                for electrode_name, signal in data['eeg_data'].items()
            }
            
            return True
        except Exception as e:
            print(f"加载EEG数据失败: {e}")
            return False