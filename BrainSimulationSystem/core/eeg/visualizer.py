"""
EEG可视化模块

实现EEG信号的可视化功能，包括时域波形绘制、地形图绘制等
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode, EEGFrequencyBand, ElectrodeManager


class EEGVisualizer:
    """EEG信号可视化器"""
    
    def __init__(self):
        """初始化EEG可视化器"""
        self.electrode_manager = ElectrodeManager()
    
    def plot_eeg(self, eeg_data: Dict[EEGElectrode, np.ndarray], sampling_rate: int,
                electrodes: List[EEGElectrode] = None, time_range: Tuple[float, float] = None,
                title: str = "EEG Signal") -> Figure:
        """
        绘制EEG信号
        
        Args:
            eeg_data: 电极到EEG信号的映射
            sampling_rate: 采样率(Hz)
            electrodes: 要绘制的电极列表，如果为None则绘制所有电极
            time_range: 时间范围(秒)，如果为None则绘制全部时间
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        if not eeg_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No EEG data available", ha='center', va='center')
            return fig
        
        # 默认绘制所有电极
        if electrodes is None:
            electrodes = list(eeg_data.keys())
        
        # 创建时间点
        first_electrode = next(iter(eeg_data.values()))
        num_samples = len(first_electrode)
        time_points = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
        
        # 确定时间范围
        if time_range is None:
            start_idx, end_idx = 0, len(time_points)
            t_plot = time_points
        else:
            start_time, end_time = time_range
            start_idx = max(0, int(start_time * sampling_rate))
            end_idx = min(len(time_points), int(end_time * sampling_rate))
            t_plot = time_points[start_idx:end_idx]
        
        # 创建图形
        fig, axes = plt.subplots(len(electrodes), 1, figsize=(12, len(electrodes)), sharex=True)
        if len(electrodes) == 1:
            axes = [axes]
        
        # 绘制每个电极的信号
        for i, electrode in enumerate(electrodes):
            if electrode in eeg_data:
                signal_data = eeg_data[electrode][start_idx:end_idx]
                axes[i].plot(t_plot, signal_data)
                axes[i].set_ylabel(f"{electrode.name} (μV)")
                # 设置y轴范围
                max_val = max(100, np.max(np.abs(signal_data)) * 1.1)
                axes[i].set_ylim(-max_val, max_val)
                # 添加网格
                axes[i].grid(True, linestyle='--', alpha=0.6)
        
        # 设置x轴标签
        axes[-1].set_xlabel("Time (s)")
        
        # 设置标题
        fig.suptitle(title)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def plot_topographic_map(self, values: Dict[EEGElectrode, float], 
                            title: str = "Topographic Map") -> Figure:
        """
        绘制地形图
        
        Args:
            values: 电极到值的映射
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        if not values:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 提取电极坐标和值
        x_coords = []
        y_coords = []
        value_list = []
        
        for electrode, value in values.items():
            x, y, _ = self.electrode_manager.get_position(electrode)
            x_coords.append(x)
            y_coords.append(y)
            value_list.append(value)
        
        # 绘制头部轮廓
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # 绘制鼻子和耳朵标记
        ax.plot([0, 0], [0.9, 1.1], 'k-', linewidth=2)  # 鼻子
        ax.plot([-1.1, -1], [0, 0], 'k-', linewidth=2)  # 左耳
        ax.plot([1, 1.1], [0, 0], 'k-', linewidth=2)    # 右耳
        
        # 绘制电极点
        scatter = ax.scatter(x_coords, y_coords, c=value_list, cmap='jet', s=50, zorder=5)
        
        # 添加电极标签
        for electrode in values.keys():
            x, y, _ = self.electrode_manager.get_position(electrode)
            ax.text(x, y, electrode.name, ha='center', va='center', fontsize=8)
        
        # 设置坐标轴
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("Value")
        
        # 设置标题
        ax.set_title(title)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def plot_band_power_topography(self, eeg_data: Dict[EEGElectrode, np.ndarray], 
                                  band: EEGFrequencyBand, sampling_rate: int,
                                  title: str = None) -> Figure:
        """
        绘制频段功率地形图
        
        Args:
            eeg_data: 电极到EEG信号的映射
            band: 频段
            sampling_rate: 采样率(Hz)
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        from BrainSimulationSystem.core.eeg.analyzer import EEGAnalyzer
        
        analyzer = EEGAnalyzer(sampling_rate)
        
        # 计算每个电极的频段功率
        band_powers = {}
        for electrode, signal in eeg_data.items():
            band_powers_dict = analyzer.calculate_band_power(signal)
            band_powers[electrode] = band_powers_dict[band]
        
        # 设置标题
        if title is None:
            title = f"{band.name} Band Power Topography"
        
        # 绘制地形图
        return self.plot_topographic_map(band_powers, title)
    
    def plot_coherence_matrix(self, eeg_data: Dict[EEGElectrode, np.ndarray], 
                             band: EEGFrequencyBand, sampling_rate: int,
                             title: str = None) -> Figure:
        """
        绘制相干性矩阵
        
        Args:
            eeg_data: 电极到EEG信号的映射
            band: 频段
            sampling_rate: 采样率(Hz)
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        from BrainSimulationSystem.core.eeg.analyzer import EEGAnalyzer
        
        analyzer = EEGAnalyzer(sampling_rate)
        
        # 获取电极列表
        electrodes = list(eeg_data.keys())
        n_electrodes = len(electrodes)
        
        # 计算相干性矩阵
        coherence_matrix = np.zeros((n_electrodes, n_electrodes))
        
        for i, electrode1 in enumerate(electrodes):
            for j, electrode2 in enumerate(electrodes):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                elif i < j:
                    band_coherence = analyzer.calculate_band_coherence(
                        eeg_data[electrode1], eeg_data[electrode2]
                    )
                    coherence_matrix[i, j] = band_coherence[band]
                    coherence_matrix[j, i] = coherence_matrix[i, j]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制相干性矩阵
        im = ax.imshow(coherence_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Coherence")
        
        # 设置刻度标签
        ax.set_xticks(np.arange(n_electrodes))
        ax.set_yticks(np.arange(n_electrodes))
        ax.set_xticklabels([electrode.name for electrode in electrodes])
        ax.set_yticklabels([electrode.name for electrode in electrodes])
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 设置标题
        if title is None:
            title = f"{band.name} Band Coherence Matrix"
        ax.set_title(title)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def plot_3d_brain(self, values: Dict[EEGElectrode, float], 
                     title: str = "3D Brain Visualization") -> Figure:
        """
        绘制3D脑模型
        
        Args:
            values: 电极到值的映射
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "3D plotting not available", ha='center', va='center')
            return fig
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建球体表面
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 0.9 * np.outer(np.cos(u), np.sin(v))
        y = 0.9 * np.outer(np.sin(u), np.sin(v))
        z = 0.9 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 绘制半透明脑模型
        ax.plot_surface(x, y, z, color='gray', alpha=0.1)
        
        # 提取电极坐标和值
        x_coords = []
        y_coords = []
        z_coords = []
        value_list = []
        
        for electrode, value in values.items():
            x, y, z = self.electrode_manager.get_position(electrode)
            # 将2D坐标投影到球面
            norm = np.sqrt(x**2 + y**2 + 0.0001)
            z_coord = z if z != 0 else 0.9 * np.sqrt(1 - (x**2 + y**2) / 0.81)
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z_coord)
            value_list.append(value)
        
        # 绘制电极点
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=value_list, cmap='jet', s=100)
        
        # 添加电极标签
        for i, electrode in enumerate(values.keys()):
            ax.text(x_coords[i], y_coords[i], z_coords[i], electrode.name, fontsize=8)
        
        # 设置坐标轴
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect([1, 1, 1])
        
        # 移除坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("Value")
        
        # 设置标题
        ax.set_title(title)
        
        # 调整布局
        fig.tight_layout()
        
        return fig