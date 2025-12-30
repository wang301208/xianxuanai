"""
fMRI/BOLD信号模拟模块

实现功能性磁共振成像(fMRI)和血氧水平依赖(BOLD)信号的模拟，包括:
1. 血流动力学响应函数(HRF)建模
2. 神经活动到BOLD信号的转换
3. 体素级别的信号生成
4. 噪声和伪影模拟
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

from BrainSimulationSystem.core.connectome import BrainRegion, ConnectomeData


class HemodynamicResponseFunction:
    """血流动力学响应函数"""
    
    def __init__(self, sampling_rate: float = 1.0):
        """
        初始化HRF
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
    
    def canonical_hrf(self, duration: float = 30.0) -> np.ndarray:
        """
        生成标准HRF
        
        Args:
            duration: 持续时间(秒)
            
        Returns:
            HRF时间序列
        """
        # 时间点
        t = np.linspace(0, duration, int(duration * self.sampling_rate), endpoint=False)
        
        # 双伽马函数参数
        a1, a2 = 6, 16  # 时间延迟
        b1, b2 = 1, 1   # 色散
        c = 1/6         # 比例因子
        
        # 计算HRF
        hrf = (t**a1 * np.exp(-t/b1)) / (np.max(t**a1 * np.exp(-t/b1))) - \
              c * (t**a2 * np.exp(-t/b2)) / (np.max(t**a2 * np.exp(-t/b2)))
        
        # 归一化
        hrf = hrf / np.max(hrf)
        
        return hrf
    
    def balloon_windkessel_model(self, neural_activity: np.ndarray, dt: float = 0.1) -> Dict[str, np.ndarray]:
        """
        使用气球-风柜模型计算BOLD响应
        
        Args:
            neural_activity: 神经活动时间序列
            dt: 时间步长(秒)
            
        Returns:
            包含各变量时间序列的字典
        """
        # 模型参数
        alpha = 0.32  # 血流效率
        tau_s = 1.54  # 信号衰减时间常数(秒)
        tau_f = 2.46  # 自动调节时间常数(秒)
        tau_0 = 1.0   # 静息状态转换时间(秒)
        epsilon = 0.5 # 神经活动效率
        
        # 初始化变量
        n_steps = len(neural_activity)
        s = np.zeros(n_steps)  # 血流信号
        f = np.ones(n_steps)   # 血流
        v = np.ones(n_steps)   # 血容量
        q = np.ones(n_steps)   # 脱氧血红蛋白
        
        # 计算BOLD响应
        for i in range(1, n_steps):
            # 血流信号
            ds = epsilon * neural_activity[i-1] - s[i-1] / tau_s - (f[i-1] - 1) / tau_f
            s[i] = s[i-1] + ds * dt
            
            # 血流
            df = s[i-1]
            f[i] = f[i-1] + df * dt
            
            # 血容量
            dv = (f[i-1] - v[i-1]**(1/alpha)) / tau_0
            v[i] = v[i-1] + dv * dt
            
            # 脱氧血红蛋白
            dq = (f[i-1] * (1 - (1 - 0.4)**(1/f[i-1])) / 0.4 - q[i-1] * v[i-1]**(1/alpha-1)) / tau_0
            q[i] = q[i-1] + dq * dt
        
        # 计算BOLD信号
        bold = 100 * 0.03 * (7 * (1 - q) + 2 * (1 - q/v) + (1 - v))
        
        return {
            'signal': s,
            'flow': f,
            'volume': v,
            'deoxy': q,
            'bold': bold
        }
    
    def convolve_with_hrf(self, neural_activity: np.ndarray) -> np.ndarray:
        """
        将神经活动与HRF卷积
        
        Args:
            neural_activity: 神经活动时间序列
            
        Returns:
            BOLD信号
        """
        # 生成HRF
        hrf = self.canonical_hrf()
        
        # 卷积
        bold = np.convolve(neural_activity, hrf, mode='full')
        
        # 截断到与输入相同长度
        bold = bold[:len(neural_activity)]
        
        return bold


class BOLDSimulator:
    """BOLD信号模拟器"""
    
    def __init__(self, tr: float = 2.0, spatial_resolution: float = 3.0):
        """
        初始化BOLD模拟器
        
        Args:
            tr: 重复时间(秒)
            spatial_resolution: 空间分辨率(毫米)
        """
        self.tr = tr
        self.spatial_resolution = spatial_resolution
        self.hrf = HemodynamicResponseFunction(sampling_rate=1.0/tr)
        
        # 噪声参数
        self.thermal_noise_level = 0.1
        self.physiological_noise_level = 0.05
        self.drift_level = 0.02
        
        # 空间平滑参数
        self.spatial_smoothing_fwhm = 6.0  # 全宽半高(毫米)
    
    def simulate_voxel_timeseries(self, neural_activity: np.ndarray, 
                                 noise_level: float = None) -> np.ndarray:
        """
        模拟单个体素的时间序列
        
        Args:
            neural_activity: 神经活动时间序列
            noise_level: 噪声水平，如果为None则使用默认值
            
        Returns:
            体素时间序列
        """
        # 使用默认噪声水平
        if noise_level is None:
            noise_level = self.thermal_noise_level
        
        # 生成BOLD信号
        bold = self.hrf.convolve_with_hrf(neural_activity)
        
        # 添加噪声
        n_timepoints = len(bold)
        
        # 热噪声(高斯)
        thermal_noise = np.random.normal(0, noise_level, n_timepoints)
        
        # 生理噪声(低频)
        t = np.arange(n_timepoints) * self.tr
        cardiac_freq = 1.1  # Hz
        respiratory_freq = 0.3  # Hz
        physiological_noise = self.physiological_noise_level * (
            np.sin(2 * np.pi * cardiac_freq * t) + 
            np.sin(2 * np.pi * respiratory_freq * t)
        )
        
        # 低频漂移
        drift = self.drift_level * np.linspace(0, 1, n_timepoints)
        
        # 合并信号和噪声
        timeseries = bold + thermal_noise + physiological_noise + drift
        
        return timeseries
    
    def simulate_brain_volume(self, region_activities: Dict[BrainRegion, np.ndarray],
                             connectome: ConnectomeData,
                             volume_shape: Tuple[int, int, int] = (64, 64, 40)) -> np.ndarray:
        """
        模拟整个大脑体积的BOLD信号
        
        Args:
            region_activities: 脑区活动时间序列
            connectome: 脑连接组数据
            volume_shape: 体积形状(x, y, z)
            
        Returns:
            4D体积时间序列(t, x, y, z)
        """
        # 获取时间点数量
        n_timepoints = len(next(iter(region_activities.values())))
        
        # 创建4D体积
        volume = np.zeros((n_timepoints, *volume_shape))
        
        # 为每个脑区创建体素掩码
        region_masks = {}
        for region in region_activities.keys():
            # 获取脑区坐标(MNI空间)
            coord = connectome.get_region_coordinate(region)
            
            # 转换为体积索引
            x = int((coord[0] + 100) / 200 * volume_shape[0])
            y = int((coord[1] + 100) / 200 * volume_shape[1])
            z = int((coord[2] + 100) / 200 * volume_shape[2])
            
            # 确保索引在有效范围内
            x = max(0, min(x, volume_shape[0] - 1))
            y = max(0, min(y, volume_shape[1] - 1))
            z = max(0, min(z, volume_shape[2] - 1))
            
            # 创建高斯掩码
            sigma = connectome.get_region_volume(region)**(1/3) / (2 * self.spatial_resolution)
            mask = np.zeros(volume_shape)
            
            # 设置掩码范围
            radius = int(3 * sigma)
            x_min, x_max = max(0, x - radius), min(volume_shape[0], x + radius + 1)
            y_min, y_max = max(0, y - radius), min(volume_shape[1], y + radius + 1)
            z_min, z_max = max(0, z - radius), min(volume_shape[2], z + radius + 1)
            
            # 填充掩码
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    for k in range(z_min, z_max):
                        dist_sq = (i - x)**2 + (j - y)**2 + (k - z)**2
                        mask[i, j, k] = np.exp(-dist_sq / (2 * sigma**2))
            
            # 归一化掩码
            if np.sum(mask) > 0:
                mask = mask / np.sum(mask)
            
            region_masks[region] = mask
        
        # 为每个体素生成时间序列
        for region, activity in region_activities.items():
            mask = region_masks[region]
            
            # 对于掩码中的每个非零体素
            for i in range(volume_shape[0]):
                for j in range(volume_shape[1]):
                    for k in range(volume_shape[2]):
                        if mask[i, j, k] > 0:
                            # 缩放活动
                            scaled_activity = activity * mask[i, j, k]
                            
                            # 生成体素时间序列
                            voxel_timeseries = self.simulate_voxel_timeseries(scaled_activity)
                            
                            # 添加到体积
                            volume[:, i, j, k] += voxel_timeseries
        
        # 应用空间平滑
        if self.spatial_smoothing_fwhm > 0:
            sigma = self.spatial_smoothing_fwhm / (2.355 * self.spatial_resolution)
            for t in range(n_timepoints):
                volume[t] = self._spatial_smooth(volume[t], sigma)
        
        return volume
    
    def _spatial_smooth(self, volume: np.ndarray, sigma: float) -> np.ndarray:
        """
        对体积应用空间高斯平滑
        
        Args:
            volume: 3D体积
            sigma: 高斯核标准差
            
        Returns:
            平滑后的体积
        """
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(volume, sigma)
    
    def extract_roi_timeseries(self, volume: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        从体积中提取ROI时间序列
        
        Args:
            volume: 4D体积时间序列(t, x, y, z)
            roi_mask: 3D ROI掩码
            
        Returns:
            ROI时间序列
        """
        # 确保掩码与体积形状匹配
        if roi_mask.shape != volume.shape[1:]:
            raise ValueError("ROI mask shape does not match volume shape")
        
        # 提取时间序列
        timeseries = np.zeros(volume.shape[0])
        
        for t in range(volume.shape[0]):
            # 计算加权平均
            timeseries[t] = np.sum(volume[t] * roi_mask) / np.sum(roi_mask)
        
        return timeseries
    
    def plot_timeseries(self, timeseries: np.ndarray, title: str = "BOLD Time Series") -> plt.Figure:
        """
        绘制时间序列
        
        Args:
            timeseries: 时间序列
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 时间点
        t = np.arange(len(timeseries)) * self.tr
        
        # 绘制时间序列
        ax.plot(t, timeseries)
        
        # 设置标签
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BOLD Signal (%)")
        
        # 设置标题
        ax.set_title(title)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def plot_volume_slice(self, volume: np.ndarray, slice_idx: int, axis: int = 2,
                         timepoint: int = 0, title: str = None) -> plt.Figure:
        """
        绘制体积切片
        
        Args:
            volume: 4D体积时间序列(t, x, y, z)
            slice_idx: 切片索引
            axis: 切片轴(0=x, 1=y, 2=z)
            timepoint: 时间点索引
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 提取切片
        if axis == 0:
            slice_data = volume[timepoint, slice_idx, :, :]
            ax_labels = ('Y', 'Z')
        elif axis == 1:
            slice_data = volume[timepoint, :, slice_idx, :]
            ax_labels = ('X', 'Z')
        else:
            slice_data = volume[timepoint, :, :, slice_idx]
            ax_labels = ('X', 'Y')
        
        # 绘制切片
        im = ax.imshow(slice_data.T, cmap='viridis', origin='lower')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("BOLD Signal (%)")
        
        # 设置标签
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        
        # 设置标题
        if title is None:
            axis_name = ['X', 'Y', 'Z'][axis]
            title = f"BOLD Volume Slice ({axis_name}={slice_idx}, t={timepoint})"
        ax.set_title(title)
        
        # 调整布局
        fig.tight_layout()
        
        return fig