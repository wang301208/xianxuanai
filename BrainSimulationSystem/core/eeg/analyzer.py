"""
EEG分析模块

实现EEG信号的分析功能，包括功率谱分析、频段功率计算、相干性分析等
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode, EEGFrequencyBand


class EEGAnalyzer:
    """EEG信号分析器"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        初始化EEG分析器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
    
    def calculate_power_spectrum(self, eeg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱
        
        Args:
            eeg_signal: EEG信号
            
        Returns:
            频率和功率谱密度
        """
        # 计算功率谱密度
        frequencies, psd = signal.welch(eeg_signal, fs=self.sampling_rate, nperseg=256)
        
        return frequencies, psd
    
    def calculate_band_power(self, eeg_signal: np.ndarray) -> Dict[EEGFrequencyBand, float]:
        """
        计算频段功率
        
        Args:
            eeg_signal: EEG信号
            
        Returns:
            频段到功率的映射
        """
        frequencies, psd = self.calculate_power_spectrum(eeg_signal)
        
        # 计算各频段功率
        band_powers = {}
        for band in EEGFrequencyBand:
            # 找到频段范围内的索引
            idx_band = np.logical_and(frequencies >= band.low_freq, frequencies <= band.high_freq)
            # 计算平均功率
            if np.any(idx_band):
                band_powers[band] = np.mean(psd[idx_band])
            else:
                band_powers[band] = 0.0
        
        return band_powers
    
    def calculate_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算两个信号之间的相干性
        
        Args:
            signal1: 第一个信号
            signal2: 第二个信号
            
        Returns:
            频率和相干性
        """
        # 计算相干性
        frequencies, coherence = signal.coherence(signal1, signal2, fs=self.sampling_rate)
        
        return frequencies, coherence
    
    def calculate_band_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict[EEGFrequencyBand, float]:
        """
        计算两个信号在各频段的相干性
        
        Args:
            signal1: 第一个信号
            signal2: 第二个信号
            
        Returns:
            频段到相干性的映射
        """
        frequencies, coherence = self.calculate_coherence(signal1, signal2)
        
        # 计算各频段相干性
        band_coherence = {}
        for band in EEGFrequencyBand:
            # 找到频段范围内的索引
            idx_band = np.logical_and(frequencies >= band.low_freq, frequencies <= band.high_freq)
            # 计算平均相干性
            if np.any(idx_band):
                band_coherence[band] = np.mean(coherence[idx_band])
            else:
                band_coherence[band] = 0.0
        
        return band_coherence
    
    def calculate_cross_correlation(self, signal1: np.ndarray, signal2: np.ndarray, 
                                   max_lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算两个信号的互相关
        
        Args:
            signal1: 第一个信号
            signal2: 第二个信号
            max_lag: 最大滞后(样本数)，如果为None则使用信号长度
            
        Returns:
            滞后和互相关
        """
        if max_lag is None:
            max_lag = len(signal1)
        
        # 计算互相关
        correlation = signal.correlate(signal1, signal2, mode='full')
        
        # 计算滞后
        lags = np.arange(-max_lag, max_lag + 1)
        
        # 归一化
        correlation /= np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
        
        # 截取中心部分
        center = len(correlation) // 2
        correlation = correlation[center - max_lag:center + max_lag + 1]
        
        return lags, correlation
    
    def calculate_time_frequency(self, eeg_signal: np.ndarray, 
                                window_size: int = 256, overlap: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算时频分析
        
        Args:
            eeg_signal: EEG信号
            window_size: 窗口大小(样本数)
            overlap: 重叠样本数
            
        Returns:
            时间、频率和功率谱密度
        """
        # 计算短时傅里叶变换
        frequencies, times, spectrogram = signal.spectrogram(
            eeg_signal, fs=self.sampling_rate, nperseg=window_size, noverlap=overlap
        )
        
        return times, frequencies, spectrogram
    
    def calculate_entropy(self, eeg_signal: np.ndarray, bins: int = 100) -> float:
        """
        计算信号熵
        
        Args:
            eeg_signal: EEG信号
            bins: 直方图箱数
            
        Returns:
            信号熵
        """
        # 计算直方图
        hist, _ = np.histogram(eeg_signal, bins=bins)
        
        # 归一化
        hist = hist / np.sum(hist)
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def calculate_hjorth_parameters(self, eeg_signal: np.ndarray) -> Tuple[float, float, float]:
        """
        计算Hjorth参数(活动度、移动度、复杂度)
        
        Args:
            eeg_signal: EEG信号
            
        Returns:
            活动度、移动度、复杂度
        """
        # 计算一阶差分
        diff1 = np.diff(eeg_signal)
        # 计算二阶差分
        diff2 = np.diff(diff1)
        
        # 计算方差
        var0 = np.var(eeg_signal)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        
        # 计算Hjorth参数
        activity = var0
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility
        
        return activity, mobility, complexity
    
    def plot_power_spectrum(self, eeg_signal: np.ndarray, title: str = "Power Spectrum") -> plt.Figure:
        """
        绘制功率谱
        
        Args:
            eeg_signal: EEG信号
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        frequencies, psd = self.calculate_power_spectrum(eeg_signal)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制功率谱
        ax.semilogy(frequencies, psd)
        
        # 标记频段
        band_colors = {
            EEGFrequencyBand.DELTA: 'blue',
            EEGFrequencyBand.THETA: 'green',
            EEGFrequencyBand.ALPHA: 'red',
            EEGFrequencyBand.BETA: 'purple',
            EEGFrequencyBand.GAMMA: 'orange'
        }
        
        y_min, y_max = ax.get_ylim()
        
        for band, color in band_colors.items():
            ax.axvspan(band.low_freq, band.high_freq, alpha=0.2, color=color)
            ax.text((band.low_freq + band.high_freq) / 2, y_max * 0.8,
                   band.name, ha='center', color=color)
        
        # 设置标签
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (μV²/Hz)")
        
        # 设置标题
        ax.set_title(title)
        
        # 设置x轴范围
        ax.set_xlim(0, 50)  # 限制到50Hz以便更好地查看低频
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def plot_time_frequency(self, eeg_signal: np.ndarray, title: str = "Time-Frequency Analysis") -> plt.Figure:
        """
        绘制时频分析
        
        Args:
            eeg_signal: EEG信号
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        times, frequencies, spectrogram = self.calculate_time_frequency(eeg_signal)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制时频图
        pcm = ax.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='gouraud', cmap='viridis')
        
        # 设置标签
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        
        # 设置标题
        ax.set_title(title)
        
        # 设置y轴范围
        ax.set_ylim(0, 50)  # 限制到50Hz以便更好地查看低频
        
        # 添加颜色条
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Power (dB)")
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig