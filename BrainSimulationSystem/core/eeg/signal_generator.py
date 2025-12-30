"""
EEG信号生成模块

实现不同频段脑电波的生成，包括δ, θ, α, β, γ波
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode, EEGFrequencyBand
from BrainSimulationSystem.core.connectome import BrainRegion


class SignalGenerator:
    """EEG信号生成器"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        初始化信号生成器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        
        # 频段基线振幅
        self.band_amplitudes = {
            EEGFrequencyBand.DELTA: 30.0,
            EEGFrequencyBand.THETA: 20.0,
            EEGFrequencyBand.ALPHA: 25.0,
            EEGFrequencyBand.BETA: 10.0,
            EEGFrequencyBand.GAMMA: 5.0
        }
    
    def generate_band_signal(self, band: EEGFrequencyBand, duration: float, 
                            amplitude_modulation: float = 1.0) -> np.ndarray:
        """
        生成特定频段的EEG信号
        
        Args:
            band: 频段
            duration: 持续时间(秒)
            amplitude_modulation: 振幅调制因子
            
        Returns:
            生成的信号
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # 在频段范围内生成多个频率分量
        num_components = 5
        signal_components = []
        
        for i in range(num_components):
            # 在频段范围内均匀分布频率
            freq = band.low_freq + (band.high_freq - band.low_freq) * (i / (num_components - 1))
            # 振幅随频率降低
            amp = self.band_amplitudes[band] * amplitude_modulation * (1.0 - 0.5 * i / num_components)
            # 随机相位
            phase = 2 * np.pi * np.random.random()
            # 生成正弦波
            component = amp * np.sin(2 * np.pi * freq * t + phase)
            signal_components.append(component)
        
        # 合并所有分量
        band_signal = np.sum(signal_components, axis=0)
        
        # 添加少量高斯噪声
        noise = np.random.normal(0, self.band_amplitudes[band] * 0.05, num_samples)
        
        return band_signal + noise
    
    def generate_eeg_signal(self, duration: float, region_activities: Dict[BrainRegion, float],
                           electrode_regions: List[BrainRegion]) -> np.ndarray:
        """
        生成EEG信号
        
        Args:
            duration: 持续时间(秒)
            region_activities: 脑区活动水平字典
            electrode_regions: 与电极相关的脑区列表
            
        Returns:
            生成的EEG信号
        """
        # 计算电极处的平均活动水平
        if electrode_regions:
            avg_activity = sum(region_activities.get(region, 0.5) for region in electrode_regions) / len(electrode_regions)
        else:
            avg_activity = 0.5
        
        # 基于活动水平调整各频段的振幅
        delta_amp = 1.5 - avg_activity  # δ波在低活动时增强
        theta_amp = 1.0 + 0.5 * (avg_activity - 0.5)  # θ波在中等活动时增强
        alpha_amp = 1.0 + 0.8 * (1.0 - avg_activity)  # α波在放松时增强
        beta_amp = 1.0 + avg_activity  # β波在活动时增强
        gamma_amp = 0.5 + 1.5 * avg_activity  # γ波在高活动时显著增强
        
        # 生成各频段信号
        delta_signal = self.generate_band_signal(EEGFrequencyBand.DELTA, duration, delta_amp)
        theta_signal = self.generate_band_signal(EEGFrequencyBand.THETA, duration, theta_amp)
        alpha_signal = self.generate_band_signal(EEGFrequencyBand.ALPHA, duration, alpha_amp)
        beta_signal = self.generate_band_signal(EEGFrequencyBand.BETA, duration, beta_amp)
        gamma_signal = self.generate_band_signal(EEGFrequencyBand.GAMMA, duration, gamma_amp)
        
        # 合并所有频段信号
        eeg_signal = delta_signal + theta_signal + alpha_signal + beta_signal + gamma_signal
        
        return eeg_signal
    
    def set_band_amplitude(self, band: EEGFrequencyBand, amplitude: float) -> None:
        """
        设置频段基线振幅
        
        Args:
            band: 频段
            amplitude: 振幅
        """
        self.band_amplitudes[band] = amplitude
    
    def get_band_amplitude(self, band: EEGFrequencyBand) -> float:
        """
        获取频段基线振幅
        
        Args:
            band: 频段
            
        Returns:
            振幅
        """
        return self.band_amplitudes[band]
    
    def generate_custom_signal(self, frequencies: List[float], amplitudes: List[float], 
                              phases: List[float], duration: float) -> np.ndarray:
        """
        生成自定义信号
        
        Args:
            frequencies: 频率列表(Hz)
            amplitudes: 振幅列表
            phases: 相位列表(弧度)
            duration: 持续时间(秒)
            
        Returns:
            生成的信号
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # 确保列表长度一致
        n = min(len(frequencies), len(amplitudes), len(phases))
        
        # 生成信号
        signal = np.zeros(num_samples)
        for i in range(n):
            signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])
        
        return signal