"""
单细胞记录接口模块 - 基础部分

实现单个神经元的电生理记录基础功能
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

from BrainSimulationSystem.core.neurons import Neuron, HodgkinHuxleyNeuron


class IntracellularRecording:
    """细胞内记录模拟基础类"""
    
    def __init__(self, sampling_rate: float = 10000.0):
        """
        初始化细胞内记录
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # 记录参数
        self.electrode_resistance = 5.0  # 电极电阻(MΩ)
        self.electrode_capacitance = 4.0  # 电极电容(pF)
        self.access_resistance = 15.0  # 通路电阻(MΩ)
        self.series_resistance_compensation = 0.7  # 串联电阻补偿(0-1)
        
        # 噪声参数
        self.thermal_noise_level = 0.1  # 热噪声水平(mV)
        self.line_noise_frequency = 50.0  # 电源噪声频率(Hz)
        self.line_noise_amplitude = 0.05  # 电源噪声幅度(mV)
    
    def record_membrane_potential(self, neuron: Neuron, duration: float, 
                               input_current: Optional[Union[float, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        记录神经元膜电位
        
        Args:
            neuron: 神经元对象
            duration: 记录持续时间(ms)
            input_current: 输入电流(nA)，可以是常数或时间序列
            
        Returns:
            包含记录数据的字典
        """
        # 计算时间点数量
        n_points = int(duration * self.sampling_rate / 1000)
        
        # 创建时间序列
        t = np.arange(n_points) * self.dt * 1000  # 转换为ms
        
        # 准备输入电流
        if input_current is None:
            input_current = np.zeros(n_points)
        elif isinstance(input_current, (int, float)):
            input_current = np.ones(n_points) * input_current
        
        # 确保输入电流长度正确
        if len(input_current) != n_points:
            raise ValueError(f"Input current length ({len(input_current)}) does not match time points ({n_points})")
        
        # 记录膜电位
        v_m = np.zeros(n_points)
        spikes = np.zeros(n_points, dtype=bool)
        
        # 重置神经元状态
        neuron.reset()
        
        # 模拟记录
        for i in range(n_points):
            # 更新神经元
            spike = neuron.step(self.dt, float(input_current[i]), current_time=float(i) * float(self.dt))
            
            # 记录膜电位和尖峰
            v_m[i] = neuron.V
            spikes[i] = spike
            
            # 添加电极和放大器效应
            if i > 0:
                # 电极低通滤波
                tau_electrode = self.electrode_resistance * self.electrode_capacitance * 1e-6  # 转换为ms
                v_m[i] = v_m[i-1] + (v_m[i] - v_m[i-1]) * (1 - np.exp(-self.dt / tau_electrode))
                
                # 串联电阻效应
                if not self.series_resistance_compensation == 1.0:
                    v_drop = input_current[i] * self.access_resistance * (1 - self.series_resistance_compensation)
                    v_m[i] -= v_drop
        
        # 添加噪声
        v_m = self._add_recording_noise(v_m, t)
        
        return {
            'time': t,
            'membrane_potential': v_m,
            'input_current': input_current,
            'spikes': spikes
        }
    
    def _add_recording_noise(self, v_m: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        添加记录噪声
        
        Args:
            v_m: 膜电位
            t: 时间序列(ms)
            
        Returns:
            添加噪声后的膜电位
        """
        # 热噪声(高斯)
        thermal_noise = np.random.normal(0, self.thermal_noise_level, len(v_m))
        
        # 电源噪声(正弦)
        line_noise = self.line_noise_amplitude * np.sin(2 * np.pi * self.line_noise_frequency * t / 1000)
        
        # 1/f噪声
        n_points = len(v_m)
        f = np.fft.fftfreq(n_points, self.dt)
        f[0] = 1e-10  # 避免除以零
        
        # 生成1/f噪声频谱
        spectrum = np.random.normal(0, 1, n_points) / np.sqrt(np.abs(f))
        spectrum[0] = 0  # 移除DC分量
        
        # 转换回时域
        pink_noise = np.real(np.fft.ifft(spectrum))
        
        # 缩放噪声
        pink_noise = pink_noise / np.std(pink_noise) * self.thermal_noise_level * 0.5
        
        # 合并噪声
        v_m_noisy = v_m + thermal_noise + line_noise + pink_noise
        
        return v_m_noisy
    
    def analyze_membrane_potential(self, recording: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        分析膜电位记录
        
        Args:
            recording: 记录数据字典
            
        Returns:
            分析结果字典
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        spikes = recording['spikes']
        
        # 计算基本统计量
        v_mean = np.mean(v_m)
        v_std = np.std(v_m)
        v_min = np.min(v_m)
        v_max = np.max(v_m)
        
        # 计算静息电位(使用非尖峰时间点)
        non_spike_mask = ~spikes
        if np.any(non_spike_mask):
            v_rest = np.mean(v_m[non_spike_mask])
        else:
            v_rest = v_mean
        
        # 检测尖峰
        spike_times = t[spikes]
        n_spikes = len(spike_times)
        
        # 计算尖峰频率
        if n_spikes > 1:
            spike_intervals = np.diff(spike_times)
            mean_isi = np.mean(spike_intervals)
            cv_isi = np.std(spike_intervals) / mean_isi if mean_isi > 0 else 0
            firing_rate = 1000 / mean_isi if mean_isi > 0 else 0  # Hz
        else:
            mean_isi = 0
            cv_isi = 0
            firing_rate = 0
        
        # 计算膜时间常数(使用电流阶跃响应)
        tau_m = self._estimate_membrane_time_constant(recording)
        
        # 计算输入电阻
        r_input = self._estimate_input_resistance(recording)
        
        # 返回分析结果
        return {
            'v_rest': v_rest,
            'v_mean': v_mean,
            'v_std': v_std,
            'v_min': v_min,
            'v_max': v_max,
            'n_spikes': n_spikes,
            'spike_times': spike_times,
            'mean_isi': mean_isi,
            'cv_isi': cv_isi,
            'firing_rate': firing_rate,
            'tau_m': tau_m,
            'r_input': r_input,
            'threshold': self._estimate_spike_thresholds(recording),
            'amplitude': self._estimate_spike_amplitudes(recording),
            'width': self._estimate_spike_widths(recording),
            'ahp_amplitude': self._estimate_ahp_amplitudes(recording)
        }
    
    def _estimate_spike_thresholds(self, recording: Dict[str, np.ndarray]) -> np.ndarray:
        """
        估计尖峰阈值
        
        Args:
            recording: 记录数据字典
            
        Returns:
            尖峰阈值数组(mV)
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        spikes = recording['spikes']
        
        # 找到尖峰时间点
        spike_indices = np.where(spikes)[0]
        
        # 计算每个尖峰的阈值
        thresholds = []
        
        for idx in spike_indices:
            # 提取尖峰前的膜电位
            pre_spike_idx = max(0, idx - int(2/self.dt))
            pre_spike_v = v_m[pre_spike_idx:idx+1]
            
            if len(pre_spike_v) < 3:
                continue
            
            # 计算二阶导数
            d2v = np.diff(np.diff(pre_spike_v))
            
            # 找到二阶导数最大点
            max_d2v_idx = np.argmax(d2v) if len(d2v) > 0 else 0
            
            # 阈值是二阶导数最大点对应的膜电位
            threshold_idx = pre_spike_idx + max_d2v_idx + 1
            threshold = v_m[threshold_idx]
            
            thresholds.append(threshold)
        
        return np.array(thresholds)
    
    def _estimate_spike_amplitudes(self, recording: Dict[str, np.ndarray]) -> np.ndarray:
        """
        估计尖峰振幅
        
        Args:
            recording: 记录数据字典
            
        Returns:
            尖峰振幅数组(mV)
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        spikes = recording['spikes']
        
        # 找到尖峰时间点
        spike_indices = np.where(spikes)[0]
        
        # 计算每个尖峰的振幅
        amplitudes = []
        
        for idx in spike_indices:
            # 提取尖峰前的膜电位作为基线
            pre_spike_idx = max(0, idx - int(2/self.dt))
            baseline = np.mean(v_m[pre_spike_idx:pre_spike_idx+int(1/self.dt)])
            
            # 提取尖峰周围的膜电位
            spike_window = min(len(v_m) - idx, int(5/self.dt))
            spike_v = v_m[idx:idx+spike_window]
            
            if len(spike_v) < 2:
                continue
            
            # 尖峰峰值
            peak = np.max(spike_v)
            
            # 计算振幅
            amplitude = peak - baseline
            
            amplitudes.append(amplitude)
        
        return np.array(amplitudes)
    
    def _estimate_spike_widths(self, recording: Dict[str, np.ndarray]) -> np.ndarray:
        """
        估计尖峰宽度
        
        Args:
            recording: 记录数据字典
            
        Returns:
            尖峰宽度数组(ms)
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        spikes = recording['spikes']
        
        # 找到尖峰时间点
        spike_indices = np.where(spikes)[0]
        
        # 计算每个尖峰的宽度
        widths = []
        
        for idx in spike_indices:
            # 提取尖峰周围的膜电位
            spike_window = min(len(v_m) - idx, int(10/self.dt))
            spike_v = v_m[idx:idx+spike_window]
            spike_t = t[idx:idx+spike_window]
            
            if len(spike_v) < 5:
                continue
            
            # 找到尖峰峰值
            peak_idx = np.argmax(spike_v)
            peak_v = spike_v[peak_idx]
            
            # 计算半高点
            half_height = peak_v / 2
            
            # 找到上升和下降穿过半高点的时间
            rising_indices = np.where(spike_v[:peak_idx] < half_height)[0]
            falling_indices = np.where(spike_v[peak_idx:] < half_height)[0]
            
            if len(rising_indices) == 0 or len(falling_indices) == 0:
                continue
            
            rising_idx = rising_indices[-1]
            falling_idx = falling_indices[0] + peak_idx
            
            # 计算宽度
            width = spike_t[falling_idx] - spike_t[rising_idx]
            
            widths.append(width)
        
        return np.array(widths)
    
    def _estimate_ahp_amplitudes(self, recording: Dict[str, np.ndarray]) -> np.ndarray:
        """
        估计后超极化振幅
        
        Args:
            recording: 记录数据字典
            
        Returns:
            后超极化振幅数组(mV)
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        spikes = recording['spikes']
        
        # 找到尖峰时间点
        spike_indices = np.where(spikes)[0]
        
        # 计算每个尖峰的后超极化振幅
        ahp_amplitudes = []
        
        for idx in spike_indices:
            # 提取尖峰前的膜电位作为基线
            pre_spike_idx = max(0, idx - int(2/self.dt))
            baseline = np.mean(v_m[pre_spike_idx:pre_spike_idx+int(1/self.dt)])
            
            # 提取尖峰后的膜电位
            post_spike_idx = min(len(v_m) - 1, idx + int(5/self.dt))
            post_spike_v = v_m[idx:post_spike_idx]
            
            if len(post_spike_v) < 5:
                continue
            
            # 找到后超极化最低点
            min_idx = np.argmin(post_spike_v)
            ahp_v = post_spike_v[min_idx]
            
            # 计算后超极化振幅
            ahp_amplitude = baseline - ahp_v
            
            ahp_amplitudes.append(ahp_amplitude)
        
        return np.array(ahp_amplitudes)
    
    def plot_recording(self, recording: Dict[str, np.ndarray], title: str = "Intracellular Recording") -> plt.Figure:
        """
        绘制记录结果
        
        Args:
            recording: 记录数据字典
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        i_in = recording['input_current']
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 绘制膜电位
        ax1.plot(t, v_m)
        ax1.set_ylabel("Membrane Potential (mV)")
        ax1.set_title(title)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制输入电流
        ax2.plot(t, i_in)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Input Current (nA)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig


class MultiElectrodeArray:
    """多电极阵列支持"""
    
    def __init__(self, n_electrodes=4, spacing=100.0):
        """
        初始化多电极阵列
        
        Args:
            n_electrodes: 电极数量
            spacing: 电极间距(μm)
        """
        self.electrodes = [ExtracellularRecording() for _ in range(n_electrodes)]
        self.spacing = spacing
        self.positions = np.array([
            [i * spacing, 0] for i in range(n_electrodes)
        ])  # 线性排列
        
    def record_network(self, neurons, duration):
        """
        记录神经元网络活动
        
        Args:
            neurons: 神经元列表(需包含位置信息)
            duration: 记录时长(ms)
            
        Returns:
            各电极记录数据的列表
        """
        recordings = []
        for i, electrode in enumerate(self.electrodes):
            # 找到最近的神经元
            dists = [np.linalg.norm(neuron.position - self.positions[i]) 
                    for neuron in neurons]
            nearest_neuron = neurons[np.argmin(dists)]
            
            # 记录(距离影响信号幅度)
            recording = electrode.record_spikes(
                nearest_neuron, duration)
            recording['electrode_pos'] = self.positions[i]
            recordings.append(recording)
        
        return recordings


class ExtracellularRecording:
    """细胞外记录模拟"""
    
    def __init__(self, sampling_rate: float = 30000.0):
        """
        初始化细胞外记录
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # 记录参数
        self.electrode_impedance = 1.0  # 电极阻抗(MΩ)
        self.electrode_distance = 50.0  # 电极距离(μm)
        self.electrode_radius = 10.0  # 电极半径(μm)
        
        # 噪声参数
        self.thermal_noise_level = 5.0  # 热噪声水平(μV)
        self.biological_noise_level = 10.0  # 生物噪声水平(μV)
        self.line_noise_frequency = 50.0  # 电源噪声频率(Hz)
        self.line_noise_amplitude = 2.0  # 电源噪声幅度(μV)
    
    def record_spikes(self, neuron: Neuron, duration: float, 
                     input_current: Optional[Union[float, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        记录神经元尖峰
        
        Args:
            neuron: 神经元对象
            duration: 记录持续时间(ms)
            input_current: 输入电流(nA)，可以是常数或时间序列
            
        Returns:
            包含记录数据的字典
        """
        # 计算时间点数量
        n_points = int(duration * self.sampling_rate / 1000)
        
        # 创建时间序列
        t = np.arange(n_points) * self.dt * 1000  # 转换为ms
        
        # 准备输入电流
        if input_current is None:
            input_current = np.zeros(n_points)
        elif isinstance(input_current, (int, float)):
            input_current = np.ones(n_points) * input_current
        
        # 确保输入电流长度正确
        if len(input_current) != n_points:
            raise ValueError(f"Input current length ({len(input_current)}) does not match time points ({n_points})")
        
        # 记录膜电位和尖峰
        v_m = np.zeros(n_points)
        spikes = np.zeros(n_points, dtype=bool)
        
        # 重置神经元状态
        neuron.reset()
        
        # 模拟记录
        for i in range(n_points):
            # 更新神经元
            spike = neuron.step(self.dt, float(input_current[i]), current_time=float(i) * float(self.dt))
            
            # 记录膜电位和尖峰
            v_m[i] = neuron.V
            spikes[i] = spike
        
        # 生成细胞外电位
        extracellular_potential = self._generate_extracellular_potential(v_m, spikes)
        
        # 添加噪声
        extracellular_potential = self._add_recording_noise(extracellular_potential, t)
        
        return {
            'time': t,
            'extracellular_potential': extracellular_potential,
            'membrane_potential': v_m,
            'input_current': input_current,
            'spikes': spikes
        }
    
    def _generate_extracellular_potential(self, v_m: np.ndarray, spikes: np.ndarray) -> np.ndarray:
        """
        生成细胞外电位
        
        Args:
            v_m: 膜电位
            spikes: 尖峰标记
            
        Returns:
            细胞外电位
        """
        # 计算膜电流(简化模型)
        i_m = np.diff(v_m, prepend=v_m[0])
        
        # 计算细胞外电位(使用点源近似)
        # V_ext = I_m / (4 * pi * sigma * r)，其中sigma是电导率，r是距离
        sigma = 0.3  # S/m
        r = self.electrode_distance * 1e-6  # 转换为m
        
        # 缩放因子
        scale_factor = 1 / (4 * np.pi * sigma * r)
        
        # 生成细胞外电位(μV)
        extracellular_potential = -i_m * scale_factor * 1e6
        
        # 添加尖峰波形
        for i in np.where(spikes)[0]:
            # 尖峰波形(三相波形)
            spike_waveform = self._generate_spike_waveform()
            
            # 添加尖峰波形
            start_idx = i
            end_idx = min(start_idx + len(spike_waveform), len(extracellular_potential))
            waveform_len = end_idx - start_idx
            
            extracellular_potential[start_idx:end_idx] += spike_waveform[:waveform_len]
        
        return extracellular_potential
    
    def _generate_spike_waveform(self) -> np.ndarray:
        """
        生成尖峰波形
        
        Returns:
            尖峰波形
        """
        # 尖峰持续时间(ms)
        spike_duration = 2.0
        
        # 计算波形长度
        n_points = int(spike_duration * self.sampling_rate / 1000)
        
        # 创建时间序列
        t = np.arange(n_points) * self.dt * 1000  # 转换为ms
        
        # 生成三相波形
        waveform = -70 * np.exp(-(t - 0.5)**2 / 0.1) + 30 * np.exp(-(t - 1.0)**2 / 0.2) - 20 * np.exp(-(t - 1.5)**2 / 0.4)
        
        return waveform
    
    def _add_recording_noise(self, potential: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        添加记录噪声
        
        Args:
            potential: 细胞外电位
            t: 时间序列(ms)
            
        Returns:
            添加噪声后的细胞外电位
        """
        # 热噪声(高斯)
        thermal_noise = np.random.normal(0, self.thermal_noise_level, len(potential))
        
        # 电源噪声(正弦)
        line_noise = self.line_noise_amplitude * np.sin(2 * np.pi * self.line_noise_frequency * t / 1000)
        
        # 生物噪声(1/f噪声)
        n_points = len(potential)
        f = np.fft.fftfreq(n_points, self.dt)
        f[0] = 1e-10  # 避免除以零
        
        # 生成1/f噪声频谱
        spectrum = np.random.normal(0, 1, n_points) / np.sqrt(np.abs(f))
        spectrum[0] = 0  # 移除DC分量
        
        # 转换回时域
        biological_noise = np.real(np.fft.ifft(spectrum))
        
        # 缩放噪声
        biological_noise = biological_noise / np.std(biological_noise) * self.biological_noise_level
        
        # 合并噪声
        potential_noisy = potential + thermal_noise + line_noise + biological_noise
        
        return potential_noisy
    
    def detect_spikes(self, recording: Dict[str, np.ndarray], threshold: float = 4.0) -> Dict[str, np.ndarray]:
        """
        检测细胞外记录中的尖峰
        
        Args:
            recording: 记录数据字典
            threshold: 检测阈值(标准差倍数)
            
        Returns:
            包含检测结果的字典
        """
        # 提取数据
        t = recording['time']
        v_ext = recording['extracellular_potential']
        
        # 计算噪声标准差
        noise_std = np.std(v_ext)
        
        # 检测尖峰
        spike_indices = np.where(v_ext > threshold * noise_std)[0]
        
        # 去除重复检测(同一尖峰的多个采样点)
        spike_indices = spike_indices[np.diff(spike_indices, prepend=-100) > 5]
        
        # 计算尖峰时间
        spike_times = t[spike_indices]
        
        # 计算尖峰波形
        spike_waveforms = []
        for idx in spike_indices:
            start_idx = max(0, idx - int(1/self.dt))
            end_idx = min(len(v_ext), idx + int(2/self.dt))
            spike_waveforms.append(v_ext[start_idx:end_idx])
        
        return {
            'spike_times': spike_times,
            'spike_indices': spike_indices,
            'spike_waveforms': spike_waveforms,
            'threshold': threshold * noise_std
        }
    
    def plot_recording(self, recording: Dict[str, np.ndarray], title: str = "Extracellular Recording") -> plt.Figure:
        """
        绘制记录结果
        
        Args:
            recording: 记录数据字典
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        # 提取数据
        t = recording['time']
        v_ext = recording['extracellular_potential']
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制细胞外电位
        ax.plot(t, v_ext)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Extracellular Potential (μV)")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def _estimate_membrane_time_constant(self, recording: Dict[str, np.ndarray]) -> float:
        """
        估计膜时间常数
        
        Args:
            recording: 记录数据字典
            
        Returns:
            膜时间常数(ms)
        """
        # 提取数据
        t = recording['time']
        v_m = recording['membrane_potential']
        i_in = recording['input_current']
        
        # 查找电流阶跃
        di = np.diff(i_in)
        step_indices = np.where(np.abs(di) > 0.1)[0]
        
        if len(step_indices) == 0:
            return 0
        
        # 使用第一个阶跃
        step_idx = step_indices[0]
        
        # 提取阶跃响应
        t_step = t[step_idx:step_idx+int(20/self.dt)]  # 20ms窗口
        v_step = v_m[step_idx:step_idx+int(20/self.dt)]
        
        if len(t_step) < 10:
            return 0
        
        # 减去基线
        v_baseline = np.mean(v_m[max(0, step_idx-int(5/self.dt)):step_idx])
        v_step = v_step - v_baseline
        
        # 归一化
        v_step = v_step / v_step[-1] if v_step[-1] != 0 else v_step
        
        # 指数拟合
        try:
            from scipy.optimize import curve_fit
            
            def exp_func(t, tau):
                return 1 - np.exp(-t/tau)
            
            t_fit = t_step - t_step[0]
            popt, _ = curve_fit(exp_func, t_fit, v_step, p0=[5], bounds=(0, 100))
            tau_m = popt[0]
        except:
            # 简单估计
            idx_63 = np.argmin(np.abs(v_step - 0.63))
            tau_m = t_step[idx_63] - t_step[0] if idx_63 > 0 else 0
        
        return tau_m
    
    def _estimate_input_resistance(self, recording: Dict[str, np.ndarray]) -> float:
        """
        估计输入电阻
        
        Args:
            recording: 记录数据字典
            
        Returns:
            输入电阻(MΩ)
        """
        # 提取数据
        v_m = recording['membrane_potential']
        i_in = recording['input_current']
        
        # 查找电流阶跃
        di = np.diff(i_in)
        step_indices = np.where(np.abs(di) > 0.1)[0]
        
        if len(step_indices) == 0:
            return 0
        
        # 使用第一个阶跃
        step_idx = step_indices[0]
        
        # 计算电流变化
        i_before = np.mean(i_in[max(0, step_idx-int(5/self.dt)):step_idx])
        i_after = np.mean(i_in[step_idx+int(50/self.dt):step_idx+int(100/self.dt)])
        di = i_after - i_before
        
        if abs(di) < 1e-10:
            return 0
        
        # 计算电压变化
        v_before = np.mean(v_m[max(0, step_idx-int(5/self.dt)):step_idx])
        v_after = np.mean(v_m[step_idx+int(50/self.dt):step_idx+int(100/self.dt)])
        dv = v_after - v_before
        
        # 计算输入电阻
        r_input = dv / di  # mV/nA = MΩ
        
        return r_input
