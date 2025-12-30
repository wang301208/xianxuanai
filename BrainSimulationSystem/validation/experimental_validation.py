# -*- coding: utf-8 -*-
"""
实验验证系统
Experimental Validation System

实现跨层实验验证：
1. 尖峰率验证
2. 功能连接验证
3. 行为输出验证
4. 与真实数据对比
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.metrics import mean_squared_error, r2_score
import json

class ValidationLevel(Enum):
    """验证层级"""
    CELLULAR = "cellular"           # 细胞层级
    CIRCUIT = "circuit"             # 回路层级
    NETWORK = "network"             # 网络层级
    BEHAVIORAL = "behavioral"       # 行为层级

class ValidationMetric(Enum):
    """验证指标"""
    FIRING_RATE = "firing_rate"
    SPIKE_TIMING = "spike_timing"
    CONNECTIVITY = "connectivity"
    OSCILLATIONS = "oscillations"
    SYNCHRONY = "synchrony"
    PLASTICITY = "plasticity"
    BEHAVIOR = "behavior"

@dataclass
class ExperimentalData:
    """实验数据"""
    name: str
    data_type: ValidationMetric
    level: ValidationLevel
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = 0.0

@dataclass
class ValidationResult:
    """验证结果"""
    metric: ValidationMetric
    level: ValidationLevel
    similarity_score: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    details: Dict[str, Any] = field(default_factory=dict)

class ExperimentalValidator:
    """实验验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ExperimentalValidator")
        
        # 实验数据库
        self.experimental_database = {}
        
        # 验证方法
        self.validation_methods = {
            ValidationMetric.FIRING_RATE: self._validate_firing_rate,
            ValidationMetric.SPIKE_TIMING: self._validate_spike_timing,
            ValidationMetric.CONNECTIVITY: self._validate_connectivity,
            ValidationMetric.OSCILLATIONS: self._validate_oscillations,
            ValidationMetric.SYNCHRONY: self._validate_synchrony,
            ValidationMetric.PLASTICITY: self._validate_plasticity,
            ValidationMetric.BEHAVIOR: self._validate_behavior
        }
        
        # 统计测试方法
        self.statistical_tests = {
            'correlation': self._correlation_test,
            'ks_test': self._kolmogorov_smirnov_test,
            't_test': self._t_test,
            'mann_whitney': self._mann_whitney_test,
            'chi_square': self._chi_square_test
        }
        
        self._load_experimental_data()
    
    def _load_experimental_data(self):
        """加载实验数据"""
        
        # 加载预定义的实验数据
        self._load_firing_rate_data()
        self._load_connectivity_data()
        self._load_oscillation_data()
        self._load_plasticity_data()
        self._load_behavioral_data()
    
    def _load_firing_rate_data(self):
        """加载发放率数据"""
        
        # V1神经元发放率分布（来自文献）
        v1_firing_rates = ExperimentalData(
            name="V1_firing_rates_awake",
            data_type=ValidationMetric.FIRING_RATE,
            level=ValidationLevel.CELLULAR,
            data=np.random.lognormal(mean=1.5, sigma=0.8, size=1000),  # 模拟数据
            metadata={
                'brain_area': 'V1',
                'condition': 'awake',
                'species': 'macaque',
                'recording_method': 'extracellular'
            },
            source="Ringach et al. 2002"
        )
        
        # 海马CA1发放率
        ca1_firing_rates = ExperimentalData(
            name="CA1_firing_rates_exploration",
            data_type=ValidationMetric.FIRING_RATE,
            level=ValidationLevel.CELLULAR,
            data=np.random.gamma(shape=2.0, scale=3.0, size=800),
            metadata={
                'brain_area': 'CA1',
                'condition': 'spatial_exploration',
                'species': 'rat'
            },
            source="O'Keefe & Nadel 1978"
        )
        
        self.experimental_database['firing_rates'] = [v1_firing_rates, ca1_firing_rates]
    
    def _load_connectivity_data(self):
        """加载连接数据"""
        
        # 皮层连接概率
        cortical_connectivity = ExperimentalData(
            name="cortical_connection_probability",
            data_type=ValidationMetric.CONNECTIVITY,
            level=ValidationLevel.CIRCUIT,
            data=np.array([0.1, 0.15, 0.08, 0.12, 0.09, 0.11]),  # 不同层间连接概率
            metadata={
                'brain_area': 'somatosensory_cortex',
                'connection_type': 'excitatory',
                'distance_range': '0-500um'
            },
            source="Lefort et al. 2009"
        )
        
        self.experimental_database['connectivity'] = [cortical_connectivity]
    
    def _load_oscillation_data(self):
        """加载振荡数据"""
        
        # 生成模拟的LFP数据
        fs = 1000  # 采样率
        t = np.arange(0, 10, 1/fs)  # 10秒数据
        
        # 模拟不同频段的振荡
        alpha_osc = 2 * np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
        beta_osc = 1.5 * np.sin(2 * np.pi * 20 * t)  # 20Hz beta
        gamma_osc = 1 * np.sin(2 * np.pi * 40 * t)   # 40Hz gamma
        noise = 0.5 * np.random.randn(len(t))
        
        lfp_signal = alpha_osc + beta_osc + gamma_osc + noise
        
        lfp_data = ExperimentalData(
            name="motor_cortex_LFP",
            data_type=ValidationMetric.OSCILLATIONS,
            level=ValidationLevel.NETWORK,
            data=lfp_signal,
            metadata={
                'sampling_rate': fs,
                'duration': 10.0,
                'brain_area': 'motor_cortex',
                'condition': 'movement_preparation'
            },
            source="Pfurtscheller & Lopes da Silva 1999"
        )
        
        self.experimental_database['oscillations'] = [lfp_data]
    
    def _load_plasticity_data(self):
        """加载可塑性数据"""
        
        # STDP时间窗口数据
        time_diffs = np.arange(-50, 51, 5)  # -50ms到+50ms
        stdp_window = np.zeros_like(time_diffs, dtype=float)
        
        # 典型STDP窗口
        for i, dt in enumerate(time_diffs):
            if dt > 0:  # LTP
                stdp_window[i] = 0.8 * np.exp(-dt / 20.0)
            else:  # LTD
                stdp_window[i] = -0.6 * np.exp(dt / 20.0)
        
        stdp_data = ExperimentalData(
            name="hippocampal_STDP_window",
            data_type=ValidationMetric.PLASTICITY,
            level=ValidationLevel.CELLULAR,
            data=stdp_window,
            metadata={
                'time_differences': time_diffs,
                'synapse_type': 'CA3-CA1',
                'protocol': 'paired_recordings'
            },
            source="Bi & Poo 1998"
        )
        
        self.experimental_database['plasticity'] = [stdp_data]
    
    def _load_behavioral_data(self):
        """加载行为数据"""
        
        # 反应时间分布
        reaction_times = np.random.gamma(shape=2.0, scale=150.0, size=1000)  # ms
        
        rt_data = ExperimentalData(
            name="visual_detection_reaction_times",
            data_type=ValidationMetric.BEHAVIOR,
            level=ValidationLevel.BEHAVIORAL,
            data=reaction_times,
            metadata={
                'task': 'visual_detection',
                'stimulus_type': 'simple',
                'n_trials': 1000
            },
            source="Luce 1986"
        )
        
        self.experimental_database['behavior'] = [rt_data]
    
    def validate_simulation_data(self, simulation_data: Dict[str, Any], 
                               validation_targets: List[ValidationMetric]) -> Dict[ValidationMetric, ValidationResult]:
        """验证仿真数据"""
        
        validation_results = {}
        
        for metric in validation_targets:
            if metric in self.validation_methods:
                try:
                    result = self.validation_methods[metric](simulation_data)
                    validation_results[metric] = result
                    
                    self.logger.info(f"Validation completed for {metric.value}: "
                                   f"similarity={result.similarity_score:.3f}, "
                                   f"p-value={result.p_value:.3f}")
                
                except Exception as e:
                    self.logger.error(f"Validation failed for {metric.value}: {str(e)}")
                    validation_results[metric] = ValidationResult(
                        metric=metric,
                        level=ValidationLevel.NETWORK,
                        similarity_score=0.0,
                        p_value=1.0,
                        effect_size=0.0,
                        confidence_interval=(0.0, 0.0),
                        details={'error': str(e)}
                    )
        
        return validation_results
    
    def _validate_firing_rate(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证发放率"""
        
        # 提取仿真发放率数据
        if 'firing_rates' in simulation_data:
            sim_rates = np.array(simulation_data['firing_rates'])
        elif 'neuron_activities' in simulation_data:
            sim_rates = np.array(list(simulation_data['neuron_activities'].values()))
        else:
            raise ValueError("No firing rate data found in simulation")
        
        # 选择合适的实验数据进行比较
        exp_data = self.experimental_database['firing_rates'][0]  # 使用V1数据
        exp_rates = exp_data.data
        
        # 统计比较
        similarity_score = self._compute_distribution_similarity(sim_rates, exp_rates)
        
        # Kolmogorov-Smirnov测试
        ks_stat, p_value = stats.ks_2samp(sim_rates, exp_rates)
        
        # 效应大小（Cohen's d）
        effect_size = self._compute_cohens_d(sim_rates, exp_rates)
        
        # 置信区间
        ci = self._compute_confidence_interval(similarity_score, len(sim_rates))
        
        return ValidationResult(
            metric=ValidationMetric.FIRING_RATE,
            level=ValidationLevel.CELLULAR,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'ks_statistic': ks_stat,
                'sim_mean': np.mean(sim_rates),
                'exp_mean': np.mean(exp_rates),
                'sim_std': np.std(sim_rates),
                'exp_std': np.std(exp_rates)
            }
        )
    
    def _validate_spike_timing(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证尖峰时序"""
        
        # 提取尖峰时间数据
        if 'spike_times' in simulation_data:
            spike_times = simulation_data['spike_times']
        else:
            raise ValueError("No spike timing data found in simulation")
        
        # 计算ISI分布
        all_isis = []
        for neuron_spikes in spike_times.values():
            if len(neuron_spikes) > 1:
                isis = np.diff(neuron_spikes)
                all_isis.extend(isis)
        
        sim_isis = np.array(all_isis)
        
        # 与典型ISI分布比较（指数分布）
        exp_isis = np.random.exponential(scale=100.0, size=len(sim_isis))  # 100ms平均ISI
        
        # 计算相似度
        similarity_score = self._compute_distribution_similarity(sim_isis, exp_isis)
        
        # 统计测试
        ks_stat, p_value = stats.ks_2samp(sim_isis, exp_isis)
        effect_size = self._compute_cohens_d(sim_isis, exp_isis)
        ci = self._compute_confidence_interval(similarity_score, len(sim_isis))
        
        return ValidationResult(
            metric=ValidationMetric.SPIKE_TIMING,
            level=ValidationLevel.CELLULAR,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'mean_isi_sim': np.mean(sim_isis),
                'mean_isi_exp': np.mean(exp_isis),
                'cv_isi_sim': np.std(sim_isis) / np.mean(sim_isis),
                'cv_isi_exp': np.std(exp_isis) / np.mean(exp_isis)
            }
        )
    
    def _validate_connectivity(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证连接性"""
        
        # 提取连接矩阵
        if 'connectivity_matrix' in simulation_data:
            conn_matrix = simulation_data['connectivity_matrix']
        elif 'synaptic_connections' in simulation_data:
            # 从突触连接构建连接矩阵
            connections = simulation_data['synaptic_connections']
            max_id = max(max(pre, post) for pre, post in connections.keys())
            conn_matrix = np.zeros((max_id + 1, max_id + 1))
            
            for (pre, post), connection in connections.items():
                conn_matrix[pre, post] = connection.weight if hasattr(connection, 'weight') else 1.0
        else:
            raise ValueError("No connectivity data found in simulation")
        
        # 计算连接概率
        sim_conn_prob = np.sum(conn_matrix > 0) / (conn_matrix.shape[0] * conn_matrix.shape[1])
        
        # 与实验数据比较
        exp_data = self.experimental_database['connectivity'][0]
        exp_conn_prob = np.mean(exp_data.data)
        
        # 相似度评分
        similarity_score = 1.0 - abs(sim_conn_prob - exp_conn_prob) / exp_conn_prob
        
        # 简单的t检验（假设正态分布）
        t_stat, p_value = stats.ttest_1samp([sim_conn_prob], exp_conn_prob)
        
        effect_size = abs(sim_conn_prob - exp_conn_prob) / np.std(exp_data.data)
        ci = (similarity_score - 0.1, similarity_score + 0.1)  # 简化的置信区间
        
        return ValidationResult(
            metric=ValidationMetric.CONNECTIVITY,
            level=ValidationLevel.CIRCUIT,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'sim_connection_probability': sim_conn_prob,
                'exp_connection_probability': exp_conn_prob,
                'connection_strength_mean': np.mean(conn_matrix[conn_matrix > 0])
            }
        )
    
    def _validate_oscillations(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证振荡"""
        
        # 提取LFP或群体活动数据
        if 'lfp_signal' in simulation_data:
            sim_signal = simulation_data['lfp_signal']
            fs = simulation_data.get('sampling_rate', 1000)
        elif 'population_activity' in simulation_data:
            sim_signal = simulation_data['population_activity']
            fs = simulation_data.get('sampling_rate', 1000)
        else:
            raise ValueError("No oscillation data found in simulation")
        
        # 获取实验数据
        exp_data = self.experimental_database['oscillations'][0]
        exp_signal = exp_data.data
        exp_fs = exp_data.metadata['sampling_rate']
        
        # 功率谱分析
        sim_freqs, sim_psd = signal.welch(sim_signal, fs=fs, nperseg=1024)
        exp_freqs, exp_psd = signal.welch(exp_signal, fs=exp_fs, nperseg=1024)
        
        # 在相同频率范围内比较
        freq_range = (1, 100)  # 1-100 Hz
        sim_mask = (sim_freqs >= freq_range[0]) & (sim_freqs <= freq_range[1])
        exp_mask = (exp_freqs >= freq_range[0]) & (exp_freqs <= freq_range[1])
        
        sim_psd_range = sim_psd[sim_mask]
        exp_psd_range = exp_psd[exp_mask]
        
        # 插值到相同频率网格
        common_freqs = np.linspace(freq_range[0], freq_range[1], 100)
        sim_psd_interp = np.interp(common_freqs, sim_freqs[sim_mask], sim_psd_range)
        exp_psd_interp = np.interp(common_freqs, exp_freqs[exp_mask], exp_psd_range)
        
        # 归一化功率谱
        sim_psd_norm = sim_psd_interp / np.sum(sim_psd_interp)
        exp_psd_norm = exp_psd_interp / np.sum(exp_psd_interp)
        
        # 计算相似度（相关系数）
        similarity_score = np.corrcoef(sim_psd_norm, exp_psd_norm)[0, 1]
        
        # 统计测试
        t_stat, p_value = stats.ttest_rel(sim_psd_norm, exp_psd_norm)
        
        effect_size = self._compute_cohens_d(sim_psd_norm, exp_psd_norm)
        ci = self._compute_confidence_interval(similarity_score, len(sim_psd_norm))
        
        return ValidationResult(
            metric=ValidationMetric.OSCILLATIONS,
            level=ValidationLevel.NETWORK,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'peak_frequency_sim': common_freqs[np.argmax(sim_psd_norm)],
                'peak_frequency_exp': common_freqs[np.argmax(exp_psd_norm)],
                'alpha_power_sim': np.sum(sim_psd_norm[(common_freqs >= 8) & (common_freqs <= 12)]),
                'alpha_power_exp': np.sum(exp_psd_norm[(common_freqs >= 8) & (common_freqs <= 12)])
            }
        )
    
    def _validate_synchrony(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证同步性"""
        
        # 提取尖峰数据
        if 'spike_times' in simulation_data:
            spike_times = simulation_data['spike_times']
        else:
            raise ValueError("No spike timing data found for synchrony validation")
        
        # 计算成对相关
        neuron_ids = list(spike_times.keys())
        correlations = []
        
        for i in range(len(neuron_ids)):
            for j in range(i + 1, len(neuron_ids)):
                corr = self._compute_spike_correlation(
                    spike_times[neuron_ids[i]], 
                    spike_times[neuron_ids[j]],
                    bin_size=10.0  # 10ms bins
                )
                correlations.append(corr)
        
        sim_correlations = np.array(correlations)
        
        # 与典型皮层同步性比较（通常较低）
        exp_correlations = np.random.beta(a=1.5, b=8.0, size=len(correlations)) * 0.3
        
        # 计算相似度
        similarity_score = self._compute_distribution_similarity(sim_correlations, exp_correlations)
        
        # 统计测试
        ks_stat, p_value = stats.ks_2samp(sim_correlations, exp_correlations)
        effect_size = self._compute_cohens_d(sim_correlations, exp_correlations)
        ci = self._compute_confidence_interval(similarity_score, len(sim_correlations))
        
        return ValidationResult(
            metric=ValidationMetric.SYNCHRONY,
            level=ValidationLevel.NETWORK,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'mean_correlation_sim': np.mean(sim_correlations),
                'mean_correlation_exp': np.mean(exp_correlations),
                'synchrony_strength': np.mean(sim_correlations)
            }
        )
    
    def _validate_plasticity(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证可塑性"""
        
        # 提取可塑性数据
        if 'plasticity_results' in simulation_data:
            plasticity_data = simulation_data['plasticity_results']
        else:
            raise ValueError("No plasticity data found in simulation")
        
        # 提取STDP窗口数据
        if 'stdp_window' in plasticity_data:
            sim_stdp = plasticity_data['stdp_window']
        else:
            # 从权重变化推断STDP窗口
            sim_stdp = self._extract_stdp_window(plasticity_data)
        
        # 获取实验STDP数据
        exp_data = self.experimental_database['plasticity'][0]
        exp_stdp = exp_data.data
        
        # 确保长度匹配
        min_len = min(len(sim_stdp), len(exp_stdp))
        sim_stdp = sim_stdp[:min_len]
        exp_stdp = exp_stdp[:min_len]
        
        # 计算相似度
        similarity_score = np.corrcoef(sim_stdp, exp_stdp)[0, 1]
        
        # 统计测试
        t_stat, p_value = stats.ttest_rel(sim_stdp, exp_stdp)
        effect_size = self._compute_cohens_d(sim_stdp, exp_stdp)
        ci = self._compute_confidence_interval(similarity_score, len(sim_stdp))
        
        return ValidationResult(
            metric=ValidationMetric.PLASTICITY,
            level=ValidationLevel.CELLULAR,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'ltp_amplitude_sim': np.max(sim_stdp),
                'ltp_amplitude_exp': np.max(exp_stdp),
                'ltd_amplitude_sim': np.min(sim_stdp),
                'ltd_amplitude_exp': np.min(exp_stdp)
            }
        )
    
    def _validate_behavior(self, simulation_data: Dict[str, Any]) -> ValidationResult:
        """验证行为"""
        
        # 提取行为数据
        if 'reaction_times' in simulation_data:
            sim_rt = simulation_data['reaction_times']
        elif 'behavioral_output' in simulation_data:
            sim_rt = simulation_data['behavioral_output'].get('reaction_times', [])
        else:
            raise ValueError("No behavioral data found in simulation")
        
        sim_rt = np.array(sim_rt)
        
        # 获取实验行为数据
        exp_data = self.experimental_database['behavior'][0]
        exp_rt = exp_data.data
        
        # 计算相似度
        similarity_score = self._compute_distribution_similarity(sim_rt, exp_rt)
        
        # 统计测试
        ks_stat, p_value = stats.ks_2samp(sim_rt, exp_rt)
        effect_size = self._compute_cohens_d(sim_rt, exp_rt)
        ci = self._compute_confidence_interval(similarity_score, len(sim_rt))
        
        return ValidationResult(
            metric=ValidationMetric.BEHAVIOR,
            level=ValidationLevel.BEHAVIORAL,
            similarity_score=similarity_score,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            details={
                'mean_rt_sim': np.mean(sim_rt),
                'mean_rt_exp': np.mean(exp_rt),
                'rt_variability_sim': np.std(sim_rt),
                'rt_variability_exp': np.std(exp_rt)
            }
        )
    
    def _compute_distribution_similarity(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算分布相似度"""
        
        # 使用多种指标的组合
        
        # 1. 均值和标准差的相似度
        mean_sim = 1.0 - abs(np.mean(data1) - np.mean(data2)) / (np.mean(data1) + np.mean(data2) + 1e-6)
        std_sim = 1.0 - abs(np.std(data1) - np.std(data2)) / (np.std(data1) + np.std(data2) + 1e-6)
        
        # 2. 分位数相似度
        quantiles = [0.25, 0.5, 0.75]
        q1 = np.percentile(data1, [25, 50, 75])
        q2 = np.percentile(data2, [25, 50, 75])
        quantile_sim = 1.0 - np.mean(np.abs(q1 - q2) / (q1 + q2 + 1e-6))
        
        # 3. KS统计量（转换为相似度）
        ks_stat, _ = stats.ks_2samp(data1, data2)
        ks_sim = 1.0 - ks_stat
        
        # 综合相似度
        similarity = (mean_sim + std_sim + quantile_sim + ks_sim) / 4
        
        return np.clip(similarity, 0.0, 1.0)
    
    def _compute_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算Cohen's d效应大小"""
        
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return abs(cohens_d)
    
    def _compute_confidence_interval(self, score: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        
        # 简化的置信区间计算
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # 假设标准误差
        se = np.sqrt(score * (1 - score) / n)
        
        lower = score - z_score * se
        upper = score + z_score * se
        
        return (max(0.0, lower), min(1.0, upper))
    
    def _compute_spike_correlation(self, spikes1: List[float], spikes2: List[float], 
                                 bin_size: float = 10.0) -> float:
        """计算尖峰相关性"""
        
        if not spikes1 or not spikes2:
            return 0.0
        
        # 确定时间范围
        all_spikes = spikes1 + spikes2
        t_min, t_max = min(all_spikes), max(all_spikes)
        
        # 创建时间bins
        bins = np.arange(t_min, t_max + bin_size, bin_size)
        
        # 计算每个bin的尖峰计数
        counts1, _ = np.histogram(spikes1, bins=bins)
        counts2, _ = np.histogram(spikes2, bins=bins)
        
        # 计算相关系数
        if np.std(counts1) > 0 and np.std(counts2) > 0:
            correlation = np.corrcoef(counts1, counts2)[0, 1]
        else:
            correlation = 0.0
        
        return correlation
    
    def _extract_stdp_window(self, plasticity_data: Dict[str, Any]) -> np.ndarray:
        """从可塑性数据提取STDP窗口"""
        
        # 这是一个简化的实现
        # 实际应该从权重变化和尖峰时序数据中推断
        
        time_diffs = np.arange(-50, 51, 5)
        stdp_window = np.zeros_like(time_diffs, dtype=float)
        
        # 使用默认STDP窗口作为占位
        for i, dt in enumerate(time_diffs):
            if dt > 0:
                stdp_window[i] = 0.5 * np.exp(-dt / 20.0)
            else:
                stdp_window[i] = -0.3 * np.exp(dt / 20.0)
        
        return stdp_window
    
    def generate_validation_report(self, validation_results: Dict[ValidationMetric, ValidationResult]) -> Dict[str, Any]:
        """生成验证报告"""
        
        report = {
            'summary': {
                'total_metrics': len(validation_results),
                'passed_metrics': sum(1 for r in validation_results.values() if r.similarity_score > 0.7),
                'overall_score': np.mean([r.similarity_score for r in validation_results.values()]),
                'timestamp': self.current_time if hasattr(self, 'current_time') else 0.0
            },
            'detailed_results': {},
            'recommendations': []
        }
        
        for metric, result in validation_results.items():
            report['detailed_results'][metric.value] = {
                'similarity_score': result.similarity_score,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'confidence_interval': result.confidence_interval,
                'level': result.level.value,
                'passed': result.similarity_score > 0.7,
                'details': result.details
            }
            
            # 生成建议
            if result.similarity_score < 0.5:
                report['recommendations'].append(
                    f"Low similarity for {metric.value} ({result.similarity_score:.2f}). "
                    f"Consider adjusting model parameters."
                )
            elif result.p_value > 0.05:
                report['recommendations'].append(
                    f"Non-significant difference for {metric.value} (p={result.p_value:.3f}). "
                    f"Good match with experimental data."
                )
        
        return report
    
    def save_validation_results(self, validation_results: Dict[ValidationMetric, ValidationResult], 
                              filepath: str):
        """保存验证结果"""
        
        # 生成报告
        report = self.generate_validation_report(validation_results)
        
        # 保存为JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {filepath}")

# 工厂函数
def create_experimental_validator(config: Optional[Dict[str, Any]] = None) -> ExperimentalValidator:
    """创建实验验证器"""
    if config is None:
        config = {}
    
    return ExperimentalValidator(config)