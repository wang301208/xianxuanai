# -*- coding: utf-8 -*-
"""
听觉皮层处理系统
Auditory Cortex Processing System

实现层级听觉处理：
1. A1: 频率调谐、时间模式
2. 听觉流分离
3. 语音处理
4. 音乐处理
5. 空间听觉
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

class AuditoryCortex:
    """听觉皮层"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AuditoryCortex")
        
        # 听觉皮层区域
        self.areas = {
            'A1': {  # 初级听觉皮层
                'neurons': {},
                'frequency_columns': {},
                'temporal_patterns': {},
                'binaural_processing': {}
            },
            'belt_areas': {  # 带状区域
                'neurons': {},
                'complex_sounds': {},
                'auditory_objects': {}
            },
            'parabelt_areas': {  # 副带状区域
                'neurons': {},
                'auditory_memory': {},
                'cross_modal_integration': {}
            },
            'superior_temporal_gyrus': {  # 上颞回
                'neurons': {},
                'speech_processing': {},
                'phoneme_detection': {}
            }
        }
        
        # 频率分析
        self.frequency_range = (20, 20000)  # Hz
        self.frequency_channels = 128
        self.frequency_map = np.logspace(np.log10(self.frequency_range[0]), 
                                       np.log10(self.frequency_range[1]), 
                                       self.frequency_channels)
        
        # 时间分析
        self.temporal_window = 50  # ms
        self.temporal_resolution = 1  # ms
        
        self._initialize_auditory_cortex()
    
    def _initialize_auditory_cortex(self):
        """初始化听觉皮层"""
        
        # 初始化A1
        for i in range(800):
            self.areas['A1']['neurons'][i] = {
                'best_frequency': np.random.choice(self.frequency_map),
                'frequency_bandwidth': np.random.uniform(0.1, 2.0),  # 倍频程
                'temporal_modulation_preference': np.random.uniform(1, 100),  # Hz
                'binaural_preference': np.random.choice(['left', 'right', 'binaural']),
                'firing_rate': 0.0,
                'membrane_potential': -70.0,
                'adaptation_state': 1.0
            }
        
        # 初始化其他区域
        for area in ['belt_areas', 'parabelt_areas', 'superior_temporal_gyrus']:
            neuron_count = {'belt_areas': 600, 'parabelt_areas': 400, 'superior_temporal_gyrus': 500}[area]
            for i in range(neuron_count):
                self.areas[area]['neurons'][i] = {
                    'feature_selectivity': np.random.random(20),
                    'temporal_integration_window': np.random.uniform(10, 500),  # ms
                    'firing_rate': 0.0,
                    'membrane_potential': -70.0
                }
    
    def process_auditory_input(self, auditory_input) -> Dict[str, Any]:
        """处理听觉输入"""
        
        # 预处理音频数据
        audio_data = auditory_input.data
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # 转为单声道
        
        # 频谱分析
        frequency_analysis = self._frequency_analysis(audio_data)
        
        # A1处理：频率调谐和时间模式
        a1_response = self._a1_processing(frequency_analysis, audio_data)
        
        # 带状区域处理：复杂声音
        belt_response = self._belt_processing(a1_response, frequency_analysis)
        
        # 副带状区域处理：听觉记忆和跨模态整合
        parabelt_response = self._parabelt_processing(belt_response)
        
        # 上颞回处理：语音处理
        stg_response = self._stg_processing(belt_response, frequency_analysis)
        
        # 听觉流分离
        stream_separation = self._auditory_stream_separation(a1_response, belt_response)
        
        # 空间听觉处理
        spatial_processing = self._spatial_auditory_processing(a1_response)
        
        return {
            'frequency_analysis': frequency_analysis,
            'A1_response': a1_response,
            'belt_response': belt_response,
            'parabelt_response': parabelt_response,
            'STG_response': stg_response,
            'stream_separation': stream_separation,
            'spatial_processing': spatial_processing,
            'speech_features': stg_response.get('speech_features', {'phoneme_clarity': 0.0, 'detected_phoneme': 'unknown'})
        }
    
    def _frequency_analysis(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """频谱分析"""
        
        # 短时傅里叶变换
        window_size = 1024
        hop_size = 512
        
        # 简化的频谱分析
        if len(audio_data) < window_size:
            # 零填充
            padded_audio = np.pad(audio_data, (0, window_size - len(audio_data)), 'constant')
        else:
            padded_audio = audio_data[:window_size]
        
        # FFT
        fft_result = np.fft.fft(padded_audio)
        magnitude_spectrum = np.abs(fft_result[:window_size//2])
        phase_spectrum = np.angle(fft_result[:window_size//2])
        
        # 频率轴
        sample_rate = 44100  # 假设采样率
        frequencies = np.fft.fftfreq(window_size, 1/sample_rate)[:window_size//2]
        
        # Mel滤波器组
        mel_filters = self._create_mel_filterbank(frequencies)
        mel_spectrum = np.dot(mel_filters, magnitude_spectrum)
        
        # 基频检测
        fundamental_frequency = self._detect_fundamental_frequency(magnitude_spectrum, frequencies)
        
        return {
            'magnitude_spectrum': magnitude_spectrum,
            'phase_spectrum': phase_spectrum,
            'frequencies': frequencies,
            'mel_spectrum': mel_spectrum,
            'fundamental_frequency': fundamental_frequency,
            'spectral_centroid': self._compute_spectral_centroid(magnitude_spectrum, frequencies),
            'spectral_rolloff': self._compute_spectral_rolloff(magnitude_spectrum, frequencies),
            'zero_crossing_rate': self._compute_zero_crossing_rate(audio_data)
        }
    
    def _create_mel_filterbank(self, frequencies: np.ndarray) -> np.ndarray:
        """创建Mel滤波器组"""
        
        n_filters = 40
        
        # Mel尺度转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Mel频率点
        mel_min = hz_to_mel(frequencies[0])
        mel_max = hz_to_mel(frequencies[-1])
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)
        
        # 创建滤波器
        filters = np.zeros((n_filters, len(frequencies)))
        
        for i in range(n_filters):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            for j, freq in enumerate(frequencies):
                if left <= freq <= center:
                    filters[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filters[i, j] = (right - freq) / (right - center)
        
        return filters
    
    def _detect_fundamental_frequency(self, magnitude_spectrum: np.ndarray, 
                                    frequencies: np.ndarray) -> float:
        """检测基频"""
        
        # 简化的基频检测
        # 找到最大峰值
        peak_idx = np.argmax(magnitude_spectrum)
        
        if peak_idx < len(frequencies):
            fundamental_freq = frequencies[peak_idx]
        else:
            fundamental_freq = 0.0
        
        return fundamental_freq
    
    def _compute_spectral_centroid(self, magnitude_spectrum: np.ndarray, 
                                 frequencies: np.ndarray) -> float:
        """计算频谱重心"""
        
        if np.sum(magnitude_spectrum) == 0:
            return 0.0
        
        centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
        return centroid
    
    def _compute_spectral_rolloff(self, magnitude_spectrum: np.ndarray, 
                                frequencies: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """计算频谱滚降点"""
        
        total_energy = np.sum(magnitude_spectrum)
        if total_energy == 0:
            return 0.0
        
        cumulative_energy = np.cumsum(magnitude_spectrum)
        rolloff_threshold = rolloff_percent * total_energy
        
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return frequencies[rolloff_idx[0]]
        else:
            return frequencies[-1]
    
    def _compute_zero_crossing_rate(self, audio_data: np.ndarray) -> float:
        """计算过零率"""
        
        if len(audio_data) < 2:
            return 0.0
        
        # 计算符号变化
        sign_changes = np.diff(np.sign(audio_data))
        zero_crossings = np.sum(np.abs(sign_changes)) / 2
        
        # 归一化
        zcr = zero_crossings / len(audio_data)
        return zcr
    
    def _a1_processing(self, frequency_analysis: Dict[str, Any], 
                      audio_data: np.ndarray) -> Dict[str, Any]:
        """A1处理：频率调谐和时间模式"""
        
        magnitude_spectrum = frequency_analysis['magnitude_spectrum']
        frequencies = frequency_analysis['frequencies']
        
        # 更新A1神经元活动
        neuron_responses = {}
        
        for neuron_id, neuron in self.areas['A1']['neurons'].items():
            best_freq = neuron['best_frequency']
            bandwidth = neuron['frequency_bandwidth']
            
            # 找到最接近的频率索引
            freq_idx = np.argmin(np.abs(frequencies - best_freq))
            
            # 计算频率调谐响应
            freq_response = self._compute_frequency_tuning(
                magnitude_spectrum, frequencies, best_freq, bandwidth
            )
            
            # 时间调制响应
            temporal_response = self._compute_temporal_modulation(
                audio_data, neuron['temporal_modulation_preference']
            )
            
            # 综合响应
            total_response = freq_response * temporal_response * neuron['adaptation_state']
            neuron['firing_rate'] = max(0, total_response)
            neuron_responses[neuron_id] = neuron['firing_rate']
            
            # 适应性更新
            if neuron['firing_rate'] > 0.5:
                neuron['adaptation_state'] *= 0.95  # 适应性降低
            else:
                neuron['adaptation_state'] = min(1.0, neuron['adaptation_state'] * 1.01)  # 恢复
        
        # 频率图谱
        frequency_map = self._create_frequency_map(neuron_responses)
        
        # 时间模式检测
        temporal_patterns = self._detect_temporal_patterns(neuron_responses)
        
        return {
            'neuron_responses': neuron_responses,
            'frequency_map': frequency_map,
            'temporal_patterns': temporal_patterns,
            'tonotopic_organization': self._analyze_tonotopic_organization(neuron_responses)
        }
    
    def _compute_frequency_tuning(self, magnitude_spectrum: np.ndarray, 
                                frequencies: np.ndarray, 
                                best_freq: float, 
                                bandwidth: float) -> float:
        """计算频率调谐响应"""
        
        # 高斯调谐曲线
        freq_diff = np.abs(frequencies - best_freq) / best_freq
        tuning_response = np.exp(-(freq_diff**2) / (2 * (bandwidth/2)**2))
        
        # 加权响应
        weighted_response = np.sum(tuning_response * magnitude_spectrum)
        
        # 归一化
        if np.sum(magnitude_spectrum) > 0:
            normalized_response = weighted_response / np.sum(magnitude_spectrum)
        else:
            normalized_response = 0.0
        
        return normalized_response
    
    def _compute_temporal_modulation(self, audio_data: np.ndarray, 
                                   preferred_modulation: float) -> float:
        """计算时间调制响应"""
        
        # 简化的时间调制检测
        if len(audio_data) < 10:
            return 0.0
        
        # 包络提取
        envelope = np.abs(audio_data)
        
        # 包络的频谱分析
        if len(envelope) > 1:
            envelope_fft = np.fft.fft(envelope)
            envelope_magnitude = np.abs(envelope_fft[:len(envelope)//2])
            
            # 调制频率轴
            sample_rate = 44100
            modulation_freqs = np.fft.fftfreq(len(envelope), 1/sample_rate)[:len(envelope)//2]
            
            # 找到最接近偏好调制频率的响应
            if len(modulation_freqs) > 0:
                mod_idx = np.argmin(np.abs(modulation_freqs - preferred_modulation))
                modulation_response = envelope_magnitude[mod_idx] if mod_idx < len(envelope_magnitude) else 0.0
            else:
                modulation_response = 0.0
        else:
            modulation_response = 0.0
        
        return min(1.0, modulation_response / 1000.0)  # 归一化
    
    def _create_frequency_map(self, neuron_responses: Dict[int, float]) -> np.ndarray:
        """创建频率图谱"""
        
        # 创建频率-响应映射
        frequency_map = np.zeros(len(self.frequency_map))
        
        for neuron_id, response in neuron_responses.items():
            if neuron_id in self.areas['A1']['neurons']:
                neuron = self.areas['A1']['neurons'][neuron_id]
                best_freq = neuron['best_frequency']
                
                # 找到对应的频率索引
                freq_idx = np.argmin(np.abs(self.frequency_map - best_freq))
                frequency_map[freq_idx] += response
        
        return frequency_map
    
    def _detect_temporal_patterns(self, neuron_responses: Dict[int, float]) -> Dict[str, Any]:
        """检测时间模式"""
        
        # 简化的时间模式检测
        response_values = list(neuron_responses.values())
        
        if not response_values:
            return {'pattern_strength': 0.0, 'pattern_type': 'none'}
        
        # 计算响应的时间统计
        mean_response = np.mean(response_values)
        std_response = np.std(response_values)
        
        # 模式分类
        if std_response < 0.1:
            pattern_type = 'sustained'
        elif std_response > 0.5:
            pattern_type = 'transient'
        else:
            pattern_type = 'modulated'
        
        return {
            'pattern_strength': std_response,
            'pattern_type': pattern_type,
            'mean_activity': mean_response,
            'temporal_coherence': 1.0 / (1.0 + std_response)
        }
    
    def _analyze_tonotopic_organization(self, neuron_responses: Dict[int, float]) -> Dict[str, Any]:
        """分析音调拓扑组织"""
        
        # 按最佳频率排序神经元
        sorted_neurons = []
        for neuron_id, response in neuron_responses.items():
            if neuron_id in self.areas['A1']['neurons']:
                neuron = self.areas['A1']['neurons'][neuron_id]
                sorted_neurons.append((neuron['best_frequency'], response))
        
        sorted_neurons.sort(key=lambda x: x[0])  # 按频率排序
        
        # 计算拓扑组织指标
        if len(sorted_neurons) > 1:
            frequencies = [x[0] for x in sorted_neurons]
            responses = [x[1] for x in sorted_neurons]
            
            # 频率梯度
            freq_gradient = np.gradient(frequencies)
            
            # 组织性指标
            organization_index = 1.0 / (1.0 + np.std(freq_gradient))
        else:
            organization_index = 0.0
        
        return {
            'organization_index': organization_index,
            'frequency_range': (min(frequencies), max(frequencies)) if sorted_neurons else (0, 0),
            'active_channels': len([r for r in neuron_responses.values() if r > 0.1])
        }
    
    def _belt_processing(self, a1_response: Dict[str, Any], 
                        frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """带状区域处理：复杂声音"""
        
        # 听觉对象形成
        auditory_objects = self._form_auditory_objects(a1_response, frequency_analysis)
        
        # 声音分类
        sound_classification = self._classify_sounds(frequency_analysis)
        
        # 音调处理
        pitch_processing = self._process_pitch(frequency_analysis)
        
        return {
            'auditory_objects': auditory_objects,
            'sound_classification': sound_classification,
            'pitch_processing': pitch_processing,
            'complex_feature_extraction': self._extract_complex_features(a1_response)
        }
    
    def _form_auditory_objects(self, a1_response: Dict[str, Any], 
                             frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """形成听觉对象"""
        
        frequency_map = a1_response['frequency_map']
        temporal_patterns = a1_response['temporal_patterns']
        
        # 简化的听觉对象检测
        # 基于频率连续性和时间一致性
        
        # 找到活跃的频率区域
        active_regions = frequency_map > np.percentile(frequency_map, 75)
        
        # 计算对象数量（连通区域）
        from scipy import ndimage
        labeled_regions, num_objects = ndimage.label(active_regions)
        
        # 对象特征
        objects = {}
        for obj_id in range(1, num_objects + 1):
            obj_mask = labeled_regions == obj_id
            obj_frequencies = self.frequency_map[obj_mask]
            obj_responses = frequency_map[obj_mask]
            
            if len(obj_frequencies) > 0:
                objects[obj_id] = {
                    'frequency_range': (np.min(obj_frequencies), np.max(obj_frequencies)),
                    'mean_frequency': np.mean(obj_frequencies),
                    'strength': np.mean(obj_responses),
                    'bandwidth': np.max(obj_frequencies) - np.min(obj_frequencies),
                    'temporal_coherence': temporal_patterns['temporal_coherence']
                }
        
        return {
            'num_objects': num_objects,
            'objects': objects,
            'object_saliency': self._compute_object_saliency(objects)
        }
    
    def _compute_object_saliency(self, objects: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """计算对象显著性"""
        
        saliency = {}
        
        for obj_id, obj_features in objects.items():
            # 基于强度、带宽和时间一致性的显著性
            strength_saliency = obj_features['strength']
            bandwidth_saliency = min(1.0, obj_features['bandwidth'] / 1000.0)  # 归一化
            temporal_saliency = obj_features['temporal_coherence']
            
            # 综合显著性
            total_saliency = (strength_saliency + bandwidth_saliency + temporal_saliency) / 3
            saliency[obj_id] = total_saliency
        
        return saliency
    
    def _classify_sounds(self, frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """声音分类"""
        
        # 基于频谱特征的简化分类
        spectral_centroid = frequency_analysis['spectral_centroid']
        spectral_rolloff = frequency_analysis['spectral_rolloff']
        zero_crossing_rate = frequency_analysis['zero_crossing_rate']
        fundamental_freq = frequency_analysis['fundamental_frequency']
        
        # 简化的分类规则
        if zero_crossing_rate > 0.1:
            sound_type = 'noise'
            confidence = min(1.0, zero_crossing_rate * 2)
        elif fundamental_freq > 80 and fundamental_freq < 1000:
            sound_type = 'speech'
            confidence = 0.8
        elif fundamental_freq > 200:
            sound_type = 'music'
            confidence = 0.7
        else:
            sound_type = 'environmental'
            confidence = 0.5
        
        return {
            'sound_type': sound_type,
            'confidence': confidence,
            'spectral_features': {
                'centroid': spectral_centroid,
                'rolloff': spectral_rolloff,
                'zcr': zero_crossing_rate,
                'f0': fundamental_freq
            }
        }
    
    def _process_pitch(self, frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """音调处理"""
        
        fundamental_freq = frequency_analysis['fundamental_frequency']
        
        # 音调高度
        if fundamental_freq > 0:
            # 转换为MIDI音符
            midi_note = 69 + 12 * np.log2(fundamental_freq / 440.0)
            
            # 音调类别
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_class = note_names[int(midi_note) % 12]
            octave = int(midi_note) // 12 - 1
        else:
            midi_note = 0
            note_class = 'unknown'
            octave = 0
        
        # 音调稳定性
        pitch_stability = 1.0 / (1.0 + np.abs(fundamental_freq - 440.0) / 440.0) if fundamental_freq > 0 else 0.0
        
        return {
            'fundamental_frequency': fundamental_freq,
            'midi_note': midi_note,
            'note_class': note_class,
            'octave': octave,
            'pitch_stability': pitch_stability,
            'pitch_salience': min(1.0, fundamental_freq / 1000.0) if fundamental_freq > 0 else 0.0
        }
    
    def _extract_complex_features(self, a1_response: Dict[str, Any]) -> Dict[str, Any]:
        """提取复杂特征"""
        
        temporal_patterns = a1_response['temporal_patterns']
        frequency_map = a1_response['frequency_map']
        
        # 谐波结构
        harmonic_strength = self._analyze_harmonic_structure(frequency_map)
        
        # 时间动态
        temporal_dynamics = {
            'onset_strength': temporal_patterns['pattern_strength'],
            'rhythmic_regularity': temporal_patterns['temporal_coherence'],
            'temporal_complexity': 1.0 - temporal_patterns['temporal_coherence']
        }
        
        # 频谱复杂度
        spectral_complexity = np.std(frequency_map) / (np.mean(frequency_map) + 1e-6)
        
        return {
            'harmonic_strength': harmonic_strength,
            'temporal_dynamics': temporal_dynamics,
            'spectral_complexity': spectral_complexity,
            'overall_complexity': (harmonic_strength + spectral_complexity + temporal_dynamics['temporal_complexity']) / 3
        }
    
    def _analyze_harmonic_structure(self, frequency_map: np.ndarray) -> float:
        """分析谐波结构"""
        
        # 简化的谐波分析
        # 寻找峰值
        peaks = []
        for i in range(1, len(frequency_map) - 1):
            if frequency_map[i] > frequency_map[i-1] and frequency_map[i] > frequency_map[i+1]:
                if frequency_map[i] > np.percentile(frequency_map, 75):
                    peaks.append((i, frequency_map[i]))
        
        if len(peaks) < 2:
            return 0.0
        
        # 检查谐波关系
        peak_freqs = [self.frequency_map[p[0]] for p in peaks]
        peak_freqs.sort()
        
        # 计算频率比
        harmonic_score = 0.0
        for i in range(1, len(peak_freqs)):
            ratio = peak_freqs[i] / peak_freqs[0]
            # 检查是否接近整数倍
            nearest_integer = round(ratio)
            if abs(ratio - nearest_integer) < 0.1:
                harmonic_score += 1.0
        
        # 归一化
        harmonic_strength = harmonic_score / max(1, len(peak_freqs) - 1)
        
        return min(1.0, harmonic_strength)
    
    def _parabelt_processing(self, belt_response: Dict[str, Any]) -> Dict[str, Any]:
        """副带状区域处理：听觉记忆和跨模态整合"""
        
        # 听觉工作记忆
        auditory_memory = self._auditory_working_memory(belt_response)
        
        # 听觉注意力
        auditory_attention = self._auditory_attention(belt_response)
        
        return {
            'auditory_memory': auditory_memory,
            'auditory_attention': auditory_attention,
            'cross_modal_readiness': self._assess_cross_modal_readiness(belt_response)
        }
    
    def _auditory_working_memory(self, belt_response: Dict[str, Any]) -> Dict[str, Any]:
        """听觉工作记忆"""
        
        auditory_objects = belt_response['auditory_objects']
        
        # 简化的工作记忆模型
        memory_capacity = 4  # 典型的工作记忆容量
        
        # 选择最显著的对象进入工作记忆
        if 'object_saliency' in auditory_objects:
            sorted_objects = sorted(auditory_objects['object_saliency'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            memory_objects = sorted_objects[:memory_capacity]
            memory_load = len(memory_objects) / memory_capacity
        else:
            memory_objects = []
            memory_load = 0.0
        
        return {
            'memory_objects': memory_objects,
            'memory_load': memory_load,
            'memory_capacity': memory_capacity,
            'maintenance_strength': 1.0 - memory_load * 0.2  # 负载越高维持越困难
        }
    
    def _auditory_attention(self, belt_response: Dict[str, Any]) -> Dict[str, Any]:
        """听觉注意力"""
        
        auditory_objects = belt_response['auditory_objects']
        
        # 基于显著性的注意力分配
        if 'object_saliency' in auditory_objects and auditory_objects['object_saliency']:
            max_saliency_obj = max(auditory_objects['object_saliency'].items(), key=lambda x: x[1])
            attention_focus = max_saliency_obj[0]
            attention_strength = max_saliency_obj[1]
        else:
            attention_focus = None
            attention_strength = 0.0
        
        return {
            'attention_focus': attention_focus,
            'attention_strength': attention_strength,
            'attention_selectivity': attention_strength * 0.8,
            'distractor_suppression': attention_strength * 0.6
        }
    
    def _assess_cross_modal_readiness(self, belt_response: Dict[str, Any]) -> Dict[str, float]:
        """评估跨模态整合准备度"""
        
        sound_classification = belt_response['sound_classification']
        
        # 不同声音类型的跨模态整合倾向
        cross_modal_weights = {
            'speech': 0.9,  # 语音高度依赖视觉
            'music': 0.3,   # 音乐较少依赖其他模态
            'environmental': 0.7,  # 环境声音常与视觉结合
            'noise': 0.1    # 噪声很少跨模态整合
        }
        
        sound_type = sound_classification['sound_type']
        confidence = sound_classification['confidence']
        
        readiness = cross_modal_weights.get(sound_type, 0.5) * confidence
        
        return {
            'visual_integration_readiness': readiness,
            'tactile_integration_readiness': readiness * 0.6,
            'overall_readiness': readiness
        }
    
    def _stg_processing(self, belt_response: Dict[str, Any], 
                       frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """上颞回处理：语音处理"""
        
        sound_classification = belt_response['sound_classification']
        
        # 语音特征提取
        if sound_classification['sound_type'] == 'speech':
            speech_features = self._extract_speech_features(frequency_analysis)
            phoneme_detection = self._detect_phonemes(speech_features)
        else:
            speech_features = {'formants': [], 'voicing': 0.0}
            phoneme_detection = {'detected_phoneme': 'unknown', 'confidence': 0.0}
        
        return {
            'speech_features': speech_features,
            'phoneme_detection': phoneme_detection,
            'speech_intelligibility': self._assess_speech_intelligibility(speech_features)
        }
    
    def _extract_speech_features(self, frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取语音特征"""
        
        magnitude_spectrum = frequency_analysis['magnitude_spectrum']
        frequencies = frequency_analysis['frequencies']
        fundamental_freq = frequency_analysis['fundamental_frequency']
        
        # 共振峰检测（简化）
        formants = self._detect_formants(magnitude_spectrum, frequencies)
        
        # 浊音检测
        voicing = 1.0 if fundamental_freq > 80 and fundamental_freq < 500 else 0.0
        
        # 语音清晰度
        phoneme_clarity = self._compute_phoneme_clarity(magnitude_spectrum, formants)
        
        return {
            'formants': formants,
            'voicing': voicing,
            'phoneme_clarity': phoneme_clarity,
            'fundamental_frequency': fundamental_freq
        }
    
    def _detect_formants(self, magnitude_spectrum: np.ndarray, 
                        frequencies: np.ndarray) -> List[float]:
        """检测共振峰"""
        
        # 简化的共振峰检测
        # 寻找频谱峰值
        peaks = []
        
        # 只在语音频率范围内寻找 (300-3000 Hz)
        speech_range = (frequencies >= 300) & (frequencies <= 3000)
        speech_spectrum = magnitude_spectrum[speech_range]
        speech_freqs = frequencies[speech_range]
        
        if len(speech_spectrum) == 0:
            return []
        
        # 寻找局部最大值
        for i in range(1, len(speech_spectrum) - 1):
            if (speech_spectrum[i] > speech_spectrum[i-1] and 
                speech_spectrum[i] > speech_spectrum[i+1] and
                speech_spectrum[i] > np.percentile(speech_spectrum, 70)):
                peaks.append(speech_freqs[i])
        
        # 返回前3个最强的峰值作为共振峰
        if peaks:
            peak_strengths = []
            for peak_freq in peaks:
                peak_idx = np.argmin(np.abs(speech_freqs - peak_freq))
                peak_strengths.append(speech_spectrum[peak_idx])
            
            # 按强度排序
            sorted_peaks = [x for _, x in sorted(zip(peak_strengths, peaks), reverse=True)]
            return sorted_peaks[:3]
        else:
            return []
    
    def _compute_phoneme_clarity(self, magnitude_spectrum: np.ndarray, 
                               formants: List[float]) -> float:
        """计算音素清晰度"""
        
        if not formants:
            return 0.0
        
        # 基于共振峰强度和分离度的清晰度
        formant_strength = len(formants) / 3.0  # 归一化到0-1
        
        # 共振峰分离度
        if len(formants) > 1:
            formant_separation = np.mean(np.diff(sorted(formants)))
            separation_score = min(1.0, formant_separation / 500.0)  # 归一化
        else:
            separation_score = 0.0
        
        clarity = (formant_strength + separation_score) / 2
        return clarity
    
    def _detect_phonemes(self, speech_features: Dict[str, Any]) -> Dict[str, Any]:
        """检测音素"""
        
        formants = speech_features['formants']
        voicing = speech_features['voicing']
        
        # 简化的音素分类
        if not formants:
            return {'detected_phoneme': 'unknown', 'confidence': 0.0}
        
        # 基于前两个共振峰的简单分类
        if len(formants) >= 2:
            f1, f2 = formants[0], formants[1]
            
            # 简化的元音分类规则
            if f1 < 400 and f2 > 2000:
                phoneme = 'i'
            elif f1 < 400 and f2 < 1000:
                phoneme = 'u'
            elif f1 > 600 and f2 > 1500:
                phoneme = 'a'
            elif f1 > 400 and f1 < 600:
                phoneme = 'e'
            else:
                phoneme = 'o'
            
            confidence = speech_features['phoneme_clarity']
        else:
            # 辅音或未知
            if voicing < 0.5:
                phoneme = 'consonant_unvoiced'
            else:
                phoneme = 'consonant_voiced'
            confidence = 0.3
        
        return {
            'detected_phoneme': phoneme,
            'confidence': confidence,
            'voicing': voicing
        }
    
    def _assess_speech_intelligibility(self, speech_features: Dict[str, Any]) -> Dict[str, float]:
        """评估语音可懂度"""
        
        formants = speech_features['formants']
        voicing = speech_features['voicing']
        phoneme_clarity = speech_features['phoneme_clarity']
        
        # 可懂度因子
        formant_factor = len(formants) / 3.0  # 共振峰完整性
        clarity_factor = phoneme_clarity  # 清晰度
        voicing_factor = abs(voicing - 0.5) * 2  # 浊音清晰度
        
        # 综合可懂度
        overall_intelligibility = (formant_factor + clarity_factor + voicing_factor) / 3
        
        return {
            'overall_intelligibility': overall_intelligibility,
            'formant_clarity': formant_factor,
            'phoneme_clarity': clarity_factor,
            'voicing_clarity': voicing_factor
        }
    
    def _auditory_stream_separation(self, a1_response: Dict[str, Any], 
                                  belt_response: Dict[str, Any]) -> Dict[str, Any]:
        """听觉流分离"""
        
        auditory_objects = belt_response['auditory_objects']
        
        # 基于频率和时间连续性的流分离
        streams = self._separate_auditory_streams(auditory_objects)
        
        # 流跟踪
        stream_tracking = self._track_auditory_streams(streams)
        
        return {
            'streams': streams,
            'stream_tracking': stream_tracking,
            'stream_segregation_strength': self._compute_segregation_strength(streams)
        }
    
    def _separate_auditory_streams(self, auditory_objects: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """分离听觉流"""
        
        objects = auditory_objects.get('objects', {})
        
        # 简化的流分离：基于频率相似性
        streams = {}
        stream_id = 0
        
        for obj_id, obj_features in objects.items():
            # 寻找相似的现有流
            assigned = False
            
            for existing_stream_id, stream_info in streams.items():
                # 检查频率相似性
                freq_diff = abs(obj_features['mean_frequency'] - stream_info['mean_frequency'])
                
                if freq_diff < 200:  # 200 Hz阈值
                    # 添加到现有流
                    stream_info['objects'].append(obj_id)
                    stream_info['mean_frequency'] = np.mean([
                        stream_info['mean_frequency'], obj_features['mean_frequency']
                    ])
                    assigned = True
                    break
            
            if not assigned:
                # 创建新流
                streams[stream_id] = {
                    'objects': [obj_id],
                    'mean_frequency': obj_features['mean_frequency'],
                    'strength': obj_features['strength'],
                    'coherence': obj_features['temporal_coherence']
                }
                stream_id += 1
        
        return streams
    
    def _track_auditory_streams(self, streams: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """跟踪听觉流"""
        
        # 流的稳定性和连续性
        stream_stability = {}
        
        for stream_id, stream_info in streams.items():
            # 基于对象数量和一致性的稳定性
            num_objects = len(stream_info['objects'])
            coherence = stream_info['coherence']
            
            stability = min(1.0, num_objects / 3.0) * coherence
            stream_stability[stream_id] = stability
        
        return {
            'stream_stability': stream_stability,
            'num_active_streams': len(streams),
            'dominant_stream': max(stream_stability.keys(), key=lambda x: stream_stability[x]) if stream_stability else None
        }
    
    def _compute_segregation_strength(self, streams: Dict[int, Dict[str, Any]]) -> float:
        """计算分离强度"""
        
        if len(streams) < 2:
            return 0.0
        
        # 基于流间频率差异的分离强度
        frequencies = [stream_info['mean_frequency'] for stream_info in streams.values()]
        
        # 计算频率差异
        freq_diffs = []
        for i in range(len(frequencies)):
            for j in range(i+1, len(frequencies)):
                freq_diffs.append(abs(frequencies[i] - frequencies[j]))
        
        if freq_diffs:
            mean_freq_diff = np.mean(freq_diffs)
            segregation_strength = min(1.0, mean_freq_diff / 1000.0)  # 归一化
        else:
            segregation_strength = 0.0
        
        return segregation_strength
    
    def _spatial_auditory_processing(self, a1_response: Dict[str, Any]) -> Dict[str, Any]:
        """空间听觉处理"""
        
        # 简化的空间听觉（需要双耳输入）
        # 这里提供单耳的占位实现
        
        return {
            'azimuth_estimate': 0.0,  # 方位角估计
            'elevation_estimate': 0.0,  # 仰角估计
            'distance_estimate': 1.0,  # 距离估计
            'spatial_confidence': 0.5,  # 空间定位置信度
            'binaural_processing': {
                'itd': 0.0,  # 双耳时间差
                'ild': 0.0,  # 双耳强度差
                'spectral_cues': {}  # 频谱线索
            }
        }