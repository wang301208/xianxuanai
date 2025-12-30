"""
EEG伪影模块

实现EEG记录中常见的伪影模拟，如眨眼、肌电、心电等
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode


@dataclass
class EEGArtifact:
    """EEG伪影配置"""
    name: str
    amplitude: float
    frequency: float
    duration: float
    probability: float
    affected_electrodes: List[EEGElectrode]


class ArtifactGenerator:
    """EEG伪影生成器"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        初始化伪影生成器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        
        # 常见EEG伪影
        self.artifacts = [
            EEGArtifact(
                name="眨眼",
                amplitude=100.0,
                frequency=0.2,
                duration=0.4,
                probability=0.05,
                affected_electrodes=[
                    EEGElectrode.FP1, EEGElectrode.FP2, 
                    EEGElectrode.F3, EEGElectrode.F4, EEGElectrode.FZ
                ]
            ),
            EEGArtifact(
                name="咬肌",
                amplitude=50.0,
                frequency=20.0,
                duration=1.0,
                probability=0.02,
                affected_electrodes=[
                    EEGElectrode.T3, EEGElectrode.T4, 
                    EEGElectrode.F7, EEGElectrode.F8
                ]
            ),
            EEGArtifact(
                name="心电",
                amplitude=15.0,
                frequency=1.2,
                duration=0.2,
                probability=0.1,
                affected_electrodes=list(EEGElectrode)  # 影响所有电极
            )
        ]
    
    def add_artifacts(self, eeg_signals: Dict[EEGElectrode, np.ndarray], duration: float) -> Dict[EEGElectrode, np.ndarray]:
        """
        向EEG信号添加伪影
        
        Args:
            eeg_signals: 电极到EEG信号的映射
            duration: 信号持续时间(秒)
            
        Returns:
            添加伪影后的EEG信号
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # 复制输入信号
        result_signals = {electrode: signal.copy() for electrode, signal in eeg_signals.items()}
        
        # 对每种伪影
        for artifact in self.artifacts:
            # 确定伪影发生的时间点
            num_occurrences = int(duration * artifact.probability)
            if num_occurrences > 0:
                occurrence_times = np.random.uniform(0, duration - artifact.duration, num_occurrences)
                
                # 对每个发生时间
                for start_time in occurrence_times:
                    start_idx = int(start_time * self.sampling_rate)
                    end_idx = min(start_idx + int(artifact.duration * self.sampling_rate), num_samples)
                    
                    # 生成伪影波形
                    if artifact.name == "眨眼":
                        # 眨眼伪影: 高斯脉冲
                        t_artifact = np.linspace(-1, 1, end_idx - start_idx)
                        artifact_wave = artifact.amplitude * np.exp(-4 * t_artifact**2)
                    elif artifact.name == "咬肌":
                        # 咬肌伪影: 高频振荡
                        t_artifact = t[start_idx:end_idx]
                        artifact_wave = artifact.amplitude * np.sin(2 * np.pi * artifact.frequency * t_artifact)
                        # 添加随机噪声
                        artifact_wave += np.random.normal(0, artifact.amplitude * 0.2, end_idx - start_idx)
                    elif artifact.name == "心电":
                        # 心电伪影: QRS波形
                        t_artifact = np.linspace(0, 1, end_idx - start_idx)
                        artifact_wave = np.zeros_like(t_artifact)
                        # 创建QRS波形
                        q_idx = int(0.2 * len(t_artifact))
                        r_idx = int(0.3 * len(t_artifact))
                        s_idx = int(0.4 * len(t_artifact))
                        t_idx = int(0.7 * len(t_artifact))
                        
                        artifact_wave[q_idx] = -artifact.amplitude * 0.3
                        artifact_wave[r_idx] = artifact.amplitude
                        artifact_wave[s_idx] = -artifact.amplitude * 0.5
                        artifact_wave[t_idx] = artifact.amplitude * 0.4
                        
                        # 平滑波形
                        artifact_wave = np.convolve(artifact_wave, np.ones(5)/5, mode='same')
                    else:
                        # 默认: 正弦波
                        t_artifact = t[start_idx:end_idx]
                        artifact_wave = artifact.amplitude * np.sin(2 * np.pi * artifact.frequency * t_artifact)
                    
                    # 将伪影添加到受影响的电极
                    for electrode in artifact.affected_electrodes:
                        if electrode in result_signals:
                            result_signals[electrode][start_idx:end_idx] += artifact_wave
        
        return result_signals
    
    def add_custom_artifact(self, artifact: EEGArtifact) -> None:
        """
        添加自定义伪影
        
        Args:
            artifact: 伪影配置
        """
        self.artifacts.append(artifact)
    
    def remove_artifact(self, artifact_name: str) -> bool:
        """
        移除伪影
        
        Args:
            artifact_name: 伪影名称
            
        Returns:
            是否成功移除
        """
        for i, artifact in enumerate(self.artifacts):
            if artifact.name == artifact_name:
                self.artifacts.pop(i)
                return True
        return False
    
    def get_artifacts(self) -> List[EEGArtifact]:
        """
        获取所有伪影
        
        Returns:
            伪影列表
        """
        return self.artifacts.copy()