"""
EEG模拟模块

提供完整的EEG信号模拟、分析和可视化功能
"""

from BrainSimulationSystem.core.eeg.electrode import EEGElectrode, EEGFrequencyBand, ElectrodeManager
from BrainSimulationSystem.core.eeg.signal_generator import SignalGenerator
from BrainSimulationSystem.core.eeg.artifacts import EEGArtifact, ArtifactGenerator
from BrainSimulationSystem.core.eeg.analyzer import EEGAnalyzer
from BrainSimulationSystem.core.eeg.visualizer import EEGVisualizer
from BrainSimulationSystem.core.eeg.simulator import EEGSimulator

__all__ = [
    'EEGElectrode',
    'EEGFrequencyBand',
    'ElectrodeManager',
    'SignalGenerator',
    'EEGArtifact',
    'ArtifactGenerator',
    'EEGAnalyzer',
    'EEGVisualizer',
    'EEGSimulator'
]