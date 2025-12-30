"""
实验数据比较模块

提供将模拟结果与实验数据进行比较的功能，包括:
1. 数据导入和预处理
2. 统计分析和相似度计算
3. 可视化比较结果
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import json

from BrainSimulationSystem.core.eeg import EEGElectrode, EEGFrequencyBand


class ExperimentalDataLoader:
    """实验数据加载器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录，如果为None则使用默认目录
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'experimental')
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 已加载的数据
        self.loaded_data = {}
    
    def load_eeg_data(self, file_name: str) -> Dict[str, Any]:
        """
        加载EEG实验数据
        
        Args:
            file_name: 文件名
            
        Returns:
            加载的数据
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return {}
        
        try:
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(file_name)[1].lower()
            
            if ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif ext == '.npy':
                data = {'data': np.load(file_path)}
            elif ext == '.npz':
                data = dict(np.load(file_path))
            elif ext == '.csv':
                import pandas as pd
                data = {'data': pd.read_csv(file_path)}
            else:
                print(f"不支持的文件格式: {ext}")
                return {}
            
            # 缓存数据
            self.loaded_data[file_name] = data
            
            return data
        
        except Exception as e:
            print(f"加载数据失败: {e}")
            return {}
    
    def load_fmri_data(self, file_name: str) -> Dict[str, Any]:
        """
        加载fMRI实验数据
        
        Args:
            file_name: 文件名
            
        Returns:
            加载的数据
        """
        # 与EEG数据加载类似，但可能需要特殊处理
        return self.load_eeg_data(file_name)
    
    def get_available_datasets(self) -> List[str]:
        """
        获取可用的数据集
        
        Returns:
            数据集文件名列表
        """
        if not os.path.exists(self.data_dir):
            return []
        
        return [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]


class DataComparison:
    """数据比较工具"""
    
    def __init__(self):
        """初始化数据比较工具"""
        pass
    
    def calculate_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        计算两组数据的相关系数
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            
        Returns:
            相关系数
        """
        # 确保数据长度一致
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # 计算Pearson相关系数
        correlation, p_value = stats.pearsonr(data1, data2)
        
        return correlation
    
    def calculate_rmse(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        计算均方根误差
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            
        Returns:
            均方根误差
        """
        # 确保数据长度一致
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # 计算均方根误差
        rmse = np.sqrt(np.mean((data1 - data2) ** 2))
        
        return rmse
    
    def calculate_similarity_metrics(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """
        计算多种相似度指标
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            
        Returns:
            相似度指标字典
        """
        # 确保数据长度一致
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # 计算各种指标
        correlation, _ = stats.pearsonr(data1, data2)
        rmse = np.sqrt(np.mean((data1 - data2) ** 2))
        mae = np.mean(np.abs(data1 - data2))
        r2 = 1 - (np.sum((data1 - data2) ** 2) / np.sum((data1 - np.mean(data1)) ** 2))
        
        return {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_comparison(self, data1: np.ndarray, data2: np.ndarray, 
                       labels: Tuple[str, str] = ('Simulated', 'Experimental'),
                       title: str = "Data Comparison") -> plt.Figure:
        """
        绘制数据比较图
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            labels: 数据标签
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        # 确保数据长度一致
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # 计算相似度指标
        metrics = self.calculate_similarity_metrics(data1, data2)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制时间序列
        x = np.arange(min_len)
        ax1.plot(x, data1, label=labels[0])
        ax1.plot(x, data2, label=labels[1])
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Value")
        ax1.set_title(f"{title} - Time Series")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 添加指标文本
        metrics_text = (
            f"Correlation: {metrics['correlation']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"MAE: {metrics['mae']:.3f}\n"
            f"R²: {metrics['r2']:.3f}"
        )
        ax1.text(0.02, 0.02, metrics_text, transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # 绘制散点图
        ax2.scatter(data1, data2, alpha=0.5)
        ax2.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])
        ax2.set_title(f"{title} - Scatter Plot")
        
        # 添加对角线
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加网格
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        fig.tight_layout()
        
        return fig
    
    def compare_eeg_band_powers(self, simulated_powers: Dict[EEGElectrode, Dict[EEGFrequencyBand, float]],
                               experimental_powers: Dict[EEGElectrode, Dict[EEGFrequencyBand, float]]) -> plt.Figure:
        """
        比较EEG频段功率
        
        Args:
            simulated_powers: 模拟的频段功率
            experimental_powers: 实验的频段功率
            
        Returns:
            matplotlib图形对象
        """
        # 找到共有的电极
        common_electrodes = set(simulated_powers.keys()) & set(experimental_powers.keys())
        
        if not common_electrodes:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No common electrodes found", ha='center', va='center')
            return fig
        
        # 创建图形
        fig, axes = plt.subplots(len(common_electrodes), 1, figsize=(10, 2 * len(common_electrodes)))
        if len(common_electrodes) == 1:
            axes = [axes]
        
        # 频段列表
        bands = list(EEGFrequencyBand)
        band_names = [band.name for band in bands]
        
        # 绘制每个电极的频段功率
        for i, electrode in enumerate(sorted(common_electrodes, key=lambda e: e.name)):
            # 提取数据
            sim_powers = [simulated_powers[electrode].get(band, 0.0) for band in bands]
            exp_powers = [experimental_powers[electrode].get(band, 0.0) for band in bands]
            
            # 设置x位置
            x = np.arange(len(bands))
            width = 0.35
            
            # 绘制条形图
            axes[i].bar(x - width/2, sim_powers, width, label='Simulated')
            axes[i].bar(x + width/2, exp_powers, width, label='Experimental')
            
            # 设置标签
            axes[i].set_ylabel("Power")
            axes[i].set_title(f"Electrode {electrode.name}")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(band_names)
            axes[i].legend()
            
            # 添加网格
            axes[i].grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # 设置总标题
        fig.suptitle("EEG Band Power Comparison")
        
        # 调整布局
        fig.tight_layout()
        
        return fig