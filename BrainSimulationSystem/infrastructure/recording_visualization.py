"""
可扩展记录与可视化系统

提供：
- 多模态数据记录（尖峰、电压、连接权重、神经调质等）
- 实时与离线可视化
- 交互式分析工具
- 大规模数据存储与检索
- 3D脑区可视化
- 网络拓扑分析
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
import time
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import pickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import networkx as nx
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
import io
import base64

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover
    sns = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.express as px  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    import plotly.offline as pyo  # type: ignore
except ImportError:  # pragma: no cover
    go = None  # type: ignore
    px = None  # type: ignore
    make_subplots = None  # type: ignore
    pyo = None  # type: ignore

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
except ImportError:  # pragma: no cover
    PCA = None  # type: ignore
    TSNE = None  # type: ignore
    KMeans = None  # type: ignore

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

class RecordingType(Enum):
    """记录类型"""
    SPIKE_TIMES = "spike_times"
    MEMBRANE_VOLTAGE = "membrane_voltage"
    SYNAPTIC_CURRENT = "synaptic_current"
    SYNAPTIC_WEIGHTS = "synaptic_weights"
    NEUROMODULATOR = "neuromodulator"
    CALCIUM_CONCENTRATION = "calcium_concentration"
    NETWORK_ACTIVITY = "network_activity"
    BEHAVIORAL_STATE = "behavioral_state"
    EXTERNAL_STIMULUS = "external_stimulus"

class VisualizationType(Enum):
    """可视化类型"""
    RASTER_PLOT = "raster_plot"
    VOLTAGE_TRACE = "voltage_trace"
    FIRING_RATE = "firing_rate"
    CONNECTIVITY_MATRIX = "connectivity_matrix"
    NETWORK_GRAPH = "network_graph"
    BRAIN_REGION_3D = "brain_region_3d"
    PHASE_SPACE = "phase_space"
    SPECTROGRAM = "spectrogram"
    CORRELATION_MATRIX = "correlation_matrix"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class StorageFormat(Enum):
    """存储格式"""
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    PARQUET = "parquet"
    NPZ = "npz"
    JSON = "json"
    CSV = "csv"

@dataclass
class RecordingConfig:
    """记录配置"""
    recording_id: str
    recording_type: RecordingType
    target_populations: List[str]
    sampling_rate: float  # Hz
    output_directory: str = "."
    buffer_size: int = 10000
    storage_format: StorageFormat = StorageFormat.HDF5
    compression: bool = True
    real_time: bool = False
    
@dataclass
class DataPoint:
    """数据点"""
    timestamp: float
    neuron_id: int
    value: Union[float, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecordingSession:
    """记录会话"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    recordings: Dict[str, 'DataRecorder'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class DataRecorder(ABC):
    """数据记录器基类"""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        self.is_recording = False
        self.buffer = []
        self.logger = logging.getLogger(f"Recorder.{config.recording_type.value}")
        
        # 存储后端
        self.storage_backend = self._create_storage_backend()
        
    @abstractmethod
    def _create_storage_backend(self):
        """创建存储后端"""
        pass
    
    @abstractmethod
    def record_data_point(self, data_point: DataPoint):
        """记录数据点"""
        self.logger.info("记录数据点")
    
    def start_recording(self):
        """开始记录"""
        self.is_recording = True
        self.logger.info(f"开始记录: {self.config.recording_id}")
    
    def stop_recording(self):
        """停止记录"""
        self.is_recording = False
        self._flush_buffer()
        self.logger.info(f"停止记录: {self.config.recording_id}")
    
    def _flush_buffer(self):
        """刷新缓冲区"""
        if self.buffer:
            self.storage_backend.write_batch(self.buffer)
            self.buffer.clear()

class SpikeRecorder(DataRecorder):
    """尖峰记录器"""
    
    def __init__(self, config: RecordingConfig):
        super().__init__(config)
        self.spike_times = {}  # neuron_id -> [spike_times]
    
    def _create_storage_backend(self):
        if self.config.storage_format == StorageFormat.HDF5:
            return HDF5SpikeStorage(self.config)
        elif self.config.storage_format == StorageFormat.SQLITE:
            return SQLiteSpikeStorage(self.config)
        else:
            raise ValueError(f"不支持的存储格式: {self.config.storage_format}")
    
    def record_data_point(self, data_point: DataPoint):
        """记录尖峰数据"""
        if not self.is_recording:
            return
        
        neuron_id = data_point.neuron_id
        spike_time = data_point.timestamp
        
        # 添加到内存缓冲
        if neuron_id not in self.spike_times:
            self.spike_times[neuron_id] = []
        self.spike_times[neuron_id].append(spike_time)
        
        # 添加到批处理缓冲
        self.buffer.append(data_point)
        
        # 检查是否需要刷新缓冲区
        if len(self.buffer) >= self.config.buffer_size:
            self._flush_buffer()
    
    def get_spike_trains(self, start_time: float = 0.0, 
                        end_time: Optional[float] = None) -> Dict[int, np.ndarray]:
        """获取尖峰序列"""
        spike_trains = {}
        
        for neuron_id, times in self.spike_times.items():
            times_array = np.array(times)
            
            # 时间窗口过滤
            if end_time is not None:
                mask = (times_array >= start_time) & (times_array <= end_time)
            else:
                mask = times_array >= start_time
            
            spike_trains[neuron_id] = times_array[mask]
        
        return spike_trains

class VoltageRecorder(DataRecorder):
    """电压记录器"""
    
    def __init__(self, config: RecordingConfig):
        super().__init__(config)
        self.voltage_traces = {}  # neuron_id -> [(time, voltage)]
    
    def _create_storage_backend(self):
        if self.config.storage_format == StorageFormat.HDF5:
            return HDF5VoltageStorage(self.config)
        else:
            raise ValueError(f"不支持的存储格式: {self.config.storage_format}")
    
    def record_data_point(self, data_point: DataPoint):
        """记录电压数据"""
        if not self.is_recording:
            return
        
        neuron_id = data_point.neuron_id
        timestamp = data_point.timestamp
        voltage = data_point.value
        
        # 添加到内存缓冲
        if neuron_id not in self.voltage_traces:
            self.voltage_traces[neuron_id] = []
        self.voltage_traces[neuron_id].append((timestamp, voltage))
        
        # 添加到批处理缓冲
        self.buffer.append(data_point)
        
        if len(self.buffer) >= self.config.buffer_size:
            self._flush_buffer()

class WeightRecorder(DataRecorder):
    """权重记录器"""
    
    def __init__(self, config: RecordingConfig):
        super().__init__(config)
        self.weight_history = {}  # (pre_id, post_id) -> [(time, weight)]
    
    def _create_storage_backend(self):
        if self.config.storage_format == StorageFormat.HDF5:
            return HDF5WeightStorage(self.config)
        else:
            raise ValueError(f"不支持的存储格式: {self.config.storage_format}")
    
    def record_data_point(self, data_point: DataPoint):
        """记录权重数据"""
        if not self.is_recording:
            return
        
        # 假设 data_point.metadata 包含连接信息
        pre_id = data_point.metadata.get('pre_neuron_id')
        post_id = data_point.metadata.get('post_neuron_id')
        
        if pre_id is not None and post_id is not None:
            connection_key = (pre_id, post_id)
            timestamp = data_point.timestamp
            weight = data_point.value
            
            if connection_key not in self.weight_history:
                self.weight_history[connection_key] = []
            self.weight_history[connection_key].append((timestamp, weight))
        
        self.buffer.append(data_point)
        
        if len(self.buffer) >= self.config.buffer_size:
            self._flush_buffer()

# 存储后端实现
class HDF5SpikeStorage:
    """HDF5尖峰存储"""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = output_dir / f"{config.recording_id}_spikes.h5"
        self.h5_file = None
        self._initialize_file()
    
    def _initialize_file(self):
        """初始化HDF5文件"""
        if h5py is None:
            raise ImportError("需要安装 h5py 库来支持 HDF5 存储")
        self.h5_file = h5py.File(self.file_path, 'w')
        
        # 创建数据集
        self.spike_times_dataset = self.h5_file.create_dataset(
            'spike_times', (0,), maxshape=(None,), dtype='f8'
        )
        self.neuron_ids_dataset = self.h5_file.create_dataset(
            'neuron_ids', (0,), maxshape=(None,), dtype='i4'
        )
        
        # 存储元数据
        self.h5_file.attrs['recording_type'] = self.config.recording_type.value
        self.h5_file.attrs['sampling_rate'] = self.config.sampling_rate
    
    def write_batch(self, data_points: List[DataPoint]):
        """批量写入数据"""
        if not data_points:
            return
        
        spike_times = [dp.timestamp for dp in data_points]
        neuron_ids = [dp.neuron_id for dp in data_points]
        
        # 扩展数据集
        current_size = self.spike_times_dataset.shape[0]
        new_size = current_size + len(spike_times)
        
        self.spike_times_dataset.resize((new_size,))
        self.neuron_ids_dataset.resize((new_size,))
        
        # 写入数据
        self.spike_times_dataset[current_size:new_size] = spike_times
        self.neuron_ids_dataset[current_size:new_size] = neuron_ids
        
        # 刷新到磁盘
        self.h5_file.flush()
    
    def close(self):
        """关闭文件"""
        if self.h5_file:
            self.h5_file.close()

class HDF5VoltageStorage:
    """HDF5电压存储"""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = output_dir / f"{config.recording_id}_voltages.h5"
        self.h5_file = None
        self._initialize_file()
    
    def _initialize_file(self):
        """初始化HDF5文件"""
        if h5py is None:
            raise ImportError("需要安装 h5py 库来支持 HDF5 存储")
        self.h5_file = h5py.File(self.file_path, 'w')
        
        # 为每个神经元创建数据集组
        self.neuron_groups = {}
    
    def write_batch(self, data_points: List[DataPoint]):
        """批量写入数据"""
        # 按神经元ID分组
        neuron_data = {}
        for dp in data_points:
            neuron_id = dp.neuron_id
            if neuron_id not in neuron_data:
                neuron_data[neuron_id] = {'times': [], 'voltages': []}
            neuron_data[neuron_id]['times'].append(dp.timestamp)
            neuron_data[neuron_id]['voltages'].append(dp.value)
        
        # 写入每个神经元的数据
        for neuron_id, data in neuron_data.items():
            if neuron_id not in self.neuron_groups:
                group = self.h5_file.create_group(f'neuron_{neuron_id}')
                group.create_dataset('times', (0,), maxshape=(None,), dtype='f8')
                group.create_dataset('voltages', (0,), maxshape=(None,), dtype='f4')
                self.neuron_groups[neuron_id] = group
            
            group = self.neuron_groups[neuron_id]
            
            # 扩展数据集
            current_size = group['times'].shape[0]
            new_size = current_size + len(data['times'])
            
            group['times'].resize((new_size,))
            group['voltages'].resize((new_size,))
            
            # 写入数据
            group['times'][current_size:new_size] = data['times']
            group['voltages'][current_size:new_size] = data['voltages']
        
        self.h5_file.flush()
    
    def close(self):
        """关闭文件"""
        if self.h5_file:
            self.h5_file.close()

class HDF5WeightStorage:
    """HDF5权重存储"""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = output_dir / f"{config.recording_id}_weights.h5"
        self.h5_file = None
        self._initialize_file()
    
    def _initialize_file(self):
        """初始化HDF5文件"""
        if h5py is None:
            raise ImportError("需要安装 h5py 库来支持 HDF5 存储")
        self.h5_file = h5py.File(self.file_path, 'w')
        self.connection_groups = {}
    
    def write_batch(self, data_points: List[DataPoint]):
        """批量写入数据"""
        # 按连接分组
        connection_data = {}
        for dp in data_points:
            pre_id = dp.metadata.get('pre_neuron_id')
            post_id = dp.metadata.get('post_neuron_id')
            
            if pre_id is not None and post_id is not None:
                conn_key = f"{pre_id}_{post_id}"
                if conn_key not in connection_data:
                    connection_data[conn_key] = {'times': [], 'weights': []}
                connection_data[conn_key]['times'].append(dp.timestamp)
                connection_data[conn_key]['weights'].append(dp.value)
        
        # 写入每个连接的数据
        for conn_key, data in connection_data.items():
            if conn_key not in self.connection_groups:
                group = self.h5_file.create_group(f'connection_{conn_key}')
                group.create_dataset('times', (0,), maxshape=(None,), dtype='f8')
                group.create_dataset('weights', (0,), maxshape=(None,), dtype='f4')
                self.connection_groups[conn_key] = group
            
            group = self.connection_groups[conn_key]
            
            # 扩展数据集
            current_size = group['times'].shape[0]
            new_size = current_size + len(data['times'])
            
            group['times'].resize((new_size,))
            group['weights'].resize((new_size,))
            
            # 写入数据
            group['times'][current_size:new_size] = data['times']
            group['weights'][current_size:new_size] = data['weights']
        
        self.h5_file.flush()
    
    def close(self):
        """关闭文件"""
        if self.h5_file:
            self.h5_file.close()

class SQLiteSpikeStorage:
    """SQLite尖峰存储"""
    
    def __init__(self, config: RecordingConfig):
        self.config = config
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_dir / f"{config.recording_id}_spikes.db"
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化数据库"""
        self.connection = sqlite3.connect(str(self.db_path))
        cursor = self.connection.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                neuron_id INTEGER NOT NULL,
                metadata TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON spikes(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_neuron_id ON spikes(neuron_id)')
        
        self.connection.commit()
    
    def write_batch(self, data_points: List[DataPoint]):
        """批量写入数据"""
        cursor = self.connection.cursor()
        
        data_to_insert = [
            (dp.timestamp, dp.neuron_id, json.dumps(dp.metadata))
            for dp in data_points
        ]
        
        cursor.executemany(
            'INSERT INTO spikes (timestamp, neuron_id, metadata) VALUES (?, ?, ?)',
            data_to_insert
        )
        
        self.connection.commit()
    
    def close(self):
        """关闭数据库"""
        if self.connection:
            self.connection.close()

class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, output_directory: str = "./visualizations"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DataVisualizer")
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_raster_plot(self, spike_trains: Dict[int, np.ndarray], 
                          title: str = "尖峰栅格图", 
                          time_window: Optional[Tuple[float, float]] = None,
                          save_path: Optional[str] = None) -> go.Figure:
        """创建尖峰栅格图"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (neuron_id, spike_times) in enumerate(spike_trains.items()):
            if time_window:
                mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
                spike_times = spike_times[mask]
            
            if len(spike_times) > 0:
                fig.add_trace(go.Scatter(
                    x=spike_times,
                    y=[neuron_id] * len(spike_times),
                    mode='markers',
                    marker=dict(
                        symbol='line-ns',
                        size=8,
                        color=colors[i % len(colors)]
                    ),
                    name=f'神经元 {neuron_id}',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="时间 (ms)",
            yaxis_title="神经元ID",
            height=600,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_firing_rate_plot(self, spike_trains: Dict[int, np.ndarray],
                               bin_size: float = 10.0,
                               title: str = "发放频率",
                               save_path: Optional[str] = None) -> go.Figure:
        """创建发放频率图"""
        # 计算总体发放频率
        all_spikes = np.concatenate(list(spike_trains.values()))
        
        if len(all_spikes) == 0:
            return go.Figure()
        
        # 创建时间窗口
        min_time = np.min(all_spikes)
        max_time = np.max(all_spikes)
        bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        # 计算每个时间窗口的发放频率
        firing_rates = []
        bin_centers = []
        
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            
            # 计算该时间窗口内的尖峰数
            spike_count = np.sum((all_spikes >= bin_start) & (all_spikes < bin_end))
            firing_rate = spike_count / (bin_size / 1000.0) / len(spike_trains)  # Hz
            
            firing_rates.append(firing_rate)
            bin_centers.append((bin_start + bin_end) / 2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=firing_rates,
            mode='lines+markers',
            name='发放频率',
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="时间 (ms)",
            yaxis_title="发放频率 (Hz)",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_voltage_trace_plot(self, voltage_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                 title: str = "膜电位轨迹",
                                 save_path: Optional[str] = None) -> go.Figure:
        """创建膜电位轨迹图"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (neuron_id, (times, voltages)) in enumerate(voltage_data.items()):
            fig.add_trace(go.Scatter(
                x=times,
                y=voltages,
                mode='lines',
                name=f'神经元 {neuron_id}',
                line=dict(
                    width=1.5,
                    color=colors[i % len(colors)]
                )
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="时间 (ms)",
            yaxis_title="膜电位 (mV)",
            height=500,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_connectivity_matrix(self, connectivity_matrix: np.ndarray,
                                  neuron_labels: Optional[List[str]] = None,
                                  title: str = "连接矩阵",
                                  save_path: Optional[str] = None) -> go.Figure:
        """创建连接矩阵热图"""
        fig = go.Figure(data=go.Heatmap(
            z=connectivity_matrix,
            x=neuron_labels,
            y=neuron_labels,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="目标神经元",
            yaxis_title="源神经元",
            height=600,
            width=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_network_graph(self, adjacency_matrix: np.ndarray,
                           positions: Optional[Dict[int, Tuple[float, float]]] = None,
                           title: str = "网络图",
                           save_path: Optional[str] = None) -> go.Figure:
        """创建网络图"""
        # 创建NetworkX图
        G = nx.from_numpy_array(adjacency_matrix)
        
        # 计算布局
        if positions is None:
            positions = nx.spring_layout(G, k=1, iterations=50)
        
        # 提取边
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 提取节点
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'神经元 {node}')
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # 添加节点
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="节点度数"
                ),
                line=dict(width=2)
            ),
            showlegend=False
        ))
        
        # 计算节点度数用于着色
        node_degrees = [G.degree(node) for node in G.nodes()]
        fig.data[1].marker.color = node_degrees
        
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="网络连接可视化",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#888', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_3d_brain_visualization(self, brain_regions: Dict[str, Dict[str, Any]],
                                     activity_data: Optional[Dict[str, float]] = None,
                                     title: str = "3D脑区可视化",
                                     save_path: Optional[str] = None) -> go.Figure:
        """创建3D脑区可视化"""
        fig = go.Figure()
        
        for region_name, region_data in brain_regions.items():
            # 假设每个脑区有位置和大小信息
            position = region_data.get('position', [0, 0, 0])
            size = region_data.get('size', 1.0)
            
            # 获取活动强度
            activity = activity_data.get(region_name, 0.0) if activity_data else 0.5
            
            # 创建球体表示脑区
            fig.add_trace(go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode='markers',
                marker=dict(
                    size=size * 20,
                    color=activity,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="活动强度")
                ),
                text=region_name,
                name=region_name,
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_correlation_matrix(self, spike_trains: Dict[int, np.ndarray],
                                 bin_size: float = 10.0,
                                 title: str = "神经元相关性矩阵",
                                 save_path: Optional[str] = None) -> go.Figure:
        """创建神经元相关性矩阵"""
        # 将尖峰序列转换为二进制时间序列
        if not spike_trains:
            return go.Figure()
        
        all_spikes = np.concatenate(list(spike_trains.values()))
        if len(all_spikes) == 0:
            return go.Figure()
        
        min_time = np.min(all_spikes)
        max_time = np.max(all_spikes)
        time_bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        # 为每个神经元创建二进制时间序列
        neuron_ids = list(spike_trains.keys())
        binary_series = []
        
        for neuron_id in neuron_ids:
            spikes = spike_trains[neuron_id]
            binary_trace = np.histogram(spikes, bins=time_bins)[0]
            binary_series.append(binary_trace)
        
        # 计算相关性矩阵
        binary_matrix = np.array(binary_series)
        correlation_matrix = np.corrcoef(binary_matrix)
        
        # 处理NaN值
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=[f'N{nid}' for nid in neuron_ids],
            y=[f'N{nid}' for nid in neuron_ids],
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title="相关系数")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="神经元",
            yaxis_title="神经元",
            height=600,
            width=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dimensionality_reduction_plot(self, spike_trains: Dict[int, np.ndarray],
                                           method: str = "pca",
                                           bin_size: float = 10.0,
                                           title: str = "降维可视化",
                                           save_path: Optional[str] = None) -> go.Figure:
        """创建降维可视化"""
        # 准备数据
        if not spike_trains:
            return go.Figure()
        
        all_spikes = np.concatenate(list(spike_trains.values()))
        if len(all_spikes) == 0:
            return go.Figure()
        
        min_time = np.min(all_spikes)
        max_time = np.max(all_spikes)
        time_bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        # 创建特征矩阵（时间窗口 x 神经元）
        neuron_ids = list(spike_trains.keys())
        feature_matrix = []
        
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            
            bin_features = []
            for neuron_id in neuron_ids:
                spikes = spike_trains[neuron_id]
                spike_count = np.sum((spikes >= bin_start) & (spikes < bin_end))
                bin_features.append(spike_count)
            
            feature_matrix.append(bin_features)
        
        feature_matrix = np.array(feature_matrix)
        
        if feature_matrix.shape[0] < 2:
            return go.Figure()
        
        # 应用降维方法
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(feature_matrix)
            explained_variance = reducer.explained_variance_ratio_
            subtitle = f"PCA (解释方差: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})"
        elif method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(feature_matrix)
            subtitle = "t-SNE"
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        # 创建可视化
        fig = go.Figure()
        
        # 使用时间作为颜色编码
        time_points = [(time_bins[i] + time_bins[i+1]) / 2 for i in range(len(time_bins)-1)]
        
        fig.add_trace(go.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='markers+lines',
            marker=dict(
                size=6,
                color=time_points,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="时间 (ms)")
            ),
            line=dict(width=1, color='rgba(0,0,0,0.3)'),
            text=[f'时间: {t:.1f} ms' for t in time_points],
            hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{title}<br><sub>{subtitle}</sub>",
            xaxis_title="第一主成分" if method.lower() == "pca" else "维度1",
            yaxis_title="第二主成分" if method.lower() == "pca" else "维度2",
            height=600,
            width=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, update_interval: float = 100.0):  # ms
        self.update_interval = update_interval
        self.is_running = False
        self.data_buffer = {}
        self.figures = {}
        self.logger = logging.getLogger("RealTimeVisualizer")
    
    def start_real_time_visualization(self, visualization_types: List[VisualizationType]):
        """开始实时可视化"""
        self.is_running = True
        
        for viz_type in visualization_types:
            if viz_type == VisualizationType.RASTER_PLOT:
                self._setup_real_time_raster()
            elif viz_type == VisualizationType.FIRING_RATE:
                self._setup_real_time_firing_rate()
    
    def _setup_real_time_raster(self):
        """设置实时栅格图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        self.figures['raster'] = (fig, ax)
        
        def update_raster(frame):
            if not self.is_running:
                return
            
            ax.clear()
            
            # 从缓冲区获取数据
            spike_data = self.data_buffer.get('spikes', {})
            
            for neuron_id, spike_times in spike_data.items():
                if len(spike_times) > 0:
                    # 只显示最近的数据
                    recent_spikes = spike_times[-1000:]  # 最近1000个尖峰
                    ax.scatter(recent_spikes, [neuron_id] * len(recent_spikes), 
                             s=1, alpha=0.7)
            
            ax.set_xlabel('时间 (ms)')
            ax.set_ylabel('神经元ID')
            ax.set_title('实时尖峰栅格图')
            
        # 创建动画
        ani = animation.FuncAnimation(fig, update_raster, 
                                    interval=self.update_interval, 
                                    blit=False)
        
        plt.show()
    
    def _setup_real_time_firing_rate(self):
        """设置实时发放频率图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self.figures['firing_rate'] = (fig, ax)
        
        def update_firing_rate(frame):
            if not self.is_running:
                return
            
            ax.clear()
            
            # 计算发放频率
            spike_data = self.data_buffer.get('spikes', {})
            
            if spike_data:
                all_spikes = []
                for spike_times in spike_data.values():
                    all_spikes.extend(spike_times)
                
                if all_spikes:
                    # 计算最近时间窗口的发放频率
                    current_time = max(all_spikes)
                    window_size = 100.0  # ms
                    
                    time_bins = np.arange(current_time - 1000, current_time + window_size, window_size)
                    firing_rates = []
                    
                    for i in range(len(time_bins) - 1):
                        bin_start = time_bins[i]
                        bin_end = time_bins[i + 1]
                        
                        spike_count = sum(1 for t in all_spikes if bin_start <= t < bin_end)
                        firing_rate = spike_count / (window_size / 1000.0) / len(spike_data)
                        firing_rates.append(firing_rate)
                    
                    ax.plot(time_bins[:-1], firing_rates, 'b-', linewidth=2)
            
            ax.set_xlabel('时间 (ms)')
            ax.set_ylabel('发放频率 (Hz)')
            ax.set_title('实时发放频率')
            ax.grid(True, alpha=0.3)
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update_firing_rate,
                                    interval=self.update_interval,
                                    blit=False)
        
        plt.show()
    
    def update_data(self, data_type: str, data: Any):
        """更新实时数据"""
        self.data_buffer[data_type] = data
    
    def stop_real_time_visualization(self):
        """停止实时可视化"""
        self.is_running = False

class RecordingManager:
    """记录管理器"""
    
    def __init__(self):
        self.active_sessions = {}
        self.recorders = {}
        self.logger = logging.getLogger("RecordingManager")
    
    def create_recording_session(self, session_id: str, 
                               recording_configs: List[RecordingConfig]) -> RecordingSession:
        """创建记录会话"""
        session = RecordingSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # 创建记录器
        for config in recording_configs:
            if config.recording_type == RecordingType.SPIKE_TIMES:
                recorder = SpikeRecorder(config)
            elif config.recording_type == RecordingType.MEMBRANE_VOLTAGE:
                recorder = VoltageRecorder(config)
            elif config.recording_type == RecordingType.SYNAPTIC_WEIGHTS:
                recorder = WeightRecorder(config)
            else:
                self.logger.warning(f"不支持的记录类型: {config.recording_type}")
                continue
            
            session.recordings[config.recording_id] = recorder
            self.recorders[config.recording_id] = recorder
        
        self.active_sessions[session_id] = session
        self.logger.info(f"创建记录会话: {session_id}")
        
        return session
    
    def start_recording(self, session_id: str):
        """开始记录"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            for recorder in session.recordings.values():
                recorder.start_recording()
            self.logger.info(f"开始记录会话: {session_id}")
    
    def stop_recording(self, session_id: str):
        """停止记录"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            
            for recorder in session.recordings.values():
                recorder.stop_recording()
            
            self.logger.info(f"停止记录会话: {session_id}")
    
    def record_spike(self, recording_id: str, neuron_id: int, timestamp: float):
        """记录尖峰"""
        if recording_id in self.recorders:
            data_point = DataPoint(
                timestamp=timestamp,
                neuron_id=neuron_id,
                value=1.0  # 尖峰事件
            )
            self.recorders[recording_id].record_data_point(data_point)
    
    def record_voltage(self, recording_id: str, neuron_id: int, 
                      timestamp: float, voltage: float):
        """记录电压"""
        if recording_id in self.recorders:
            data_point = DataPoint(
                timestamp=timestamp,
                neuron_id=neuron_id,
                value=voltage
            )
            self.recorders[recording_id].record_data_point(data_point)
    
    def record_weight(self, recording_id: str, pre_neuron_id: int, 
                     post_neuron_id: int, timestamp: float, weight: float):
        """记录权重"""
        if recording_id in self.recorders:
            data_point = DataPoint(
                timestamp=timestamp,
                neuron_id=0,  # 权重记录不需要特定神经元ID
                value=weight,
                metadata={
                    'pre_neuron_id': pre_neuron_id,
                    'post_neuron_id': post_neuron_id
                }
            )
            self.recorders[recording_id].record_data_point(data_point)

