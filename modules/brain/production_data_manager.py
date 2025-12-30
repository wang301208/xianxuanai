"""
生产级神经数据管理系统

提供高性能的神经数据存储、压缩、索引和查询功能，
支持大规模神经网络数据的实时处理和分析。
"""

import asyncio
import hashlib
import json
import logging
import lz4.frame
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from uuid import uuid4

import h5py


class DataType(Enum):
    """数据类型"""
    SPIKE_TRAIN = "spike_train"
    NEURAL_ACTIVITY = "neural_activity"
    SYNAPTIC_WEIGHTS = "synaptic_weights"
    MEMBRANE_POTENTIAL = "membrane_potential"
    CONNECTIVITY_MATRIX = "connectivity_matrix"
    METADATA = "metadata"


class CompressionMethod(Enum):
    """压缩方法"""
    NONE = "none"
    LZ4 = "lz4"
    GZIP = "gzip"
    ZSTD = "zstd"
    NEURAL_SPECIFIC = "neural_specific"


@dataclass
class DataRecord:
    """数据记录"""
    record_id: str = field(default_factory=lambda: str(uuid4()))
    neuron_id: str = ""
    data_type: DataType = DataType.SPIKE_TRAIN
    data: Union[np.ndarray, Dict, List] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_method: CompressionMethod = CompressionMethod.LZ4
    compressed_size: int = 0
    original_size: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """压缩比"""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


@dataclass
class QueryPattern:
    """查询模式"""
    pattern_data: np.ndarray
    tolerance: float = 0.1
    max_results: int = 100
    time_window: Optional[Tuple[datetime, datetime]] = None
    neuron_ids: Optional[List[str]] = None


@dataclass
class StorageStats:
    """存储统计"""
    total_records: int = 0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    average_compression_ratio: float = 1.0
    storage_nodes: int = 0
    query_count: int = 0
    average_query_time: float = 0.0


class NeuralDataCompressor:
    """
    神经数据压缩器
    
    提供多种压缩算法，针对神经数据特性进行优化。
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compression_stats: Dict[CompressionMethod, Dict[str, float]] = {}
    
    def compress_data(self, data: Union[np.ndarray, Dict, List], 
                     method: CompressionMethod = CompressionMethod.LZ4) -> Tuple[bytes, float]:
        """
        压缩数据
        
        Args:
            data: 要压缩的数据
            method: 压缩方法
            
        Returns:
            (压缩后的数据, 压缩比)
        """
        start_time = time.perf_counter()
        
        try:
            # 序列化数据
            serialized_data = self._serialize_data(data)
            original_size = len(serialized_data)
            
            # 执行压缩
            if method == CompressionMethod.LZ4:
                compressed_data = lz4.frame.compress(serialized_data)
            elif method == CompressionMethod.NEURAL_SPECIFIC:
                compressed_data = self._neural_specific_compression(data, serialized_data)
            else:
                compressed_data = serialized_data  # 无压缩
            
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # 记录统计信息
            compression_time = time.perf_counter() - start_time
            self._update_compression_stats(method, compression_ratio, compression_time)
            
            return compressed_data, compression_ratio
            
        except Exception as e:
            self.logger.error(f"数据压缩失败: {e}")
            raise
    
    def decompress_data(self, compressed_data: bytes, 
                       method: CompressionMethod = CompressionMethod.LZ4) -> Any:
        """
        解压缩数据
        
        Args:
            compressed_data: 压缩的数据
            method: 压缩方法
            
        Returns:
            解压缩后的数据
        """
        try:
            # 执行解压缩
            if method == CompressionMethod.LZ4:
                decompressed_data = lz4.frame.decompress(compressed_data)
            elif method == CompressionMethod.NEURAL_SPECIFIC:
                decompressed_data = self._neural_specific_decompression(compressed_data)
            else:
                decompressed_data = compressed_data  # 无压缩
            
            # 反序列化数据
            return self._deserialize_data(decompressed_data)
            
        except Exception as e:
            self.logger.error(f"数据解压缩失败: {e}")
            raise
    
    def _serialize_data(self, data: Any) -> bytes:
        """序列化数据"""
        if isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            # 尝试作为JSON解析
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 假设是numpy数组
            return np.frombuffer(data, dtype=np.float64)
    
    def _neural_specific_compression(self, original_data: Any, serialized_data: bytes) -> bytes:
        """神经数据特定压缩"""
        if isinstance(original_data, np.ndarray):
            # 对于稀疏的尖峰数据，使用差分编码
            if original_data.dtype in [np.int32, np.int64]:
                diff_data = np.diff(original_data, prepend=0)
                return lz4.frame.compress(diff_data.tobytes())
        
        # 默认使用LZ4
        return lz4.frame.compress(serialized_data)
    
    def _neural_specific_decompression(self, compressed_data: bytes) -> bytes:
        """神经数据特定解压缩"""
        # 这里应该实现对应的解压缩逻辑
        # 为简化，直接使用LZ4解压缩
        return lz4.frame.decompress(compressed_data)
    
    def _update_compression_stats(self, method: CompressionMethod, 
                                 ratio: float, time_taken: float) -> None:
        """更新压缩统计信息"""
        if method not in self.compression_stats:
            self.compression_stats[method] = {
                'total_operations': 0,
                'total_ratio': 0.0,
                'total_time': 0.0
            }
        
        stats = self.compression_stats[method]
        stats['total_operations'] += 1
        stats['total_ratio'] += ratio
        stats['total_time'] += time_taken
    
    def get_compression_stats(self) -> Dict[str, Dict[str, float]]:
        """获取压缩统计信息"""
        result = {}
        for method, stats in self.compression_stats.items():
            if stats['total_operations'] > 0:
                result[method.value] = {
                    'average_ratio': stats['total_ratio'] / stats['total_operations'],
                    'average_time': stats['total_time'] / stats['total_operations'],
                    'total_operations': stats['total_operations']
                }
        return result


class NeuralDataIndexer:
    """
    神经数据索引器
    
    为神经数据创建高效的索引，支持快速模式匹配和查询。
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # 索引存储
        self.pattern_index: Dict[str, List[str]] = {}  # 模式哈希 -> 记录ID列表
        self.temporal_index: Dict[str, List[Tuple[datetime, str]]] = {}  # 神经元ID -> (时间, 记录ID)
        self.metadata_index: Dict[str, Dict[str, List[str]]] = {}  # 元数据键 -> 值 -> 记录ID列表
        
        # 索引锁
        self.index_lock = threading.RLock()
    
    def add_to_index(self, record: DataRecord) -> None:
        """添加记录到索引"""
        with self.index_lock:
            try:
                # 模式索引
                if isinstance(record.data, np.ndarray) and record.data_type == DataType.SPIKE_TRAIN:
                    patterns = self._extract_patterns(record.data)
                    for pattern_hash in patterns:
                        if pattern_hash not in self.pattern_index:
                            self.pattern_index[pattern_hash] = []
                        self.pattern_index[pattern_hash].append(record.record_id)
                
                # 时间索引
                if record.neuron_id not in self.temporal_index:
                    self.temporal_index[record.neuron_id] = []
                self.temporal_index[record.neuron_id].append((record.timestamp, record.record_id))
                
                # 元数据索引
                for key, value in record.metadata.items():
                    if key not in self.metadata_index:
                        self.metadata_index[key] = {}
                    
                    value_str = str(value)
                    if value_str not in self.metadata_index[key]:
                        self.metadata_index[key][value_str] = []
                    
                    self.metadata_index[key][value_str].append(record.record_id)
                
                self.logger.debug(f"记录已添加到索引: {record.record_id}")
                
            except Exception as e:
                self.logger.error(f"索引添加失败: {e}")
    
    def remove_from_index(self, record_id: str, neuron_id: str) -> None:
        """从索引中移除记录"""
        with self.index_lock:
            try:
                # 从模式索引中移除
                for pattern_list in self.pattern_index.values():
                    if record_id in pattern_list:
                        pattern_list.remove(record_id)
                
                # 从时间索引中移除
                if neuron_id in self.temporal_index:
                    self.temporal_index[neuron_id] = [
                        (ts, rid) for ts, rid in self.temporal_index[neuron_id]
                        if rid != record_id
                    ]
                
                # 从元数据索引中移除
                for key_dict in self.metadata_index.values():
                    for value_list in key_dict.values():
                        if record_id in value_list:
                            value_list.remove(record_id)
                
                self.logger.debug(f"记录已从索引移除: {record_id}")
                
            except Exception as e:
                self.logger.error(f"索引移除失败: {e}")
    
    def search_patterns(self, query_pattern: QueryPattern) -> List[str]:
        """搜索匹配的模式"""
        with self.index_lock:
            try:
                # 生成查询模式的哈希
                query_hashes = self._extract_patterns(query_pattern.pattern_data)
                
                # 收集匹配的记录ID
                matching_records = set()
                for pattern_hash in query_hashes:
                    if pattern_hash in self.pattern_index:
                        matching_records.update(self.pattern_index[pattern_hash])
                
                # 限制结果数量
                result = list(matching_records)[:query_pattern.max_results]
                
                self.logger.debug(f"模式搜索完成，找到 {len(result)} 个匹配")
                return result
                
            except Exception as e:
                self.logger.error(f"模式搜索失败: {e}")
                return []
    
    def search_by_time_range(self, neuron_id: str, start_time: datetime, 
                           end_time: datetime) -> List[str]:
        """按时间范围搜索"""
        with self.index_lock:
            if neuron_id not in self.temporal_index:
                return []
            
            matching_records = []
            for timestamp, record_id in self.temporal_index[neuron_id]:
                if start_time <= timestamp <= end_time:
                    matching_records.append(record_id)
            
            return matching_records
    
    def search_by_metadata(self, metadata_filters: Dict[str, Any]) -> List[str]:
        """按元数据搜索"""
        with self.index_lock:
            if not metadata_filters:
                return []
            
            # 获取第一个过滤条件的结果
            first_key, first_value = next(iter(metadata_filters.items()))
            first_value_str = str(first_value)
            
            if (first_key not in self.metadata_index or 
                first_value_str not in self.metadata_index[first_key]):
                return []
            
            matching_records = set(self.metadata_index[first_key][first_value_str])
            
            # 应用其他过滤条件
            for key, value in list(metadata_filters.items())[1:]:
                value_str = str(value)
                if (key not in self.metadata_index or 
                    value_str not in self.metadata_index[key]):
                    return []
                
                matching_records &= set(self.metadata_index[key][value_str])
            
            return list(matching_records)
    
    def _extract_patterns(self, data: np.ndarray) -> List[str]:
        """从数据中提取模式哈希"""
        patterns = []
        
        if len(data) < self.window_size:
            # 数据太短，使用整个数据作为模式
            pattern_hash = hashlib.md5(data.tobytes()).hexdigest()
            patterns.append(pattern_hash)
        else:
            # 使用滑动窗口提取模式
            for i in range(len(data) - self.window_size + 1):
                window = data[i:i + self.window_size]
                pattern_hash = hashlib.md5(window.tobytes()).hexdigest()
                patterns.append(pattern_hash)
        
        return patterns
    
    def get_index_stats(self) -> Dict[str, int]:
        """获取索引统计信息"""
        with self.index_lock:
            return {
                'pattern_count': len(self.pattern_index),
                'neuron_count': len(self.temporal_index),
                'metadata_keys': len(self.metadata_index)
            }


class DistributedNeuralStorage:
    """
    分布式神经数据存储
    
    支持多节点数据存储，提供高可用性和负载均衡。
    """
    
    def __init__(self, storage_nodes: List[Path], replication_factor: int = 2):
        self.storage_nodes = [Path(node) for node in storage_nodes]
        self.replication_factor = min(replication_factor, len(self.storage_nodes))
        self.logger = logging.getLogger(__name__)
        
        # 创建存储目录
        for node in self.storage_nodes:
            node.mkdir(parents=True, exist_ok=True)
        
        # 节点健康状态
        self.node_health: Dict[Path, bool] = {node: True for node in self.storage_nodes}
        
        # 数据分布映射
        self.data_distribution: Dict[str, List[Path]] = {}
        
        # 存储锁
        self.storage_lock = threading.RLock()
    
    def store_data(self, record_id: str, data: bytes) -> bool:
        """存储数据到分布式节点"""
        with self.storage_lock:
            try:
                # 选择存储节点
                target_nodes = self._select_storage_nodes(record_id)
                
                if not target_nodes:
                    self.logger.error("没有可用的存储节点")
                    return False
                
                # 存储到选定的节点
                successful_stores = 0
                for node in target_nodes:
                    if self._store_to_node(node, record_id, data):
                        successful_stores += 1
                
                # 记录数据分布
                self.data_distribution[record_id] = target_nodes
                
                # 检查是否满足最小副本数
                min_replicas = max(1, self.replication_factor // 2)
                success = successful_stores >= min_replicas
                
                if success:
                    self.logger.debug(f"数据存储成功: {record_id} -> {successful_stores} 个节点")
                else:
                    self.logger.error(f"数据存储失败: {record_id} -> 只有 {successful_stores} 个节点成功")
                
                return success
                
            except Exception as e:
                self.logger.error(f"数据存储异常: {e}")
                return False
    
    def retrieve_data(self, record_id: str) -> Optional[bytes]:
        """从分布式节点检索数据"""
        with self.storage_lock:
            if record_id not in self.data_distribution:
                self.logger.warning(f"未找到数据分布信息: {record_id}")
                return None
            
            # 尝试从各个节点读取数据
            for node in self.data_distribution[record_id]:
                if not self.node_health.get(node, False):
                    continue
                
                data = self._retrieve_from_node(node, record_id)
                if data is not None:
                    return data
            
            self.logger.error(f"无法从任何节点检索数据: {record_id}")
            return None
    
    def delete_data(self, record_id: str) -> bool:
        """删除分布式数据"""
        with self.storage_lock:
            if record_id not in self.data_distribution:
                return True  # 数据不存在，视为删除成功
            
            # 从所有节点删除数据
            successful_deletes = 0
            for node in self.data_distribution[record_id]:
                if self._delete_from_node(node, record_id):
                    successful_deletes += 1
            
            # 清理分布信息
            del self.data_distribution[record_id]
            
            self.logger.debug(f"数据删除完成: {record_id} -> {successful_deletes} 个节点")
            return successful_deletes > 0
    
    def _select_storage_nodes(self, record_id: str) -> List[Path]:
        """选择存储节点"""
        # 基于记录ID的哈希选择节点
        hash_value = int(hashlib.md5(record_id.encode()).hexdigest(), 16)
        
        # 选择健康的节点
        healthy_nodes = [node for node, health in self.node_health.items() if health]
        
        if not healthy_nodes:
            return []
        
        # 确定性地选择节点
        selected_nodes = []
        for i in range(self.replication_factor):
            node_index = (hash_value + i) % len(healthy_nodes)
            selected_nodes.append(healthy_nodes[node_index])
        
        return selected_nodes
    
    def _store_to_node(self, node: Path, record_id: str, data: bytes) -> bool:
        """存储数据到指定节点"""
        try:
            file_path = node / f"{record_id}.dat"
            with open(file_path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            self.logger.error(f"节点存储失败 {node}: {e}")
            self.node_health[node] = False
            return False
    
    def _retrieve_from_node(self, node: Path, record_id: str) -> Optional[bytes]:
        """从指定节点检索数据"""
        try:
            file_path = node / f"{record_id}.dat"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return f.read()
            return None
        except Exception as e:
            self.logger.error(f"节点检索失败 {node}: {e}")
            self.node_health[node] = False
            return None
    
    def _delete_from_node(self, node: Path, record_id: str) -> bool:
        """从指定节点删除数据"""
        try:
            file_path = node / f"{record_id}.dat"
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            self.logger.error(f"节点删除失败 {node}: {e}")
            return False
    
    def get_storage_stats(self) -> StorageStats:
        """获取存储统计信息"""
        with self.storage_lock:
            total_records = len(self.data_distribution)
            healthy_nodes = sum(1 for health in self.node_health.values() if health)
            
            # 计算总存储大小
            total_size = 0
            for node in self.storage_nodes:
                if self.node_health.get(node, False):
                    try:
                        for file_path in node.glob("*.dat"):
                            total_size += file_path.stat().st_size
                    except Exception:
                        pass
            
            return StorageStats(
                total_records=total_records,
                total_size_bytes=total_size,
                storage_nodes=healthy_nodes
            )


class ProductionNeuralDataManager:
    """
    生产级神经数据管理系统
    
    集成数据压缩、索引、存储和查询功能的完整数据管理解决方案。
    """
    
    def __init__(self, storage_nodes: List[str], window_size: int = 100):
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.compressor = NeuralDataCompressor()
        self.indexer = NeuralDataIndexer(window_size)
        self.storage = DistributedNeuralStorage([Path(node) for node in storage_nodes])
        
        # 任务执行器
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # 缓存
        self.cache: Dict[str, DataRecord] = {}
        self.cache_lock = threading.RLock()
        self.max_cache_size = 1000
        
        # 统计信息
        self.stats = StorageStats()
        self.query_times: List[float] = []
    
    def store_spike_train(self, neuron_id: str, spike_data: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None,
                         compression_method: CompressionMethod = CompressionMethod.LZ4) -> Tuple[str, float]:
        """
        存储尖峰序列数据
        
        Args:
            neuron_id: 神经元ID
            spike_data: 尖峰数据
            metadata: 元数据
            compression_method: 压缩方法
            
        Returns:
            (记录ID, 压缩比)
        """
        try:
            # 创建数据记录
            record = DataRecord(
                neuron_id=neuron_id,
                data_type=DataType.SPIKE_TRAIN,
                data=spike_data,
                metadata=metadata or {},
                compression_method=compression_method
            )
            
            # 压缩数据
            compressed_data, compression_ratio = self.compressor.compress_data(
                spike_data, compression_method
            )
            
            record.original_size = spike_data.nbytes
            record.compressed_size = len(compressed_data)
            
            # 存储到分布式存储
            if not self.storage.store_data(record.record_id, compressed_data):
                raise RuntimeError("数据存储失败")
            
            # 添加到索引
            self.indexer.add_to_index(record)
            
            # 添加到缓存
            self._add_to_cache(record)
            
            # 更新统计信息
            self._update_stats(record)
            
            self.logger.info(f"尖峰序列存储成功: {neuron_id} -> {record.record_id}")
            return record.record_id, compression_ratio
            
        except Exception as e:
            self.logger.error(f"尖峰序列存储失败: {e}")
            raise
    
    def query_neural_patterns(self, query_pattern: QueryPattern) -> Tuple[List[str], float]:
        """
        查询神经模式
        
        Args:
            query_pattern: 查询模式
            
        Returns:
            (匹配的记录ID列表, 查询时间)
        """
        start_time = time.perf_counter()
        
        try:
            # 执行模式搜索
            matching_records = self.indexer.search_patterns(query_pattern)
            
            # 应用时间窗口过滤
            if query_pattern.time_window:
                start_time_filter, end_time_filter = query_pattern.time_window
                filtered_records = []
                
                for record_id in matching_records:
                    record = self._get_record_metadata(record_id)
                    if (record and start_time_filter <= record.timestamp <= end_time_filter):
                        filtered_records.append(record_id)
                
                matching_records = filtered_records
            
            # 应用神经元ID过滤
            if query_pattern.neuron_ids:
                neuron_filtered = []
                for record_id in matching_records:
                    record = self._get_record_metadata(record_id)
                    if record and record.neuron_id in query_pattern.neuron_ids:
                        neuron_filtered.append(record_id)
                
                matching_records = neuron_filtered
            
            query_time = time.perf_counter() - start_time
            
            # 记录查询统计
            self.query_times.append(query_time)
            if len(self.query_times) > 1000:
                self.query_times = self.query_times[-500:]
            
            self.stats.query_count += 1
            if self.query_times:
                self.stats.average_query_time = sum(self.query_times) / len(self.query_times)
            
            self.logger.debug(f"模式查询完成: {len(matching_records)} 个匹配，耗时 {query_time:.4f}s")
            return matching_records, query_time
            
        except Exception as e:
            self.logger.error(f"模式查询失败: {e}")
            return [], time.perf_counter() - start_time
    
    def retrieve_data(self, record_id: str) -> Optional[DataRecord]:
        """检索数据记录"""
        try:
            # 检查缓存
            with self.cache_lock:
                if record_id in self.cache:
                    return self.cache[record_id]
            
            # 从存储检索
            compressed_data = self.storage.retrieve_data(record_id)
            if compressed_data is None:
                return None
            
            # 这里需要从某处获取记录元数据
            # 在实际实现中，应该将元数据单独存储
            # 为简化，返回基本记录
            record = DataRecord(record_id=record_id)
            
            self.logger.debug(f"数据检索成功: {record_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"数据检索失败: {e}")
            return None
    
    def delete_data(self, record_id: str, neuron_id: str) -> bool:
        """删除数据记录"""
        try:
            # 从存储删除
            if not self.storage.delete_data(record_id):
                return False
            
            # 从索引删除
            self.indexer.remove_from_index(record_id, neuron_id)
            
            # 从缓存删除
            with self.cache_lock:
                self.cache.pop(record_id, None)
            
            self.logger.info(f"数据删除成功: {record_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据删除失败: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        storage_stats = self.storage.get_storage_stats()
        compression_stats = self.compressor.get_compression_stats()
        index_stats = self.indexer.get_index_stats()
        
        return {
            'storage': storage_stats.__dict__,
            'compression': compression_stats,
            'indexing': index_stats,
            'cache_size': len(self.cache),
            'query_performance': {
                'total_queries': self.stats.query_count,
                'average_query_time': self.stats.average_query_time
            }
        }
    
    def _add_to_cache(self, record: DataRecord) -> None:
        """添加记录到缓存"""
        with self.cache_lock:
            # 如果缓存已满，移除最旧的记录
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[record.record_id] = record
    
    def _get_record_metadata(self, record_id: str) -> Optional[DataRecord]:
        """获取记录元数据"""
        # 在实际实现中，应该从元数据存储中获取
        # 这里简化处理
        with self.cache_lock:
            return self.cache.get(record_id)
    
    def _update_stats(self, record: DataRecord) -> None:
        """更新统计信息"""
        self.stats.total_records += 1
        self.stats.total_size_bytes += record.original_size
        self.stats.compressed_size_bytes += record.compressed_size
        
        if self.stats.total_size_bytes > 0:
            self.stats.average_compression_ratio = (
                self.stats.compressed_size_bytes / self.stats.total_size_bytes
            )


# 便捷函数
def create_data_manager(storage_nodes: List[str], window_size: int = 100) -> ProductionNeuralDataManager:
    """创建数据管理器实例"""
    return ProductionNeuralDataManager(storage_nodes, window_size)


def store_spike_train(neuron_id: str, spike_data: np.ndarray, 
                     compressor: NeuralDataCompressor, indexer: NeuralDataIndexer,
                     storage: DistributedNeuralStorage) -> float:
    """存储尖峰序列（兼容性函数）"""
    manager = ProductionNeuralDataManager([])
    manager.compressor = compressor
    manager.indexer = indexer
    manager.storage = storage
    
    record_id, compression_ratio = manager.store_spike_train(neuron_id, spike_data)
    return compression_ratio


def query_neural_patterns(pattern_data: np.ndarray, indexer: NeuralDataIndexer,
                         storage: DistributedNeuralStorage, 
                         compressor: NeuralDataCompressor) -> int:
    """查询神经模式（兼容性函数）"""
    query_pattern = QueryPattern(pattern_data=pattern_data)
    
    manager = ProductionNeuralDataManager([])
    manager.indexer = indexer
    manager.storage = storage
    manager.compressor = compressor
    
    matching_records, _ = manager.query_neural_patterns(query_pattern)
    return len(matching_records)