"""
稀疏连接矩阵实现
Sparse Connection Matrix Implementation
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

from .enhanced_connectivity import ConnectionType

class SparseConnectionMatrix:
    """稀疏连接矩阵"""
    
    def __init__(self, n_neurons: int, dtype=np.float32):
        self.n_neurons = n_neurons
        self.dtype = dtype
        
        # 使用CSR格式的稀疏矩阵
        self.weight_matrix = sp.csr_matrix((n_neurons, n_neurons), dtype=dtype)
        self.delay_matrix = sp.csr_matrix((n_neurons, n_neurons), dtype=dtype)
        
        # 连接类型矩阵（使用整数编码）
        self.type_matrix = sp.csr_matrix((n_neurons, n_neurons), dtype=np.int8)
        
        # 统计信息
        self.connection_count = 0
        self.connection_types = {}
        
        self.logger = logging.getLogger("SparseConnectionMatrix")
    
    def add_connection(self, pre_id: int, post_id: int, weight: float, 
                      delay: float, connection_type: ConnectionType):
        """添加连接"""
        
        if pre_id >= self.n_neurons or post_id >= self.n_neurons:
            raise ValueError(f"神经元ID超出范围: {pre_id}, {post_id}")
        
        # 更新权重矩阵
        self.weight_matrix[pre_id, post_id] = weight
        
        # 更新延迟矩阵
        self.delay_matrix[pre_id, post_id] = delay
        
        # 更新类型矩阵
        type_code = list(ConnectionType).index(connection_type)
        self.type_matrix[pre_id, post_id] = type_code
        
        # 更新统计
        self.connection_count += 1
        type_name = connection_type.value
        self.connection_types[type_name] = self.connection_types.get(type_name, 0) + 1
        
        self.logger.debug(f"添加连接: {pre_id} -> {post_id}, 权重: {weight:.3f}, 延迟: {delay:.3f}")
    
    def get_connections(self, pre_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取突触前神经元的所有连接"""
        
        # 获取非零元素
        row = self.weight_matrix.getrow(pre_id)
        post_ids = row.indices
        weights = row.data
        
        # 获取对应的延迟
        delay_row = self.delay_matrix.getrow(pre_id)
        delays = delay_row.data
        
        return post_ids, weights, delays
    
    def get_input_connections(self, post_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取突触后神经元的所有输入连接"""

        # SciPy may return a CSR matrix for getcol(); in that case `.indices` are
        # column indices (always 0 for a single-column view). Convert to COO so
        # we can reliably read the row indices as the presynaptic neuron IDs.
        weight_col = self.weight_matrix.getcol(post_id).tocoo()
        pre_ids = weight_col.row.astype(np.int32, copy=False)
        weights = weight_col.data

        delay_col = self.delay_matrix.getcol(post_id).tocoo()
        delays = delay_col.data

        # Ensure deterministic ordering by presynaptic id.
        if pre_ids.size:
            order = np.argsort(pre_ids)
            pre_ids = pre_ids[order]
            weights = weights[order]
            if delays.size == order.size:
                delays = delays[order]

        return pre_ids, weights, delays
    
    def update_weight(self, pre_id: int, post_id: int, new_weight: float):
        """更新连接权重"""
        self.weight_matrix[pre_id, post_id] = new_weight
    
    def remove_connection(self, pre_id: int, post_id: int):
        """移除连接"""
        
        if self.weight_matrix[pre_id, post_id] != 0:
            self.weight_matrix[pre_id, post_id] = 0
            self.delay_matrix[pre_id, post_id] = 0
            self.type_matrix[pre_id, post_id] = 0
            self.connection_count -= 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取连接统计"""
        
        return {
            'total_neurons': self.n_neurons,
            'total_connections': self.connection_count,
            'connection_density': self.connection_count / (self.n_neurons ** 2),
            'connection_types': self.connection_types,
            'memory_usage_mb': (self.weight_matrix.data.nbytes + 
                              self.delay_matrix.data.nbytes + 
                              self.type_matrix.data.nbytes) / 1024 / 1024
        }
    
    def save(self, filepath: str):
        """保存连接矩阵"""
        
        if H5PY_AVAILABLE:
            with h5py.File(filepath, 'w') as f:
                # 保存稀疏矩阵
                grp = f.create_group('sparse_matrices')
                
                # 权重矩阵
                weight_grp = grp.create_group('weights')
                weight_grp.create_dataset('data', data=self.weight_matrix.data)
                weight_grp.create_dataset('indices', data=self.weight_matrix.indices)
                weight_grp.create_dataset('indptr', data=self.weight_matrix.indptr)
                weight_grp.attrs['shape'] = self.weight_matrix.shape
                
                # 延迟矩阵
                delay_grp = grp.create_group('delays')
                delay_grp.create_dataset('data', data=self.delay_matrix.data)
                delay_grp.create_dataset('indices', data=self.delay_matrix.indices)
                delay_grp.create_dataset('indptr', data=self.delay_matrix.indptr)
                
                # 类型矩阵
                type_grp = grp.create_group('types')
                type_grp.create_dataset('data', data=self.type_matrix.data)
                type_grp.create_dataset('indices', data=self.type_matrix.indices)
                type_grp.create_dataset('indptr', data=self.type_matrix.indptr)
                
                # 元数据
                f.attrs['n_neurons'] = self.n_neurons
                f.attrs['connection_count'] = self.connection_count
        else:
            # 使用pickle作为备选
            data = {
                'weight_matrix': self.weight_matrix,
                'delay_matrix': self.delay_matrix,
                'type_matrix': self.type_matrix,
                'n_neurons': self.n_neurons,
                'connection_count': self.connection_count,
                'connection_types': self.connection_types
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    
    def load(self, filepath: str):
        """加载连接矩阵"""
        
        if H5PY_AVAILABLE and filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                # 加载权重矩阵
                weight_grp = f['sparse_matrices/weights']
                shape = tuple(weight_grp.attrs['shape'])
                self.weight_matrix = sp.csr_matrix(
                    (weight_grp['data'][:], weight_grp['indices'][:], weight_grp['indptr'][:]),
                    shape=shape
                )
                
                # 加载延迟矩阵
                delay_grp = f['sparse_matrices/delays']
                self.delay_matrix = sp.csr_matrix(
                    (delay_grp['data'][:], delay_grp['indices'][:], delay_grp['indptr'][:]),
                    shape=shape
                )
                
                # 加载类型矩阵
                type_grp = f['sparse_matrices/types']
                self.type_matrix = sp.csr_matrix(
                    (type_grp['data'][:], type_grp['indices'][:], type_grp['indptr'][:]),
                    shape=shape
                )
                
                # 加载元数据
                self.n_neurons = f.attrs['n_neurons']
                self.connection_count = f.attrs['connection_count']
        else:
            # 使用pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.weight_matrix = data['weight_matrix']
                self.delay_matrix = data['delay_matrix']
                self.type_matrix = data['type_matrix']
                self.n_neurons = data['n_neurons']
                self.connection_count = data['connection_count']
                self.connection_types = data['connection_types']
