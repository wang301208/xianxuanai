"""
增强连接层测试
Enhanced Connectivity Tests
"""

import unittest
import numpy as np
from typing import Dict, Any, List

from BrainSimulationSystem.core.enhanced_connectivity import (
    ConnectionType, DelayType, ConnectionParameters, AxonParameters,
    EnhancedConnectivityManager, create_enhanced_connectivity_manager
)
from BrainSimulationSystem.core.sparse_matrix import SparseConnectionMatrix
from BrainSimulationSystem.core.regional_connectivity import (
    RegionalConnectivityMatrix, LongRangeAxon
)
from BrainSimulationSystem.core.probabilistic_connector import ProbabilisticConnector

class TestEnhancedConnectivity(unittest.TestCase):
    """增强连接测试类"""
    
    def setUp(self):
        """设置测试环境"""
        
        self.config = {
            'connector': {
                'seed': 42
            },
            'graph_database': {
                'enabled': False  # 测试时禁用图数据库
            }
        }
        
        self.n_neurons = 1000
        self.regions = ['cortex', 'thalamus', 'hippocampus']
        
        self.connectivity_manager = EnhancedConnectivityManager(self.config)
        self.connectivity_manager.initialize(self.n_neurons, self.regions)
    
    def test_connectivity_manager_initialization(self):
        """测试连接管理器初始化"""
        
        # 检查初始化状态
        self.assertIsNotNone(self.connectivity_manager.sparse_matrix)
        self.assertIsNotNone(self.connectivity_manager.regional_matrix)
        self.assertEqual(len(self.connectivity_manager.regions), 3)
        
        # 检查稀疏矩阵
        self.assertEqual(self.connectivity_manager.sparse_matrix.n_neurons, self.n_neurons)
        
        # 检查区域矩阵
        self.assertEqual(self.connectivity_manager.regional_matrix.n_regions, 3)
    
    def test_neuron_position_and_region_setting(self):
        """测试神经元位置和区域设置"""
        
        # 设置神经元位置
        positions = [
            (0.0, 0.0, 0.0),
            (100.0, 100.0, 0.0),
            (200.0, 200.0, 0.0)
        ]
        
        for i, pos in enumerate(positions):
            self.connectivity_manager.set_neuron_position(i, pos)
        
        # 检查位置设置
        self.assertEqual(len(self.connectivity_manager.neuron_positions), 3)
        self.assertEqual(self.connectivity_manager.neuron_positions[0], positions[0])
        
        # 设置神经元区域
        regions = ['cortex', 'thalamus', 'hippocampus']
        for i, region in enumerate(regions):
            self.connectivity_manager.set_neuron_region(i, region)
        
        # 检查区域设置
        self.assertEqual(len(self.connectivity_manager.neuron_regions), 3)
        self.assertEqual(self.connectivity_manager.neuron_regions[0], 'cortex')
    
    def test_local_connections(self):
        """测试局部连接"""
        
        # 设置测试神经元
        test_neurons = list(range(10))
        for i in test_neurons:
            self.connectivity_manager.set_neuron_region(i, 'cortex')
        
        # 创建连接参数
        connection_params = ConnectionParameters(
            connection_type=ConnectionType.LOCAL,
            probability=0.3,
            weight_mean=1.0,
            weight_std=0.2,
            delay_mean=1.0,
            delay_std=0.1
        )
        
        # 添加局部连接
        self.connectivity_manager.add_local_connections(test_neurons, connection_params)
        
        # 检查连接统计
        stats = self.connectivity_manager.get_statistics()
        self.assertGreater(stats['total_connections'], 0)
        self.assertIn('local', stats['connections_by_type'])
        
        # 检查稀疏矩阵
        weight_matrix = self.connectivity_manager.get_connection_matrix()
        self.assertIsNotNone(weight_matrix)
        self.assertGreater(weight_matrix.nnz, 0)  # 非零元素数量
    
    def test_regional_connections(self):
        """测试区域间连接"""
        
        # 设置测试神经元
        cortex_neurons = list(range(10))
        thalamus_neurons = list(range(10, 20))
        
        for i in cortex_neurons:
            self.connectivity_manager.set_neuron_region(i, 'cortex')
        for i in thalamus_neurons:
            self.connectivity_manager.set_neuron_region(i, 'thalamus')
        
        # 创建区域间连接参数
        connection_params = ConnectionParameters(
            connection_type=ConnectionType.INTERREGION,
            probability=0.1,
            weight_mean=0.5,
            weight_std=0.1,
            delay_mean=5.0,
            delay_std=1.0,
            conduction_velocity=2.0
        )
        
        # 添加区域间连接
        self.connectivity_manager.add_regional_connections(
            'cortex', 'thalamus', connection_params
        )
        
        # 检查连接统计
        stats = self.connectivity_manager.get_statistics()
        self.assertIn('interregion', stats['connections_by_type'])
        
        # 检查区域连接统计
        regional_stats = self.connectivity_manager.regional_matrix.get_regional_statistics()
        self.assertGreater(regional_stats['long_range_axons'], 0)

class TestSparseConnectionMatrix(unittest.TestCase):
    """稀疏连接矩阵测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.n_neurons = 100
        self.matrix = SparseConnectionMatrix(self.n_neurons)
    
    def test_matrix_initialization(self):
        """测试矩阵初始化"""
        
        self.assertEqual(self.matrix.n_neurons, self.n_neurons)
        self.assertEqual(self.matrix.connection_count, 0)
        self.assertEqual(self.matrix.weight_matrix.shape, (self.n_neurons, self.n_neurons))
    
    def test_add_connection(self):
        """测试添加连接"""
        
        # 添加连接
        self.matrix.add_connection(0, 1, 1.5, 2.0, ConnectionType.LOCAL)
        
        # 检查连接计数
        self.assertEqual(self.matrix.connection_count, 1)
        
        # 检查权重矩阵
        self.assertEqual(self.matrix.weight_matrix[0, 1], 1.5)
        
        # 检查延迟矩阵
        self.assertEqual(self.matrix.delay_matrix[0, 1], 2.0)
        
        # 检查连接类型统计
        self.assertIn('local', self.matrix.connection_types)
        self.assertEqual(self.matrix.connection_types['local'], 1)
    
    def test_get_connections(self):
        """测试获取连接"""
        
        # 添加多个连接
        connections = [
            (0, 1, 1.0, 1.5),
            (0, 2, 2.0, 2.5),
            (0, 3, 1.5, 2.0)
        ]
        
        for pre, post, weight, delay in connections:
            self.matrix.add_connection(pre, post, weight, delay, ConnectionType.LOCAL)
        
        # 获取神经元0的输出连接
        post_ids, weights, delays = self.matrix.get_connections(0)
        
        self.assertEqual(len(post_ids), 3)
        self.assertIn(1, post_ids)
        self.assertIn(2, post_ids)
        self.assertIn(3, post_ids)
        
        # 检查权重和延迟
        idx_1 = np.where(post_ids == 1)[0][0]
        self.assertEqual(weights[idx_1], 1.0)
        self.assertEqual(delays[idx_1], 1.5)
    
    def test_get_input_connections(self):
        """测试获取输入连接"""
        
        # 添加输入连接
        input_connections = [
            (1, 0, 1.0, 1.5),
            (2, 0, 2.0, 2.5),
            (3, 0, 1.5, 2.0)
        ]
        
        for pre, post, weight, delay in input_connections:
            self.matrix.add_connection(pre, post, weight, delay, ConnectionType.LOCAL)
        
        # 获取神经元0的输入连接
        pre_ids, weights, delays = self.matrix.get_input_connections(0)
        
        self.assertEqual(len(pre_ids), 3)
        self.assertIn(1, pre_ids)
        self.assertIn(2, pre_ids)
        self.assertIn(3, pre_ids)
    
    def test_update_and_remove_connection(self):
        """测试更新和移除连接"""
        
        # 添加连接
        self.matrix.add_connection(0, 1, 1.0, 2.0, ConnectionType.LOCAL)
        
        # 更新权重
        self.matrix.update_weight(0, 1, 2.0)
        self.assertEqual(self.matrix.weight_matrix[0, 1], 2.0)
        
        # 移除连接
        self.matrix.remove_connection(0, 1)
        self.assertEqual(self.matrix.weight_matrix[0, 1], 0.0)
        self.assertEqual(self.matrix.connection_count, 0)
    
    def test_statistics(self):
        """测试统计信息"""
        
        # 添加一些连接
        for i in range(10):
            self.matrix.add_connection(i, (i+1) % self.n_neurons, 1.0, 1.0, ConnectionType.LOCAL)
        
        stats = self.matrix.get_statistics()
        
        self.assertEqual(stats['total_neurons'], self.n_neurons)
        self.assertEqual(stats['total_connections'], 10)
        self.assertAlmostEqual(stats['connection_density'], 10 / (self.n_neurons ** 2), places=6)
        self.assertIn('local', stats['connection_types'])

class TestRegionalConnectivity(unittest.TestCase):
    """区域连接测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.regions = ['V1', 'V2', 'MT', 'LGN']
        self.regional_matrix = RegionalConnectivityMatrix(self.regions)
    
    def test_regional_matrix_initialization(self):
        """测试区域矩阵初始化"""
        
        self.assertEqual(self.regional_matrix.n_regions, 4)
        self.assertIn('V1', self.regional_matrix.region_to_index)
        self.assertEqual(self.regional_matrix.region_to_index['V1'], 0)
        
        # 检查矩阵形状
        self.assertEqual(self.regional_matrix.connection_strength.shape, (4, 4))
        self.assertEqual(self.regional_matrix.connection_probability.shape, (4, 4))
        self.assertEqual(self.regional_matrix.connection_delay.shape, (4, 4))
    
    def test_set_connection(self):
        """测试设置区域连接"""
        
        # 设置V1到V2的连接
        self.regional_matrix.set_connection('V1', 'V2', 0.8, 0.3, 5.0)
        
        # 检查连接参数
        strength, prob, delay = self.regional_matrix.get_connection_parameters('V1', 'V2')
        self.assertEqual(strength, 0.8)
        self.assertEqual(prob, 0.3)
        self.assertEqual(delay, 5.0)
    
    def test_long_range_axon(self):
        """测试长程轴突"""
        
        # 创建轴突参数
        axon_params = AxonParameters(
            length=5000.0,
            diameter=2.0,
            conduction_velocity=2.0,
            myelination=True
        )
        
        # 添加长程轴突
        self.regional_matrix.add_long_range_axon(1, 'V1', 'MT', axon_params)
        
        # 检查轴突
        key = (1, 'V1', 'MT')
        self.assertIn(key, self.regional_matrix.long_range_axons)
        
        axon = self.regional_matrix.long_range_axons[key]
        self.assertEqual(axon.source_id, 1)
        self.assertEqual(axon.target_region, 'MT')
        self.assertGreater(axon.conduction_delay, 0)
    
    def test_regional_statistics(self):
        """测试区域统计"""
        
        # 设置一些连接
        connections = [
            ('V1', 'V2', 0.8, 0.3, 5.0),
            ('V2', 'MT', 0.6, 0.2, 8.0),
            ('LGN', 'V1', 0.9, 0.4, 3.0)
        ]
        
        for source, target, strength, prob, delay in connections:
            self.regional_matrix.set_connection(source, target, strength, prob, delay)
        
        stats = self.regional_matrix.get_regional_statistics()
        
        self.assertEqual(stats['total_regions'], 4)
        self.assertEqual(stats['total_connections'], 3)
        self.assertGreater(stats['mean_strength'], 0)
        self.assertGreater(stats['mean_delay'], 0)

class TestLongRangeAxon(unittest.TestCase):
    """长程轴突测试"""
    
    def test_axon_initialization(self):
        """测试轴突初始化"""
        
        params = AxonParameters(
            length=10000.0,
            diameter=3.0,
            conduction_velocity=5.0,
            myelination=True
        )
        
        axon = LongRangeAxon(source_id=1, target_region='target', params=params)
        
        self.assertEqual(axon.source_id, 1)
        self.assertEqual(axon.target_region, 'target')
        self.assertGreater(axon.conduction_delay, 0)
    
    def test_myelination_effect(self):
        """测试髓鞘化效应"""
        
        # 有髓轴突
        myelinated_params = AxonParameters(
            length=5000.0,
            diameter=2.0,
            conduction_velocity=3.0,
            myelination=True
        )
        
        # 无髓轴突
        unmyelinated_params = AxonParameters(
            length=5000.0,
            diameter=2.0,
            conduction_velocity=3.0,
            myelination=False
        )
        
        myelinated_axon = LongRangeAxon(1, 'target', myelinated_params)
        unmyelinated_axon = LongRangeAxon(2, 'target', unmyelinated_params)
        
        # 有髓轴突应该传导更快（延迟更小）
        self.assertLess(myelinated_axon.conduction_delay, unmyelinated_axon.conduction_delay)
    
    def test_axon_branching(self):
        """测试轴突分支"""
        
        params = AxonParameters(length=5000.0, diameter=2.0)
        axon = LongRangeAxon(source_id=1, target_region='main_target', params=params)
        
        # 添加分支
        axon.add_branch(target_id=10, branch_length=2000.0)
        axon.add_branch(target_id=11, branch_length=3000.0)
        
        self.assertEqual(len(axon.branches), 2)
        
        # 检查分支延迟
        delay_10 = axon.get_total_delay(10)
        delay_11 = axon.get_total_delay(11)
        
        self.assertGreater(delay_10, axon.conduction_delay)
        self.assertGreater(delay_11, axon.conduction_delay)
        self.assertNotEqual(delay_10, delay_11)  # 不同分支应该有不同延迟

class TestProbabilisticConnector(unittest.TestCase):
    """概率连接器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {'seed': 42}
        self.connector = ProbabilisticConnector(self.config)
    
    def test_connection_generation(self):
        """测试连接生成"""
        
        source_neurons = [0, 1, 2]
        target_neurons = [3, 4, 5]
        
        connection_params = ConnectionParameters(
            connection_type=ConnectionType.LOCAL,
            probability=0.5,
            weight_mean=1.0,
            weight_std=0.2,
            delay_mean=2.0,
            delay_std=0.5
        )
        
        connections = self.connector.generate_connections(
            source_neurons, target_neurons, connection_params
        )
        
        # 检查连接格式
        for conn in connections:
            self.assertEqual(len(conn), 4)  # (source, target, weight, delay)
            source, target, weight, delay = conn
            self.assertIn(source, source_neurons)
            self.assertIn(target, target_neurons)
            self.assertGreater(weight, 0)
            self.assertGreater(delay, 0)
    
    def test_reproducibility(self):
        """测试可重现性"""
        
        source_neurons = [0, 1]
        target_neurons = [2, 3]
        
        connection_params = ConnectionParameters(
            connection_type=ConnectionType.LOCAL,
            probability=0.8
        )
        
        # 生成两次连接
        connections1 = self.connector.generate_connections(
            source_neurons, target_neurons, connection_params
        )
        
        # 重新初始化连接器（相同种子）
        connector2 = ProbabilisticConnector(self.config)
        connections2 = connector2.generate_connections(
            source_neurons, target_neurons, connection_params
        )
        
        # 应该生成相同的连接
        self.assertEqual(len(connections1), len(connections2))
        for conn1, conn2 in zip(connections1, connections2):
            self.assertEqual(conn1, conn2)

if __name__ == '__main__':
    unittest.main()