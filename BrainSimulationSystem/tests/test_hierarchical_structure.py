"""
分层结构测试
Hierarchical Structure Tests
"""

import unittest
import numpy as np
from typing import Dict, Any

from BrainSimulationSystem.core.hierarchical_structure import (
    StructuralHierarchy, StructuralLevel, NeuronDensity, ConnectionDensity,
    create_hierarchical_structure
)

class TestHierarchicalStructure(unittest.TestCase):
    """分层结构测试类"""
    
    def setUp(self):
        """设置测试环境"""
        
        self.config = {
            'total_neurons': 1000000,
            'brain_regions': [
                {
                    'name': 'primary_visual_cortex',
                    'neurons': 500000,
                    'volume': 2000.0,
                    'subregions': [
                        {'name': 'V1_L1', 'neurons': 25000, 'type': 'molecular_layer'},
                        {'name': 'V1_L2_3', 'neurons': 175000, 'type': 'supragranular'},
                        {'name': 'V1_L4', 'neurons': 125000, 'type': 'granular'},
                        {'name': 'V1_L5', 'neurons': 125000, 'type': 'infragranular'},
                        {'name': 'V1_L6', 'neurons': 50000, 'type': 'deep_layer'}
                    ]
                },
                {
                    'name': 'thalamus',
                    'neurons': 300000,
                    'volume': 1000.0
                }
            ],
            'columns_per_subregion': 5,
            'microcircuits_per_column': 3
        }
        
        self.hierarchy = StructuralHierarchy(self.config)
    
    def test_hierarchy_creation(self):
        """测试层级结构创建"""
        
        # 检查全脑级别
        whole_brain = self.hierarchy.hierarchy[StructuralLevel.WHOLE_BRAIN]
        self.assertEqual(whole_brain['total_neurons'], 1000000)
        self.assertEqual(len(whole_brain['regions']), 2)
        
        # 检查脑区
        regions = whole_brain['regions']
        v1_region = next(r for r in regions if r['name'] == 'primary_visual_cortex')
        self.assertEqual(v1_region['total_neurons'], 500000)
        self.assertEqual(len(v1_region['subregions']), 5)
        
        # 检查子区域
        v1_l4 = next(sr for sr in v1_region['subregions'] if sr['name'] == 'V1_L4')
        self.assertEqual(v1_l4['total_neurons'], 125000)
        self.assertEqual(len(v1_l4['columns']), 5)
        
        # 检查皮层柱
        column = v1_l4['columns'][0]
        self.assertEqual(len(column['microcircuits']), 3)
        
        # 检查微回路
        microcircuit = column['microcircuits'][0]
        self.assertGreater(len(microcircuit['layers']), 0)
    
    def test_neuron_density(self):
        """测试神经元密度参数"""
        
        density = NeuronDensity(total_neurons=1000)
        
        # 检查默认比例
        self.assertEqual(density.excitatory_ratio, 0.8)
        self.assertEqual(density.inhibitory_ratio, 0.2)
        
        # 检查层分布
        layer_sum = sum(density.layer_distribution.values())
        self.assertAlmostEqual(layer_sum, 1.0, places=2)
        
        # 检查中间神经元亚型
        interneuron_sum = sum(density.interneuron_subtypes.values())
        self.assertAlmostEqual(interneuron_sum, 1.0, places=2)
    
    def test_connection_density(self):
        """测试连接密度参数"""
        
        density = ConnectionDensity()
        
        # 检查默认连接概率
        self.assertEqual(density.local_connection_prob, 0.1)
        self.assertEqual(density.long_range_prob, 0.01)
        
        # 检查层间连接
        self.assertIn(('L4', 'L2_3'), density.interlayer_connections)
        self.assertGreater(density.interlayer_connections[('L4', 'L2_3')], 0)
    
    def test_structure_retrieval(self):
        """测试结构检索"""
        
        # 获取所有区域
        regions = self.hierarchy.get_structure_at_level(StructuralLevel.REGION)
        self.assertEqual(len(regions), 2)
        
        # 获取所有柱
        columns = self.hierarchy.get_structure_at_level(StructuralLevel.COLUMN)
        expected_columns = 5 * 5 + 5 * 2  # V1: 5子区域×5柱 + 丘脑: 2子区域×5柱
        self.assertEqual(len(columns), expected_columns)
        
        # 获取所有微柱
        minicolumns = self.hierarchy.get_structure_at_level(StructuralLevel.MINICOLUMN)
        self.assertGreater(len(minicolumns), 0)
    
    def test_connection_probability_calculation(self):
        """测试连接概率计算"""
        
        # 获取两个微柱
        minicolumns = self.hierarchy.get_structure_at_level(StructuralLevel.MINICOLUMN)
        self.assertGreaterEqual(len(minicolumns), 2)
        
        mc1, mc2 = minicolumns[0], minicolumns[1]
        
        # 计算连接概率
        prob = self.hierarchy.calculate_connection_probability(mc1, mc2)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
        # 同一微柱内应该有更高的连接概率
        if mc1['name'] == mc2['name']:
            self.assertGreater(prob, 0.1)
    
    def test_neuron_allocation(self):
        """测试神经元分配"""
        
        allocation = self.hierarchy.get_neuron_allocation()
        
        # 检查分配结果
        self.assertGreater(len(allocation), 0)
        
        # 检查每个微柱的分配
        for minicolumn_name, alloc in allocation.items():
            self.assertIn('pyramidal', alloc)
            self.assertIn('pv', alloc)
            self.assertIn('sst', alloc)
            self.assertIn('neuron_id_range', alloc)
            
            # 检查神经元ID范围
            start_id, end_id = alloc['neuron_id_range']
            self.assertLess(start_id, end_id)
    
    def test_connection_matrix_template(self):
        """测试连接矩阵模板"""
        
        template = self.hierarchy.get_connection_matrix_template()
        
        # 检查模板结构
        self.assertIn('total_neurons', template)
        self.assertIn('minicolumns', template)
        self.assertIn('connection_probabilities', template)
        self.assertIn('estimated_synapses', template)
        
        # 检查神经元总数
        self.assertGreater(template['total_neurons'], 0)
        
        # 检查连接概率
        self.assertIsInstance(template['connection_probabilities'], dict)
        
        # 检查突触估算
        self.assertGreaterEqual(template['estimated_synapses'], 0)
    
    def test_structure_summary_export(self):
        """测试结构摘要导出"""
        
        summary = self.hierarchy.export_structure_summary()
        
        # 检查摘要结构
        self.assertIn('hierarchy_levels', summary)
        self.assertIn('total_statistics', summary)
        self.assertIn('allocation_summary', summary)
        self.assertIn('connection_template', summary)
        
        # 检查层级统计
        levels = summary['hierarchy_levels']
        self.assertIn('whole_brain', levels)
        self.assertIn('region', levels)
        self.assertIn('column', levels)
        
        # 检查总体统计
        total_stats = summary['total_statistics']
        self.assertIn('total_neurons', total_stats)
        self.assertIn('total_regions', total_stats)
        self.assertIn('estimated_synapses', total_stats)
    
    def test_factory_function(self):
        """测试工厂函数"""
        
        hierarchy = create_hierarchical_structure(self.config)
        self.assertIsInstance(hierarchy, StructuralHierarchy)
        
        # 检查配置是否正确应用
        whole_brain = hierarchy.hierarchy[StructuralLevel.WHOLE_BRAIN]
        self.assertEqual(whole_brain['total_neurons'], self.config['total_neurons'])

class TestNeuronDensityParameters(unittest.TestCase):
    """神经元密度参数测试"""
    
    def test_default_parameters(self):
        """测试默认参数"""
        
        density = NeuronDensity(total_neurons=1000)
        
        # 检查基本参数
        self.assertEqual(density.total_neurons, 1000)
        self.assertEqual(density.excitatory_ratio + density.inhibitory_ratio, 1.0)
        
        # 检查层分布总和
        layer_sum = sum(density.layer_distribution.values())
        self.assertAlmostEqual(layer_sum, 1.0, places=2)
        
        # 检查中间神经元亚型总和
        interneuron_sum = sum(density.interneuron_subtypes.values())
        self.assertAlmostEqual(interneuron_sum, 1.0, places=2)
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        
        custom_layer_dist = {
            'L1': 0.1,
            'L2_3': 0.3,
            'L4': 0.2,
            'L5': 0.3,
            'L6': 0.1
        }
        
        density = NeuronDensity(
            total_neurons=2000,
            excitatory_ratio=0.85,
            layer_distribution=custom_layer_dist
        )
        
        self.assertEqual(density.total_neurons, 2000)
        self.assertEqual(density.excitatory_ratio, 0.85)
        self.assertEqual(density.inhibitory_ratio, 0.15)
        self.assertEqual(density.layer_distribution, custom_layer_dist)

if __name__ == '__main__':
    unittest.main()