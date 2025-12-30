"""
分层神经网络结构实现
Hierarchical Neural Network Structure Implementation

建立分层图谱：区域→子区→柱/微回路
确定每层神经元数量与连接密度
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

class StructuralLevel(Enum):
    """结构层级枚举"""
    WHOLE_BRAIN = "whole_brain"
    REGION = "region"
    SUBREGION = "subregion"
    COLUMN = "column"
    MICROCIRCUIT = "microcircuit"
    LAYER = "layer"
    MINICOLUMN = "minicolumn"

@dataclass
class NeuronDensity:
    """神经元密度参数"""
    total_neurons: int
    excitatory_ratio: float = 0.8
    inhibitory_ratio: Optional[float] = None
    
    # 细胞类型分布
    pyramidal_ratio: float = 0.7
    interneuron_subtypes: Dict[str, float] = field(default_factory=lambda: {
        'pv': 0.4,      # Parvalbumin+
        'sst': 0.3,     # Somatostatin+
        'vip': 0.2,     # VIP+
        'other': 0.1    # 其他类型
    })
    
    # 层特异性密度
    layer_distribution: Dict[str, float] = field(default_factory=lambda: {
        'L1': 0.05,
        'L2_3': 0.35,
        'L4': 0.25,
        'L5': 0.25,
        'L6': 0.10
    })

    def __post_init__(self):
        # 默认按兴奋性比例推导抑制性比例，避免浮点误差影响断言
        if self.inhibitory_ratio is None:
            self.inhibitory_ratio = round(1.0 - float(self.excitatory_ratio), 10)

@dataclass
class ConnectionDensity:
    """连接密度参数"""
    # 局部连接概率
    local_connection_prob: float = 0.1
    
    # 层间连接概率矩阵
    interlayer_connections: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        ('L4', 'L2_3'): 0.4,
        ('L2_3', 'L5'): 0.3,
        ('L4', 'L5'): 0.2,
        ('L5', 'L6'): 0.3,
        ('L6', 'L4'): 0.2,
        ('L5', 'L2_3'): 0.2,
        ('L2_3', 'L1'): 0.3
    })
    
    # 长程连接概率
    long_range_prob: float = 0.01
    
    # 连接距离衰减参数
    distance_decay: float = 0.1  # 每μm的衰减率

class StructuralHierarchy:
    """结构层级管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hierarchy = {}
        self.logger = logging.getLogger("StructuralHierarchy")
        
        # 初始化层级结构
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """构建层级结构"""
        
        # 全脑级别
        self.hierarchy[StructuralLevel.WHOLE_BRAIN] = {
            'name': 'whole_brain',
            'total_neurons': self.config.get('total_neurons', 86_000_000_000),
            'regions': []
        }
        
        # 脑区级别
        brain_regions = self.config.get('brain_regions', [])
        for region_config in brain_regions:
            region = self._create_brain_region(region_config)
            self.hierarchy[StructuralLevel.WHOLE_BRAIN]['regions'].append(region)
    
    def _create_brain_region(self, region_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建脑区结构"""
        
        region_name = region_config['name']
        total_neurons = region_config.get('neurons', 1_000_000)
        
        region = {
            'name': region_name,
            'level': StructuralLevel.REGION,
            'total_neurons': total_neurons,
            'volume': region_config.get('volume', 1000.0),  # mm³
            'subregions': [],
            'density': NeuronDensity(total_neurons),
            'connectivity': ConnectionDensity()
        }
        
        # 创建子区域
        subregions = region_config.get('subregions', [])
        if not subregions:
            # 默认创建皮层子区域
            subregions = self._create_default_subregions(region_name, total_neurons)
        
        for subregion_config in subregions:
            subregion = self._create_subregion(subregion_config, region)
            region['subregions'].append(subregion)
        
        return region
    
    def _create_default_subregions(self, region_name: str, total_neurons: int) -> List[Dict[str, Any]]:
        """创建默认子区域配置"""
        
        if 'cortex' in region_name.lower() or 'cortical' in region_name.lower():
            # 皮层区域：按层分布
            return [
                {
                    'name': f'{region_name}_L1',
                    'neurons': int(total_neurons * 0.05),
                    'type': 'molecular_layer'
                },
                {
                    'name': f'{region_name}_L2_3',
                    'neurons': int(total_neurons * 0.35),
                    'type': 'supragranular'
                },
                {
                    'name': f'{region_name}_L4',
                    'neurons': int(total_neurons * 0.25),
                    'type': 'granular'
                },
                {
                    'name': f'{region_name}_L5',
                    'neurons': int(total_neurons * 0.25),
                    'type': 'infragranular'
                },
                {
                    'name': f'{region_name}_L6',
                    'neurons': int(total_neurons * 0.10),
                    'type': 'deep_layer'
                }
            ]
        else:
            # 非皮层区域：按功能分布
            return [
                {
                    'name': f'{region_name}_core',
                    'neurons': int(total_neurons * 0.6),
                    'type': 'core'
                },
                {
                    'name': f'{region_name}_shell',
                    'neurons': int(total_neurons * 0.4),
                    'type': 'shell'
                }
            ]
    
    def _create_subregion(self, subregion_config: Dict[str, Any], parent_region: Dict[str, Any]) -> Dict[str, Any]:
        """创建子区域结构"""
        
        subregion_name = subregion_config['name']
        neurons = subregion_config.get('neurons', 100_000)
        
        subregion = {
            'name': subregion_name,
            'level': StructuralLevel.SUBREGION,
            'parent_region': parent_region['name'],
            'total_neurons': neurons,
            'type': subregion_config.get('type', 'generic'),
            'columns': [],
            'density': NeuronDensity(neurons),
            'connectivity': ConnectionDensity()
        }
        
        # 创建皮层柱
        columns_per_subregion = self.config.get('columns_per_subregion', 10)
        neurons_per_column = neurons // columns_per_subregion
        
        for i in range(columns_per_subregion):
            column = self._create_cortical_column(
                f'{subregion_name}_col_{i}',
                neurons_per_column,
                subregion
            )
            subregion['columns'].append(column)
        
        return subregion
    
    def _create_cortical_column(self, column_name: str, neurons: int, parent_subregion: Dict[str, Any]) -> Dict[str, Any]:
        """创建皮层柱结构"""
        
        column = {
            'name': column_name,
            'level': StructuralLevel.COLUMN,
            'parent_subregion': parent_subregion['name'],
            'total_neurons': neurons,
            'diameter': 50.0,  # μm
            'height': 2000.0,  # μm
            'microcircuits': [],
            'density': NeuronDensity(neurons),
            'connectivity': ConnectionDensity(),
            'position': self._generate_column_position()
        }
        
        # 创建微回路
        microcircuits_per_column = self.config.get('microcircuits_per_column', 5)
        neurons_per_microcircuit = neurons // microcircuits_per_column
        
        for i in range(microcircuits_per_column):
            microcircuit = self._create_microcircuit(
                f'{column_name}_mc_{i}',
                neurons_per_microcircuit,
                column
            )
            column['microcircuits'].append(microcircuit)
        
        return column
    
    def _create_microcircuit(self, microcircuit_name: str, neurons: int, parent_column: Dict[str, Any]) -> Dict[str, Any]:
        """创建微回路结构"""
        
        microcircuit = {
            'name': microcircuit_name,
            'level': StructuralLevel.MICROCIRCUIT,
            'parent_column': parent_column['name'],
            'total_neurons': neurons,
            'layers': [],
            'density': NeuronDensity(neurons),
            'connectivity': ConnectionDensity(),
            'functional_type': self._determine_microcircuit_type()
        }
        
        # 创建层结构
        layer_distribution = microcircuit['density'].layer_distribution
        
        for layer_name, proportion in layer_distribution.items():
            layer_neurons = int(neurons * proportion)
            if layer_neurons > 0:
                layer = self._create_layer(
                    f'{microcircuit_name}_{layer_name}',
                    layer_neurons,
                    layer_name,
                    microcircuit
                )
                microcircuit['layers'].append(layer)
        
        return microcircuit
    
    def _create_layer(self, layer_name: str, neurons: int, layer_type: str, parent_microcircuit: Dict[str, Any]) -> Dict[str, Any]:
        """创建层结构"""
        
        layer = {
            'name': layer_name,
            'level': StructuralLevel.LAYER,
            'parent_microcircuit': parent_microcircuit['name'],
            'layer_type': layer_type,
            'total_neurons': neurons,
            'thickness': self._get_layer_thickness(layer_type),
            'minicolumns': [],
            'density': NeuronDensity(neurons),
            'connectivity': ConnectionDensity()
        }
        
        # 创建微柱
        minicolumns_per_layer = max(1, neurons // 100)  # 每个微柱约100个神经元
        neurons_per_minicolumn = neurons // minicolumns_per_layer
        
        for i in range(minicolumns_per_layer):
            minicolumn = self._create_minicolumn(
                f'{layer_name}_mini_{i}',
                neurons_per_minicolumn,
                layer
            )
            layer['minicolumns'].append(minicolumn)
        
        return layer
    
    def _create_minicolumn(self, minicolumn_name: str, neurons: int, parent_layer: Dict[str, Any]) -> Dict[str, Any]:
        """创建微柱结构"""
        
        return {
            'name': minicolumn_name,
            'level': StructuralLevel.MINICOLUMN,
            'parent_layer': parent_layer['name'],
            'total_neurons': neurons,
            'diameter': 30.0,  # μm
            'density': NeuronDensity(neurons),
            'connectivity': ConnectionDensity(),
            'neuron_ids': []  # 将在神经元创建时填充
        }
    
    def _generate_column_position(self) -> Tuple[float, float, float]:
        """生成皮层柱位置"""
        return (
            np.random.uniform(0, 5000),  # x (μm)
            np.random.uniform(0, 5000),  # y (μm)
            0.0                          # z (μm, 皮层表面)
        )
    
    def _determine_microcircuit_type(self) -> str:
        """确定微回路功能类型"""
        types = ['canonical', 'inhibitory', 'excitatory', 'mixed']
        weights = [0.4, 0.2, 0.2, 0.2]
        return np.random.choice(types, p=weights)
    
    def _get_layer_thickness(self, layer_type: str) -> float:
        """获取层厚度"""
        thickness_map = {
            'L1': 150.0,
            'L2_3': 400.0,
            'L4': 300.0,
            'L5': 600.0,
            'L6': 550.0
        }
        return thickness_map.get(layer_type, 300.0)
    
    def get_structure_at_level(self, level: StructuralLevel) -> List[Dict[str, Any]]:
        """获取指定层级的所有结构"""
        
        structures = []
        
        def traverse_hierarchy(node, current_level):
            if current_level == level:
                structures.append(node)
                return
            
            # 递归遍历子结构
            if 'regions' in node:
                for region in node['regions']:
                    traverse_hierarchy(region, StructuralLevel.REGION)
            elif 'subregions' in node:
                for subregion in node['subregions']:
                    traverse_hierarchy(subregion, StructuralLevel.SUBREGION)
            elif 'columns' in node:
                for column in node['columns']:
                    traverse_hierarchy(column, StructuralLevel.COLUMN)
            elif 'microcircuits' in node:
                for microcircuit in node['microcircuits']:
                    traverse_hierarchy(microcircuit, StructuralLevel.MICROCIRCUIT)
            elif 'layers' in node:
                for layer in node['layers']:
                    traverse_hierarchy(layer, StructuralLevel.LAYER)
            elif 'minicolumns' in node:
                for minicolumn in node['minicolumns']:
                    traverse_hierarchy(minicolumn, StructuralLevel.MINICOLUMN)
        
        traverse_hierarchy(self.hierarchy[StructuralLevel.WHOLE_BRAIN], StructuralLevel.WHOLE_BRAIN)
        return structures
    
    def calculate_connection_probability(self, source_structure: Dict[str, Any], 
                                       target_structure: Dict[str, Any]) -> float:
        """计算两个结构间的连接概率"""
        
        source_level = source_structure['level']
        target_level = target_structure['level']
        
        # 同层级连接
        if source_level == target_level:
            if source_level == StructuralLevel.MINICOLUMN:
                return 0.3  # 微柱内高连接
            elif source_level == StructuralLevel.LAYER:
                return 0.1  # 层内中等连接
            elif source_level == StructuralLevel.COLUMN:
                return 0.05  # 柱间低连接
            else:
                return 0.01  # 其他同级低连接
        
        # 跨层级连接
        if source_level == StructuralLevel.LAYER and target_level == StructuralLevel.LAYER:
            # 层间连接使用预定义概率
            source_type = source_structure.get('layer_type', '')
            target_type = target_structure.get('layer_type', '')
            
            connectivity = source_structure.get('connectivity', ConnectionDensity())
            return connectivity.interlayer_connections.get((source_type, target_type), 0.0)
        
        # 长程连接
        if (source_level in [StructuralLevel.REGION, StructuralLevel.SUBREGION] and
            target_level in [StructuralLevel.REGION, StructuralLevel.SUBREGION]):
            
            # 计算距离衰减
            if 'position' in source_structure and 'position' in target_structure:
                distance = self._calculate_distance(
                    source_structure['position'],
                    target_structure['position']
                )
                decay_factor = np.exp(-distance * 0.001)  # 距离衰减
                return 0.01 * decay_factor
        
        return 0.001  # 默认极低连接概率
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                           pos2: Tuple[float, float, float]) -> float:
        """计算3D距离"""
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    
    def get_neuron_allocation(self) -> Dict[str, Dict[str, int]]:
        """获取神经元分配方案"""
        
        allocation = {}
        
        # 遍历所有微柱，分配神经元ID
        minicolumns = self.get_structure_at_level(StructuralLevel.MINICOLUMN)
        
        neuron_id = 0
        for minicolumn in minicolumns:
            total_neurons = minicolumn['total_neurons']
            density = minicolumn['density']
            
            # 计算细胞类型分布
            excitatory_count = int(total_neurons * density.excitatory_ratio)
            inhibitory_count = total_neurons - excitatory_count
            
            # 分配兴奋性神经元
            pyramidal_count = int(excitatory_count * density.pyramidal_ratio)
            other_exc_count = excitatory_count - pyramidal_count
            
            # 分配抑制性神经元
            interneuron_counts = {}
            for subtype, ratio in density.interneuron_subtypes.items():
                count = int(inhibitory_count * ratio)
                interneuron_counts[subtype] = count
            
            # 记录分配
            allocation[minicolumn['name']] = {
                'pyramidal': pyramidal_count,
                'other_excitatory': other_exc_count,
                **interneuron_counts,
                'neuron_id_range': (neuron_id, neuron_id + total_neurons)
            }
            
            # 更新神经元ID范围
            minicolumn['neuron_ids'] = list(range(neuron_id, neuron_id + total_neurons))
            neuron_id += total_neurons
        
        return allocation
    
    def get_connection_matrix_template(self) -> Dict[str, Any]:
        """生成连接矩阵模板"""
        
        # 获取所有微柱
        minicolumns = self.get_structure_at_level(StructuralLevel.MINICOLUMN)
        
        # 计算总神经元数
        total_neurons = sum(mc['total_neurons'] for mc in minicolumns)
        
        # 生成连接概率矩阵
        connection_probabilities = {}
        
        for i, source_mc in enumerate(minicolumns):
            for j, target_mc in enumerate(minicolumns):
                if i != j:  # 不包括自连接
                    prob = self.calculate_connection_probability(source_mc, target_mc)
                    if prob > 0.001:  # 只记录有意义的连接
                        key = (source_mc['name'], target_mc['name'])
                        connection_probabilities[key] = prob
        
        return {
            'total_neurons': total_neurons,
            'minicolumns': minicolumns,
            'connection_probabilities': connection_probabilities,
            'estimated_synapses': self._estimate_total_synapses(connection_probabilities, minicolumns)
        }
    
    def _estimate_total_synapses(self, connection_probs: Dict[Tuple[str, str], float],
                                minicolumns: List[Dict[str, Any]]) -> int:
        """估算总突触数量"""
        
        total_synapses = 0
        
        # 创建微柱名称到神经元数的映射
        mc_neurons = {mc['name']: mc['total_neurons'] for mc in minicolumns}
        
        for (source_name, target_name), prob in connection_probs.items():
            source_neurons = mc_neurons[source_name]
            target_neurons = mc_neurons[target_name]
            
            # 估算连接数：源神经元数 × 目标神经元数 × 连接概率
            estimated_connections = int(source_neurons * target_neurons * prob)
            total_synapses += estimated_connections
        
        return total_synapses
    
    def export_structure_summary(self) -> Dict[str, Any]:
        """导出结构摘要"""
        
        summary = {
            'hierarchy_levels': {},
            'total_statistics': {},
            'allocation_summary': self.get_neuron_allocation(),
            'connection_template': self.get_connection_matrix_template()
        }
        
        # 统计各层级数量
        for level in StructuralLevel:
            structures = self.get_structure_at_level(level)
            summary['hierarchy_levels'][level.value] = {
                'count': len(structures),
                'total_neurons': sum(s.get('total_neurons', 0) for s in structures)
            }
        
        # 总体统计
        summary['total_statistics'] = {
            'total_neurons': summary['hierarchy_levels']['whole_brain']['total_neurons'],
            'total_regions': summary['hierarchy_levels']['region']['count'],
            'total_columns': summary['hierarchy_levels']['column']['count'],
            'total_minicolumns': summary['hierarchy_levels']['minicolumn']['count'],
            'estimated_synapses': summary['connection_template']['estimated_synapses']
        }
        
        return summary

def create_hierarchical_structure(config: Dict[str, Any]) -> StructuralHierarchy:
    """创建分层结构的工厂函数"""
    return StructuralHierarchy(config)
