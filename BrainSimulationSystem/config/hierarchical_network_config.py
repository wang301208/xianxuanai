"""
分层神经网络配置
Hierarchical Neural Network Configuration

完整的分层神经网络架构配置文件
包含结构层、细胞层、连接层的所有参数
"""

from typing import Dict, Any, List
from BrainSimulationSystem.core.multi_neuron_models import NeuronType
from BrainSimulationSystem.core.enhanced_connectivity import ConnectionType, DelayType

# 全局配置
GLOBAL_CONFIG = {
    'simulation': {
        'dt': 0.1,  # ms
        'duration': 1000.0,  # ms
        'seed': 42
    },
    'logging': {
        'level': 'INFO',
        'file': 'hierarchical_network.log'
    }
}

# 结构层配置
STRUCTURAL_CONFIG = {
    'total_neurons': 100000,  # 总神经元数量
    
    # 脑区定义
    'brain_regions': [
        {
            'name': 'primary_visual_cortex',
            'neurons': 50000,
            'volume': 5000.0,  # mm³
            'position': (0.0, 0.0, 0.0),
            'subregions': [
                {
                    'name': 'V1_L1',
                    'neurons': 2500,
                    'type': 'molecular_layer',
                    'thickness': 150.0
                },
                {
                    'name': 'V1_L2_3',
                    'neurons': 17500,
                    'type': 'supragranular',
                    'thickness': 400.0
                },
                {
                    'name': 'V1_L4',
                    'neurons': 12500,
                    'type': 'granular',
                    'thickness': 300.0
                },
                {
                    'name': 'V1_L5',
                    'neurons': 12500,
                    'type': 'infragranular',
                    'thickness': 600.0
                },
                {
                    'name': 'V1_L6',
                    'neurons': 5000,
                    'type': 'deep_layer',
                    'thickness': 550.0
                }
            ]
        },
        {
            'name': 'lateral_geniculate_nucleus',
            'neurons': 20000,
            'volume': 800.0,
            'position': (0.0, -15.0, -10.0),
            'subregions': [
                {
                    'name': 'LGN_magnocellular',
                    'neurons': 8000,
                    'type': 'magno_pathway'
                },
                {
                    'name': 'LGN_parvocellular',
                    'neurons': 12000,
                    'type': 'parvo_pathway'
                }
            ]
        },
        {
            'name': 'superior_colliculus',
            'neurons': 15000,
            'volume': 600.0,
            'position': (0.0, -20.0, -5.0),
            'subregions': [
                {
                    'name': 'SC_superficial',
                    'neurons': 9000,
                    'type': 'visual_layer'
                },
                {
                    'name': 'SC_deep',
                    'neurons': 6000,
                    'type': 'motor_layer'
                }
            ]
        },
        {
            'name': 'hippocampus',
            'neurons': 15000,
            'volume': 400.0,
            'position': (-10.0, -15.0, -8.0),
            'subregions': [
                {
                    'name': 'CA1',
                    'neurons': 6000,
                    'type': 'pyramidal_field'
                },
                {
                    'name': 'CA3',
                    'neurons': 3000,
                    'type': 'pyramidal_field'
                },
                {
                    'name': 'DG',
                    'neurons': 6000,
                    'type': 'granule_field'
                }
            ]
        }
    ],
    
    # 层级结构参数
    'columns_per_subregion': 10,
    'microcircuits_per_column': 5,
    'minicolumns_per_layer': 20,
    
    # 神经元密度参数
    'neuron_density': {
        'excitatory_ratio': 0.8,
        'inhibitory_ratio': 0.2,
        'pyramidal_ratio': 0.7,
        'interneuron_subtypes': {
            'pv': 0.4,      # Parvalbumin+
            'sst': 0.3,     # Somatostatin+
            'vip': 0.2,     # VIP+
            'other': 0.1
        },
        'layer_distribution': {
            'L1': 0.05,
            'L2_3': 0.35,
            'L4': 0.25,
            'L5': 0.25,
            'L6': 0.10
        }
    },
    
    # 连接密度参数
    'connection_density': {
        'local_connection_prob': 0.1,
        'interlayer_connections': {
            ('L4', 'L2_3'): 0.4,
            ('L2_3', 'L5'): 0.3,
            ('L4', 'L5'): 0.2,
            ('L5', 'L6'): 0.3,
            ('L6', 'L4'): 0.2,
            ('L5', 'L2_3'): 0.2,
            ('L2_3', 'L1'): 0.3
        },
        'long_range_prob': 0.01,
        'distance_decay': 0.1
    }
}

# 细胞层配置
CELLULAR_CONFIG = {
    # 神经元类型分布
    'neuron_type_distribution': {
        'V1_L2_3': {
            NeuronType.PYRAMIDAL_L23: 0.7,
            NeuronType.PV_INTERNEURON: 0.12,
            NeuronType.SST_INTERNEURON: 0.09,
            NeuronType.VIP_INTERNEURON: 0.06,
            NeuronType.ADAPTIVE_EXPONENTIAL: 0.03  # 其他兴奋性
        },
        'V1_L4': {
            NeuronType.ADAPTIVE_EXPONENTIAL: 0.75,  # 星形细胞
            NeuronType.PV_INTERNEURON: 0.15,
            NeuronType.SST_INTERNEURON: 0.10
        },
        'V1_L5': {
            NeuronType.PYRAMIDAL_L5A: 0.4,
            NeuronType.PYRAMIDAL_L5B: 0.3,
            NeuronType.PV_INTERNEURON: 0.15,
            NeuronType.SST_INTERNEURON: 0.15
        },
        'V1_L6': {
            NeuronType.PYRAMIDAL_L6: 0.8,
            NeuronType.PV_INTERNEURON: 0.1,
            NeuronType.SST_INTERNEURON: 0.1
        },
        'LGN': {
            NeuronType.HODGKIN_HUXLEY: 0.9,  # 丘脑中继神经元
            NeuronType.PV_INTERNEURON: 0.1   # 丘脑中间神经元
        },
        'hippocampus': {
            NeuronType.MULTI_COMPARTMENT: 0.8,  # 海马锥体细胞
            NeuronType.IZHIKEVICH: 0.2          # 海马中间神经元
        }
    },
    
    # 神经元参数
    'neuron_parameters': {
        NeuronType.LIF: {
            'tau_m': 20.0,
            'V_rest': -70.0,
            'V_thresh': -50.0,
            'V_reset': -70.0,
            't_ref': 2.0,
            'R_m': 10.0
        },
        NeuronType.HODGKIN_HUXLEY: {
            'C_m': 1.0,
            'g_Na': 120.0,
            'g_K': 36.0,
            'g_L': 0.3,
            'E_Na': 50.0,
            'E_K': -77.0,
            'E_L': -54.4
        },
        NeuronType.ADAPTIVE_EXPONENTIAL: {
            'C': 281.0,
            'g_L': 30.0,
            'E_L': -70.6,
            'V_T': -50.4,
            'Delta_T': 2.0,
            'a': 4.0,
            'tau_w': 144.0,
            'b': 80.5,
            'V_thresh': 0.0,
            'V_reset': -70.6
        },
        NeuronType.IZHIKEVICH: {
            'a': 0.02,
            'b': 0.2,
            'c': -65.0,
            'd': 8.0,
            'V_thresh': 30.0
        },
        NeuronType.MULTI_COMPARTMENT: {
            'morphology': {
                'soma': {
                    'length': 20.0,
                    'diameter': 20.0,
                    'Na_density': 120.0,
                    'K_density': 36.0
                },
                'basal_dendrite': {
                    'length': 300.0,
                    'diameter': 2.0,
                    'Na_density': 20.0,
                    'K_density': 10.0
                },
                'apical_dendrite': {
                    'length': 600.0,
                    'diameter': 3.0,
                    'Na_density': 15.0,
                    'K_density': 8.0,
                    'Ca_density': 5.0
                },
                'axon': {
                    'length': 1000.0,
                    'diameter': 1.0,
                    'Na_density': 500.0,
                    'K_density': 200.0
                }
            }
        },
        # 特化细胞类型参数
        NeuronType.PYRAMIDAL_L23: {
            'C': 150.0,
            'g_L': 10.0,
            'E_L': -70.0,
            'V_T': -50.0,
            'Delta_T': 2.0,
            'a': 2.0,
            'tau_w': 300.0,
            'b': 60.0
        },
        NeuronType.PYRAMIDAL_L5A: {
            'C': 200.0,
            'g_L': 12.0,
            'E_L': -70.0,
            'V_T': -52.0,
            'Delta_T': 2.5,
            'a': 4.0,
            'tau_w': 200.0,
            'b': 100.0
        },
        NeuronType.PYRAMIDAL_L5B: {
            'C': 250.0,
            'g_L': 15.0,
            'E_L': -70.0,
            'V_T': -50.0,
            'Delta_T': 3.0,
            'a': 6.0,
            'tau_w': 150.0,
            'b': 150.0
        },
        NeuronType.PV_INTERNEURON: {
            'a': 0.1,
            'b': 0.2,
            'c': -65.0,
            'd': 2.0,
            'V_thresh': 30.0
        },
        NeuronType.SST_INTERNEURON: {
            'a': 0.02,
            'b': 0.25,
            'c': -65.0,
            'd': 0.05,
            'V_thresh': 30.0
        },
        NeuronType.VIP_INTERNEURON: {
            'a': 0.1,
            'b': 0.2,
            'c': -50.0,
            'd': 2.0,
            'V_thresh': 30.0
        }
    },
    
    # 胶质细胞配置
    'glial_cells': {
        'astrocyte_ratio': 0.1,  # 相对于神经元的比例
        'microglia_ratio': 0.05,
        'oligodendrocyte_ratio': 0.03,
        
        'astrocyte_parameters': {
            'territory_radius': 50.0,
            'Ca_threshold': 0.5,
            'glucose_supply_rate': 0.1
        },
        
        'microglia_parameters': {
            'territory_radius': 30.0,
            'activation_threshold': 0.3,
            'surveillance_speed': 1.0
        }
    },
    
    # 神经调质系统
    'neuromodulation': {
        'systems': ['dopamine', 'serotonin', 'acetylcholine', 'norepinephrine'],
        'release_rates': {
            'dopamine': 0.01,
            'serotonin': 0.005,
            'acetylcholine': 0.02,
            'norepinephrine': 0.01
        },
        'clearance_rates': {
            'dopamine': 0.1,
            'serotonin': 0.05,
            'acetylcholine': 0.5,
            'norepinephrine': 0.08
        },
        'target_regions': {
            'dopamine': ['V1_L5', 'hippocampus'],
            'serotonin': ['V1_L2_3', 'V1_L5'],
            'acetylcholine': ['V1_L4', 'LGN'],
            'norepinephrine': ['V1_L1', 'V1_L6']
        }
    }
}

# 连接层配置
CONNECTIVITY_CONFIG = {
    # 稀疏矩阵配置
    'sparse_matrix': {
        'dtype': 'float32',
        'format': 'csr',
        'memory_limit_gb': 8.0
    },
    
    # 图数据库配置
    'graph_database': {
        'enabled': False,  # 默认禁用
        'uri': 'bolt://localhost:7687',
        'username': 'neo4j',
        'password': 'password'
    },
    
    # 概率连接器配置
    'connector': {
        'seed': 42,
        'batch_size': 1000
    },
    
    # 连接类型参数
    'connection_parameters': {
        # 局部连接
        ConnectionType.LOCAL: {
            'probability': 0.1,
            'weight_mean': 1.0,
            'weight_std': 0.3,
            'delay_mean': 1.0,
            'delay_std': 0.2,
            'delay_type': DelayType.GAUSSIAN,
            'plasticity_enabled': True,
            'learning_rate': 0.01
        },
        
        # 层间连接
        ConnectionType.INTERLAYER: {
            'probability': 0.05,
            'weight_mean': 0.8,
            'weight_std': 0.2,
            'delay_mean': 2.0,
            'delay_std': 0.5,
            'delay_type': DelayType.GAUSSIAN,
            'plasticity_enabled': True,
            'learning_rate': 0.005
        },
        
        # 柱间连接
        ConnectionType.INTERCOLUMN: {
            'probability': 0.02,
            'weight_mean': 0.5,
            'weight_std': 0.15,
            'delay_mean': 3.0,
            'delay_std': 1.0,
            'delay_type': DelayType.DISTANCE_DEPENDENT,
            'distance_decay': 0.1,
            'max_distance': 500.0
        },
        
        # 区域间连接
        ConnectionType.INTERREGION: {
            'probability': 0.005,
            'weight_mean': 0.3,
            'weight_std': 0.1,
            'delay_mean': 10.0,
            'delay_std': 3.0,
            'delay_type': DelayType.DISTANCE_DEPENDENT,
            'conduction_velocity': 2.0,
            'myelination': True
        },
        
        # 长程连接
        ConnectionType.LONG_RANGE: {
            'probability': 0.001,
            'weight_mean': 0.2,
            'weight_std': 0.05,
            'delay_mean': 20.0,
            'delay_std': 5.0,
            'delay_type': DelayType.DISTANCE_DEPENDENT,
            'conduction_velocity': 5.0,
            'myelination': True
        },
        
        # 反馈连接
        ConnectionType.FEEDBACK: {
            'probability': 0.03,
            'weight_mean': -0.5,  # 抑制性
            'weight_std': 0.1,
            'delay_mean': 5.0,
            'delay_std': 1.5,
            'delay_type': DelayType.GAUSSIAN
        },
        
        # 前馈连接
        ConnectionType.FEEDFORWARD: {
            'probability': 0.08,
            'weight_mean': 1.2,
            'weight_std': 0.4,
            'delay_mean': 2.5,
            'delay_std': 0.8,
            'delay_type': DelayType.GAUSSIAN
        }
    },
    
    # 区域间连接矩阵
    'regional_connections': {
        ('LGN_parvocellular', 'V1_L4'): {
            'strength': 0.9,
            'probability': 0.3,
            'delay': 3.0,
            'connection_type': ConnectionType.FEEDFORWARD
        },
        ('LGN_magnocellular', 'V1_L4'): {
            'strength': 0.8,
            'probability': 0.25,
            'delay': 2.5,
            'connection_type': ConnectionType.FEEDFORWARD
        },
        ('V1_L6', 'LGN_parvocellular'): {
            'strength': 0.4,
            'probability': 0.1,
            'delay': 8.0,
            'connection_type': ConnectionType.FEEDBACK
        },
        ('V1_L5', 'SC_superficial'): {
            'strength': 0.6,
            'probability': 0.15,
            'delay': 12.0,
            'connection_type': ConnectionType.FEEDFORWARD
        },
        ('V1_L2_3', 'hippocampus'): {
            'strength': 0.3,
            'probability': 0.05,
            'delay': 25.0,
            'connection_type': ConnectionType.LONG_RANGE
        }
    },
    
    # 轴突参数
    'axon_parameters': {
        'local_axon': {
            'length': 500.0,
            'diameter': 0.5,
            'conduction_velocity': 0.5,
            'myelination': False,
            'nodes_of_ranvier': 0
        },
        'projection_axon': {
            'length': 5000.0,
            'diameter': 2.0,
            'conduction_velocity': 2.0,
            'myelination': True,
            'nodes_of_ranvier': 25
        },
        'long_range_axon': {
            'length': 20000.0,
            'diameter': 5.0,
            'conduction_velocity': 10.0,
            'myelination': True,
            'nodes_of_ranvier': 100
        }
    }
}

# 完整配置
HIERARCHICAL_NETWORK_CONFIG = {
    'global': GLOBAL_CONFIG,
    'structure': STRUCTURAL_CONFIG,
    'cellular': CELLULAR_CONFIG,
    'connectivity': CONNECTIVITY_CONFIG
}

def get_config() -> Dict[str, Any]:
    """获取完整的分层网络配置"""
    return HIERARCHICAL_NETWORK_CONFIG

def get_structure_config() -> Dict[str, Any]:
    """获取结构层配置"""
    return STRUCTURAL_CONFIG

def get_cellular_config() -> Dict[str, Any]:
    """获取细胞层配置"""
    return CELLULAR_CONFIG

def get_connectivity_config() -> Dict[str, Any]:
    """获取连接层配置"""
    return CONNECTIVITY_CONFIG

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置完整性"""
    
    required_sections = ['global', 'structure', 'cellular', 'connectivity']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"缺少配置节: {section}")
    
    # 验证神经元总数一致性
    structure_config = config['structure']
    total_neurons = structure_config['total_neurons']
    
    region_neurons = sum(
        region['neurons'] for region in structure_config['brain_regions']
    )
    
    if region_neurons != total_neurons:
        raise ValueError(f"神经元总数不一致: {total_neurons} vs {region_neurons}")
    
    # 验证神经元类型分布
    cellular_config = config['cellular']
    for region_name, distribution in cellular_config['neuron_type_distribution'].items():
        total_ratio = sum(distribution.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"区域 {region_name} 神经元类型分布总和不为1: {total_ratio}")
    
    return True

# 配置验证
if __name__ == '__main__':
    config = get_config()
    try:
        validate_config(config)
        print("配置验证通过")
        
        # 打印配置摘要
        print(f"总神经元数: {config['structure']['total_neurons']}")
        print(f"脑区数量: {len(config['structure']['brain_regions'])}")
        print(f"神经元类型数: {len(config['cellular']['neuron_parameters'])}")
        print(f"连接类型数: {len(config['connectivity']['connection_parameters'])}")
        
    except ValueError as e:
        print(f"配置验证失败: {e}")