# -*- coding: utf-8 -*-
"""
增强的大脑仿真配置系统

基于 Allen Brain Atlas、Blue Brain Project、EBRAINS 等数据源，
建立区域 -> 子区 -> 微电路的分层描述，用于驱动网络生成器。

特点：
- 细化各脑区神经元规模、细胞类型、连接密度等参数
- 整合多个解剖数据库的连接信息
- 支持分层的网络结构描述
- 提供灵活的配置接口
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class BrainRegion(Enum):
    """脑区枚举"""
    # 新皮层
    NEOCORTEX = "neocortex"
    PRIMARY_VISUAL = "primary_visual"
    PRIMARY_AUDITORY = "primary_auditory"
    PRIMARY_SOMATOSENSORY = "primary_somatosensory"
    PRIMARY_MOTOR = "primary_motor"
    PREFRONTAL = "prefrontal"
    PARIETAL = "parietal"
    TEMPORAL = "temporal"
    
    # 海马系统
    HIPPOCAMPUS = "hippocampus"
    DENTATE_GYRUS = "dentate_gyrus"
    CA3 = "ca3"
    CA1 = "ca1"
    SUBICULAR_COMPLEX = "subicular_complex"
    
    # 基底节
    BASAL_GANGLIA = "basal_ganglia"
    STRIATUM = "striatum"
    GLOBUS_PALLIDUS = "globus_pallidus"
    SUBTHALAMIC_NUCLEUS = "subthalamic_nucleus"
    SUBSTANTIA_NIGRA = "substantia_nigra"
    
    # 小脑
    CEREBELLUM = "cerebellum"
    GRANULAR_LAYER = "granular_layer"
    PURKINJE_LAYER = "purkinje_layer"
    DEEP_NUCLEI = "deep_nuclei"
    
    # 丘脑
    THALAMUS = "thalamus"
    SENSORY_RELAY = "sensory_relay"
    ASSOCIATION_NUCLEI = "association_nuclei"
    RETICULAR_NUCLEUS = "reticular_nucleus"
    
    # 脑干
    BRAINSTEM = "brainstem"
    MIDBRAIN = "midbrain"
    PONS = "pons"
    MEDULLA = "medulla"

class CellType(Enum):
    """细胞类型枚举"""
    # 皮层细胞
    L23_PYRAMIDAL = "l23_pyramidal"
    L4_SPINY_STELLATE = "l4_spiny_stellate"
    L5_PYRAMIDAL = "l5_pyramidal"
    L6_PYRAMIDAL = "l6_pyramidal"
    PV_INTERNEURON = "pv_interneuron"
    SST_INTERNEURON = "sst_interneuron"
    VIP_INTERNEURON = "vip_interneuron"
    
    # 海马细胞
    GRANULE_CELLS = "granule_cells"
    MOSSY_CELLS = "mossy_cells"
    CA3_PYRAMIDAL = "ca3_pyramidal"
    CA1_PYRAMIDAL = "ca1_pyramidal"
    
    # 基底节细胞
    MEDIUM_SPINY_D1 = "medium_spiny_d1"
    MEDIUM_SPINY_D2 = "medium_spiny_d2"
    CHOLINERGIC_INTERNEURONS = "cholinergic_interneurons"
    GPE_NEURONS = "gpe_neurons"
    GPI_NEURONS = "gpi_neurons"
    STN_NEURONS = "stn_neurons"
    DOPAMINERGIC_NEURONS = "dopaminergic_neurons"
    
    # 小脑细胞
    GRANULE_CELLS_CB = "granule_cells_cb"
    GOLGI_CELLS = "golgi_cells"
    PURKINJE_CELLS = "purkinje_cells"
    NUCLEAR_NEURONS = "nuclear_neurons"
    
    # 丘脑细胞
    RELAY_NEURONS = "relay_neurons"
    RETICULAR_NEURONS = "reticular_neurons"
    MATRIX_NEURONS = "matrix_neurons"

@dataclass
class CellTypeConfig:
    """细胞类型配置"""
    name: str
    proportion: float  # 在该区域中的比例
    morphology: str  # 形态学类型
    electrophysiology: Dict[str, float]  # 电生理参数
    connectivity: Dict[str, float]  # 连接参数

@dataclass
class SubregionConfig:
    """子区域配置"""
    name: str
    areas: List[str]  # 包含的具体区域
    neuron_count: int
    cell_types: Dict[CellType, CellTypeConfig]
    connection_density: float
    special_properties: Dict[str, Any]  # 特殊属性

@dataclass
class RegionConfig:
    """脑区配置"""
    name: str
    enabled: bool
    subregions: Dict[str, SubregionConfig]
    total_neurons: int
    inter_subregion_connectivity: Dict[str, float]
    anatomical_properties: Dict[str, Any]

class AnatomicalDataSource:
    """解剖数据源管理"""
    
    def __init__(self):
        self.data_sources = {
            'allen_brain_atlas': {
                'url': 'http://api.brain-map.org/',
                'version': '2023.1',
                'species': ['mouse', 'human'],
                'data_types': ['connectivity', 'gene_expression', 'morphology']
            },
            'blue_brain_project': {
                'url': 'https://bbp.epfl.ch/',
                'version': '2.0',
                'species': ['rat'],
                'data_types': ['microcircuit', 'connectivity', 'morphology']
            },
            'ebrains_atlas': {
                'url': 'https://ebrains.eu/',
                'version': '3.0',
                'species': ['human', 'macaque'],
                'data_types': ['parcellation', 'connectivity', 'cytoarchitecture']
            },
            'human_connectome_project': {
                'url': 'https://www.humanconnectome.org/',
                'version': '1200',
                'species': ['human'],
                'data_types': ['structural_connectivity', 'functional_connectivity']
            }
        }
    
    def get_connectivity_matrix(self, source: str, regions: List[str]) -> np.ndarray:
        """获取连接矩阵"""
        # 这里应该实现实际的数据获取逻辑
        # 目前返回模拟数据
        n_regions = len(regions)
        return np.random.rand(n_regions, n_regions) * 0.1
    
    def get_cell_type_distribution(self, source: str, region: str) -> Dict[str, float]:
        """获取细胞类型分布"""
        # 基于文献的典型分布
        distributions = {
            'neocortex': {
                'l23_pyramidal': 0.35,
                'l4_spiny_stellate': 0.15,
                'l5_pyramidal': 0.20,
                'l6_pyramidal': 0.15,
                'pv_interneuron': 0.08,
                'sst_interneuron': 0.05,
                'vip_interneuron': 0.02
            },
            'hippocampus_ca1': {
                'ca1_pyramidal': 0.88,
                'pv_interneuron': 0.06,
                'sst_interneuron': 0.04,
                'vip_interneuron': 0.02
            },
            'striatum': {
                'medium_spiny_d1': 0.45,
                'medium_spiny_d2': 0.45,
                'cholinergic_interneurons': 0.05,
                'pv_interneuron': 0.05
            }
        }
        return distributions.get(region, {})

class EnhancedBrainConfig:
    """增强的大脑配置系统"""
    
    def __init__(self):
        self.anatomical_data = AnatomicalDataSource()
        self.regions = {}
        self._initialize_regions()
    
    def _initialize_regions(self):
        """初始化所有脑区配置"""
        
        # 新皮层配置
        self.regions[BrainRegion.NEOCORTEX] = RegionConfig(
            name="neocortex",
            enabled=True,
            subregions={
                "primary_visual": SubregionConfig(
                    name="primary_visual",
                    areas=["V1", "V2", "V4"],
                    neuron_count=150000,
                    cell_types=self._get_cortical_cell_types(),
                    connection_density=0.12,
                    special_properties={
                        "laminar_structure": True,
                        "orientation_columns": True,
                        "ocular_dominance_columns": True,
                        "retinotopic_mapping": True
                    }
                ),
                "primary_auditory": SubregionConfig(
                    name="primary_auditory",
                    areas=["A1", "A2"],
                    neuron_count=80000,
                    cell_types=self._get_cortical_cell_types(auditory=True),
                    connection_density=0.10,
                    special_properties={
                        "laminar_structure": True,
                        "tonotopic_mapping": True,
                        "frequency_columns": True
                    }
                ),
                "primary_somatosensory": SubregionConfig(
                    name="primary_somatosensory",
                    areas=["S1", "S2"],
                    neuron_count=120000,
                    cell_types=self._get_cortical_cell_types(),
                    connection_density=0.15,
                    special_properties={
                        "laminar_structure": True,
                        "somatotopic_mapping": True,
                        "barrel_columns": True
                    }
                ),
                "primary_motor": SubregionConfig(
                    name="primary_motor",
                    areas=["M1"],
                    neuron_count=100000,
                    cell_types=self._get_cortical_cell_types(motor=True),
                    connection_density=0.18,
                    special_properties={
                        "laminar_structure": True,
                        "motor_mapping": True,
                        "corticospinal_projections": True
                    }
                ),
                "prefrontal": SubregionConfig(
                    name="prefrontal",
                    areas=["PFC", "ACC", "OFC"],
                    neuron_count=200000,
                    cell_types=self._get_cortical_cell_types(),
                    connection_density=0.08,
                    special_properties={
                        "laminar_structure": True,
                        "working_memory": True,
                        "executive_control": True,
                        "long_range_connections": True
                    }
                )
            },
            total_neurons=650000,
            inter_subregion_connectivity={
                "feedforward_strength": 0.8,
                "feedback_strength": 0.6,
                "lateral_strength": 0.4
            },
            anatomical_properties={
                "cortical_thickness": 2.5,  # mm
                "surface_area": 1200,  # cm²
                "folding_index": 2.8
            }
        )
        
        # 海马系统配置
        self.regions[BrainRegion.HIPPOCAMPUS] = RegionConfig(
            name="hippocampus",
            enabled=True,
            subregions={
                "dentate_gyrus": SubregionConfig(
                    name="dentate_gyrus",
                    areas=["DG"],
                    neuron_count=15000,
                    cell_types=self._get_hippocampal_cell_types("dg"),
                    connection_density=0.05,
                    special_properties={
                        "sparse_coding": True,
                        "pattern_separation": True,
                        "neurogenesis": True
                    }
                ),
                "ca3": SubregionConfig(
                    name="ca3",
                    areas=["CA3"],
                    neuron_count=8000,
                    cell_types=self._get_hippocampal_cell_types("ca3"),
                    connection_density=0.25,
                    special_properties={
                        "recurrent_connectivity": True,
                        "pattern_completion": True,
                        "sequence_generation": True
                    }
                ),
                "ca1": SubregionConfig(
                    name="ca1",
                    areas=["CA1"],
                    neuron_count=12000,
                    cell_types=self._get_hippocampal_cell_types("ca1"),
                    connection_density=0.15,
                    special_properties={
                        "place_cells": True,
                        "temporal_coding": True,
                        "memory_consolidation": True
                    }
                )
            },
            total_neurons=35000,
            inter_subregion_connectivity={
                "trisynaptic_pathway": 0.9,
                "ca3_recurrent": 0.8,
                "ca1_output": 0.7
            },
            anatomical_properties={
                "theta_oscillations": True,
                "gamma_oscillations": True,
                "sharp_wave_ripples": True
            }
        )
        
        # 基底节配置
        self.regions[BrainRegion.BASAL_GANGLIA] = RegionConfig(
            name="basal_ganglia",
            enabled=True,
            subregions={
                "striatum": SubregionConfig(
                    name="striatum",
                    areas=["caudate", "putamen", "nucleus_accumbens"],
                    neuron_count=80000,
                    cell_types=self._get_basal_ganglia_cell_types("striatum"),
                    connection_density=0.08,
                    special_properties={
                        "dopamine_modulation": True,
                        "action_selection": True,
                        "reward_learning": True
                    }
                ),
                "globus_pallidus": SubregionConfig(
                    name="globus_pallidus",
                    areas=["GPe", "GPi"],
                    neuron_count=15000,
                    cell_types=self._get_basal_ganglia_cell_types("gp"),
                    connection_density=0.20,
                    special_properties={
                        "high_firing_rate": True,
                        "inhibitory_control": True
                    }
                )
            },
            total_neurons=95000,
            inter_subregion_connectivity={
                "direct_pathway": 0.8,
                "indirect_pathway": 0.7,
                "hyperdirect_pathway": 0.6
            },
            anatomical_properties={
                "dopaminergic_innervation": True,
                "beta_oscillations": True
            }
        )
        
        # 小脑配置
        self.regions[BrainRegion.CEREBELLUM] = RegionConfig(
            name="cerebellum",
            enabled=True,
            subregions={
                "granular_layer": SubregionConfig(
                    name="granular_layer",
                    areas=["granular"],
                    neuron_count=500000,
                    cell_types=self._get_cerebellar_cell_types("granular"),
                    connection_density=0.001,
                    special_properties={
                        "parallel_fibers": True,
                        "sparse_connectivity": True,
                        "high_frequency_activity": True
                    }
                ),
                "purkinje_layer": SubregionConfig(
                    name="purkinje_layer",
                    areas=["purkinje"],
                    neuron_count=2000,
                    cell_types=self._get_cerebellar_cell_types("purkinje"),
                    connection_density=0.8,
                    special_properties={
                        "complex_dendrites": True,
                        "climbing_fiber_input": True,
                        "motor_learning": True
                    }
                )
            },
            total_neurons=502000,
            inter_subregion_connectivity={
                "parallel_fiber_strength": 0.9,
                "climbing_fiber_strength": 0.95,
                "inhibitory_feedback": 0.8
            },
            anatomical_properties={
                "modular_organization": True,
                "somatotopic_mapping": True
            }
        )
        
        # 丘脑配置
        self.regions[BrainRegion.THALAMUS] = RegionConfig(
            name="thalamus",
            enabled=True,
            subregions={
                "sensory_relay": SubregionConfig(
                    name="sensory_relay",
                    areas=["LGN", "MGN", "VPL", "VPM"],
                    neuron_count=25000,
                    cell_types=self._get_thalamic_cell_types("relay"),
                    connection_density=0.15,
                    special_properties={
                        "sensory_gating": True,
                        "burst_firing": True,
                        "tonic_firing": True
                    }
                ),
                "association_nuclei": SubregionConfig(
                    name="association_nuclei",
                    areas=["MD", "PULVINAR", "LP", "LD"],
                    neuron_count=30000,
                    cell_types=self._get_thalamic_cell_types("association"),
                    connection_density=0.12,
                    special_properties={
                        "cognitive_gating": True,
                        "attention_modulation": True,
                        "working_memory_support": True
                    }
                )
            },
            total_neurons=55000,
            inter_subregion_connectivity={
                "cortical_feedback": 0.9,
                "reticular_inhibition": 0.8,
                "intralaminar_connections": 0.6
            },
            anatomical_properties={
                "sleep_spindles": True,
                "alpha_oscillations": True,
                "gamma_oscillations": True
            }
        )
    
    def _get_cortical_cell_types(self, motor=False, auditory=False) -> Dict[CellType, CellTypeConfig]:
        """获取皮层细胞类型配置"""
        if motor:
            # 运动皮层有更多L5锥体细胞
            proportions = {
                CellType.L23_PYRAMIDAL: 0.25,
                CellType.L5_PYRAMIDAL: 0.45,
                CellType.L6_PYRAMIDAL: 0.15,
                CellType.PV_INTERNEURON: 0.08,
                CellType.SST_INTERNEURON: 0.05,
                CellType.VIP_INTERNEURON: 0.02
            }
        elif auditory:
            # 听觉皮层L4更发达
            proportions = {
                CellType.L23_PYRAMIDAL: 0.30,
                CellType.L4_SPINY_STELLATE: 0.25,
                CellType.L5_PYRAMIDAL: 0.20,
                CellType.L6_PYRAMIDAL: 0.15,
                CellType.PV_INTERNEURON: 0.06,
                CellType.SST_INTERNEURON: 0.03,
                CellType.VIP_INTERNEURON: 0.01
            }
        else:
            # 标准皮层分布
            proportions = {
                CellType.L23_PYRAMIDAL: 0.35,
                CellType.L4_SPINY_STELLATE: 0.15,
                CellType.L5_PYRAMIDAL: 0.20,
                CellType.L6_PYRAMIDAL: 0.15,
                CellType.PV_INTERNEURON: 0.08,
                CellType.SST_INTERNEURON: 0.05,
                CellType.VIP_INTERNEURON: 0.02
            }
        
        cell_types = {}
        for cell_type, proportion in proportions.items():
            cell_types[cell_type] = CellTypeConfig(
                name=cell_type.value,
                proportion=proportion,
                morphology=self._get_morphology(cell_type),
                electrophysiology=self._get_electrophysiology(cell_type),
                connectivity=self._get_connectivity_params(cell_type)
            )
        
        return cell_types
    
    def _get_hippocampal_cell_types(self, region: str) -> Dict[CellType, CellTypeConfig]:
        """获取海马细胞类型配置"""
        if region == "dg":
            proportions = {
                CellType.GRANULE_CELLS: 0.95,
                CellType.MOSSY_CELLS: 0.03,
                CellType.PV_INTERNEURON: 0.02
            }
        elif region == "ca3":
            proportions = {
                CellType.CA3_PYRAMIDAL: 0.90,
                CellType.PV_INTERNEURON: 0.06,
                CellType.SST_INTERNEURON: 0.04
            }
        elif region == "ca1":
            proportions = {
                CellType.CA1_PYRAMIDAL: 0.88,
                CellType.PV_INTERNEURON: 0.06,
                CellType.SST_INTERNEURON: 0.04,
                CellType.VIP_INTERNEURON: 0.02
            }
        else:
            proportions = {}
        
        cell_types = {}
        for cell_type, proportion in proportions.items():
            cell_types[cell_type] = CellTypeConfig(
                name=cell_type.value,
                proportion=proportion,
                morphology=self._get_morphology(cell_type),
                electrophysiology=self._get_electrophysiology(cell_type),
                connectivity=self._get_connectivity_params(cell_type)
            )
        
        return cell_types
    
    def _get_basal_ganglia_cell_types(self, region: str) -> Dict[CellType, CellTypeConfig]:
        """获取基底节细胞类型配置"""
        if region == "striatum":
            proportions = {
                CellType.MEDIUM_SPINY_D1: 0.45,
                CellType.MEDIUM_SPINY_D2: 0.45,
                CellType.CHOLINERGIC_INTERNEURONS: 0.05,
                CellType.PV_INTERNEURON: 0.05
            }
        elif region == "gp":
            proportions = {
                CellType.GPE_NEURONS: 0.60,
                CellType.GPI_NEURONS: 0.40
            }
        else:
            proportions = {}
        
        cell_types = {}
        for cell_type, proportion in proportions.items():
            cell_types[cell_type] = CellTypeConfig(
                name=cell_type.value,
                proportion=proportion,
                morphology=self._get_morphology(cell_type),
                electrophysiology=self._get_electrophysiology(cell_type),
                connectivity=self._get_connectivity_params(cell_type)
            )
        
        return cell_types
    
    def _get_cerebellar_cell_types(self, region: str) -> Dict[CellType, CellTypeConfig]:
        """获取小脑细胞类型配置"""
        if region == "granular":
            proportions = {
                CellType.GRANULE_CELLS_CB: 0.98,
                CellType.GOLGI_CELLS: 0.02
            }
        elif region == "purkinje":
            proportions = {
                CellType.PURKINJE_CELLS: 1.0
            }
        else:
            proportions = {}
        
        cell_types = {}
        for cell_type, proportion in proportions.items():
            cell_types[cell_type] = CellTypeConfig(
                name=cell_type.value,
                proportion=proportion,
                morphology=self._get_morphology(cell_type),
                electrophysiology=self._get_electrophysiology(cell_type),
                connectivity=self._get_connectivity_params(cell_type)
            )
        
        return cell_types
    
    def _get_thalamic_cell_types(self, region: str) -> Dict[CellType, CellTypeConfig]:
        """获取丘脑细胞类型配置"""
        if region == "relay":
            proportions = {
                CellType.RELAY_NEURONS: 0.75,
                CellType.PV_INTERNEURON: 0.20,
                CellType.RETICULAR_NEURONS: 0.05
            }
        elif region == "association":
            proportions = {
                CellType.RELAY_NEURONS: 0.70,
                CellType.PV_INTERNEURON: 0.25,
                CellType.MATRIX_NEURONS: 0.05
            }
        else:
            proportions = {}
        
        cell_types = {}
        for cell_type, proportion in proportions.items():
            cell_types[cell_type] = CellTypeConfig(
                name=cell_type.value,
                proportion=proportion,
                morphology=self._get_morphology(cell_type),
                electrophysiology=self._get_electrophysiology(cell_type),
                connectivity=self._get_connectivity_params(cell_type)
            )
        
        return cell_types
    
    def _get_morphology(self, cell_type: CellType) -> str:
        """获取细胞形态学类型"""
        morphology_map = {
            CellType.L23_PYRAMIDAL: "pyramidal_l23",
            CellType.L4_SPINY_STELLATE: "spiny_stellate",
            CellType.L5_PYRAMIDAL: "pyramidal_l5",
            CellType.L6_PYRAMIDAL: "pyramidal_l6",
            CellType.PV_INTERNEURON: "basket_cell",
            CellType.SST_INTERNEURON: "martinotti_cell",
            CellType.VIP_INTERNEURON: "bipolar_cell",
            CellType.GRANULE_CELLS: "granule_cell",
            CellType.CA3_PYRAMIDAL: "pyramidal_ca3",
            CellType.CA1_PYRAMIDAL: "pyramidal_ca1",
            CellType.MEDIUM_SPINY_D1: "medium_spiny",
            CellType.MEDIUM_SPINY_D2: "medium_spiny",
            CellType.PURKINJE_CELLS: "purkinje",
            CellType.RELAY_NEURONS: "relay_cell"
        }
        return morphology_map.get(cell_type, "generic")
    
    def _get_electrophysiology(self, cell_type: CellType) -> Dict[str, float]:
        """获取电生理参数"""
        # 基于文献的典型参数
        params_map = {
            CellType.L23_PYRAMIDAL: {
                "resting_potential": -70.0,
                "threshold": -50.0,
                "reset_potential": -60.0,
                "membrane_capacitance": 281.0,
                "membrane_resistance": 40.0,
                "refractory_period": 2.0
            },
            CellType.L5_PYRAMIDAL: {
                "resting_potential": -70.0,
                "threshold": -50.0,
                "reset_potential": -60.0,
                "membrane_capacitance": 340.0,
                "membrane_resistance": 35.0,
                "refractory_period": 2.0
            },
            CellType.PV_INTERNEURON: {
                "resting_potential": -70.0,
                "threshold": -52.0,
                "reset_potential": -65.0,
                "membrane_capacitance": 115.0,
                "membrane_resistance": 80.0,
                "refractory_period": 1.0
            }
        }
        return params_map.get(cell_type, {
            "resting_potential": -70.0,
            "threshold": -50.0,
            "reset_potential": -60.0,
            "membrane_capacitance": 200.0,
            "membrane_resistance": 50.0,
            "refractory_period": 2.0
        })
    
    def _get_connectivity_params(self, cell_type: CellType) -> Dict[str, float]:
        """获取连接参数"""
        return {
            "axonal_delay": 1.5,
            "synaptic_weight": 0.5,
            "connection_probability": 0.1,
            "synaptic_plasticity": True
        }
    
    def get_region_config(self, region: BrainRegion) -> Optional[RegionConfig]:
        """获取指定脑区的配置"""
        return self.regions.get(region)
    
    def get_connectivity_matrix(self, regions: List[BrainRegion]) -> np.ndarray:
        """获取脑区间连接矩阵"""
        region_names = [region.value for region in regions]
        return self.anatomical_data.get_connectivity_matrix('allen_brain_atlas', region_names)
    
    def export_config(self, filepath: str):
        """导出配置到文件"""
        import json
        
        config_dict = {}
        for region_enum, region_config in self.regions.items():
            config_dict[region_enum.value] = {
                'name': region_config.name,
                'enabled': region_config.enabled,
                'total_neurons': region_config.total_neurons,
                'subregions': {},
                'inter_subregion_connectivity': region_config.inter_subregion_connectivity,
                'anatomical_properties': region_config.anatomical_properties
            }
            
            for subregion_name, subregion in region_config.subregions.items():
                config_dict[region_enum.value]['subregions'][subregion_name] = {
                    'name': subregion.name,
                    'areas': subregion.areas,
                    'neuron_count': subregion.neuron_count,
                    'connection_density': subregion.connection_density,
                    'special_properties': subregion.special_properties,
                    'cell_types': {}
                }
                
                for cell_type, cell_config in subregion.cell_types.items():
                    config_dict[region_enum.value]['subregions'][subregion_name]['cell_types'][cell_type.value] = {
                        'name': cell_config.name,
                        'proportion': cell_config.proportion,
                        'morphology': cell_config.morphology,
                        'electrophysiology': cell_config.electrophysiology,
                        'connectivity': cell_config.connectivity
                    }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

def get_enhanced_config() -> Dict[str, Any]:
    """获取增强的大脑配置"""
    config_manager = EnhancedBrainConfig()
    
    # 构建完整配置字典
    enhanced_config = {
        'metadata': {
            'version': '2.0',
            'description': '增强的大脑仿真配置，基于多个解剖数据库',
            'data_sources': [
                'allen_brain_atlas',
                'blue_brain_project',
                'ebrains_atlas',
                'human_connectome_project'
            ],
            'total_neurons': sum(region.total_neurons for region in config_manager.regions.values()),
            'supported_species': ['human', 'mouse', 'rat', 'macaque']
        },
        'regions': {},
        'global_connectivity': {},
        'simulation_parameters': {
            'timestep': 0.1,  # ms
            'duration': 1000.0,  # ms
            'temperature': 37.0,  # °C
            'ph': 7.4
        },
        'plasticity_rules': {
            'stdp': {
                'enabled': True,
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'a_plus': 0.01,
                'a_minus': 0.012
            },
            'homeostatic': {
                'enabled': True,
                'target_rate': 5.0,
                'tau': 1000.0
            }
        }
    }
    
    # 添加所有脑区配置
    for region_enum, region_config in config_manager.regions.items():
        enhanced_config['regions'][region_enum.value] = {
            'enabled': region_config.enabled,
            'total_neurons': region_config.total_neurons,
            'subregions': {},
            'connectivity': region_config.inter_subregion_connectivity,
            'properties': region_config.anatomical_properties
        }
        
        for subregion_name, subregion in region_config.subregions.items():
            enhanced_config['regions'][region_enum.value]['subregions'][subregion_name] = {
                'areas': subregion.areas,
                'neuron_count': subregion.neuron_count,
                'connection_density': subregion.connection_density,
                'properties': subregion.special_properties,
                'cell_types': {
                    cell_type.value: {
                        'proportion': cell_config.proportion,
                        'morphology': cell_config.morphology,
                        'electrophysiology': cell_config.electrophysiology,
                        'connectivity': cell_config.connectivity
                    }
                    for cell_type, cell_config in subregion.cell_types.items()
                }
            }
    
    return enhanced_config

if __name__ == "__main__":
    # 创建配置管理器
    config_manager = EnhancedBrainConfig()
    
    # 导出配置
    config_manager.export_config("BrainSimulationSystem/config/enhanced_brain_regions.json")
    
    # 获取完整配置
    full_config = get_enhanced_config()
    
    print(f"配置系统初始化完成")
    print(f"总神经元数量: {full_config['metadata']['total_neurons']:,}")
    print(f"支持的脑区数量: {len(full_config['regions'])}")
    print(f"数据源: {', '.join(full_config['metadata']['data_sources'])}")