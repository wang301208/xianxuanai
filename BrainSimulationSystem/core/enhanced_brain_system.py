"""
Enhanced Brain Simulation System

集成细胞多样性、血管系统和生理脑区的完整大脑仿真系统
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import logging

from .cell_diversity import CellType, CellPopulationManager, EnhancedNeuron, Astrocyte, Microglia
from .vascular_system import VascularNetwork, VesselType
from .physiological_regions import (
    BrainRegion,
    BrainRegionNetwork,
    PrimaryVisualCortex,
    PrimarySomatosensoryCortex,
    PrimaryMotorCortex,
    HippocampusCA1,
    HippocampusCA3,
    DentateGyrus,
    ThalamusLGN,
    ThalamusVPL,
    Striatum,
    SubstantiaNigra,
    LocusCoeruleus,
    RapheNuclei,
    PrefrontalCortex,
    RegionParameters,
    LayerSpecification
)
from .module_interface import ModuleBus, ModuleSignal, ModuleTopic
from .body_model import create_musculoskeletal_model
from .gpu_acceleration import configure_gpu_acceleration, get_gpu_accelerator
from .performance_monitor import PerformanceMonitor

class EnhancedBrainSimulation:
    """增强型大脑仿真系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化脑区网络
        self.brain_network = BrainRegionNetwork()
        runtime_cfg = self.config.get('runtime', {}) if isinstance(self.config, dict) else {}
        gpu_cfg = runtime_cfg.get('gpu')
        if gpu_cfg is None:
            gpu_cfg = self.config.get('gpu')
        self._gpu_config = dict(gpu_cfg or {})
        configure_gpu_acceleration(self._gpu_config)
        accelerator = get_gpu_accelerator()
        self._gpu_enabled = accelerator is not None and getattr(accelerator, "available", False)

        perf_cfg = runtime_cfg.get('performance') or self.config.get('performance')
        self.performance_monitor = PerformanceMonitor(perf_cfg or {})
        parallel_cfg = runtime_cfg.get('parallel')
        if parallel_cfg is None:
            parallel_cfg = runtime_cfg.get('parallelism')
        if parallel_cfg is None:
            parallel_cfg = self.config.get('parallel')
        if parallel_cfg is None:
            parallel_cfg = {}
        self.brain_network.configure_parallelism(parallel_cfg)
        # 默认使用宏观模式以保证测试与快速迭代性能；需要细胞级仿真时可在配置中显式设置为 micro
        self.default_region_mode = self.config.get('default_region_mode', 'macro')
        self.brain_network.set_default_mode(self.default_region_mode)
        self.auto_macro_cortical = self.config.get('auto_macro_cortical', True)
        
        # 全局状态
        self.simulation_time = 0.0
        self.global_metabolic_state = 1.0
        self.global_inflammation = 0.0
        
        # 性能监控
        self.performance_metrics = {
            'update_times': [],
            'cell_counts': {},
            'vascular_flow_rates': [],
            'neurotransmitter_levels': {},
            'region_update_times': {},
            'recommendations': []
        }

        # ģ��ͨ��
        self.module_bus = ModuleBus()
        for topic in ModuleTopic:
            self.module_bus.register_topic(topic)
        self._top_down_modulators: Dict[BrainRegion, Dict[str, float]] = {}
        self._pending_alerts: List[ModuleSignal] = []
        self._pending_pfc_alert = False
        
        # 初始化系统
        self._initialize_brain_regions()
        self._setup_inter_region_connections()
        self._initialize_sensorimotor_loop()
        
        self.logger.info("Enhanced brain simulation system initialized")
    
    def _initialize_brain_regions(self):
        """初始化各个脑区"""
        
        def visual_cortex_layers() -> List[LayerSpecification]:
            return [
                LayerSpecification('L1', 0.07, {
                    CellType.VIP_INTERNEURON: 0.35,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.35,
                    CellType.MICROGLIA_RAMIFIED: 0.15
                }, {'L2/3': 0.6, 'L5': 0.2, 'corticocortical_feedback': 0.2}),
                LayerSpecification('L2/3', 0.23, {
                    CellType.PYRAMIDAL_L23: 0.65,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L4': 0.25, 'L5': 0.35, 'corticocortical_feedforward': 0.4}),
                LayerSpecification('L4', 0.25, {
                    CellType.PYRAMIDAL_L23: 0.55,
                    CellType.PV_INTERNEURON: 0.25,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.1
                }, {'L2/3': 0.45, 'L5': 0.35, 'L6': 0.2, 'thalamic_input': 0.8}),
                LayerSpecification('L5', 0.25, {
                    CellType.PYRAMIDAL_L5A: 0.35,
                    CellType.PYRAMIDAL_L5B: 0.35,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L2/3': 0.25, 'L6': 0.35, 'subcortical_output': 0.4}),
                LayerSpecification('L6', 0.2, {
                    CellType.PYRAMIDAL_L6: 0.5,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.2
                }, {'L4': 0.4, 'corticothalamic': 0.6})
            ]

        def somatosensory_layers() -> List[LayerSpecification]:
            return [
                LayerSpecification('L1', 0.06, {
                    CellType.VIP_INTERNEURON: 0.3,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.4,
                    CellType.MICROGLIA_RAMIFIED: 0.1
                }, {'L2/3': 0.6, 'L5': 0.2, 'corticocortical_feedback': 0.2}),
                LayerSpecification('L2/3', 0.24, {
                    CellType.PYRAMIDAL_L23: 0.6,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L4': 0.3, 'L5': 0.4, 'corticocortical_feedforward': 0.3}),
                LayerSpecification('L4', 0.3, {
                    CellType.PYRAMIDAL_L23: 0.5,
                    CellType.PV_INTERNEURON: 0.25,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.1
                }, {'L2/3': 0.4, 'L5': 0.4, 'thalamic_input': 0.9}),
                LayerSpecification('L5', 0.24, {
                    CellType.PYRAMIDAL_L5A: 0.4,
                    CellType.PYRAMIDAL_L5B: 0.35,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.1
                }, {'L2/3': 0.2, 'L6': 0.4, 'subcortical_output': 0.4}),
                LayerSpecification('L6', 0.16, {
                    CellType.PYRAMIDAL_L6: 0.45,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.2
                }, {'L4': 0.35, 'corticothalamic': 0.65})
            ]

        def motor_layers() -> List[LayerSpecification]:
            return [
                LayerSpecification('L1', 0.05, {
                    CellType.VIP_INTERNEURON: 0.25,
                    CellType.PV_INTERNEURON: 0.25,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.4,
                    CellType.MICROGLIA_RAMIFIED: 0.1
                }, {'L2/3': 0.6, 'L5': 0.2, 'corticocortical_feedback': 0.2}),
                LayerSpecification('L2/3', 0.27, {
                    CellType.PYRAMIDAL_L23: 0.55,
                    CellType.PV_INTERNEURON: 0.25,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L5': 0.5, 'L6': 0.2, 'corticocortical_feedforward': 0.3}),
                LayerSpecification('L5', 0.45, {
                    CellType.PYRAMIDAL_L5A: 0.3,
                    CellType.PYRAMIDAL_L5B: 0.4,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L2/3': 0.25, 'L6': 0.35, 'subcortical_output': 0.4}),
                LayerSpecification('L6', 0.23, {
                    CellType.PYRAMIDAL_L6: 0.5,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.2
                }, {'L5': 0.4, 'corticothalamic': 0.6})
            ]

        def pfc_layers() -> List[LayerSpecification]:
            return [
                LayerSpecification('L1', 0.08, {
                    CellType.VIP_INTERNEURON: 0.3,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.4,
                    CellType.MICROGLIA_RAMIFIED: 0.1
                }, {'L2/3': 0.6, 'L5': 0.2, 'corticocortical_feedback': 0.2}),
                LayerSpecification('L2/3', 0.27, {
                    CellType.PYRAMIDAL_L23: 0.6,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.05
                }, {'L5': 0.45, 'L6': 0.25, 'corticocortical_feedforward': 0.3}),
                LayerSpecification('L5', 0.35, {
                    CellType.PYRAMIDAL_L5A: 0.3,
                    CellType.PYRAMIDAL_L5B: 0.35,
                    CellType.PV_INTERNEURON: 0.15,
                    CellType.SST_INTERNEURON: 0.1,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.1
                }, {'L2/3': 0.3, 'L6': 0.3, 'subcortical_output': 0.4}),
                LayerSpecification('L6', 0.3, {
                    CellType.PYRAMIDAL_L6: 0.45,
                    CellType.PV_INTERNEURON: 0.2,
                    CellType.SST_INTERNEURON: 0.15,
                    CellType.ASTROCYTE_PROTOPLASMIC: 0.2
                }, {'L4': 0.3, 'corticothalamic': 0.7})
            ]

        region_definitions = [
            # 丘脑中继核
            (
                RegionParameters(
                    region_name=BrainRegion.THALAMUS_LGN,
                    dimensions=(1500.0, 1500.0, 800.0),
                    layers=None,
                    neurotransmitter_systems={
                        'glutamate': 0.9,
                        'gaba': 0.7
                    },
                    oscillation_frequencies={
                        'alpha': (8, 13),
                        'gamma': (30, 60)
                    },
                    local_connectivity=0.06,
                    long_range_targets=[BrainRegion.PRIMARY_VISUAL_CORTEX],
                    vascular_density=520.0
                ),
                ThalamusLGN
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.THALAMUS_VPL,
                    dimensions=(1600.0, 1600.0, 900.0),
                    layers=None,
                    neurotransmitter_systems={
                        'glutamate': 0.95,
                        'gaba': 0.75
                    },
                    oscillation_frequencies={
                        'beta': (15, 30),
                        'alpha': (8, 13)
                    },
                    local_connectivity=0.05,
                    long_range_targets=[BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX],
                    vascular_density=540.0
                ),
                ThalamusVPL
            ),
            # 感觉皮层
            (
                RegionParameters(
                    region_name=BrainRegion.PRIMARY_VISUAL_CORTEX,
                    dimensions=(2000.0, 2000.0, 1200.0),
                    layers=visual_cortex_layers(),
                    neurotransmitter_systems={
                        'glutamate': 1.0,
                        'gaba': 0.8,
                        'acetylcholine': 0.3
                    },
                    oscillation_frequencies={
                        'gamma': (30, 80),
                        'beta': (13, 30),
                        'alpha': (8, 13)
                    },
                    local_connectivity=0.1,
                    long_range_targets=[
                        BrainRegion.PREFRONTAL_CORTEX,
                        BrainRegion.THALAMUS_LGN
                    ],
                    vascular_density=800.0
                ),
                PrimaryVisualCortex
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                    dimensions=(2200.0, 2200.0, 1100.0),
                    layers=somatosensory_layers(),
                    neurotransmitter_systems={
                        'glutamate': 1.1,
                        'gaba': 0.9,
                        'acetylcholine': 0.3
                    },
                    oscillation_frequencies={
                        'beta': (15, 30),
                        'gamma': (30, 80),
                        'alpha': (8, 13)
                    },
                    local_connectivity=0.09,
                    long_range_targets=[
                        BrainRegion.PREFRONTAL_CORTEX,
                        BrainRegion.PRIMARY_MOTOR_CORTEX
                    ],
                    vascular_density=820.0
                ),
                PrimarySomatosensoryCortex
            ),
            # 运动皮层
            (
                RegionParameters(
                    region_name=BrainRegion.PRIMARY_MOTOR_CORTEX,
                    dimensions=(2400.0, 2400.0, 1200.0),
                    layers=motor_layers(),
                    neurotransmitter_systems={
                        'glutamate': 1.0,
                        'gaba': 0.8,
                        'dopamine': 0.4
                    },
                    oscillation_frequencies={
                        'beta': (13, 30),
                        'gamma': (30, 90)
                    },
                    local_connectivity=0.1,
                    long_range_targets=[BrainRegion.STRIATUM],
                    vascular_density=850.0
                ),
                PrimaryMotorCortex
            ),
            # 海马环路
            (
                RegionParameters(
                    region_name=BrainRegion.DENTATE_GYRUS,
                    dimensions=(2800.0, 900.0, 450.0),
                    layers=None,
                    neurotransmitter_systems={
                        'glutamate': 1.1,
                        'gaba': 0.7
                    },
                    oscillation_frequencies={
                        'theta': (4, 8),
                        'gamma': (30, 80)
                    },
                    local_connectivity=0.04,
                    long_range_targets=[BrainRegion.HIPPOCAMPUS_CA3],
                    vascular_density=580.0
                ),
                DentateGyrus
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.HIPPOCAMPUS_CA3,
                    dimensions=(2600.0, 900.0, 450.0),
                    layers=None,
                    neurotransmitter_systems={
                        'glutamate': 1.1,
                        'gaba': 0.6
                    },
                    oscillation_frequencies={
                        'theta': (4, 8),
                        'gamma': (30, 100)
                    },
                    local_connectivity=0.07,
                    long_range_targets=[BrainRegion.HIPPOCAMPUS_CA1],
                    vascular_density=560.0
                ),
                HippocampusCA3
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.HIPPOCAMPUS_CA1,
                    dimensions=(3000.0, 1000.0, 500.0),
                    layers=None,
                    neurotransmitter_systems={
                        'glutamate': 1.2,
                        'gaba': 0.6,
                        'acetylcholine': 0.8
                    },
                    oscillation_frequencies={
                        'theta': (4, 12),
                        'gamma': (25, 100),
                        'ripples': (150, 250)
                    },
                    local_connectivity=0.05,
                    long_range_targets=[BrainRegion.PREFRONTAL_CORTEX],
                    vascular_density=600.0
                ),
                HippocampusCA1
            ),
            # 前额叶
            (
                RegionParameters(
                    region_name=BrainRegion.PREFRONTAL_CORTEX,
                    dimensions=(4000.0, 3000.0, 1400.0),
                    layers=pfc_layers(),
                    neurotransmitter_systems={
                        'glutamate': 1.0,
                        'gaba': 0.7,
                        'dopamine': 0.5,
                        'norepinephrine': 0.4,
                        'serotonin': 0.3
                    },
                    oscillation_frequencies={
                        'gamma': (30, 100),
                        'beta': (13, 30),
                        'theta': (4, 8)
                    },
                    local_connectivity=0.08,
                    long_range_targets=[
                        BrainRegion.PRIMARY_MOTOR_CORTEX,
                        BrainRegion.STRIATUM,
                        BrainRegion.HIPPOCAMPUS_CA1
                    ],
                    vascular_density=900.0
                ),
                PrefrontalCortex
            ),
            # 基底节回路
            (
                RegionParameters(
                    region_name=BrainRegion.STRIATUM,
                    dimensions=(3500.0, 2500.0, 1500.0),
                    layers=None,
                    neurotransmitter_systems={
                        'gaba': 1.2,
                        'dopamine': 0.3
                    },
                    oscillation_frequencies={
                        'beta': (15, 30),
                        'theta': (4, 8)
                    },
                    local_connectivity=0.12,
                    long_range_targets=[
                        BrainRegion.SUBSTANTIA_NIGRA,
                        BrainRegion.THALAMUS_VPL
                    ],
                    vascular_density=700.0
                ),
                Striatum
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.SUBSTANTIA_NIGRA,
                    dimensions=(1200.0, 800.0, 600.0),
                    layers=None,
                    neurotransmitter_systems={
                        'dopamine': 1.0,
                        'gaba': 0.6
                    },
                    oscillation_frequencies={
                        'beta': (15, 30),
                        'delta': (1, 4)
                    },
                    local_connectivity=0.08,
                    long_range_targets=[
                        BrainRegion.STRIATUM,
                        BrainRegion.THALAMUS_VPL
                    ],
                    vascular_density=480.0
                ),
                SubstantiaNigra
            ),
            # 脑干调节系统
            (
                RegionParameters(
                    region_name=BrainRegion.LOCUS_COERULEUS,
                    dimensions=(600.0, 600.0, 400.0),
                    layers=None,
                    neurotransmitter_systems={
                        'norepinephrine': 1.0
                    },
                    oscillation_frequencies={
                        'theta': (4, 8),
                        'alpha': (8, 13)
                    },
                    local_connectivity=0.03,
                    long_range_targets=[
                        BrainRegion.PREFRONTAL_CORTEX,
                        BrainRegion.HIPPOCAMPUS_CA1,
                        BrainRegion.PRIMARY_VISUAL_CORTEX
                    ],
                    vascular_density=450.0
                ),
                LocusCoeruleus
            ),
            (
                RegionParameters(
                    region_name=BrainRegion.RAPHE_NUCLEI,
                    dimensions=(800.0, 800.0, 500.0),
                    layers=None,
                    neurotransmitter_systems={
                        'serotonin': 1.0
                    },
                    oscillation_frequencies={
                        'theta': (4, 8),
                        'delta': (1, 4)
                    },
                    local_connectivity=0.04,
                    long_range_targets=[
                        BrainRegion.PREFRONTAL_CORTEX,
                        BrainRegion.HIPPOCAMPUS_CA1
                    ],
                    vascular_density=460.0
                ),
                RapheNuclei
            )
        ]
        
        requested = self.config.get('brain_regions') or self.config.get('enabled_regions') or self.config.get('regions')
        profile = str(self.config.get('region_profile') or '').strip().lower()
        enabled_regions = None
        if requested:
            enabled_regions = set()
            for entry in self._ensure_iterable(requested):
                resolved = self._resolve_region_identifier(entry)
                if resolved is not None:
                    enabled_regions.add(resolved)
            if not enabled_regions:
                enabled_regions = {
                    BrainRegion.PRIMARY_VISUAL_CORTEX,
                    BrainRegion.HIPPOCAMPUS_CA1,
                    BrainRegion.PREFRONTAL_CORTEX,
                }
        elif self.config.get('full_brain') or profile in {'full', 'all', 'extended'}:
            enabled_regions = None
        else:
            enabled_regions = {
                BrainRegion.PRIMARY_VISUAL_CORTEX,
                BrainRegion.HIPPOCAMPUS_CA1,
                BrainRegion.PREFRONTAL_CORTEX,
            }

        for params, region_cls in region_definitions:
            if enabled_regions is not None and params.region_name not in enabled_regions:
                continue
            region_instance = region_cls(params)
            self.brain_network.add_region(region_instance)
            self.logger.debug("Initialized region %s", params.region_name.value)

        self._apply_region_modes()
        self.logger.info(f"Initialized {len(self.brain_network.regions)} brain regions")
    
    def _setup_inter_region_connections(self):
        """设置脑区间连接"""
        
        connection_specs = [
            {
                'source': BrainRegion.THALAMUS_LGN,
                'target': BrainRegion.PRIMARY_VISUAL_CORTEX,
                'strength': 0.9,
                'type': 'excitatory',
                'target_layer': 'L4'
            },
            {
                'source': BrainRegion.PRIMARY_VISUAL_CORTEX,
                'target': BrainRegion.THALAMUS_LGN,
                'strength': 0.5,
                'type': 'excitatory',
                'source_layer': 'L6'
            },
            {
                'source': BrainRegion.PRIMARY_VISUAL_CORTEX,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.8,
                'type': 'excitatory',
                'source_layer': 'L2/3',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.THALAMUS_VPL,
                'target': BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                'strength': 0.9,
                'type': 'excitatory',
                'target_layer': 'L4'
            },
            {
                'source': BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                'target': BrainRegion.THALAMUS_VPL,
                'strength': 0.5,
                'type': 'excitatory',
                'source_layer': 'L6'
            },
            {
                'source': BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.7,
                'type': 'excitatory',
                'source_layer': 'L2/3',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                'target': BrainRegion.PRIMARY_MOTOR_CORTEX,
                'strength': 0.75,
                'type': 'excitatory',
                'source_layer': 'L2/3',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.DENTATE_GYRUS,
                'target': BrainRegion.HIPPOCAMPUS_CA3,
                'strength': 0.6,
                'type': 'excitatory'
            },
            {
                'source': BrainRegion.HIPPOCAMPUS_CA3,
                'target': BrainRegion.HIPPOCAMPUS_CA1,
                'strength': 0.7,
                'type': 'excitatory'
            },
            {
                'source': BrainRegion.HIPPOCAMPUS_CA1,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.7,
                'type': 'excitatory',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.PREFRONTAL_CORTEX,
                'target': BrainRegion.HIPPOCAMPUS_CA1,
                'strength': 0.4,
                'type': 'excitatory',
                'source_layer': 'L5'
            },
            {
                'source': BrainRegion.PREFRONTAL_CORTEX,
                'target': BrainRegion.STRIATUM,
                'strength': 0.6,
                'type': 'excitatory',
                'source_layer': 'L5'
            },
            {
                'source': BrainRegion.PRIMARY_MOTOR_CORTEX,
                'target': BrainRegion.STRIATUM,
                'strength': 0.65,
                'type': 'excitatory',
                'source_layer': 'L5'
            },
            {
                'source': BrainRegion.STRIATUM,
                'target': BrainRegion.SUBSTANTIA_NIGRA,
                'strength': 0.55,
                'type': 'inhibitory'
            },
            {
                'source': BrainRegion.SUBSTANTIA_NIGRA,
                'target': BrainRegion.STRIATUM,
                'strength': 0.55,
                'type': 'modulatory',
                'neurotransmitter': 'dopamine'
            },
            {
                'source': BrainRegion.SUBSTANTIA_NIGRA,
                'target': BrainRegion.THALAMUS_VPL,
                'strength': 0.45,
                'type': 'inhibitory'
            },
            {
                'source': BrainRegion.THALAMUS_VPL,
                'target': BrainRegion.PRIMARY_MOTOR_CORTEX,
                'strength': 0.7,
                'type': 'excitatory',
                'target_layer': 'L4'
            },
            {
                'source': BrainRegion.THALAMUS_VPL,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.5,
                'type': 'excitatory',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.PREFRONTAL_CORTEX,
                'target': BrainRegion.PRIMARY_MOTOR_CORTEX,
                'strength': 0.65,
                'type': 'excitatory',
                'source_layer': 'L2/3',
                'target_layer': 'L2/3'
            },
            {
                'source': BrainRegion.LOCUS_COERULEUS,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.8,
                'type': 'modulatory',
                'neurotransmitter': 'norepinephrine'
            },
            {
                'source': BrainRegion.LOCUS_COERULEUS,
                'target': BrainRegion.HIPPOCAMPUS_CA1,
                'strength': 0.7,
                'type': 'modulatory',
                'neurotransmitter': 'norepinephrine'
            },
            {
                'source': BrainRegion.LOCUS_COERULEUS,
                'target': BrainRegion.PRIMARY_VISUAL_CORTEX,
                'strength': 0.6,
                'type': 'modulatory',
                'neurotransmitter': 'norepinephrine'
            },
            {
                'source': BrainRegion.RAPHE_NUCLEI,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.7,
                'type': 'modulatory',
                'neurotransmitter': 'serotonin'
            },
            {
                'source': BrainRegion.RAPHE_NUCLEI,
                'target': BrainRegion.HIPPOCAMPUS_CA1,
                'strength': 0.6,
                'type': 'modulatory',
                'neurotransmitter': 'serotonin'
            },
            {
                'source': BrainRegion.SUBSTANTIA_NIGRA,
                'target': BrainRegion.PREFRONTAL_CORTEX,
                'strength': 0.35,
                'type': 'modulatory',
                'neurotransmitter': 'dopamine'
            }
        ]
        added_connections = 0
        for spec in connection_specs:
            source = spec['source']
            target = spec['target']
            if source in self.brain_network.regions and target in self.brain_network.regions:
                self.brain_network.add_connection(
                    source,
                    target,
                    strength=spec.get('strength', 0.0),
                    connection_type=spec.get('type', 'excitatory'),
                    metadata=spec.get('metadata'),
                    source_layer=spec.get('source_layer'),
                    target_layer=spec.get('target_layer'),
                    neurotransmitter=spec.get('neurotransmitter')
                )
                added_connections += 1
            else:
                self.logger.debug(
                    "Skipped connection %s -> %s because one of the regions is missing",
                    source.value,
                    target.value
                )
        
        self.logger.info(f"Setup {added_connections} inter-region connections")

    @staticmethod
    def _resolve_region_identifier(identifier: Any) -> Optional[BrainRegion]:
        if isinstance(identifier, BrainRegion):
            return identifier
        if isinstance(identifier, str):
            token = identifier.strip().lower()
            for region in BrainRegion:
                if token in (region.name.lower(), region.value.lower()):
                    return region
        return None

    @staticmethod
    def _ensure_iterable(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _publish_signal(
        self,
        topic: ModuleTopic,
        payload: Dict[str, Any],
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModuleSignal:
        """Helper to publish a structured signal on the module bus."""
        signal = ModuleSignal(topic=topic, payload=payload, source=source, metadata=metadata or {})
        self.module_bus.publish(signal)
        return signal

    def _apply_region_modes(self):
        """根据配置应用宏/微粒度设置"""
        resolved_modes: Dict[BrainRegion, str] = {}
        for entry in self._ensure_iterable(self.config.get('macro_regions', [])):
            region = self._resolve_region_identifier(entry)
            if region:
                resolved_modes[region] = 'macro'
        for entry in self._ensure_iterable(self.config.get('micro_regions', [])):
            region = self._resolve_region_identifier(entry)
            if region:
                resolved_modes[region] = 'micro'
        if self.auto_macro_cortical:
            cortical_defaults = [
                BrainRegion.PRIMARY_VISUAL_CORTEX,
                BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
                BrainRegion.PRIMARY_MOTOR_CORTEX,
                BrainRegion.PREFRONTAL_CORTEX
            ]
            for region in cortical_defaults:
                resolved_modes.setdefault(region, 'macro')
        for region, mode in resolved_modes.items():
            if region in self.brain_network.regions:
                self.brain_network.set_region_mode(region, mode)
        if resolved_modes:
            self.logger.debug("Applied region modes: %s", self.brain_network.get_region_modes())

    def _initialize_sensorimotor_loop(self):
        """Set up the sensorimotor feedback loop (body model and cached sensory frames)."""
        self.sensorimotor_enabled = bool(self.config.get('enable_sensorimotor_loop', False))
        self.body_model = None
        self._baseline_tone: float = float(self.config.get('baseline_muscle_tone', 0.1))
        self.environment_state: Dict[str, Any] = dict(self.config.get('environment', {}) or {})
        self._last_sensory_frames: Dict[str, np.ndarray] = {
            'visual_stimulus': np.zeros((64, 64), dtype=np.float32),
            'somatosensory_stimulus': np.zeros((32, 32), dtype=np.float32)
        }
        self._last_body_feedback: Optional[Dict[str, Any]] = None
        self._last_motor_commands: Dict[str, float] = {}
        self.motor_command_map: List[Tuple[str, Optional[str]]] = []

        if not self.sensorimotor_enabled:
            return

        try:
            body_config = self.config.get('body_model', {}) or {}
            self.body_model = create_musculoskeletal_model(body_config)
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.warning(
                "Failed to initialize musculoskeletal model (%s). Sensorimotor loop disabled.",
                exc,
            )
            self.sensorimotor_enabled = False
            return

        mapping_config = self.config.get('motor_command_map')
        if mapping_config:
            resolved_map: List[Tuple[str, Optional[str]]] = []
            for entry in mapping_config:
                agonist: Optional[str] = None
                antagonist: Optional[str] = None
                if isinstance(entry, dict):
                    agonist = entry.get('agonist')
                    antagonist = entry.get('antagonist')
                elif isinstance(entry, (list, tuple)) and entry:
                    agonist = entry[0]
                    antagonist = entry[1] if len(entry) > 1 else None
                elif isinstance(entry, str):
                    agonist = entry
                if agonist:
                    resolved_map.append((str(agonist), str(antagonist) if antagonist else None))
            if resolved_map:
                self.motor_command_map = resolved_map
        if not self.motor_command_map:
            self.motor_command_map = [
                ('deltoid_left', None),
                ('biceps_left', 'triceps_left'),
                ('biceps_right', 'triceps_right'),
                ('deltoid_right', None),
                ('quadriceps_left', 'hamstring_left'),
                ('quadriceps_right', 'hamstring_right'),
                ('gastrocnemius_left', None),
                ('gastrocnemius_right', None),
            ]

        if 'targets' not in self.environment_state:
            self.environment_state['targets'] = [(0.6, 0.5)]
        if 'visual_field' not in self.environment_state:
            # (x_min, x_max, y_min, y_max) in metres around the body origin.
            self.environment_state['visual_field'] = (-1.2, 1.2, -1.2, 1.2)

    def _inject_sensor_feedback(self, external_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure sensory cortices receive the latest feedback from the body/environment."""
        if not self.sensorimotor_enabled:
            return external_inputs

        ext_inputs = dict(external_inputs or {})
        v1_payload = ext_inputs.setdefault(BrainRegion.PRIMARY_VISUAL_CORTEX.value, {})
        if 'visual_stimulus' not in v1_payload:
            v1_payload['visual_stimulus'] = self._last_sensory_frames['visual_stimulus']
        visual_frame = v1_payload.get('visual_stimulus')
        if visual_frame is not None:
            frame = np.asarray(visual_frame)
            stats = {
                'mean': float(frame.mean()),
                'max': float(frame.max()),
                'std': float(frame.std())
            }
            self._publish_signal(ModuleTopic.SENSORY_VISUAL, stats, 'sensory_feedback')

        s1_payload = ext_inputs.setdefault(BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX.value, {})
        if 'somatosensory_stimulus' not in s1_payload:
            s1_payload['somatosensory_stimulus'] = self._last_sensory_frames['somatosensory_stimulus']
        somato_frame = s1_payload.get('somatosensory_stimulus')
        if somato_frame is not None:
            grid = np.asarray(somato_frame)
            stats = {
                'mean': float(grid.mean()),
                'max': float(grid.max()),
                'std': float(grid.std())
            }
            self._publish_signal(ModuleTopic.SENSORY_SOMATOSENSORY, stats, 'sensory_feedback')

        return ext_inputs

    def _motor_vector_to_commands(self, motor_vector: Any) -> Dict[str, float]:
        """Map motor cortex output vector to body-model muscle activations."""
        commands: Dict[str, float] = {}
        if self.body_model is None or motor_vector is None:
            return commands

        vector = np.atleast_1d(np.asarray(motor_vector, dtype=np.float32))
        if vector.size == 0:
            return commands

        baseline = float(np.clip(self._baseline_tone, 0.0, 1.0))
        for agonist, antagonist in self.motor_command_map:
            if agonist and agonist in self.body_model.muscles:
                commands[agonist] = baseline
            if antagonist and antagonist in self.body_model.muscles:
                commands[antagonist] = baseline

        for idx, (agonist, antagonist) in enumerate(self.motor_command_map):
            if idx >= vector.size:
                break
            value = float(np.clip(vector[idx], -1.0, 1.0))

            if value >= 0.0:
                if agonist and agonist in self.body_model.muscles:
                    commands[agonist] = float(np.clip(0.5 * value + 0.5, 0.0, 1.0))
                if antagonist and antagonist in self.body_model.muscles:
                    commands.setdefault(antagonist, baseline * 0.5)
            else:
                if antagonist and antagonist in self.body_model.muscles:
                    commands[antagonist] = float(np.clip(-0.5 * value + 0.5, 0.0, 1.0))
                if agonist and agonist in self.body_model.muscles:
                    commands.setdefault(agonist, baseline * 0.5)

        return commands

    def _generate_sensory_inputs(self, body_feedback: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert body sensory feedback into cortical input tensors."""
        visual = self._render_visual_scene(body_feedback)
        somato = self._render_somatosensory_map(body_feedback)
        self._last_sensory_frames['visual_stimulus'] = visual
        self._last_sensory_frames['somatosensory_stimulus'] = somato
        hazard_threshold = float(self.environment_state.get('hazard_threshold', 0.9))
        if visual.size and float(visual.max()) >= hazard_threshold:
            peak_index = np.unravel_index(int(np.argmax(visual)), visual.shape)
            peak_value = float(visual[peak_index])
            location = {
                'x': float(peak_index[1] / max(visual.shape[1] - 1, 1)),
                'y': float(peak_index[0] / max(visual.shape[0] - 1, 1))
            }
            alert_payload = {'channel': 'visual', 'peak_value': peak_value, 'location': location}
            signal = self._publish_signal(ModuleTopic.CONTROL_BOTTOM_UP, alert_payload, "visual_cortex")
            self._pending_alerts.append(signal)
            self._pending_pfc_alert = True
        return {
            'visual_stimulus': visual,
            'somatosensory_stimulus': somato
        }

    def _render_visual_scene(self, body_feedback: Optional[Dict[str, Any]]) -> np.ndarray:
        """Produce a coarse visual snapshot based on end-effector poses and environment targets."""
        canvas = np.zeros((64, 64), dtype=np.float32)
        if not body_feedback:
            return canvas

        x_min, x_max, y_min, y_max = self.environment_state.get(
            'visual_field', (-1.2, 1.2, -1.2, 1.2)
        )
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)

        def _project(position: np.ndarray) -> Tuple[float, float]:
            x_norm = np.clip((position[0] - x_min) / width, 0.0, 1.0)
            y_norm = np.clip((position[1] - y_min) / height, 0.0, 1.0)
            # Invert Y for image coordinates
            return x_norm, 1.0 - y_norm

        for target in self.environment_state.get('targets', []):
            try:
                tx, ty = target
                ix = int(np.clip(tx, 0.0, 1.0) * 63)
                iy = int(np.clip(1.0 - ty, 0.0, 1.0) * 63)
                canvas[iy, ix] = 1.0
            except Exception:  # pragma: no cover - defensive
                continue

        end_effectors = body_feedback.get('end_effector_positions', {}) or {}
        for eff_name, position in end_effectors.items():
            pos_vec = np.asarray(position, dtype=np.float32)
            if pos_vec.size < 2:
                continue
            px, py = _project(pos_vec[:2])
            ix = int(np.clip(px, 0.0, 1.0) * 63)
            iy = int(np.clip(py, 0.0, 1.0) * 63)
            intensity = 0.8 if 'hand' in eff_name else 0.6
            canvas[iy, ix] = max(canvas[iy, ix], intensity)

        return canvas

    def _render_somatosensory_map(self, body_feedback: Optional[Dict[str, Any]]) -> np.ndarray:
        """Produce a coarse somatosensory activation map from body feedback signals."""
        grid = np.zeros((32, 32), dtype=np.float32)
        if not body_feedback:
            return grid

        sensory = body_feedback.get('sensory_feedback', {}) or {}
        proprio = sensory.get('proprioceptive', {}) or {}
        tactile = sensory.get('tactile', {}) or {}
        vestibular = sensory.get('vestibular', {}) or {}

        # Encode proprioception across two rows.
        for idx, state in enumerate(proprio.values()):
            if idx >= 16:
                break
            row = (idx // 8) * 8
            col = (idx % 8) * 4
            angle = float(state.get('angle', 0.0))
            norm_angle = float(np.clip((angle + np.pi) / (2.0 * np.pi), 0.0, 1.0))
            grid[row:row + 4, col:col + 4] = norm_angle

        # Encode tactile contact on the lower rows.
        for idx, state in enumerate(tactile.values()):
            row = 24 + (idx // 4) * 4
            col = (idx % 4) * 8
            force = float(state.get('force', 0.0))
            norm_force = float(np.clip(force / 50.0, 0.0, 1.0))
            grid[row:row + 4, col:col + 8] = norm_force

        linear_acc = vestibular.get('linear_acceleration')
        if linear_acc is not None:
            magnitude = float(np.linalg.norm(linear_acc) / 9.81)
            norm_mag = float(np.clip(magnitude, 0.0, 1.0))
            grid[0:4, -8:] = norm_mag

        return grid

    def _update_sensorimotor_loop(self, results: Dict[BrainRegion, Any], dt: float) -> Dict[str, Any]:
        """Run the sensorimotor closed loop after each brain update."""
        report: Dict[str, Any] = {'enabled': bool(self.sensorimotor_enabled)}
        if not self.sensorimotor_enabled or self.body_model is None:
            return report

        motor_result = results.get(BrainRegion.PRIMARY_MOTOR_CORTEX, {}) or {}
        motor_vector = motor_result.get('motor_output')
        commands = self._motor_vector_to_commands(motor_vector)
        if not commands:
            zero_vec = np.zeros(len(self.motor_command_map), dtype=np.float32)
            commands = self._motor_vector_to_commands(zero_vec)

        dt_seconds = max(float(dt), 1.0) / 1000.0
        body_feedback = self.body_model.update_dynamics(dt_seconds, commands)
        sensory_inputs = self._generate_sensory_inputs(body_feedback)

        self._last_body_feedback = body_feedback
        self._last_motor_commands = commands
        command_payload = {
            'commands': {k: float(v) for k, v in commands.items()},
            'dt': float(dt_seconds)
        }
        self._publish_signal(ModuleTopic.MOTOR_COMMAND, command_payload, 'primary_motor_cortex')
        feedback = body_feedback.get('sensory_feedback', {}) or {}
        proprio = feedback.get('proprioceptive', {}) or {}
        tactile = feedback.get('tactile', {}) or {}
        feedback_payload = {
            'proprioceptive': {k: float(v.get('angle', 0.0)) for k, v in list(proprio.items())[:4]},
            'tactile_contacts': {k: bool(v.get('contact', False)) for k, v in tactile.items()}
        }
        self._publish_signal(ModuleTopic.MOTOR_FEEDBACK, feedback_payload, 'body_model')

        report.update({
            'commands': commands,
            'body_feedback': body_feedback,
            'sensory_inputs': sensory_inputs
        })
        return report

    def _process_hierarchical_control(self, results: Dict[BrainRegion, Any]) -> Dict[str, Any]:
        """Derive top-down modulators and bottom-up alerts based on region activity."""
        report: Dict[str, Any] = {'top_down': {}, 'bottom_up': []}
        pfc_result = results.get(BrainRegion.PREFRONTAL_CORTEX)
        if pfc_result:
            macro_summary = pfc_result.get('macro_summary', {})
            exec_level = float(macro_summary.get('macro_activity', pfc_result.get('macro_activity', 0.0)))
            exec_level = max(0.0, exec_level)
            focus = min(1.0, exec_level / 2.0) if exec_level else 0.0
            report['top_down']['executive_level'] = exec_level
            report['top_down']['focus'] = focus
            self._publish_signal(ModuleTopic.COGNITIVE_STATE, {'executive_level': exec_level}, 'prefrontal_cortex')
            if focus > 0.05:
                modulation = {
                    'acetylcholine': round(1.0 + 0.5 * focus, 4),
                    'norepinephrine': round(0.3 + 0.7 * focus, 4)
                }
                targets = [
                    BrainRegion.PRIMARY_VISUAL_CORTEX,
                    BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX
                ]
                for target in targets:
                    self._top_down_modulators[target] = modulation.copy()
                self._publish_signal(
                    ModuleTopic.CONTROL_TOP_DOWN,
                    {'focus': focus, 'modulation': modulation, 'targets': ['visual', 'somatosensory']},
                    'prefrontal_cortex'
                )
        if self._pending_alerts:
            report['bottom_up'] = [signal.payload for signal in self._pending_alerts]
        return report

    def set_region_mode(self, region: Any, mode: str):
        """在运行时调整脑区粒度"""
        resolved = self._resolve_region_identifier(region)
        if not resolved:
            raise ValueError(f"Unable to resolve region identifier '{region}'")
        self.brain_network.set_region_mode(resolved, mode)

    def get_region_modes(self) -> Dict[str, str]:
        """返回当前脑区粒度映射"""
        return self.brain_network.get_region_modes()

    def get_multi_scale_state(self, granularity: str = 'auto') -> Dict[str, Any]:
        """导出多尺度网络状态"""
        return self.brain_network.get_multi_scale_state(granularity)
    
    def step(self, dt: float, external_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行一个仿真步骤"""
        
        start_time = time.time()
        
        self.module_bus.reset_cycle(self.simulation_time)
        
        # ׼���ⲿ����
        external_inputs = dict(external_inputs or {})
        if self.sensorimotor_enabled:
            external_inputs = self._inject_sensor_feedback(external_inputs)
        
        pending_modulators = self._top_down_modulators.copy()
        self._top_down_modulators.clear()
        active_alerts = self._pending_alerts.copy()
        if active_alerts:
            self._pending_alerts.clear()
        deliver_alerts = active_alerts if active_alerts else None
        
        # ����ȫ��״̬������
        global_inputs = {
            'time': self.simulation_time,
            'global_metabolic_state': self.global_metabolic_state,
            'global_inflammation': self.global_inflammation
        }
        
        # Ϊÿ������׼������
        region_inputs = {}
        for region_name in self.brain_network.regions.keys():
            region_input = global_inputs.copy()
            # 兼容外部输入使用 BrainRegion.name（测试）或 BrainRegion.value（缩写）作为键
            region_input.update(external_inputs.get(region_name.value, {}))
            region_input.update(external_inputs.get(region_name.name, {}))
            if deliver_alerts and region_name == BrainRegion.PREFRONTAL_CORTEX:
                urgent = region_input.setdefault('urgent_signals', [])
                urgent.extend(signal.payload for signal in deliver_alerts)
            if region_name in pending_modulators:
                mod_inputs = pending_modulators.pop(region_name)
                target_mods = region_input.setdefault('modulatory_inputs', {})
                for key, value in mod_inputs.items():
                    target_mods[key] = target_mods.get(key, 0.0) + float(value)
            region_inputs[region_name] = region_input
        
        self._pending_pfc_alert = False
        
        # 更新脑区网络
        results = self.brain_network.update_network(dt, region_inputs)
        
        # 更新全局状态
        self._update_global_state(results, dt)

        sensorimotor_report = self._update_sensorimotor_loop(results, dt) if self.sensorimotor_enabled else {'enabled': False}
        control_report = self._process_hierarchical_control(results)
        
        # 更新仿真时间
        self.simulation_time += dt
        
        # 记录性能指标
        update_time = time.time() - start_time
        self.performance_metrics['update_times'].append(update_time)
        if len(self.performance_metrics['update_times']) > 200:
            self.performance_metrics['update_times'] = self.performance_metrics['update_times'][-200:]

        self.performance_monitor.record_global_step(update_time)
        region_metrics = self.performance_metrics.setdefault('region_update_times', {})
        for region_name, region_result in results.items():
            elapsed = region_result.get('elapsed')
            if elapsed is None:
                continue
            history = region_metrics.setdefault(region_name.value, [])
            history.append(elapsed)
            if len(history) > 200:
                del history[:-200]
            self.performance_monitor.record_region_step(region_name.value, elapsed)

        if self.performance_monitor.should_evaluate():
            actions = self.performance_monitor.generate_actions()
            if actions:
                self.performance_metrics.setdefault('recommendations', []).append(
                    {'time': self.simulation_time, 'actions': actions}
                )
                self.performance_metrics['recommendations'] = self.performance_metrics['recommendations'][-50:]
                self._apply_performance_actions(actions)
        
        # 收集统计信息
        step_statistics = self._collect_step_statistics(results)
        
        return {
            'results': results,
            'sensorimotor': sensorimotor_report,
            'statistics': step_statistics,
            'simulation_time': self.simulation_time,
            'update_time': update_time,
            'control': control_report,
            'module_bus': self.module_bus.export_cycle(),
        }
    
    def _apply_performance_actions(self, actions: Dict[str, Dict[str, str]]) -> None:
        """Apply adaptive scheduling decisions derived from the performance monitor."""
        region_modes = actions.get('region_modes', {})
        for region_value, mode in region_modes.items():
            if mode not in ('micro', 'macro'):
                continue
            target_region = None
            for candidate in BrainRegion:
                if candidate.value == region_value or candidate.name == region_value:
                    target_region = candidate
                    break
            if target_region is None:
                continue
            try:
                current_mode = self.brain_network.get_region_mode(target_region)
                if current_mode != mode:
                    self.logger.debug(
                        "Adaptive scheduling: setting region %s mode to %s (was %s)",
                        target_region.value,
                        mode,
                        current_mode,
                    )
                    self.brain_network.set_region_mode(target_region, mode)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Failed to adjust mode for region %s: %s", region_value, exc)

        gpu_actions = actions.get('gpu', {})
        if gpu_actions.get('enable') == 'true' and not self._gpu_enabled:
            self.logger.info("Adaptive scheduling: enabling GPU acceleration due to load heuristics.")
            self._gpu_config['enabled'] = True
            configure_gpu_acceleration(self._gpu_config)
            accelerator = get_gpu_accelerator()
            self._gpu_enabled = accelerator is not None and getattr(accelerator, "available", False)
            if self._gpu_enabled:
                for region in self.brain_network.regions.values():
                    if hasattr(region, 'cell_manager'):
                        region.cell_manager._gpu_accelerator = accelerator

    def _update_global_state(self, results: Dict[BrainRegion, Any], dt: float):
        """更新全局状态"""
        
        # 计算全局代谢状态
        total_oxygen_consumption = 0.0
        total_glucose_consumption = 0.0
        
        for region_result in results.values():
            vascular_exchange = region_result.get('vascular_exchange', {})
            total_oxygen_consumption += abs(vascular_exchange.get('oxygen', 0.0))
            total_glucose_consumption += abs(vascular_exchange.get('glucose', 0.0))
        
        # 代谢状态基于供需平衡
        oxygen_supply = 100.0  # 基础供应
        glucose_supply = 50.0
        
        oxygen_balance = oxygen_supply - total_oxygen_consumption
        glucose_balance = glucose_supply - total_glucose_consumption
        
        metabolic_balance = min(oxygen_balance / oxygen_supply, glucose_balance / glucose_supply)
        self.global_metabolic_state = max(0.1, min(2.0, metabolic_balance))
        
        # 计算全局炎症水平
        total_cytokines = 0.0
        total_microglia = 0
        
        for region_name, region in self.brain_network.regions.items():
            microglia_cells = [cell for cell in region.cell_manager.cells.values() 
                             if isinstance(cell, Microglia)]
            
            for microglia in microglia_cells:
                total_cytokines += microglia.cytokine_release
                total_microglia += 1
        
        if total_microglia > 0:
            average_cytokine_level = total_cytokines / total_microglia
            self.global_inflammation = min(1.0, average_cytokine_level)
        
        # 炎症缓慢衰减
        self.global_inflammation *= np.exp(-dt / 10000.0)  # 10秒时间常数
    
    def _collect_step_statistics(self, results: Dict[BrainRegion, Any]) -> Dict[str, Any]:
        """收集步骤统计信息"""
        
        statistics = {
            'global_state': {
                'metabolic_state': self.global_metabolic_state,
                'inflammation_level': self.global_inflammation,
                'simulation_time': self.simulation_time
            },
            'regions': {}
        }
        
        for region_name, region_result in results.items():
            region = self.brain_network.regions[region_name]
            region_stats = {
                'cell_activity': {},
                'vascular_stats': {},
                'neurotransmitter_levels': region_result.get('neurotransmitters', {}),
                'oscillations': {}
            }
            
            # 细胞活动统计
            cell_results = region_result.get('cell_results', {}) or {}
            if cell_results:
                active_neurons = sum(1 for result in cell_results.values() if result.get('spike', False))
                total_cells = len(cell_results)
                activity_rate = active_neurons / total_cells if total_cells else 0.0
            else:
                macro_activity = float(region_result.get('macro_activity', 0.0) or 0.0)
                total_cells = int(region_result.get('total_neurons') or getattr(region, 'total_neurons', 0) or 0)
                activity_rate = float(np.clip(macro_activity / 5.0, 0.0, 1.0))
                active_neurons = int(activity_rate * total_cells) if total_cells else int(round(activity_rate * 100.0))

            region_stats['cell_activity'] = {
                'total_cells': int(total_cells),
                'active_neurons': int(active_neurons),
                'activity_rate': float(activity_rate),
            }
            
            # 血管统计
            vascular_stats = region.vascular_network.get_vascular_statistics()
            region_stats['vascular_stats'] = {
                'total_flow_rate': vascular_stats.get('total_flow_rate', 0.0),
                'mean_pressure': vascular_stats.get('mean_pressure', 0.0),
                'vessel_count': vascular_stats.get('total_vessels', 0)
            }
            
            # 振荡活动
            oscillations = region_result.get('oscillations', {})
            region_stats['oscillations'] = {
                band: {'frequency': osc.get('frequency', 0), 'amplitude': osc.get('amplitude', 0)}
                for band, osc in oscillations.items()
            }
            
            statistics['regions'][region_name.name] = region_stats
            if region_name.value != region_name.name:
                statistics['regions'][region_name.value] = region_stats
        
        return statistics
    
    def run_simulation(self, duration: float, dt: float = 1.0, 
                      input_sequence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行完整仿真"""
        
        self.logger.info(f"Starting simulation: duration={duration}ms, dt={dt}ms")
        
        steps = int(duration / dt)
        results_history = []
        
        for step_idx in range(steps):
            # 获取当前步骤的输入
            current_inputs = None
            if input_sequence and step_idx < len(input_sequence):
                current_inputs = input_sequence[step_idx]
            
            # 执行仿真步骤
            step_result = self.step(dt, current_inputs)
            
            # 记录结果（每10步记录一次以节省内存）
            if step_idx % 10 == 0:
                results_history.append({
                    'step': step_idx,
                    'time': step_result['simulation_time'],
                    'statistics': step_result['statistics']
                })
            
            # 进度报告
            if step_idx % (steps // 10) == 0:
                progress = (step_idx / steps) * 100
                self.logger.info(f"Simulation progress: {progress:.1f}%")
        
        # 生成最终报告
        final_report = self._generate_simulation_report(results_history)
        
        self.logger.info("Simulation completed successfully")
        
        return {
            'results_history': results_history,
            'final_report': final_report,
            'performance_metrics': self.performance_metrics
        }
    
    def _generate_simulation_report(self, results_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成仿真报告"""
        
        report = {
            'simulation_summary': {
                'total_steps': len(results_history),
                'simulation_duration': self.simulation_time,
                'average_update_time': np.mean(self.performance_metrics['update_times'])
            },
            'brain_regions': {},
            'global_trends': {}
        }
        
        # 分析每个脑区的趋势
        for region_name in self.brain_network.regions.keys():
            region_data = {
                'activity_trend': [],
                'vascular_trend': [],
                'neurotransmitter_trends': {}
            }
            
            for result in results_history:
                region_stats = result['statistics']['regions'].get(region_name.value, {})
                
                # 活动趋势
                activity_rate = region_stats.get('cell_activity', {}).get('activity_rate', 0.0)
                region_data['activity_trend'].append(activity_rate)
                
                # 血管趋势
                flow_rate = region_stats.get('vascular_stats', {}).get('total_flow_rate', 0.0)
                region_data['vascular_trend'].append(flow_rate)
                
                # 神经递质趋势
                nt_levels = region_stats.get('neurotransmitter_levels', {})
                for nt, level in nt_levels.items():
                    if nt not in region_data['neurotransmitter_trends']:
                        region_data['neurotransmitter_trends'][nt] = []
                    region_data['neurotransmitter_trends'][nt].append(level)
            
            # 计算统计量
            report['brain_regions'][region_name.value] = {
                'mean_activity': np.mean(region_data['activity_trend']),
                'activity_variability': np.std(region_data['activity_trend']),
                'mean_flow_rate': np.mean(region_data['vascular_trend']),
                'neurotransmitter_summary': {
                    nt: {
                        'mean': np.mean(trend),
                        'std': np.std(trend),
                        'final': trend[-1] if trend else 0.0
                    }
                    for nt, trend in region_data['neurotransmitter_trends'].items()
                }
            }
        
        # 全局趋势
        metabolic_trend = [result['statistics']['global_state']['metabolic_state'] 
                          for result in results_history]
        inflammation_trend = [result['statistics']['global_state']['inflammation_level'] 
                            for result in results_history]
        
        report['global_trends'] = {
            'metabolic_state': {
                'mean': np.mean(metabolic_trend),
                'std': np.std(metabolic_trend),
                'final': metabolic_trend[-1] if metabolic_trend else 1.0
            },
            'inflammation_level': {
                'mean': np.mean(inflammation_trend),
                'std': np.std(inflammation_trend),
                'final': inflammation_trend[-1] if inflammation_trend else 0.0
            }
        }
        
        return report
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        
        overview = {
            'brain_regions': {},
            'total_cells': 0,
            'total_vessels': 0,
            'system_capabilities': []
        }
        
        for region_name, region in self.brain_network.regions.items():
            # 细胞统计
            cell_stats = region.cell_manager.get_population_statistics()
            
            # 血管统计
            vascular_stats = region.vascular_network.get_vascular_statistics()
            
            region_overview = {
                'dimensions': region.params.dimensions,
                'cell_types': cell_stats['cell_type_counts'],
                'total_cells': cell_stats['total_cells'],
                'vessel_count': vascular_stats['total_vessels'],
                'vascular_volume': vascular_stats['total_volume_um3'],
                'neurotransmitter_systems': region.params.neurotransmitter_systems,
                'oscillation_bands': list(region.params.oscillation_frequencies.keys())
            }
            
            overview['brain_regions'][region_name.value] = region_overview
            overview['total_cells'] += cell_stats['total_cells']
            overview['total_vessels'] += vascular_stats['total_vessels']
        
        # 系统能力
        overview['system_capabilities'] = [
            'Multi-scale neural simulation (molecular to network)',
            'Realistic cell type diversity (10+ cell types)',
            'Vascular system with blood-brain barrier',
            'Neurotransmitter dynamics',
            'Neural oscillations',
            'Inter-region connectivity',
            'Metabolic modeling',
            'Neuroinflammation simulation',
            'Real-time performance monitoring'
        ]
        
        return overview

    def shutdown(self) -> None:
        """Release executor resources before tearing down the simulation."""
        self.brain_network.shutdown_parallelism()

def create_enhanced_brain_simulation(config: Optional[Dict[str, Any]] = None) -> EnhancedBrainSimulation:
    """创建增强型大脑仿真系统的工厂函数"""
    
    if config is None:
        config = {
            'simulation_dt': 1.0,  # ms
            'logging_level': 'INFO',
            'performance_monitoring': True,
            'save_results': True
        }
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, config.get('logging_level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return EnhancedBrainSimulation(config)

