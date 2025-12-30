"""Helpers for the FullBrainInitializationMixin responsibilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .dependencies import (
    BrainRegion,
    NeuromorphicBridge,
    NeuromorphicBackendManager,
    PartitionManager,
    SynapseManager,
    ThalamicRelay,
    connect_hippocampus_to_pfc,
    connect_thalamus_to_cortex,
    create_glutamate_synapse_config,
    create_neuromorphic_backend_manager,
    create_synapse_manager,
    get_default_integration_config,
    initialize_hippocampus_pfc,
    nengo,
    nengo_loihi,
    np,
    sim,
    LOIHI_AVAILABLE,
    SPINNAKER_AVAILABLE,
)

class FullBrainInitializationMixin:
    """Mixin generated from the legacy network implementation."""

    def initialize_sync(self) -> None:
        """Synchronously construct the full network hierarchy."""

        self._bootstrap_network()

    def _bootstrap_network(self) -> None:
        if self._is_initialized:
            return

        config = self.config

        # 配置校验（只记录警告，不影响运行）
        try:
            from ..config_validator import validate_config
            cfg_warnings = validate_config(config)
            if cfg_warnings:
                self.logger.debug(f"Config warnings: {cfg_warnings}")
                self.config_warnings = cfg_warnings
        except Exception:
            self.config_warnings = []

        # 清理并初始化核心容器
        self.brain_regions.clear()
        self.cortical_columns.clear()
        self.long_range_connections.clear()
        self.neuromorphic_backends.clear()
        self._pending_synaptic_currents.clear()
        self._pending_bridge_events.clear()
        self._last_column_inputs = {}
        self._last_synapse_currents = {}
        self._last_bridge_inputs = []
        self.total_neurons = 0
        self.total_synapses = 0
        self._column_neuron_to_global.clear()
        self._global_to_column_neuron.clear()

        self._initialize_brain_regions()
        self._create_cortical_columns()
        self._build_global_neuron_index()

        # 分区管理器接入（非侵入）：基于脑区-柱映射构建分区，供后续资源调度与检查点分片使用
        self.partition_manager = None
        if PartitionManager is not None:
            try:
                self.partition_manager = PartitionManager(self.config)
                region_map = {}
                for region, info in self.brain_regions.items():
                    for cid in info.get('columns', []):
                        region_map[cid] = region.value
                self.partition_manager.build_partitions(list(self.cortical_columns.keys()), region_map)
            except Exception as e:
                self.partition_manager = None
                self.logger.debug(f"PartitionManager init skipped: {e}")

        # 创建或复用突触管理器（用于长程连接和跨区突触闭环）
        if self.synapse_manager is None:
            self.synapse_manager = create_synapse_manager(self.config)

        self._establish_long_range_connectivity()

        # 可选：初始化丘脑-皮层与海马-PFC环路（默认关闭，配置启用后接入）
        self.thalamic_relays = {}
        try:
            phys_cfg = self.config.get('physiology', {}).get('thalamocortical', {})
            if phys_cfg.get('enabled', False) and ThalamicRelay is not None:
                self._initialize_thalamocortical(phys_cfg)
        except Exception as e:
            self.logger.debug(f"Thalamocortical init skipped: {e}")
        self.hipp_pfc_connections = {}
        try:
            hpfc_cfg = self.config.get('physiology', {}).get('hippocampus_pfc', {})
            if hpfc_cfg.get('enabled', False) and initialize_hippocampus_pfc is not None:
                self._initialize_hippocampus_pfc(hpfc_cfg)
        except Exception as e:
            self.logger.debug(f"Hippocampus-PFC init skipped: {e}")

        # 神经形态桥接器（可选）
        try:
            bridge_cfg = self.config.get('neuromorphic', {}).get('bridge', {})
            if bridge_cfg.get('enabled', False) and NeuromorphicBridge is not None:
                cfg = get_default_integration_config()
                self.neuromorphic_bridge = NeuromorphicBridge(cfg) if cfg is not None else None
                self.bridge_enabled = self.neuromorphic_bridge is not None
            else:
                self.bridge_enabled = False
        except Exception:
            self.bridge_enabled = False

        # 硬件后端管理器（可选：受配置控制，默认关闭）
        self.backend_manager = None
        self.backend_network_config = None
        nm_cfg = self.config.get('neuromorphic', {}).get('backend_manager', {})
        if nm_cfg.get('enabled', False) and create_neuromorphic_backend_manager is not None:
            try:
                self.backend_manager = create_neuromorphic_backend_manager(self.config)
                self.backend_network_config = self.build_backend_network_config()
            except Exception:
                self.backend_manager = None
                self.backend_network_config = None

        self._initialize_neuromorphic_backends()

        self._is_initialized = True

        self.logger.info(
            "Full brain network initialized with %s regions and %s columns",
            len(self.brain_regions),
            len(self.cortical_columns),
        )

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("FullBrainNetwork has not been initialized yet")

    def _initialize_brain_regions(self):
        """初始化脑区"""

        scope = self.config.get("scope")
        if not isinstance(scope, dict):
            scope = {}
            self.config["scope"] = scope

        brain_regions = scope.get("brain_regions")

        if not brain_regions:
            fallback = self.config.get("brain_regions")
            if isinstance(fallback, int):
                enum_values = list(BrainRegion)
                brain_regions = [enum.name for enum in enum_values[:max(1, min(fallback, len(enum_values)))]]
            elif isinstance(fallback, (list, tuple)):
                brain_regions = list(fallback)
            else:
                brain_regions = [
                    BrainRegion.PREFRONTAL_CORTEX.name,
                    BrainRegion.PRIMARY_VISUAL_CORTEX.name,
                    BrainRegion.PRIMARY_AUDITORY_CORTEX.name,
                    BrainRegion.PRIMARY_MOTOR_CORTEX.name,
                    BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX.name,
                ]
            scope["brain_regions"] = brain_regions

        for region_name in brain_regions:
            # 将输入的脑区名称映射到现有枚举，避免无效枚举错误
            name_key = str(region_name).upper()
            mapping = {
                'PRIMARY_VISUAL_CORTEX': 'PRIMARY_VISUAL_CORTEX',
                'V1': 'PRIMARY_VISUAL_CORTEX',
                'PRIMARY_AUDITORY_CORTEX': 'PRIMARY_AUDITORY_CORTEX',
                'A1': 'PRIMARY_AUDITORY_CORTEX',
                'PRIMARY_SOMATOSENSORY_CORTEX': 'PRIMARY_SOMATOSENSORY_CORTEX',
                'S1': 'PRIMARY_SOMATOSENSORY_CORTEX',
                'PRIMARY_MOTOR_CORTEX': 'PRIMARY_MOTOR_CORTEX',
                'M1': 'PRIMARY_MOTOR_CORTEX',
                'PREFRONTAL_CORTEX': 'PREFRONTAL_CORTEX',
                'HIPPOCAMPUS_CA1': 'HIPPOCAMPUS_CA1',
                'HIPPOCAMPUS_CA3': 'HIPPOCAMPUS_CA3',
                'DENTATE_GYRUS': 'DENTATE_GYRUS',
                'THALAMUS_LGN': 'THALAMUS_LGN',
                'THALAMUS_MD': 'THALAMUS_MD',
                'THALAMUS_VPL': 'THALAMUS_VPL',
                'THALAMUS_VPM': 'THALAMUS_VPM',
                'STRIATUM': 'STRIATUM',
                'GLOBUS_PALLIDUS': 'GLOBUS_PALLIDUS',
                'SUBSTANTIA_NIGRA': 'SUBSTANTIA_NIGRA',
                'SUBTHALAMIC_NUCLEUS': 'SUBTHALAMIC_NUCLEUS',
                'AMYGDALA': 'AMYGDALA',
                'NUCLEUS_ACCUMBENS': 'NUCLEUS_ACCUMBENS',
                'SEPTAL_NUCLEI': 'SEPTAL_NUCLEI',
            }
            try_key = mapping.get(name_key, name_key)
            try:
                # 先按成员名匹配，否则尝试按值匹配
                region_enum = BrainRegion[try_key] if try_key in BrainRegion.__members__ else BrainRegion(try_key)
            except Exception:
                # 若仍失败，回退到一个安全的默认脑区
                region_enum = BrainRegion.PREFRONTAL_CORTEX

            self.brain_regions[region_enum] = {
                'name': region_name,
                'columns': [],
                'neurons': {},
                'connectivity': {},
                'metabolic_state': 1.0,
                'neurotransmitter_levels': {
                    'glutamate': 10.0,
                    'gaba': 5.0,
                    'dopamine': 0.1,
                    'serotonin': 0.05,
                    'acetylcholine': 0.2,
                    'norepinephrine': 0.1
                }
            }

    def _create_cortical_columns(self):
        """Create cortical columns."""
        from ..cortical_column import CorticalColumn

        column_id = 0

        # 为每个皮层区域创建皮层柱
        cortical_regions = [
            BrainRegion.PRIMARY_VISUAL_CORTEX,
            BrainRegion.PRIMARY_AUDITORY_CORTEX,
            BrainRegion.PRIMARY_MOTOR_CORTEX,
            BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX,
            BrainRegion.PREFRONTAL_CORTEX
        ]

        for region in cortical_regions:
            if region in self.brain_regions:
                # 每个区域创建多个皮层柱（从配置读取，缺省为2以便测试）
                columns_per_region = int(self.config.get("scope", {}).get("columns_per_region", 2))

                for i in range(columns_per_region):
                    # 生成皮层柱位置
                    x = np.random.uniform(0, 5000)  # 5mm范围
                    y = np.random.uniform(0, 5000)
                    position = (x, y)

                    # 创建皮层柱
                    thalamic_nucleus_by_region = {
                        BrainRegion.PRIMARY_VISUAL_CORTEX: "LGN",
                        BrainRegion.PRIMARY_AUDITORY_CORTEX: "MGN",
                        BrainRegion.PRIMARY_SOMATOSENSORY_CORTEX: "VPL",
                        BrainRegion.PRIMARY_MOTOR_CORTEX: "VA/VL",
                        BrainRegion.PREFRONTAL_CORTEX: "MD",
                    }
                    column_config = dict(self.config)
                    column_config["thalamic_nucleus"] = thalamic_nucleus_by_region.get(region, "VPL")
                    column = CorticalColumn(column_id, position, column_config)
                    self.cortical_columns[column_id] = column

                    # 添加到脑区
                    self.brain_regions[region]['columns'].append(column_id)

                    column_id += 1

        self.logger.info(f"Created {len(self.cortical_columns)} cortical columns")

    def _build_global_neuron_index(self) -> None:
        """为网络分配全局神经元索引并聚合扁平映射。"""

        self.neurons.clear()
        self._column_neuron_to_global.clear()
        self._global_to_column_neuron.clear()

        global_id = 0
        for column_id, column in self.cortical_columns.items():
            for local_id, neuron in column.neurons.items():
                gid = global_id
                key = (int(column_id), int(local_id))
                self._column_neuron_to_global[key] = gid
                self._global_to_column_neuron[gid] = key
                setattr(neuron, 'global_id', gid)
                self.neurons[gid] = neuron
                global_id += 1

        self.total_neurons = global_id
        self._update_total_synapse_count()

    def _update_total_synapse_count(self) -> None:
        """刷新网络级的突触数量统计。"""

        local_synapses = sum(len(column.synapses) for column in self.cortical_columns.values())
        long_range_synapses = len(self.long_range_connections)
        self.total_synapses = int(local_synapses + long_range_synapses)

    def _establish_long_range_connectivity(self):
        """建立长程连接"""

        patterns = self.config.get("connectivity_patterns")
        if not isinstance(patterns, dict):
            patterns = {}
            self.config["connectivity_patterns"] = patterns

        connectivity = patterns.get("long_range_connectivity")
        if not connectivity:
            connectivity = {
                "cortico_cortical": 0.05,
                "thalamocortical": 0.1,
                "hippocampal": 0.05,
            }
            patterns["long_range_connectivity"] = connectivity

        # 皮层-皮层连接
        cortical_regions = list(self.brain_regions.keys())[:4]  # 前4个皮层区域

        for source_region in cortical_regions:
            for target_region in cortical_regions:
                if source_region != target_region:
                    connection_prob = connectivity["cortico_cortical"]

                    if np.random.random() < connection_prob:
                        self._create_inter_region_connection(source_region, target_region)

        self._update_total_synapse_count()
        self.logger.info(f"Established {len(self.long_range_connections)} long-range connections")

    def _create_inter_region_connection(self, source_region: BrainRegion, target_region: BrainRegion):
        """创建脑区间连接"""

        connection_id = len(self.long_range_connections)

        # 选择源和目标皮层柱
        source_columns = self.brain_regions[source_region]['columns']
        target_columns = self.brain_regions[target_region]['columns']

        if source_columns and target_columns:
            source_column = np.random.choice(source_columns)
            target_column = np.random.choice(target_columns)

            # 在源/目标柱内选择一个代表性神经元作为长程连接端点
            pre_candidates = list(self.cortical_columns[source_column].neurons.keys())
            post_candidates = list(self.cortical_columns[target_column].neurons.keys())
            if not pre_candidates or not post_candidates:
                return

            pre_neuron_id = int(np.random.choice(pre_candidates))
            post_neuron_id = int(np.random.choice(post_candidates))
            pre_key = (int(source_column), int(pre_neuron_id))
            post_key = (int(target_column), int(post_neuron_id))
            pre_global = self._column_neuron_to_global.get(pre_key)
            post_global = self._column_neuron_to_global.get(post_key)
            if pre_global is None or post_global is None:
                return

            delay_ms: float
            try:
                src_pos = getattr(self.cortical_columns[source_column], "position", None)
                dst_pos = getattr(self.cortical_columns[target_column], "position", None)
                if src_pos is not None and dst_pos is not None:
                    dx = float(src_pos[0]) - float(dst_pos[0])
                    dy = float(src_pos[1]) - float(dst_pos[1])
                    dist_um = float(np.hypot(dx, dy))
                    dist_mm = dist_um / 1000.0

                    conduction_cfg = self.config.get("physiology", {}).get("conduction", {})
                    velocity = float(
                        conduction_cfg.get("axonal_velocity_m_s", conduction_cfg.get("velocity_m_s", 5.0))
                    )
                    syn_delay = float(conduction_cfg.get("synaptic_delay_ms", 1.0))
                    if not np.isfinite(velocity) or velocity <= 0.0:
                        velocity = 5.0
                    if not np.isfinite(syn_delay) or syn_delay < 0.0:
                        syn_delay = 1.0

                    delay_ms = dist_mm / velocity + syn_delay
                    min_delay = float(conduction_cfg.get("min_delay_ms", 0.5))
                    max_delay = float(conduction_cfg.get("max_delay_ms", 25.0))
                    if not np.isfinite(min_delay) or min_delay < 0.0:
                        min_delay = 0.5
                    if not np.isfinite(max_delay) or max_delay <= min_delay:
                        max_delay = 25.0
                    delay_ms = float(np.clip(delay_ms, min_delay, max_delay))
                else:
                    delay_ms = float(np.random.uniform(5.0, 20.0))
            except Exception:
                delay_ms = float(np.random.uniform(5.0, 20.0))

            synapse_id = None
            if self.synapse_manager is not None:
                syn_cfg = create_glutamate_synapse_config(
                    weight=np.random.uniform(0.1, 1.0),
                    learning_enabled=True
                )
                if isinstance(syn_cfg, dict):
                    syn_cfg["delay"] = float(delay_ms)
                try:
                    synapse_id = self.synapse_manager.create_synapse(
                        pre_neuron_id=pre_global,
                        post_neuron_id=post_global,
                        synapse_config=syn_cfg
                    )
                except Exception as exc:
                    self.logger.debug(f"Synapse manager creation failed: {exc}")

            self.long_range_connections[connection_id] = {
                'source_region': source_region,
                'target_region': target_region,
                'source_column': source_column,
                'target_column': target_column,
                'strength': np.random.uniform(0.1, 0.5),
                'delay': float(delay_ms),
                'synapse_id': synapse_id,
                'pre_neuron_id': pre_global,
                'post_neuron_id': post_global,
            }

    def _initialize_neuromorphic_backends(self):
        """初始化神经形态硬件后端"""

        neuromorphic_config = self.config.get("neuromorphic", {})
        hardware_platforms = neuromorphic_config.get("hardware_platforms", {})

        # Intel Loihi
        if hardware_platforms.get("intel_loihi", {}).get("enabled", False) and LOIHI_AVAILABLE:
            try:
                self.neuromorphic_backends['loihi'] = self._initialize_loihi_backend()
                self.logger.info("Intel Loihi backend initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Loihi backend: {e}")

        # SpiNNaker
        if hardware_platforms.get("spinnaker", {}).get("enabled", False) and SPINNAKER_AVAILABLE:
            try:
                self.neuromorphic_backends['spinnaker'] = self._initialize_spinnaker_backend()
                self.logger.info("SpiNNaker backend initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SpiNNaker backend: {e}")

    def _initialize_loihi_backend(self):
        """初始化Loihi后端"""
        if not LOIHI_AVAILABLE:
            return None

        # 创建Loihi网络映射
        loihi_network = nengo_loihi.Network()

        # 将部分神经元映射到Loihi芯片
        mapped_neurons = {}

        # 选择一些皮层柱进行硬件映射
        columns_to_map = list(self.cortical_columns.keys())[:10]  # 映射前10个皮层柱

        for column_id in columns_to_map:
            column = self.cortical_columns[column_id]

            # 创建Loihi神经元群
            with loihi_network:
                neuron_ensemble = nengo.Ensemble(
                    n_neurons=len(column.neurons),
                    dimensions=1,
                    neuron_type=nengo_loihi.neurons.LoihiLIF()
                )

                mapped_neurons[column_id] = neuron_ensemble

        return {
            'network': loihi_network,
            'mapped_neurons': mapped_neurons,
            'simulator': None  # 将在运行时创建
        }

    def _initialize_spinnaker_backend(self):
        """初始化SpiNNaker后端"""
        if not SPINNAKER_AVAILABLE:
            return None

        # 设置SpiNNaker仿真
        sim.setup(timestep=0.1)

        # 创建神经元群
        spinnaker_populations = {}

        # 选择一些皮层柱进行SpiNNaker映射
        columns_to_map = list(self.cortical_columns.keys())[10:20]  # 映射第11-20个皮层柱

        for column_id in columns_to_map:
            column = self.cortical_columns[column_id]

            # 创建LIF神经元群
            population = sim.Population(
                len(column.neurons),
                sim.IF_curr_exp(),
                label=f"Column_{column_id}"
            )

            spinnaker_populations[column_id] = population

        return {
            'populations': spinnaker_populations,
            'projections': {},
            'simulator': sim
        }

    def _initialize_thalamocortical(self, phys_cfg: Dict[str, Any]):
        """初始化丘脑-皮层环路（最小实现）"""
        target_columns = list(self.cortical_columns.keys())[:max(1, min(4, len(self.cortical_columns)))]
        nuclei = phys_cfg.get('nuclei', ['VPL', 'VPM', 'MD', 'LGN'])
        for i, nuc in enumerate(nuclei):
            relay = ThalamicRelay(relay_id=i, nucleus=str(nuc),
                                  position=(float(np.random.uniform(0, 2000)),
                                            float(np.random.uniform(0, 2000)),
                                            float(np.random.uniform(0, 2000))),
                                  size=int(phys_cfg.get('relay_size', 50)))
            self.thalamic_relays[i] = {
                'relay': relay,
                'synapses': connect_thalamus_to_cortex(relay, self.cortical_columns, self.synapse_manager, target_columns)
            }

    def _initialize_hippocampus_pfc(self, hpfc_cfg: Dict[str, Any]):
        """初始化海马-PFC 回路（最小实现，受配置开关控制）"""
        try:
            pathway = hpfc_cfg.get('pathway', ['DG', 'CA3', 'CA1', 'PFC'])
            proj_size = int(hpfc_cfg.get('projection_size', 100))
            # 使用模块提供的初始化与连接函数（若存在）
            meta = initialize_hippocampus_pfc(pathway=pathway, projection_size=proj_size)
            conns = connect_hippocampus_to_pfc(meta, self.cortical_columns, self.synapse_manager)
            self.hipp_pfc_connections = {'meta': meta, 'synapses': conns}
        except Exception as e:
            self.hipp_pfc_connections = {}
            self.logger.debug(f"Hippocampus-PFC setup failed: {e}")

    def build_backend_network_config(self) -> Dict[str, Any]:
        """构建硬件后端可消费的网络配置（当前采集长程连接；可扩展柱内局部连接）"""
        neurons: Dict[int, Dict[str, Any]] = {}
        synapses: Dict[int, Dict[str, Any]] = {}

        # 收集神经元（从所有皮层柱聚合元数据；简化为基本电生理参数）
        for col_id, column in self.cortical_columns.items():
            for nid, neuron in column.neurons.items():
                gid = self._column_neuron_to_global.get((int(col_id), int(nid)))
                if gid is None:
                    continue
                neurons[gid] = {
                    'cell_type': getattr(neuron.cell_type, 'value', 'LIF'),
                    'threshold': getattr(neuron.params, 'threshold', -50.0),
                    'refractory_period': getattr(neuron.params, 'refractory_period', 2.0),
                    'v_init': getattr(neuron, 'membrane_potential', -70.0),
                    'bias': 0,
                    'position': neuron.position,
                    'column_id': col_id,
                }

        # 收集长程突触（已由 synapse_manager 创建）
        for conn_id, conn in self.long_range_connections.items():
            syn_id = conn.get('synapse_id')
            if syn_id is None:
                continue
            synapses[syn_id] = {
                'pre_neuron_id': conn.get('pre_neuron_id'),
                'post_neuron_id': conn.get('post_neuron_id'),
                'weight': conn.get('strength', 0.5),
                'delay': conn.get('delay', 5.0),
            }

        return {'neurons': neurons, 'synapses': synapses}

    def get_backend_network_config(self) -> Dict[str, Any]:
        """导出硬件后端网络配置"""
        return self.build_backend_network_config()
