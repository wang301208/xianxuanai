"""Helpers for the FullBrainIntegrationMixin responsibilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from .dependencies import *  # noqa: F401,F403

class FullBrainIntegrationMixin:
    """Mixin generated from the legacy network implementation."""

    def set_neuromorphic_bridge(self, bridge: Any):
        """设置神经形态桥接器，并启用事件桥接"""
        self.neuromorphic_bridge = bridge
        self.bridge_enabled = bridge is not None

    def apply_bridge_outputs(self, output_events: List[Tuple[int, float]]) -> None:
        """应用桥接器的回传事件，将其转换为外部输入电流（简单占位实现）"""

        processed: List[Dict[str, float]] = []
        if not output_events:
            self._last_bridge_outputs = []
            self._last_bridge_inputs = []
            return

        events = output_events if isinstance(output_events, (list, tuple)) else [output_events]
        events_list = list(events)
        for event in events_list:
            if isinstance(event, dict):
                gid = event.get('neuron', event.get('id'))
                amplitude = event.get('current', event.get('amplitude', event.get('value')))
            else:
                try:
                    gid, amplitude = event
                except (TypeError, ValueError):
                    continue
            try:
                gid_int = int(gid)
                amplitude_val = float(amplitude)
            except (TypeError, ValueError):
                continue
            self._pending_bridge_events.append((gid_int, amplitude_val))
            processed.append({'neuron': gid_int, 'current': amplitude_val})

        self._last_bridge_outputs = events_list
        self._last_bridge_inputs = processed

    def get_last_bridge_spikes(self) -> List[Tuple[int, float]]:
        """获取上一更新步的尖峰事件，供桥接器处理"""
        return getattr(self, "_last_bridge_spikes", [])

    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""

        stats = {
            'network_size': {
                'brain_regions': len(self.brain_regions),
                'cortical_columns': len(self.cortical_columns),
                'long_range_connections': len(self.long_range_connections),
                'total_neurons': int(self.total_neurons),
                'total_synapses': int(self.total_synapses)
            },
            'neuromorphic_backends': list(self.neuromorphic_backends.keys()),
            'performance_metrics': {
                'mean_update_time': np.mean(self.performance_metrics['update_times']) if self.performance_metrics['update_times'] else 0.0,
                'total_updates': len(self.performance_metrics['update_times'])
            }
        }

        # 后端管理器与导出配置统计（若启用）
        if hasattr(self, 'backend_manager') and self.backend_manager is not None:
            try:
                syn_count = len(self.backend_network_config.get('synapses', {})) if self.backend_network_config else 0
                neuron_count = len(self.backend_network_config.get('neurons', {})) if self.backend_network_config else 0
                stats['backend_export'] = {
                    'enabled': True,
                    'neurons': neuron_count,
                    'synapses': syn_count
                }
            except Exception:
                stats['backend_export'] = {'enabled': False}

        # 添加分区统计（若可用）
        if hasattr(self, 'partition_manager') and self.partition_manager is not None:
            try:
                stats['partitioning'] = {
                    'metadata': self.partition_manager.get_metadata(),
                    'suggested_checkpoint_shards': self.partition_manager.suggest_checkpoint_shards()
                }
            except Exception:
                stats['partitioning'] = {'available': False}

        # 计算每个脑区的神经元数量
        for region, region_info in self.brain_regions.items():
            neuron_count = 0
            for column_id in region_info['columns']:
                if column_id in self.cortical_columns:
                    neuron_count += len(self.cortical_columns[column_id].neurons)

            stats['network_size'][f'{region.value}_neurons'] = neuron_count

        # 增加细胞类型统计（全局聚合）
        try:
            cell_type_counts: Dict[str, int] = {}
            for col_id, column in self.cortical_columns.items():
                for nid, neuron in column.neurons.items():
                    ct = getattr(neuron.cell_type, 'value', str(neuron.cell_type))
                    cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
            stats['cell_types'] = {
                'total_types': len(cell_type_counts),
                'counts': cell_type_counts
            }
        except Exception:
            # 安全回退
            stats['cell_types'] = {'total_types': 0, 'counts': {}}

        # 生理环路统计（若已启用）
        if hasattr(self, 'thalamic_relays') and self.thalamic_relays:
            try:
                # 中文注释：统计丘脑-皮层与海马-PFC 回路（若启用）
                hpfc_count = 0
                if hasattr(self, 'hipp_pfc_connections') and self.hipp_pfc_connections:
                    try:
                        hpfc_count = int(len(self.hipp_pfc_connections.get('synapses', [])))
                    except Exception:
                        hpfc_count = 0
                stats['physiology'] = {
                    'thalamic_relays': len(self.thalamic_relays),
                    'thalamocortical_synapses': int(sum(len(v.get('synapses', [])) for v in self.thalamic_relays.values())),
                    'hippocampus_pfc_synapses': hpfc_count
                }
            except Exception:
                stats['physiology'] = {'enabled': False}

        # 解剖与认知绑定元数据（仅统计，不改变运行路径）
        try:
            stats['anatomy'] = anatomy_metadata()
        except Exception:
            stats['anatomy'] = {}

        try:
            scope_regions = self.config.get('scope', {}).get('brain_regions', [])
            stats['cognition_binding'] = {
                'metadata': binding_metadata(),
                'suggested': suggest_bindings_for_config(scope_regions)
            }
        except Exception:
            stats['cognition_binding'] = {}

        # 胶质-血管耦合元数据（从配置读取，仅统计）
        try:
            phys = self.config.get('physiology', {})
            gv = phys.get('glia_vascular', {}) if isinstance(phys, dict) else {}
            stats['glial_vascular'] = {
                'enabled': bool(gv.get('enabled', False)),
                'modulation_strength': float(gv.get('modulation_strength', 0.0)) if gv.get('enabled', False) else 0.0
            }
        except Exception:
            stats['glial_vascular'] = {'enabled': False, 'modulation_strength': 0.0}

        return stats

    def get_capabilities_report(self) -> Dict[str, Any]:
        """能力总览报告（仅统计，不改变运行路径）"""
        cfg = self.config or {}
        neuromorphic = cfg.get('neuromorphic', {})
        physiology = cfg.get('physiology', {})
        runtime = cfg.get('runtime', {})
        monitoring = cfg.get('monitoring', {})
        visualization = cfg.get('visualization', {})
        cognition = cfg.get('cognition', {})
        optional_modules = cfg.get('optional_modules', {})
        scope = cfg.get('scope', {})

        # 整体规模与整脑达成度占位判断
        total_neurons_cfg = int(scope.get('total_neurons', 0))
        columns_per_region = int(scope.get('columns_per_region', 0))
        is_full_brain_scale = total_neurons_cfg >= 8_000_000_000 or columns_per_region >= 100_000  # 占位阈值

        # 硬件与桥接
        bridge_enabled = bool(neuromorphic.get('bridge', {}).get('enabled', False))
        backend_enabled = bool(neuromorphic.get('backend_manager', {}).get('enabled', False))
        hardware_platforms = {k: bool(v.get('enabled', False)) for k, v in neuromorphic.get('hardware_platforms', {}).items()}
        aer_export = bool(neuromorphic.get('export', {}).get('aer', {}).get('enabled', False))
        mapping_export = bool(neuromorphic.get('export', {}).get('mapping', {}).get('enabled', False))

        # 分区/分布式/检查点
        distributed_enabled = bool(runtime.get('distributed', {}).get('enabled', False))
        checkpoint_enabled = bool(runtime.get('checkpoint', {}).get('enabled', False))
        partition_strategy = runtime.get('partition', {}).get('strategy', 'round_robin')

        # 生理与认知
        tc_enabled = bool(physiology.get('thalamocortical', {}).get('enabled', False))
        hpfc_enabled = bool(physiology.get('hippocampus_pfc', {}).get('enabled', False))
        gv_cfg = physiology.get('glia_vascular', {}) if isinstance(physiology, dict) else {}
        gv_enabled = bool(gv_cfg.get('enabled', False))
        gv_strength = float(gv_cfg.get('modulation_strength', 0.0)) if gv_enabled else 0.0

        cognition_enabled = bool(cognition.get('enabled', False))
        cognition_tasks = list(cognition.get('tasks', [])) if cognition_enabled else []

        # 监控与可视化
        monitoring_enabled = bool(monitoring.get('enabled', False))
        perf_enabled = bool(monitoring.get('performance', {}).get('enabled', False)) if isinstance(monitoring.get('performance', {}), dict) else False
        visualization_enabled = bool(visualization.get('enabled', False))

        # 当前网络内事实数据
        regions_count = len(self.brain_regions)
        columns_count = len(self.cortical_columns)
        long_range_count = len(self.long_range_connections)
        backend_export_stats = {}
        if hasattr(self, 'backend_network_config') and self.backend_network_config:
            try:
                backend_export_stats = {
                    'neurons': len(self.backend_network_config.get('neurons', {})),
                    'synapses': len(self.backend_network_config.get('synapses', {}))
                }
            except Exception:
                backend_export_stats = {}

        return {
            'full_brain_ready': {
                'is_full_brain_scale': is_full_brain_scale,
                'regions_count': regions_count,
                'cortical_columns_count': columns_count,
                'long_range_connections_count': long_range_count
            },
            'neuromorphic': {
                'backend_manager_enabled': backend_enabled,
                'hardware_platforms': hardware_platforms,
                'bridge_enabled': bridge_enabled,
                'aer_export_enabled': aer_export,
                'mapping_export_enabled': mapping_export,
                'backend_export_stats': backend_export_stats
            },
            'runtime': {
                'distributed_enabled': distributed_enabled,
                'checkpoint_enabled': checkpoint_enabled,
                'partition_strategy': partition_strategy
            },
            'physiology': {
                'thalamocortical_enabled': tc_enabled,
                'hippocampus_pfc_enabled': hpfc_enabled,
                'glia_vascular': {
                    'enabled': gv_enabled,
                    'modulation_strength': gv_strength
                }
            },
            'cognition': {
                'enabled': cognition_enabled,
                'tasks': cognition_tasks
            },
            'monitoring_visualization': {
                'monitoring_enabled': monitoring_enabled,
                'performance_enabled': perf_enabled,
                'visualization_enabled': visualization_enabled
            }
        }

    def save_network_state(self, filepath: str):
        """保存网络状态"""

        # 准备保存数据
        save_data = {
            'config': self.config,
            'brain_regions': {region.value: info for region, info in self.brain_regions.items()},
            'performance_metrics': self.performance_metrics,
            'network_statistics': self.get_network_statistics()
        }

        # 保存皮层柱状态（简化）
        column_states = {}
        for column_id, column in self.cortical_columns.items():
            column_states[column_id] = {
                'position': column.position,
                'neuron_count': len(column.neurons),
                'synapse_count': len(column.synapses),
                'astrocyte_count': len(column.astrocytes),
                'vessel_count': len(column.blood_vessels)
            }

        save_data['column_states'] = column_states

        # 使用HDF5保存大数据
        with h5py.File(filepath, 'w') as f:
            # 保存配置和统计
            config_group = f.create_group('config')
            for key, value in save_data.items():
                if key != 'column_states':
                    config_group.attrs[key] = str(value)

            # 保存皮层柱状态
            columns_group = f.create_group('columns')
            for column_id, state in column_states.items():
                column_group = columns_group.create_group(str(column_id))
                for key, value in state.items():
                    column_group.attrs[key] = value

    def load_network_state(self, filepath: str):
        """加载网络状态"""

        with h5py.File(filepath, 'r') as f:
            # 加载配置
            config_group = f['config']

            # 加载皮层柱状态
            if 'columns' in f:
                columns_group = f['columns']
                for column_id_str in columns_group.keys():
                    column_id = int(column_id_str)
                    column_group = columns_group[column_id_str]

                    # 恢复皮层柱基本信息
                    if column_id in self.cortical_columns:
                        column = self.cortical_columns[column_id]
                        # 这里可以恢复更多状态信息

        self.logger.info(f"Network state loaded from {filepath}")

    def set_synapse_manager(self, manager: SynapseManager) -> None:
        """Inject an externally managed synapse manager.

        The complete brain system owns a high level :class:`SynapseManager`
        instance that coordinates synaptic plasticity across subsystems.  When
        it hands that manager to the network we need to re-bind the existing
        long range connections so that subsequent updates continue to drive the
        correct synapses.
        """

        self.synapse_manager = manager
        if self._is_initialized and self.synapse_manager is not None:
            try:
                self._rebind_long_range_synapses()
            except Exception as exc:
                self.logger.debug(f"Rebinding long-range synapses failed: {exc}")

    def _rebind_long_range_synapses(self) -> None:
        if self.synapse_manager is None:
            return

        for conn in self.long_range_connections.values():
            pre_id = conn.get('pre_neuron_id')
            post_id = conn.get('post_neuron_id')
            if pre_id is None or post_id is None:
                continue

            syn_cfg = create_glutamate_synapse_config(
                weight=float(conn.get('strength', 0.5)),
                learning_enabled=True
            )
            try:
                synapse_id = self.synapse_manager.create_synapse(
                    pre_neuron_id=pre_id,
                    post_neuron_id=post_id,
                    synapse_config=syn_cfg
                )
                conn['synapse_id'] = synapse_id
            except Exception as exc:
                self.logger.debug(f"Failed to re-create synapse for connection {conn}: {exc}")
        self._update_total_synapse_count()

    def set_cognitive_interface(self, interface: Any) -> None:
        self.cognitive_interface = interface

    def set_cell_diversity_system(self, system: Any) -> None:
        self.cell_diversity_system = system

    def set_vascular_system(self, system: Any) -> None:
        self.vascular_system = system

    def get_brain_regions(self) -> List[str]:
        """Return registered brain region identifiers (legacy compatibility)."""
        if not self._is_initialized:
            return []
        return [region.value for region in self.brain_regions.keys()]

    def get_region_neurons(self, region: Union[str, BrainRegion]) -> List[int]:
        self._ensure_initialized()
        region_enum = self._normalize_region(region)
        if region_enum is None or region_enum not in self.brain_regions:
            return []

        neuron_ids: List[int] = []
        for column_id in self.brain_regions[region_enum].get('columns', []):
            column = self.cortical_columns.get(column_id)
            if column is None:
                continue
            for nid in column.neurons.keys():
                gid = self._column_neuron_to_global.get((int(column_id), int(nid)))
                if gid is not None:
                    neuron_ids.append(gid)
        return neuron_ids

    def _normalize_region(self, region: Union[str, BrainRegion]) -> Optional[BrainRegion]:
        if isinstance(region, BrainRegion):
            return region
        if isinstance(region, str):
            key = region.upper()
            aliases = {
                "VISUAL_CORTEX": "PRIMARY_VISUAL_CORTEX",
                "AUDITORY_CORTEX": "PRIMARY_AUDITORY_CORTEX",
                "SOMATOSENSORY_CORTEX": "PRIMARY_SOMATOSENSORY_CORTEX",
                "MOTOR_CORTEX": "PRIMARY_MOTOR_CORTEX",
                "PFC": "PREFRONTAL_CORTEX",
                "PREFRONTAL": "PREFRONTAL_CORTEX",
                "PARIETAL_CORTEX": "PREFRONTAL_CORTEX",
                "TEMPORAL_CORTEX": "PREFRONTAL_CORTEX",
                "HIPPOCAMPUS": "HIPPOCAMPUS_CA1",
                "BASAL_GANGLIA": "STRIATUM",
                "CEREBELLUM": "PREFRONTAL_CORTEX",
                "INTERNEURONS": "PREFRONTAL_CORTEX",
                "PYRAMIDAL_CELLS": "PREFRONTAL_CORTEX",
            }
            key = aliases.get(key, key)
            try:
                if key in BrainRegion.__members__:
                    return BrainRegion[key]
                return BrainRegion(region)
            except Exception:
                return BrainRegion.PREFRONTAL_CORTEX
        return None

    def get_neuron_voltages(self) -> Dict[int, float]:
        return dict(self._last_neuron_voltages)

    def get_global_activity(self) -> Dict[str, Any]:
        return dict(self._last_global_activity)

    def is_healthy(self) -> bool:
        return self._is_initialized and bool(self.brain_regions) and bool(self.cortical_columns)

    def shutdown_sync(self) -> None:
        """释放网络内部持有的外部资源。"""

        if self.backend_manager is not None:
            try:
                shutdown = getattr(self.backend_manager, 'shutdown', None)
                if callable(shutdown):
                    shutdown()
            except Exception as exc:
                self.logger.debug(f"Backend manager shutdown skipped: {exc}")

        for backend in self.neuromorphic_backends.values():
            try:
                simulator = backend.get('simulator')
                if simulator is not None:
                    shutdown = getattr(simulator, 'close', None) or getattr(simulator, 'stop', None)
                    if callable(shutdown):
                        shutdown()
            except Exception:
                pass

        self.neuromorphic_backends.clear()
        self._pending_synaptic_currents.clear()
        self._pending_bridge_events.clear()
        self._last_column_inputs = {}
        self._last_synapse_currents = {}
        self._last_bridge_inputs = []
