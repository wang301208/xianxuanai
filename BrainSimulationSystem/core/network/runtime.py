"""Helpers for the FullBrainRuntimeMixin responsibilities."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .dependencies import *  # noqa: F401,F403

class FullBrainRuntimeMixin:
    """Mixin generated from the legacy network implementation."""

    def update(self, dt: float, external_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """更新整个大脑网络"""

        import time
        start_time = time.time()

        self._ensure_initialized()

        raw_inputs = external_inputs if isinstance(external_inputs, dict) else {}
        column_inputs, input_metadata = self._prepare_column_inputs(raw_inputs)

        # ��ǰ����ʱ�䣨ms��
        current_time_ms = (self.global_step if hasattr(self, 'global_step') else 0) * dt

        results = {
            'regions': {},
            'columns': {},
            'long_range_activity': {},
            'neuromorphic_results': {},
            'global_statistics': {},
            'synapse_update': {}
        }

        # ���и���Ƥ����
        column_results = self._update_columns_parallel(dt, column_inputs)
        results['columns'] = column_results

        if self._last_column_inputs:
            results['input_currents'] = {col_id: dict(currents) for col_id, currents in self._last_column_inputs.items()}
        else:
            results['input_currents'] = {}
        if input_metadata.get('sources'):
            results['input_sources'] = input_metadata['sources']
        if input_metadata.get('sensory'):
            results['sensory'] = input_metadata['sensory']
        if input_metadata.get('bridge_events') or input_metadata.get('user_bridge_events'):
            results['bridge'] = {
                'applied_events': input_metadata.get('bridge_events', []),
                'user_events': input_metadata.get('user_bridge_events', []),
                'pending_events': len(self._pending_bridge_events),
            }
        raw_applied_currents = input_metadata.get('synapse_currents', {})
        if isinstance(raw_applied_currents, dict):
            applied_synaptic_currents: Dict[int, float] = {
                int(gid): float(val)
                for gid, val in raw_applied_currents.items()
                if isinstance(val, (int, float)) and abs(float(val)) >= 1e-12
            }
        else:
            applied_synaptic_currents = {}

        # 构建每个柱内的发放集合与膜电位映射，用于驱动突触闭环
        spikes_by_column: Dict[int, set] = {}
        neuron_voltages: Dict[int, float] = {}
        for col_id, col_res in column_results.items():
            spikes_local = col_res.get('spikes', [])
            spikes_global: List[int] = []
            for local_spike in spikes_local:
                gid = self._column_neuron_to_global.get((int(col_id), int(local_spike)))
                if gid is not None:
                    spikes_global.append(gid)
            col_res['spikes_global'] = spikes_global
            spikes_by_column[col_id] = set(spikes_global)

            voltages_global: Dict[int, float] = {}
            for nid, v in col_res.get('voltages', {}).items():
                gid = self._column_neuron_to_global.get((int(col_id), int(nid)))
                if gid is not None:
                    voltage_value = float(v)
                    voltages_global[gid] = voltage_value
                    neuron_voltages[gid] = voltage_value
            col_res['voltages_global'] = voltages_global

        # 驱动长程突触的发放事件（前/后）
        for conn_id, conn in self.long_range_connections.items():
            pre_col = conn['source_column']
            post_col = conn['target_column']
            pre_nid = conn.get('pre_neuron_id')
            post_nid = conn.get('post_neuron_id')
            syn_id = conn.get('synapse_id')
            if syn_id is None:
                continue
            # 突触前发放
            if pre_nid in spikes_by_column.get(pre_col, set()):
                self.synapse_manager.process_spike(syn_id, current_time_ms, spike_type='pre')
            # 突触后发放
            if post_nid in spikes_by_column.get(post_col, set()):
                self.synapse_manager.process_spike(syn_id, current_time_ms, spike_type='post')

        # 更新所有长程突触，形成突触-神经元闭环（电流、递质与可塑性）
        # 构建星形胶质活动图：按柱聚合的平均谷氨酸摄取，映射到柱内神经元
        astro_activities_map: Dict[int, float] = {}
        for col_id, col_res in column_results.items():
            try:
                astro_map = self.cortical_columns[col_id].astrocytes
                astro_mean = float(np.mean([a.get('glutamate_uptake', 0.0) for a in astro_map.values()])) if astro_map else 0.0
                # 将该柱的平均值赋给柱内所有神经元的全局ID
                for nid in self.cortical_columns[col_id].neurons.keys():
                    gid = self._column_neuron_to_global.get((int(col_id), int(nid)))
                    if gid is not None:
                        astro_activities_map[gid] = astro_mean
            except Exception:
                # 若获取失败，维持默认0.0
                pass

        # 中文注释：若启用胶质-血管耦合，则对传入突触闭环的膜电位进行轻微调制（不改变核心数值路径）
        try:
            gv_cfg = self.config.get('physiology', {}).get('glia_vascular', {})
            if isinstance(gv_cfg, dict) and gv_cfg.get('enabled', False):
                strength = float(gv_cfg.get('modulation_strength', 0.0))
                if strength > 0.0:
                    for nid, v in list(neuron_voltages.items()):
                        # 简化调制：按 astrocyte_activities 的映射进行小幅度衰减
                        a = float(astro_activities_map.get(nid, 0.0))
                        neuron_voltages[nid] = v * (1.0 - min(0.2, strength * a))
        except Exception as exc:
            self.logger.warning("Glia-vascular modulation skipped: %s", exc)

        attention_payload = None
        try:
            runtime_cfg = self.config.get('runtime', {}) if isinstance(self.config, dict) else {}
            attention_cfg = runtime_cfg.get('attention_module', {}) if isinstance(runtime_cfg, dict) else {}
            if isinstance(attention_cfg, dict) and attention_cfg.get('enabled', False):
                attention_module = getattr(self, '_attention_module', None)
                if attention_module is None:
                    from ..attention_module import AdaptiveGainAttentionModule

                    attention_module = AdaptiveGainAttentionModule(attention_cfg)
                    setattr(self, '_attention_module', attention_module)

                sensory_meta = input_metadata.get('sensory') if isinstance(input_metadata.get('sensory'), dict) else {}
                spike_count = 0
                for col_res in column_results.values():
                    if not isinstance(col_res, dict) or 'error' in col_res:
                        continue
                    spikes_local = col_res.get('spikes', [])
                    if isinstance(spikes_local, (list, tuple, set)):
                        spike_count += len(spikes_local)
                synchrony = self._calculate_network_synchrony(column_results)

                attention_payload = attention_module.update(
                    dt=float(dt),
                    spike_count=int(spike_count),
                    network_synchrony=float(synchrony),
                    sensory=sensory_meta,
                    external_inputs=raw_inputs,
                )
                if isinstance(attention_payload, dict) and attention_payload:
                    attention_gain = attention_payload.get('attention_gain')
                    if isinstance(attention_gain, (int, float, np.floating)):
                        setattr(self, '_attention_gain', float(attention_gain))

                    neuromodulators_out = attention_payload.get('neuromodulators')
                    if isinstance(neuromodulators_out, dict) and neuromodulators_out:
                        merged: Dict[str, float] = {}
                        for key, value in neuromodulators_out.items():
                            try:
                                merged[str(key)] = float(value)
                            except (TypeError, ValueError):
                                continue
                        existing = input_metadata.get('neuromodulators')
                        if isinstance(existing, dict):
                            merged.update(existing)
                        input_metadata['neuromodulators'] = merged
        except Exception as exc:  # pragma: no cover - attention module is optional
            self.logger.debug("Attention module update skipped: %s", exc)
            attention_payload = None

        if isinstance(attention_payload, dict) and attention_payload:
            results['attention_module'] = {
                key: attention_payload.get(key)
                for key in (
                    'attention_gain',
                    'norepinephrine',
                    'tonic_level',
                    'phasic_component',
                    'novelty',
                    'uncertainty',
                    'spike_count',
                    'network_synchrony',
                )
                if key in attention_payload
            }

        if self.synapse_manager is not None:
            neuromodulators = input_metadata.get('neuromodulators')
            syn_update_results = self.synapse_manager.update_all_synapses(
                dt=dt,
                current_time=current_time_ms,
                neuron_voltages=neuron_voltages,
                astrocyte_activities=astro_activities_map,
                neuromodulators=neuromodulators if isinstance(neuromodulators, dict) else None,
            )
            self._accumulate_synapse_currents(syn_update_results)
            results['synapse_update'] = {
                'active_synapses': len(syn_update_results),
                'stats': self.synapse_manager.get_statistics()
            }
            if applied_synaptic_currents:
                results['synapse_update']['applied_currents'] = dict(applied_synaptic_currents)
            if self._last_synapse_currents:
                results['synapse_update']['generated_currents'] = dict(self._last_synapse_currents)
            if isinstance(neuromodulators, dict) and neuromodulators:
                results['neuromodulators'] = dict(neuromodulators)
        else:
            results['synapse_update'] = {'active_synapses': 0, 'stats': {}}
            if applied_synaptic_currents:
                results['synapse_update']['applied_currents'] = dict(applied_synaptic_currents)

        # 更新脑区状态
        region_results = self._update_brain_regions(dt, column_results)
        results['regions'] = region_results

        # 处理长程连接
        long_range_results = self._process_long_range_connections(dt, column_results)
        results['long_range_activity'] = long_range_results

        # 更新神经形态硬件
        if self.neuromorphic_backends:
            neuromorphic_results = self._update_neuromorphic_backends(dt, column_results)
            results['neuromorphic_results'] = neuromorphic_results

        # 计算全局统计
        global_stats = self._calculate_global_statistics(results)
        results['global_statistics'] = global_stats

        capabilities: Optional[Dict[str, Any]] = None
        try:
            capabilities = self.get_capabilities_report()
        except Exception as exc:
            self.logger.warning("Capabilities report unavailable: %s", exc)

        if isinstance(capabilities, dict):
            results['global_statistics']['capabilities'] = capabilities

            score = 0.0

            def _safe_contribution(label: str, supplier: Callable[[], float]) -> None:
                nonlocal score
                try:
                    contribution = float(supplier())
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning("Readiness contribution '%s' skipped: %s", label, exc)
                    return
                if not np.isfinite(contribution):
                    self.logger.warning("Readiness contribution '%s' produced non-finite value: %s", label, contribution)
                    return
                score += contribution

            def _scale_contribution() -> float:
                fb = capabilities.get('full_brain_ready', {})
                if bool(fb.get('is_full_brain_scale', False)):
                    return 0.3
                cols = float(fb.get('cortical_columns_count', 0))
                lr = float(fb.get('long_range_connections_count', 0))
                return min(0.2, (cols / 1000.0) + (lr / 5000.0))

            def _physiology_contribution() -> float:
                phys = capabilities.get('physiology', {})
                contribution = 0.0
                if bool(phys.get('thalamocortical_enabled', False)):
                    contribution += 0.05
                if bool(phys.get('hippocampus_pfc_enabled', False)):
                    contribution += 0.05
                gv = phys.get('glia_vascular', {})
                if isinstance(gv, dict) and bool(gv.get('enabled', False)):
                    contribution += min(0.05, float(gv.get('modulation_strength', 0.0)))
                return contribution

            def _neuromorphic_contribution() -> float:
                nm = capabilities.get('neuromorphic', {})
                contribution = 0.0
                if bool(nm.get('bridge_enabled', False)):
                    contribution += 0.05
                if bool(nm.get('aer_export_enabled', False)):
                    contribution += 0.05
                if bool(nm.get('mapping_export_enabled', False)):
                    contribution += 0.05
                if bool(nm.get('backend_manager_enabled', False)):
                    contribution += 0.1
                return contribution

            def _runtime_contribution() -> float:
                rt = capabilities.get('runtime', {})
                mv = capabilities.get('monitoring_visualization', {})
                contribution = 0.0
                if bool(rt.get('distributed_enabled', False)):
                    contribution += 0.05
                if bool(rt.get('checkpoint_enabled', False)):
                    contribution += 0.05
                if bool(mv.get('monitoring_enabled', False)):
                    contribution += 0.05
                if bool(mv.get('performance_enabled', False)):
                    contribution += 0.05
                return contribution

            _safe_contribution('scale', _scale_contribution)
            _safe_contribution('physiology', _physiology_contribution)
            _safe_contribution('neuromorphic', _neuromorphic_contribution)
            _safe_contribution('runtime', _runtime_contribution)

            results['global_statistics']['readiness_score'] = float(max(0.0, min(1.0, score)))
        elif capabilities is not None:
            self.logger.warning("Capabilities report returned non-dict result: %r", type(capabilities))


        # 中文注释：解剖连通矩阵（脑区×脑区的长程连接计数）
        try:
            regions = list(self.brain_regions.keys())
            region_index = {r: i for i, r in enumerate(regions)}
            size = len(regions)
            matrix = [[0 for _ in range(size)] for __ in range(size)]
            strengths: List[float] = []
            for conn in self.long_range_connections.values():
                sr = conn.get('source_region')
                tr = conn.get('target_region')
                if sr in region_index and tr in region_index:
                    matrix[region_index[sr]][region_index[tr]] += 1
                try:
                    strength = float(conn.get('strength', 0.0))
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning("Skipping connection strength for %s -> %s: %s", sr, tr, exc)
                    continue
                strengths.append(strength)

            density = float(np.mean(strengths)) if strengths else 0.0
            variance = float(np.var(strengths)) if strengths else 0.0

            hist_bins: List[float] = []
            hist_counts_serialized: List[int] = []
            if strengths:
                try:
                    hist_counts_array, bin_edges = np.histogram(np.array(strengths, dtype=float), bins=10, range=(0.0, 1.0))
                    hist_bins = [float(x) for x in bin_edges.tolist()]
                    hist_counts_serialized = [int(x) for x in hist_counts_array.tolist()]
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning("Connectivity histogram computation failed: %s", exc)
                    hist_bins = []
                    hist_counts_serialized = []

            results['global_statistics']['connectivity_matrix'] = {
                'regions': [r.value for r in regions],
                'counts': matrix,
                'strength_mean': density,
                'strength_var': variance,
                'strength_hist_bins': hist_bins,
                'strength_hist_counts': hist_counts_serialized
            }
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.warning("Connectivity statistics unavailable: %s", exc)

        # 中文注释：解剖规模占位统计（不改变运行路径）
        def _safe_len(container: Any) -> int:
            try:
                return int(len(container))
            except Exception:
                return 0

        missing_anatomy_sources: List[str] = []

        brain_regions = getattr(self, 'brain_regions', None)
        if brain_regions is None:
            missing_anatomy_sources.append('brain_regions')
            brain_regions = {}

        cortical_columns = getattr(self, 'cortical_columns', None)
        if cortical_columns is None:
            missing_anatomy_sources.append('cortical_columns')
            cortical_columns = {}

        long_range_connections = getattr(self, 'long_range_connections', None)
        if long_range_connections is None:
            missing_anatomy_sources.append('long_range_connections')
            long_range_connections = {}

        anatomy_scale = {
            'regions_count': _safe_len(brain_regions),
            'cortical_columns_count': _safe_len(cortical_columns),
            'long_range_connections_count': _safe_len(long_range_connections),
        }
        results['global_statistics']['anatomy_scale'] = anatomy_scale

        if missing_anatomy_sources:
            self.logger.warning(
                "Anatomy scale computed with missing sources: %s",
                ", ".join(missing_anatomy_sources),
            )

        # 中文注释：硬件干跑统计（不执行硬件，仅输出一次统计）
        dry_enabled = (self.bridge_enabled and self.neuromorphic_bridge is not None)
        dry_enabled = dry_enabled or (hasattr(self, 'backend_manager') and self.backend_manager is not None)

        if dry_enabled:
            total_spikes = int(global_stats.get('total_spikes', 0))

            backend_config: Dict[str, Any] = {}
            if hasattr(self, 'build_backend_network_config'):
                try:
                    backend_config = self.get_backend_network_config()
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning('Backend network config unavailable: %s', exc)
                    backend_config = {}

            def _safe_collection_size(collection: Any) -> int:
                try:
                    return int(len(collection))
                except Exception:
                    return 0

            neurons_mapped = _safe_collection_size(backend_config.get('neurons', {}))
            synapses_mapped = _safe_collection_size(backend_config.get('synapses', {}))

            try:
                event_density = float(total_spikes / max(1, int(global_stats.get('total_neurons', 0))))
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Failed to compute event density: %s', exc)
                event_density = 0.0

            try:
                mapping_coverage = float(neurons_mapped / max(1, int(global_stats.get('total_neurons', 0))))
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Failed to compute mapping coverage: %s', exc)
                mapping_coverage = 0.0

            results['global_statistics']['hardware_dry_run'] = {
                'total_spikes': total_spikes,
                'neurons_mapped': neurons_mapped,
                'synapses_mapped': synapses_mapped,
                'event_density': event_density,
                'mapping_coverage': mapping_coverage,
                'resources': {
                    'estimated_cores_used': int(max(1, neurons_mapped // 10000)),
                    'estimated_memory_mb': float(max(1.0, synapses_mapped * 0.001)),
                    'bandwidth_mbps_est': float(total_spikes / 1000.0),
                },
            }

        # 中文注释：监控占位指标（仅在监控启用时写入）
        mon_cfg = self.config.get('monitoring', {})
        monitoring_enabled = bool(mon_cfg.get('enabled', False))

        if monitoring_enabled:
            latency_ms = float(global_stats.get('mean_update_time', 0.0)) * 1000.0
            bandwidth_mbps = float(global_stats.get('total_spikes', 0)) / 1000.0
            power_w = float(global_stats.get('total_spikes', 0)) * 0.0005

            results['global_statistics']['monitoring_placeholders'] = {
                'latency_ms': latency_ms,
                'bandwidth_mbps': bandwidth_mbps,
                'power_w': power_w,
            }

            perf_cfg = mon_cfg.get('performance', {})
            if isinstance(perf_cfg, dict) and perf_cfg.get('enabled', False):
                results['global_statistics']['performance'] = {
                    'latency_ms': latency_ms,
                    'throughput_spikes_per_update': float(global_stats.get('total_spikes', 0)),
                    'bandwidth_mbps': bandwidth_mbps,
                    'estimated_power_w': power_w,
                }

        # 中文注释：认知/记忆任务绑定统计（默认关闭，仅统计占位）
        cog_cfg = self.config.get('cognition', {})
        cognition_enabled = isinstance(cog_cfg, dict) and cog_cfg.get('enabled', False)

        if cognition_enabled:
            tasks = list(cog_cfg.get('tasks', [])) if isinstance(cog_cfg.get('tasks', []), (list, tuple)) else []
            results['global_statistics']['cognition_tasks'] = {
                'enabled': True,
                'tasks': tasks,
                'bindings': {'hippocampus_pfc': bool(self.hipp_pfc_connections)},
            }

        # 可选监控导出（默认关闭）
        mon_cfg = self.config.get('monitoring', {})
        monitoring_enabled = bool(mon_cfg.get('enabled', False))

        if monitoring_enabled:
            export_path = mon_cfg.get('export_path', 'BrainSimulationSystem/monitoring/metrics.json')

            try:
                from ..monitoring.metrics import export_metrics
                export_metrics(global_stats, export_path)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Failed to export monitoring metrics to %s: %s', export_path, exc)

            try:
                capabilities_for_export = self.get_capabilities_report()
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Capabilities report unavailable for export: %s', exc)
                capabilities_for_export = None

            if isinstance(capabilities_for_export, dict):
                readiness_score = float(results.get('global_statistics', {}).get('readiness_score', 0.0))
                capabilities_for_export.setdefault('readiness_score', readiness_score)

                import os
                import json

                caps_path = mon_cfg.get('capabilities_path', 'BrainSimulationSystem/monitoring/capabilities.json')
                try:
                    os.makedirs(os.path.dirname(caps_path), exist_ok=True)
                    with open(caps_path, 'w', encoding='utf-8') as f:
                        json.dump(capabilities_for_export, f, ensure_ascii=False, indent=2)
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning('Failed to export capabilities report to %s: %s', caps_path, exc)

        # 可选可视化导出（默认关闭）
        vis_cfg = self.config.get('visualization', {})
        if bool(vis_cfg.get('enabled', False)):
            csv_path = vis_cfg.get('export_path', 'BrainSimulationSystem/visualization/metrics.csv')
            try:
                from ..visualization.export import export_csv
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Visualization export unavailable: %s', exc)
            else:
                try:
                    export_csv(global_stats, csv_path)
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning('Failed to export visualization metrics to %s: %s', csv_path, exc)

        # 可选：硬件映射导出（默认关闭，读配置 neuromorphic.export.mapping）
        mapping_cfg = self.config.get('neuromorphic', {}).get('export', {}).get('mapping', {})
        if isinstance(mapping_cfg, dict) and mapping_cfg.get('enabled', False):
            mapping: Dict[str, Any] = {}
            if hasattr(self, 'build_backend_network_config'):
                try:
                    mapping = self.get_backend_network_config()
                except Exception as exc:  # pragma: no cover - defensive logging path
                    self.logger.warning('Backend mapping export skipped: %s', exc)
                    mapping = {}

            if not isinstance(mapping, dict):
                mapping = {}

            if mapping and bool(mapping_cfg.get('validate_ranges', True)):
                for sid, syn in mapping.get('synapses', {}).items():
                    try:
                        weight_val = float(syn.get('weight', 0.0))
                    except Exception as exc:  # pragma: no cover - defensive logging path
                        self.logger.warning('Synapse %s weight invalid: %s', sid, exc)
                        continue
                    if weight_val < 0.0 or weight_val > 10.0:
                        self.logger.warning('Synapse %s weight %.3f outside expected range [0, 10]', sid, weight_val)

            json_path = mapping_cfg.get('json_path', 'BrainSimulationSystem/monitoring/backend_mapping.json')
            csv_path = mapping_cfg.get('csv_path', 'BrainSimulationSystem/monitoring/backend_mapping.csv')

            try:
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(mapping, jf, ensure_ascii=False)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Failed to export mapping JSON to %s: %s', json_path, exc)

            try:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(['synapse_id', 'pre_neuron_id', 'post_neuron_id', 'weight', 'delay'])
                    for sid, syn in mapping.get('synapses', {}).items():
                        writer.writerow([
                            sid,
                            syn.get('pre_neuron_id'),
                            syn.get('post_neuron_id'),
                            syn.get('weight'),
                            syn.get('delay'),
                        ])
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Failed to export mapping CSV to %s: %s', csv_path, exc)

        # 可选：基于分区元数据生成分布式调度建议（默认关闭，非侵入式）
        dist_cfg = self.config.get('runtime', {}).get('distributed', {})
        dist_enabled = bool(dist_cfg.get('enabled', False))
        if dist_enabled and getattr(self, 'partition_manager', None) is not None:
            try:
                from ..scheduler import plan_distributed_execution
                part_meta = self.partition_manager.get_metadata()
                sched_plan = plan_distributed_execution(part_meta, self.get_network_statistics())
                results['global_statistics']['distributed_plan'] = sched_plan

                plan_path = dist_cfg.get('export_plan_path')
                if plan_path:
                    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
                    with open(plan_path, 'w', encoding='utf-8') as f:
                        json.dump(sched_plan, f, ensure_ascii=False)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Distributed plan generation failed: %s', exc)

        # 可选：硬件设备发现与最小映射建议（默认关闭，非侵入式）
        device_cfg = self.config.get('neuromorphic', {}).get('device_discovery', {})
        if bool(device_cfg.get('enabled', False)):
            try:
                from ..hardware import discover_devices, suggest_mapping
                devices = discover_devices()
                mapping_suggest = suggest_mapping(self.get_network_statistics())
                results['global_statistics']['hardware_devices'] = devices
                results['global_statistics']['hardware_mapping_suggest'] = mapping_suggest
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.logger.warning('Hardware discovery failed: %s', exc)

        # 可选监控导出（默认关闭）
        try:
            mon_cfg = self.config.get('monitoring', {})
            if mon_cfg.get('enabled', False):
                export_path = mon_cfg.get('export_path', 'BrainSimulationSystem/monitoring/metrics.json')
                # 延迟导入，避免硬依赖
                try:
                    from ..monitoring.metrics import export_metrics
                    export_metrics(global_stats, export_path)
                except Exception:
                    pass
                # 中文注释：能力总览导出（仅在监控启用时写出，默认路径）
                try:
                    caps = self.get_capabilities_report()
                    import os, json
                    caps_path = 'BrainSimulationSystem/monitoring/capabilities.json'
                    os.makedirs(os.path.dirname(caps_path), exist_ok=True)
                    with open(caps_path, 'w', encoding='utf-8') as f:
                        json.dump(caps, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        # 胶质/血管耦合的占位调制：根据柱内星形胶质谷氨酸摄取对部分神经元电流做轻微调制（后续可拓展为精确耦合）
        # 当前仅记录统计，不改变数值稳定路径
        glial_stats = {}
        for col_id, col_res in column_results.items():
            astro_activity = 0.0
            try:
                # 统计星形胶质的平均摄取水平
                astro_map = self.cortical_columns[col_id].astrocytes
                if astro_map:
                    astro_activity = float(np.mean([a.get('glutamate_uptake', 0.0) for a in astro_map.values()]))
            except Exception:
                astro_activity = 0.0
            glial_stats[col_id] = {'astro_mean_uptake': astro_activity}
        results['global_statistics']['glial_coupling'] = glial_stats

        # 记录性能指标
        update_time = time.time() - start_time
        self.performance_metrics['update_times'].append(update_time)

        # 周期性检查点保存（配置控制；无 h5py 时安全跳过）
        try:
            ckpt_cfg = self.config.get('runtime', {}).get('checkpoint', {})
            if ckpt_cfg.get('enabled', False):
                interval_ms = int(self.config.get('simulation', {}).get('save_interval', 0))
                if interval_ms > 0:
                    # global_step 是步数；使用步数与 dt 推算时间
                    sim_time_ms = (self.global_step + 1) * dt
                    if int(sim_time_ms) % interval_ms == 0 and H5PY_AVAILABLE:
                        self.save_network_state(ckpt_cfg.get('path', 'BrainSimulationSystem/checkpoints/default.ckpt'))
                        # 若配置提供检查点分片规划导出路径，则写出分片建议
                        try:
                            plan_path = ckpt_cfg.get('export_plan_path')
                            if plan_path and hasattr(self, 'partition_manager') and self.partition_manager is not None:
                                plan = self.partition_manager.suggest_checkpoint_shards()
                                import os, json
                                os.makedirs(os.path.dirname(plan_path), exist_ok=True)
                                with open(plan_path, 'w', encoding='utf-8') as f:
                                    json.dump(plan, f, ensure_ascii=False)
                        except Exception:
                            pass
        except Exception as e:
            self.logger.debug(f"Checkpoint skipped: {e}")

        # 记录用于上层访问的最新状态
        self._last_neuron_voltages = neuron_voltages
        self._last_global_activity = global_stats

        # 增加全局步数
        self.global_step = (self.global_step + 1) if hasattr(self, 'global_step') else 1


        # 若启用桥接器：将本步的尖峰事件打包给桥接器（异步执行由上层驱动）
        if self.bridge_enabled and self.neuromorphic_bridge is not None:
            # 收集本步尖峰事件（以 (neuron_id, time_ms) 格式）
            bridge_spikes = []
            for col_id, spike_set in spikes_by_column.items():
                for nid in spike_set:
                    bridge_spikes.append((int(nid), float(current_time_ms)))
            # 暂存以供外部驱动调用桥接器
            self._last_bridge_spikes = bridge_spikes
            self._last_bridge_time_ms = current_time_ms
            # 可选：AER 事件导出（默认关闭）
            try:
                aer_cfg = self.config.get('neuromorphic', {}).get('export', {}).get('aer', {})
                if aer_cfg.get('enabled', False):
                    aer_path = aer_cfg.get('path', 'BrainSimulationSystem/monitoring/aer_events.csv')
                    # 写出 CSV：列为 neuron_id,time_ms
                    try:
                        import csv, os
                        os.makedirs(os.path.dirname(aer_path), exist_ok=True)
                        with open(aer_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for nid, t in bridge_spikes:
                                writer.writerow([nid, t])
                    except Exception:
                        pass
            except Exception:
                pass

        return results

    def _prepare_column_inputs(self, raw_inputs: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Any]]:
        """Normalize external inputs into per-column current maps and capture metadata."""

        inputs_by_column: Dict[int, Dict[int, float]] = {}
        metadata: Dict[str, Any] = {
            'bridge_events': [],
            'synapse_currents': {},
            'sources': {},
            'user_bridge_events': [],
            'neuromodulators': {},
        }
        sources = metadata['sources']

        def add_current(column_id: int, neuron_id: int, current: Any, source: Optional[str] = None) -> None:
            try:
                col_key = int(column_id)
                neuron_key = int(neuron_id)
                current_value = float(current)
            except (TypeError, ValueError):
                return
            if abs(current_value) < 1e-12:
                return
            column_map = inputs_by_column.setdefault(col_key, {})
            column_map[neuron_key] = column_map.get(neuron_key, 0.0) + current_value
            if source:
                sources[source] = sources.get(source, 0) + 1

        external_inputs = raw_inputs if isinstance(raw_inputs, dict) else {}

        raw_neuromodulators = external_inputs.get('neuromodulators') if external_inputs else None
        if isinstance(raw_neuromodulators, dict):
            cleaned: Dict[str, float] = {}
            for key, value in raw_neuromodulators.items():
                try:
                    cleaned[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            metadata['neuromodulators'] = cleaned

        # Optional sensory pathways (retina/cochlea-style encoders feeding thalamic input).
        sensory_vectors: Dict[int, np.ndarray] = {}
        sensory_metadata: Dict[str, Any] = {}
        if external_inputs:
            image_input = external_inputs.get('image', external_inputs.get('visual_stimulus'))
            if image_input is not None:
                try:
                    vision_system = getattr(self, '_vision_system', None)
                    if vision_system is None:
                        from ..sensory import VisionSystem  # local import to avoid import-time costs

                        cfg = {}
                        sensory_cfg = self.config.get('sensory_pathways', {}) if isinstance(self.config, dict) else {}
                        if isinstance(sensory_cfg, dict):
                            cfg = sensory_cfg.get('vision', {}) if isinstance(sensory_cfg.get('vision', {}), dict) else {}
                        cfg = dict(cfg)
                        cfg.setdefault('thalamic_size', int(self.config.get('thalamic_size', 200)) if isinstance(self.config, dict) else 200)
                        vision_system = VisionSystem(cfg)
                        setattr(self, '_vision_system', vision_system)

                    vec = vision_system.encode(image_input)
                    gain = 1.0
                    try:
                        default_gain = float(getattr(self, '_attention_gain', 1.0))
                        gain = float(external_inputs.get('vision_gain', external_inputs.get('attention_gain', default_gain)))
                    except (TypeError, ValueError):
                        gain = 1.0
                    vec = np.clip(vec * max(0.0, min(2.0, gain)), 0.0, 1.0)

                    target_cols = []
                    try:
                        region_info = self.brain_regions.get(BrainRegion.PRIMARY_VISUAL_CORTEX)
                        if isinstance(region_info, dict):
                            target_cols = list(region_info.get('columns', []))
                    except Exception:
                        target_cols = []

                    for col_id in target_cols:
                        try:
                            col_int = int(col_id)
                        except (TypeError, ValueError):
                            continue
                        existing = sensory_vectors.get(col_int)
                        sensory_vectors[col_int] = vec if existing is None else np.clip(existing + vec, 0.0, 1.0)
                        sources['vision_system'] = sources.get('vision_system', 0) + 1
                    mean_activity = float(np.mean(vec)) if getattr(vec, 'size', 0) else 0.0
                    max_activity = float(np.max(vec)) if getattr(vec, 'size', 0) else 0.0
                    energy = float(np.linalg.norm(vec)) if getattr(vec, 'size', 0) else 0.0
                    sensory_metadata['vision'] = {
                        'target_columns': target_cols,
                        'thalamic_size': int(vec.size),
                        'gain': float(gain),
                        'mean_activity': mean_activity,
                        'max_activity': max_activity,
                        'energy': energy,
                    }
                except Exception as exc:
                    sensory_metadata['vision'] = {'error': str(exc)}

            audio_input = external_inputs.get('audio')
            if audio_input is not None:
                try:
                    auditory_system = getattr(self, '_auditory_system', None)
                    if auditory_system is None:
                        from ..sensory import AuditorySystem  # local import to avoid import-time costs

                        cfg = {}
                        sensory_cfg = self.config.get('sensory_pathways', {}) if isinstance(self.config, dict) else {}
                        if isinstance(sensory_cfg, dict):
                            cfg = sensory_cfg.get('auditory', {}) if isinstance(sensory_cfg.get('auditory', {}), dict) else {}
                        cfg = dict(cfg)
                        cfg.setdefault('thalamic_size', int(self.config.get('thalamic_size', 200)) if isinstance(self.config, dict) else 200)
                        auditory_system = AuditorySystem(cfg)
                        setattr(self, '_auditory_system', auditory_system)

                    sr = external_inputs.get('audio_sample_rate', None)
                    vec = auditory_system.encode(audio_input, sample_rate=sr if isinstance(sr, (int, float)) else None)
                    gain = 1.0
                    try:
                        default_gain = float(getattr(self, '_attention_gain', 1.0))
                        gain = float(external_inputs.get('auditory_gain', external_inputs.get('attention_gain', default_gain)))
                    except (TypeError, ValueError):
                        gain = 1.0
                    vec = np.clip(vec * max(0.0, min(2.0, gain)), 0.0, 1.0)

                    target_cols = []
                    try:
                        region_info = self.brain_regions.get(BrainRegion.PRIMARY_AUDITORY_CORTEX)
                        if isinstance(region_info, dict):
                            target_cols = list(region_info.get('columns', []))
                    except Exception:
                        target_cols = []

                    for col_id in target_cols:
                        try:
                            col_int = int(col_id)
                        except (TypeError, ValueError):
                            continue
                        existing = sensory_vectors.get(col_int)
                        sensory_vectors[col_int] = vec if existing is None else np.clip(existing + vec, 0.0, 1.0)
                        sources['auditory_system'] = sources.get('auditory_system', 0) + 1
                    mean_activity = float(np.mean(vec)) if getattr(vec, 'size', 0) else 0.0
                    max_activity = float(np.max(vec)) if getattr(vec, 'size', 0) else 0.0
                    energy = float(np.linalg.norm(vec)) if getattr(vec, 'size', 0) else 0.0
                    sensory_metadata['auditory'] = {
                        'target_columns': target_cols,
                        'thalamic_size': int(vec.size),
                        'gain': float(gain),
                        'mean_activity': mean_activity,
                        'max_activity': max_activity,
                        'energy': energy,
                    }
                except Exception as exc:
                    sensory_metadata['auditory'] = {'error': str(exc)}

        if self._pending_synaptic_currents:
            snapshot: Dict[int, float] = {
                int(gid): float(value)
                for gid, value in self._pending_synaptic_currents.items()
                if isinstance(value, (int, float)) and abs(float(value)) >= 1e-12
            }
            if snapshot:
                metadata['synapse_currents'] = snapshot.copy()
                for gid, current in snapshot.items():
                    mapping = self._global_to_column_neuron.get(gid)
                    if mapping is not None:
                        add_current(mapping[0], mapping[1], current, source='synapse_manager')
            self._pending_synaptic_currents.clear()

        def parse_bridge_event(event: Any) -> Optional[Tuple[int, float]]:
            if isinstance(event, dict):
                gid = event.get('neuron', event.get('id'))
                amplitude = event.get('current', event.get('amplitude', event.get('value')))
            else:
                try:
                    gid, amplitude = event
                except (TypeError, ValueError):
                    return None
            try:
                return int(gid), float(amplitude)
            except (TypeError, ValueError):
                return None

        user_bridge_events: List[Dict[str, float]] = []
        raw_bridge = external_inputs.get('bridge_events') if external_inputs else None
        if isinstance(raw_bridge, (list, tuple)):
            for event in raw_bridge:
                parsed = parse_bridge_event(event)
                if parsed is None:
                    continue
                gid, amplitude = parsed
                self._pending_bridge_events.append((gid, amplitude))
                user_bridge_events.append({'neuron': gid, 'current': amplitude})
        metadata['user_bridge_events'] = user_bridge_events

        bridge_applied: List[Dict[str, float]] = []
        while self._pending_bridge_events:
            gid, amplitude = self._pending_bridge_events.popleft()
            try:
                gid_int = int(gid)
                amplitude_val = float(amplitude)
            except (TypeError, ValueError):
                continue
            mapping = self._global_to_column_neuron.get(gid_int)
            if mapping is None:
                continue
            add_current(mapping[0], mapping[1], amplitude_val, source='bridge')
            bridge_applied.append({'neuron': gid_int, 'current': amplitude_val})
        metadata['bridge_events'] = bridge_applied
        self._last_bridge_inputs = list(bridge_applied)

        if external_inputs:
            for key, value in external_inputs.items():
                if value is None or key == 'bridge_events':
                    continue
                if key.startswith('column_') and isinstance(value, dict):
                    try:
                        col_id = int(key.split('_', 1)[1])
                    except (ValueError, IndexError):
                        continue
                    for neuron_id, current in value.items():
                        add_current(col_id, neuron_id, current, source=f'user_column_{col_id}')
                elif key == 'columns' and isinstance(value, dict):
                    for col_id, mapping in value.items():
                        try:
                            col_int = int(col_id)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(mapping, dict):
                            continue
                        for neuron_id, current in mapping.items():
                            add_current(col_int, neuron_id, current, source='user_columns')
                elif key == 'global' and isinstance(value, dict):
                    for gid, current in value.items():
                        try:
                            gid_int = int(gid)
                        except (TypeError, ValueError):
                            continue
                        mapping = self._global_to_column_neuron.get(gid_int)
                        if mapping is not None:
                            add_current(mapping[0], mapping[1], current, source='user_global')
                elif key == 'regions' and isinstance(value, dict):
                    for region_key, payload in value.items():
                        region_enum = self._normalize_region(region_key)
                        if region_enum is None or region_enum not in self.brain_regions:
                            continue
                        neuron_ids = self.get_region_neurons(region_enum)
                        if not neuron_ids:
                            continue
                        if isinstance(payload, dict):
                            amplitude = payload.get('current', payload.get('amplitude', payload.get('value', 0.0)))
                            distribution = str(payload.get('distribution', 'uniform')).lower()
                        else:
                            amplitude = payload
                            distribution = 'uniform'
                        try:
                            amplitude_val = float(amplitude)
                        except (TypeError, ValueError):
                            continue
                        if distribution == 'per_neuron':
                            for gid in neuron_ids:
                                mapping = self._global_to_column_neuron.get(int(gid))
                                if mapping is not None:
                                    add_current(mapping[0], mapping[1], amplitude_val, source=f'region_{region_enum.value}')
                        else:
                            per_neuron = amplitude_val / max(1, len(neuron_ids))
                            for gid in neuron_ids:
                                mapping = self._global_to_column_neuron.get(int(gid))
                                if mapping is not None:
                                    add_current(mapping[0], mapping[1], per_neuron, source=f'region_{region_enum.value}')
                elif key == 'global_scalar':
                    try:
                        amplitude_val = float(value)
                    except (TypeError, ValueError):
                        continue
                    for gid, (col_id, neuron_id) in self._global_to_column_neuron.items():
                        add_current(col_id, neuron_id, amplitude_val, source='global_scalar')

        sanitized_inputs: Dict[int, Dict[int, float]] = {}
        for col_id, mapping in inputs_by_column.items():
            sanitized = {int(nid): float(val) for nid, val in mapping.items() if abs(val) >= 1e-12}
            if sanitized:
                sanitized_inputs[col_id] = sanitized

        self._last_column_inputs = sanitized_inputs
        metadata['sources'] = {key: int(count) for key, count in sources.items()}

        executor_inputs = {f"column_{col_id}": currents for col_id, currents in sanitized_inputs.items()}
        if sensory_metadata:
            metadata['sensory'] = dict(sensory_metadata)
        if sensory_vectors:
            for col_id, vec in sensory_vectors.items():
                key = f"column_{int(col_id)}"
                column_payload = executor_inputs.setdefault(key, {})
                column_payload["__sensory__"] = vec
        return executor_inputs, metadata

    def _accumulate_synapse_currents(self, synapse_results: Dict[int, Any]) -> None:
        aggregated: Dict[int, float] = {}
        if synapse_results and self.synapse_manager is not None:
            synapses = getattr(self.synapse_manager, 'synapses', {})
            for synapse_id, result in synapse_results.items():
                # `SynapseManager.update_all_synapses` returns a mapping from
                # postsynaptic neuron id -> current (float). Older/alternate
                # backends may return per-synapse dictionaries.
                if isinstance(result, (int, float, np.floating)):
                    try:
                        gid_int = int(synapse_id)
                        current_val = float(result)
                    except (TypeError, ValueError):
                        continue
                    if abs(current_val) < 1e-12:
                        continue
                    aggregated[gid_int] = aggregated.get(gid_int, 0.0) + current_val
                    continue

                synapse = synapses.get(synapse_id)
                if synapse is None:
                    continue
                post_gid = getattr(synapse, 'post_neuron_id', None)
                if post_gid is None:
                    continue
                try:
                    gid_int = int(post_gid)
                    current_val = float(result.get('postsynaptic_current', 0.0)) if isinstance(result, dict) else 0.0
                except (TypeError, ValueError):
                    continue
                if abs(current_val) < 1e-12:
                    continue
                aggregated[gid_int] = aggregated.get(gid_int, 0.0) + current_val
        self._last_synapse_currents = aggregated
        for gid, current in aggregated.items():
            self._pending_synaptic_currents[gid] += current

    def _update_columns_parallel(self, dt: float, external_inputs: Dict[str, Any]) -> Dict[int, Any]:
        """并行更新皮层柱（线程并行，避免 Windows 进程序列化问题）"""

        # 使用线程并行更新（最小测试规模）
        max_workers = min(8, max(1, len(self.cortical_columns)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for column_id, column in self.cortical_columns.items():
                # 准备该皮层柱的输入
                column_inputs = external_inputs.get(f"column_{column_id}", {})

                # 提交更新任务
                future = executor.submit(self._update_single_column, column, dt, column_inputs)
                futures[column_id] = future

            # 收集结果
            column_results = {}
            for column_id, future in futures.items():
                try:
                    column_results[column_id] = future.result(timeout=2.0)
                except Exception as e:
                    self.logger.error(f"Error updating column {column_id}: {e}")
                    column_results[column_id] = {'error': str(e)}

        return column_results

    def _update_single_column(self, column: "CorticalColumn", dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新单个皮层柱（兼容缺省实现）"""

        if isinstance(inputs, dict) and "__sensory__" in inputs:
            sensory_payload = inputs.get("__sensory__")
            if sensory_payload is not None and hasattr(column, "process_sensory_input"):
                try:
                    vec = np.asarray(sensory_payload, dtype=float).reshape(-1)
                    if vec.size:
                        column.process_sensory_input(vec)  # type: ignore[call-arg]
                except Exception:
                    pass

        sanitized_inputs = self._apply_column_inputs(column, inputs)

        if hasattr(column, 'update'):
            try:
                return column.update(dt, sanitized_inputs)
            except TypeError:
                try:
                    return column.update(dt)
                except TypeError:
                    pass

        try:
            result = column.step(dt)
        except TypeError:
            result = column.step(dt, sanitized_inputs)

        if isinstance(result, dict):
            result.setdefault('spikes', [])
            result.setdefault('voltages', {})
            result.setdefault('weights', {})
            return result

        if result is None:
            return {'spikes': [], 'voltages': {}, 'weights': {}}

        if isinstance(result, (list, tuple, set)):
            spikes = list(result)
        else:
            spikes = []

        return {'spikes': spikes, 'voltages': {}, 'weights': {}}

    def _apply_column_inputs(self, column: "CorticalColumn", inputs: Dict[str, Any]) -> Dict[int, float]:
        """标准化皮层柱输入并缓存到列对象上。"""

        sanitized: Dict[int, float] = {}
        if isinstance(inputs, dict):
            for neuron_id, current in inputs.items():
                try:
                    nid = int(neuron_id)
                    val = float(current)
                except (TypeError, ValueError):
                    continue
                if abs(val) < 1e-12:
                    continue
                sanitized[nid] = sanitized.get(nid, 0.0) + val

        if not sanitized:
            return {}

        try:
            column._external_inputs = dict(sanitized)  # type: ignore[attr-defined]
        except Exception:
            pass

        neuron_map = getattr(column, 'neurons', {})
        for nid, current in sanitized.items():
            neuron = neuron_map.get(nid) if isinstance(neuron_map, dict) else None
            if neuron is None:
                continue
            try:
                neuron.I_ext = float(getattr(neuron, 'I_ext', 0.0)) + current
            except Exception:
                try:
                    neuron.I_ext = current
                except Exception:
                    pass
            try:
                neuron._input_current = float(getattr(neuron, '_input_current', 0.0)) + current
            except Exception:
                try:
                    neuron._input_current = current
                except Exception:
                    pass

        return sanitized

    def _update_brain_regions(self, dt: float, column_results: Dict[int, Any]) -> Dict[BrainRegion, Any]:
        """更新脑区状态"""

        region_results = {}

        for region, region_info in self.brain_regions.items():
            # 收集该脑区所有皮层柱的活动
            region_spikes = []
            region_voltages = []

            for column_id in region_info['columns']:
                if column_id in column_results:
                    column_result = column_results[column_id]
                    region_spikes.extend(column_result.get('spikes', []))
                    region_voltages.extend(column_result.get('voltages', {}).values())

            # 计算脑区级别的统计
            spike_rate = len(region_spikes) / max(1, len(region_voltages)) if region_voltages else 0.0
            mean_voltage = np.mean(region_voltages) if region_voltages else -70.0

            # 更新神经递质水平
            self._update_neurotransmitter_levels(region_info, spike_rate, dt)

            region_results[region] = {
                'spike_rate': spike_rate,
                'mean_voltage': mean_voltage,
                'active_columns': len([c for c in region_info['columns'] if c in column_results]),
                'neurotransmitter_levels': region_info['neurotransmitter_levels'].copy(),
                'metabolic_state': region_info['metabolic_state']
            }

        return region_results

    def _update_neurotransmitter_levels(self, region_info: Dict[str, Any], spike_rate: float, dt: float):
        """更新神经递质水平"""

        nt_levels = region_info['neurotransmitter_levels']

        # 基于神经活动更新神经递质
        if spike_rate > 0.1:
            # 增加谷氨酸释放
            nt_levels['glutamate'] += spike_rate * 0.1 * dt

            # 增加GABA释放（反馈抑制）
            nt_levels['gaba'] += spike_rate * 0.05 * dt

        # 神经递质清除
        clearance_rates = {
            'glutamate': 0.1,
            'gaba': 0.05,
            'dopamine': 0.02,
            'serotonin': 0.01,
            'acetylcholine': 0.2,
            'norepinephrine': 0.03
        }

        for nt, rate in clearance_rates.items():
            if nt in nt_levels:
                nt_levels[nt] *= np.exp(-rate * dt)
                nt_levels[nt] = max(0.01, nt_levels[nt])  # 最小基础水平

    def _process_long_range_connections(self, dt: float, column_results: Dict[int, Any]) -> Dict[int, Any]:
        """处理长程连接"""

        long_range_results = {}

        for connection_id, connection in self.long_range_connections.items():
            source_column = connection['source_column']
            target_column = connection['target_column']

            if source_column in column_results and target_column in column_results:
                # 获取源皮层柱的活动
                source_spikes = column_results[source_column].get('spikes', [])

                if source_spikes:
                    # 计算传递到目标的信号强度
                    signal_strength = len(source_spikes) * connection['strength']

                    # 添加延迟（简化实现）
                    long_range_results[connection_id] = {
                        'source_activity': len(source_spikes),
                        'signal_strength': signal_strength,
                        'delay': connection['delay'],
                        'target_column': target_column
                    }

        return long_range_results

    def _update_neuromorphic_backends(self, dt: float, column_results: Dict[int, Any]) -> Dict[str, Any]:
        """更新神经形态硬件后端"""

        neuromorphic_results = {}

        # 更新Loihi后端
        if 'loihi' in self.neuromorphic_backends:
            try:
                loihi_result = self._update_loihi_backend(dt, column_results)
                neuromorphic_results['loihi'] = loihi_result
            except Exception as e:
                self.logger.error(f"Loihi backend update failed: {e}")

        # 更新SpiNNaker后端
        if 'spinnaker' in self.neuromorphic_backends:
            try:
                spinnaker_result = self._update_spinnaker_backend(dt, column_results)
                neuromorphic_results['spinnaker'] = spinnaker_result
            except Exception as e:
                self.logger.error(f"SpiNNaker backend update failed: {e}")

        return neuromorphic_results

    def _update_loihi_backend(self, dt: float, column_results: Dict[int, Any]) -> Dict[str, Any]:
        """更新Loihi后端"""

        loihi_backend = self.neuromorphic_backends['loihi']

        # 简化实现：收集映射神经元的活动
        mapped_activity = {}

        for column_id, ensemble in loihi_backend['mapped_neurons'].items():
            if column_id in column_results:
                spikes = column_results[column_id].get('spikes', [])
                mapped_activity[column_id] = len(spikes)

        return {
            'mapped_columns': len(loihi_backend['mapped_neurons']),
            'total_spikes': sum(mapped_activity.values()),
            'power_consumption': sum(mapped_activity.values()) * 0.001,  # 简化功耗模型
            'chip_utilization': len(mapped_activity) / 100.0  # 假设100个芯片
        }

    def _update_spinnaker_backend(self, dt: float, column_results: Dict[int, Any]) -> Dict[str, Any]:
        """更新SpiNNaker后端"""

        spinnaker_backend = self.neuromorphic_backends['spinnaker']

        # 简化实现：收集SpiNNaker群体的活动
        population_activity = {}

        for column_id, population in spinnaker_backend['populations'].items():
            if column_id in column_results:
                spikes = column_results[column_id].get('spikes', [])
                population_activity[column_id] = len(spikes)

        return {
            'mapped_populations': len(spinnaker_backend['populations']),
            'total_spikes': sum(population_activity.values()),
            'real_time_factor': 1.0,  # 实时运行
            'core_utilization': len(population_activity) / 18.0  # 每板18核心
        }

    def _calculate_global_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算全局统计"""

        # 收集所有皮层柱的统计
        total_spikes = 0
        total_neurons = 0
        active_columns = 0

        for column_result in results['columns'].values():
            if 'error' not in column_result:
                total_spikes += len(column_result.get('spikes', []))
                total_neurons += len(column_result.get('voltages', {}))
                active_columns += 1

        # 计算全局指标
        sampling_base = max(1, total_neurons)
        global_total_neurons = self.total_neurons if self.total_neurons > 0 else total_neurons
        sampling_fraction = total_neurons / max(1, global_total_neurons) if global_total_neurons else 0.0
        sampled_spike_rate = total_spikes / sampling_base
        spike_count_estimate = total_spikes / sampling_fraction if sampling_fraction > 1e-9 else float(total_spikes)
        network_synchrony = self._calculate_network_synchrony(results['columns'])

        # 性能指标
        mean_update_time = np.mean(self.performance_metrics['update_times'][-10:]) if self.performance_metrics['update_times'] else 0.0

        return {
            'sampled_neurons': total_neurons,
            'total_neurons': global_total_neurons,
            'total_synapses': self.total_synapses,
            'total_spikes': total_spikes,
            'spike_count_estimate': float(spike_count_estimate),
            'sampled_spike_rate': float(sampled_spike_rate),
            'estimated_spike_rate': float(sampled_spike_rate),
            'sampling_fraction': float(sampling_fraction),
            'active_columns': active_columns,
            'network_synchrony': network_synchrony,
            'mean_update_time': mean_update_time,
            'neuromorphic_active': len(self.neuromorphic_backends) > 0
        }

    def _calculate_network_synchrony(self, column_results: Dict[int, Any]) -> float:
        """计算网络同步性"""

        # 简化的同步性计算
        spike_times_by_column = {}

        for column_id, result in column_results.items():
            if 'error' not in result:
                spikes = result.get('spikes', [])
                if spikes:
                    spike_times_by_column[column_id] = len(spikes)

        if len(spike_times_by_column) < 2:
            return 0.0

        # 计算皮层柱间活动的相关性
        activities = list(spike_times_by_column.values())
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)

        # 同步性 = 1 - 标准差/均值（归一化）
        synchrony = 1.0 - (std_activity / max(mean_activity, 0.1))
        return max(0.0, min(1.0, synchrony))

    def set_input(self, inputs: List[float]) -> None:
        """Record the most recent external input signal."""

        super().set_input(inputs)

    def reset(self) -> None:
        """Reset simulation state for software backends."""

        super().reset()

    def step(self, dt: float) -> Dict[str, Any]:
        """Advance the network by ``dt`` milliseconds.

        This lightweight implementation drives a leaky integrate-and-fire update
        across a representative subset of the simulated neurons so that higher
        level modules receive meaningful spike and voltage telemetry during
        smoke tests.
        """

        self._ensure_initialized()
        if not self._runtime_prepared:
            self._prepare_runtime_views()

        if not self._runtime_neurons:
            return super().step(dt)

        current_time = self.global_step * dt
        synaptic_currents: Dict[Tuple[int, int], float] = {
            key: 0.0 for key in self._runtime_neuron_index.keys()
        }

        for entry in self._runtime_synapses:
            syn = entry['synapse']
            try:
                current = syn.update(dt, current_time)
            except TypeError:
                current = syn.update(dt)
            if current != 0.0:
                post_key = entry['post_key']
                if post_key in synaptic_currents:
                    synaptic_currents[post_key] += current

        spikes: List[Dict[str, Any]] = []
        spikes_global: List[int] = []
        voltages_local: Dict[str, float] = {}
        voltages_global: Dict[int, float] = {}
        voltage_values: List[float] = []

        for idx, (column_id, neuron_id, neuron) in enumerate(self._runtime_neurons):
            key = (column_id, neuron_id)
            total_current = synaptic_currents.get(key, 0.0)

            if idx < len(self._baseline_currents):
                total_current += float(self._baseline_currents[idx])

            if self._input_buffer:
                if idx < len(self._input_buffer):
                    total_current += float(self._input_buffer[idx]) * self._input_scale
                else:
                    total_current += float(np.mean(self._input_buffer)) * self._input_scale

            if self._noise_std > 0.0:
                total_current += float(np.random.normal(0.0, self._noise_std))

            spike, voltage = self._lif_integrate(neuron, idx, total_current, dt, current_time)
            key_str = self._runtime_global_keys[idx] if idx < len(self._runtime_global_keys) else f"{column_id}:{neuron_id}"
            volts = float(voltage)
            voltages_local[key_str] = volts
            gid = None
            if idx < len(self._runtime_global_ids):
                gid = self._runtime_global_ids[idx]
                if gid is not None:
                    voltages_global[gid] = volts
            voltage_values.append(voltage)

            if spike:
                spike_entry = {
                    'column': column_id,
                    'neuron': neuron_id,
                    'time_ms': current_time + dt,
                }
                if gid is None and idx < len(self._runtime_global_ids):
                    gid = self._runtime_global_ids[idx]
                if gid is not None:
                    spike_entry['neuron_global'] = gid
                    spikes_global.append(gid)
                spikes.append(spike_entry)
                for syn_entry in self._runtime_synapses_by_pre.get((column_id, neuron_id), []):
                    syn_entry['synapse'].process_spike(current_time + dt)

        self.global_step += 1

        weights: Dict[str, float] = {}
        for entry in self._runtime_synapses[: self._weight_sample_limit]:
            weights[entry['key']] = float(entry['synapse'].weight)

        avg_voltage = float(np.mean(voltage_values)) if voltage_values else float('nan')

        if voltages_global:
            self._last_neuron_voltages = voltages_global
        else:
            self._last_neuron_voltages = {idx: float(val) for idx, val in enumerate(voltage_values)}
        self._last_global_activity = {
            'time_ms': current_time + dt,
            'spike_count': len(spikes),
            'spike_count_global': len(spikes_global),
            'avg_voltage': avg_voltage,
            'synapse_currents': dict(self._last_synapse_currents),
            'bridge_events': list(self._last_bridge_inputs),
        }
        self.performance_metrics['spike_counts'].append(len(spikes))

        return {'spikes': spikes, 'spikes_global': spikes_global, 'voltages': voltages_local, 'voltages_global': voltages_global, 'weights': weights}

    def _prepare_runtime_views(self) -> None:
        """Select a manageable subset of neurons/synapses for lightweight simulation."""

        if self._runtime_prepared:
            return

        neurons: List[Tuple[int, int, DetailedNeuron]] = []
        for column_id, column in self.cortical_columns.items():
            for neuron_id, neuron in column.neurons.items():
                neurons.append((column_id, neuron_id, neuron))

        if not neurons:
            self._runtime_prepared = True
            return

        sample_limit = max(1, min(self._max_sample_neurons, len(neurons)))
        if len(neurons) > sample_limit:
            indices = np.random.choice(len(neurons), size=sample_limit, replace=False)
            indices.sort()
            selected = [neurons[i] for i in indices]
        else:
            selected = neurons

        self._runtime_neurons = selected
        self._runtime_global_keys = []
        self._runtime_global_ids = []
        for column_id, neuron_id, _ in selected:
            self._runtime_global_keys.append(f"{column_id}:{neuron_id}")
            self._runtime_global_ids.append(self._column_neuron_to_global.get((int(column_id), int(neuron_id))))
        self._runtime_neuron_index = {
            (column_id, neuron_id): idx for idx, (column_id, neuron_id, _) in enumerate(selected)
        }

        selected_by_column: Dict[int, set] = {}
        for column_id, neuron_id, _ in selected:
            selected_by_column.setdefault(column_id, set()).add(neuron_id)

        class _RuntimeSynapse:
            __slots__ = ("weight", "delay", "_pending")

            def __init__(self, weight: float, delay: float):
                self.weight = float(weight)
                self.delay = float(delay)
                self._pending: Deque[float] = deque()

            def process_spike(self, spike_time: float) -> None:
                self._pending.append(float(spike_time) + self.delay)

            def update(self, dt: float, current_time: float) -> float:
                if not self._pending:
                    return 0.0
                window_end = float(current_time) + float(dt)
                delivered = 0.0
                while self._pending and self._pending[0] <= window_end:
                    event_time = self._pending.popleft()
                    if event_time > float(current_time) - 1e-9:
                        delivered += self.weight
                return delivered

        synapse_entries: List[Dict[str, Any]] = []
        synapses_by_pre: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for column_id, column in self.cortical_columns.items():
            selected_ids = selected_by_column.get(column_id)
            if not selected_ids:
                continue
            for synapse_id, synapse in column.synapses.items():
                pre_id: Optional[int] = None
                post_id: Optional[int] = None
                synapse_obj: Any = synapse

                if hasattr(synapse, "pre_neuron") and hasattr(synapse, "post_neuron"):
                    pre_id = int(getattr(synapse.pre_neuron, "neuron_id"))
                    post_id = int(getattr(synapse.post_neuron, "neuron_id"))
                elif isinstance(synapse, dict):
                    pre_id = synapse.get("pre")
                    post_id = synapse.get("post")
                    if pre_id is None:
                        pre_id = synapse.get("pre_neuron_id")
                    if post_id is None:
                        post_id = synapse.get("post_neuron_id")
                    try:
                        weight = float(synapse.get("weight", 0.0))
                    except Exception:
                        weight = 0.0
                    try:
                        delay = float(synapse.get("delay", 0.0))
                    except Exception:
                        delay = 0.0
                    synapse_obj = _RuntimeSynapse(weight=weight, delay=max(0.0, delay))

                if pre_id is None or post_id is None:
                    continue

                if pre_id in selected_ids and post_id in selected_ids:
                    syn_id_key = synapse_id
                    if isinstance(syn_id_key, tuple):
                        syn_key_str = ":".join(str(part) for part in syn_id_key)
                    else:
                        syn_key_str = str(syn_id_key)
                    entry = {
                        'column_id': column_id,
                        'synapse_id': synapse_id,
                        'synapse': synapse_obj,
                        'post_key': (column_id, post_id),
                        'pre_key': (column_id, pre_id),
                        'key': f"{column_id}:{syn_key_str}",
                    }
                    synapse_entries.append(entry)
                    synapses_by_pre.setdefault((column_id, pre_id), []).append(entry)

        self._runtime_synapses = synapse_entries
        self._runtime_synapses_by_pre = synapses_by_pre
        self._runtime_states = [{'refractory': 0.0} for _ in selected]

        if selected:
            currents = np.random.normal(
                loc=self._baseline_mean,
                scale=self._baseline_std,
                size=len(selected),
            ).astype(float)
            self._baseline_currents = np.maximum(currents, 0.0)
        else:
            self._baseline_currents = np.zeros(0, dtype=float)

        self._runtime_prepared = True

    def _lif_integrate(
        self,
        neuron: DetailedNeuron,
        idx: int,
        total_current: float,
        dt: float,
        current_time: float,
    ) -> Tuple[bool, float]:
        """Single step leaky integrate-and-fire update for a neuron."""

        params = getattr(neuron, "params", {})

        def _param(name: str, default: float, *aliases: str) -> float:
            if isinstance(params, dict):
                for key in (name,) + tuple(aliases):
                    if key in params:
                        try:
                            return float(params[key])
                        except Exception:
                            break
                return float(default)
            for key in (name,) + tuple(aliases):
                if hasattr(params, key):
                    try:
                        return float(getattr(params, key))
                    except Exception:
                        break
            return float(default)

        def _get_voltage() -> float:
            for attr in ("membrane_potential", "voltage", "V"):
                if hasattr(neuron, attr):
                    try:
                        return float(getattr(neuron, attr))
                    except Exception:
                        continue
            return _param("resting_potential", -65.0, "V_rest", "E_L")

        def _set_voltage(value: float) -> None:
            for attr in ("membrane_potential", "voltage", "V"):
                if hasattr(neuron, attr):
                    try:
                        setattr(neuron, attr, float(value))
                        return
                    except Exception:
                        continue

        state = self._runtime_states[idx]
        refractory = float(state.get('refractory', 0.0))
        if refractory > 0.0:
            refractory = max(0.0, refractory - dt)
            state['refractory'] = refractory
            reset_potential = _param("reset_potential", -65.0, "V_reset")
            _set_voltage(reset_potential)
            return False, reset_potential

        rest = _param("resting_potential", -65.0, "V_rest", "E_L")
        threshold = _param("threshold", -50.0, "V_thresh", "V_T")
        reset = _param("reset_potential", rest, "V_reset")
        tau = self._lif_tau if self._lif_tau > 1e-6 else 10.0
        decay = self._lif_decay

        voltage = _get_voltage()
        dv = ((-(voltage - rest) * decay) + total_current) * (dt / tau)
        voltage = float(np.clip(voltage + dv, rest - 40.0, threshold + 40.0))
        spike = voltage >= threshold

        if spike:
            voltage = reset
            state['refractory'] = _param("refractory_period", 2.0, "t_ref")
            neuron.spike_times.append(current_time + dt)
            neuron.last_spike_time = current_time + dt
        else:
            state['refractory'] = 0.0

        _set_voltage(voltage)
        return spike, voltage
