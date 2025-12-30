"""
认知架构模块
Cognitive Architecture Module

定义了 CognitiveArchitecture 类，用于构建和管理整个大脑的认知模型。
"""
import asyncio
import heapq
import logging
import os
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import networkx as nx
except ImportError:
    # 提供一个轻量级的替代品，以防 networkx 未安装
    class _FallbackGraph:
        def __init__(self, *args, **kwargs):
            self._nodes = {}
            self._edges = []
        def add_node(self, node, **attrs):
            self._nodes[node] = attrs
        def add_edge(self, u, v, **attrs):
            self._edges.append((u, v, attrs))
        def number_of_edges(self):
            return len(self._edges)
    class _FallbackNetworkX:
        DiGraph = _FallbackGraph
    nx = _FallbackNetworkX()

from .enums import BrainRegion, CognitiveFunction, OscillationBand
from .states import AttentionState, ConsciousnessState
from .regions import PhysiologicalBrainRegion
from ..core.parallel_execution import RegionParallelExecutor, RegionUpdateTask
from ..core.module_interface import ModuleBus, ModuleSignal, ModuleTopic
from .functional_topology import FunctionalTopologyRouter
from .parameter_mapping import ModuleParameterMapper

try:
    from ..memory.memory_consolidation import SleepStage as ConsolidationSleepStage
except Exception:  # pragma: no cover - optional dependency for sleep/memory coupling
    ConsolidationSleepStage = None

class CognitiveArchitecture:
    """认知架构"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 脑区网络
        self.brain_regions: Dict[BrainRegion, PhysiologicalBrainRegion] = {}
        self.region_graph = nx.DiGraph()
        
        # 认知状态
        self.attention_state = AttentionState()
        self.consciousness_state = ConsciousnessState()
        
        # 全局调节系统
        self.neuromodulation = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'acetylcholine': 0.5,
            'norepinephrine': 0.5
        }
        
        # 认知功能映射
        self.function_region_mapping = self._initialize_function_mapping()
        
        # 性能监控
        self.cognitive_load = 0.0
        self.processing_efficiency = 1.0
        
        self.logger = logging.getLogger("CognitiveArchitecture")

        runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config, dict) else {}
        update_cfg = runtime_cfg.get("region_update", {}) if isinstance(runtime_cfg, dict) else {}
        parallel_cfg = update_cfg.get("parallel", {}) if isinstance(update_cfg, dict) else {}

        # Optional functional->anatomical mapping bridge (modules/brain BrainFunctionalTopology).
        topo_cfg = runtime_cfg.get("functional_topology", {}) if isinstance(runtime_cfg, dict) else {}
        self._functional_topology_router = None
        if isinstance(topo_cfg, dict):
            try:
                if bool(topo_cfg.get("enabled", False)):
                    self._functional_topology_router = FunctionalTopologyRouter(topo_cfg, logger=self.logger)
            except Exception as exc:
                self.logger.warning("Functional topology routing disabled: %s", exc)
                self._functional_topology_router = None

        # Optional: map cognitive-module parameters into microcircuit parameters (coarse calibration bridge).
        mapping_cfg = runtime_cfg.get("parameter_mapping", {}) if isinstance(runtime_cfg, dict) else {}
        self._module_parameter_mapper = None
        self._parameter_mapping_applied: Dict[str, Dict[str, Any]] = {}
        if isinstance(mapping_cfg, dict):
            try:
                if bool(mapping_cfg.get("enabled", False)):
                    self._module_parameter_mapper = ModuleParameterMapper(mapping_cfg, logger=self.logger)
            except Exception as exc:
                self.logger.warning("Parameter mapping disabled: %s", exc)
                self._module_parameter_mapper = None

        # Standardised pub/sub interface for cognition <-> microcircuits (ModuleBus).
        bus_cfg = runtime_cfg.get("module_bus", {}) if isinstance(runtime_cfg, dict) else {}
        if not isinstance(bus_cfg, dict):
            bus_cfg = {}

        try:
            self._module_bus_enabled = bool(bus_cfg.get("enabled", True))
        except Exception:
            self._module_bus_enabled = True

        try:
            self._module_bus_manage_cycle = bool(bus_cfg.get("manage_cycle", True))
        except Exception:
            self._module_bus_manage_cycle = True

        try:
            self._module_bus_export = bool(bus_cfg.get("export_in_results", True))
        except Exception:
            self._module_bus_export = True

        try:
            self._bus_publish_cognitive_state = bool(bus_cfg.get("publish_cognitive_state", True))
        except Exception:
            self._bus_publish_cognitive_state = True

        try:
            self._bus_publish_micro_events = bool(bus_cfg.get("publish_microcircuit_events", True))
        except Exception:
            self._bus_publish_micro_events = True

        thresholds = bus_cfg.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        try:
            self._bus_v1_salience_rate_hz = float(thresholds.get("v1_salience_rate_hz", 20.0))
        except Exception:
            self._bus_v1_salience_rate_hz = 20.0
        try:
            self._bus_ca1_retrieval_rate_hz = float(thresholds.get("ca1_retrieval_rate_hz", 12.0))
        except Exception:
            self._bus_ca1_retrieval_rate_hz = 12.0
        try:
            self._bus_high_activity_rate_hz = float(thresholds.get("high_activity_rate_hz", 25.0))
        except Exception:
            self._bus_high_activity_rate_hz = 25.0

        self.module_bus: ModuleBus = ModuleBus()
        for topic in ModuleTopic:
            self.module_bus.register_topic(topic)
        self._module_bus_inbox: List[ModuleSignal] = []

        parallel_enabled = True
        try:
            parallel_enabled = bool(parallel_cfg.get("enabled", True)) if isinstance(parallel_cfg, dict) else True
        except Exception:
            parallel_enabled = True

        parallel_strategy = "auto"
        try:
            parallel_strategy = str(parallel_cfg.get("strategy", "auto") or "auto").strip().lower()
        except Exception:
            parallel_strategy = "auto"

        max_workers = None
        try:
            raw = parallel_cfg.get("max_workers") if isinstance(parallel_cfg, dict) else None
            if raw is not None:
                max_workers = int(raw)
        except Exception:
            max_workers = None

        if max_workers is None:
            try:
                max_workers = max(1, min(8, int(os.cpu_count() or 2)))
            except Exception:
                max_workers = 4

        if parallel_strategy in {"process", "distributed"}:
            self.logger.warning(
                "Parallel strategy '%s' requested for region updates; falling back to 'thread' (region state is not process-safe).",
                parallel_strategy,
            )
            parallel_strategy = "thread"

        self._region_executor = (
            RegionParallelExecutor(strategy=parallel_strategy, max_workers=max_workers, logger=self.logger)
            if parallel_enabled
            else None
        )

        self._region_update_mode = "full"
        try:
            self._region_update_mode = str(update_cfg.get("mode", "full") or "full").strip().lower()
        except Exception:
            self._region_update_mode = "full"

        if self._region_update_mode not in {"full", "event_driven"}:
            self.logger.warning(
                "Unsupported region_update.mode '%s'; falling back to 'full'.", self._region_update_mode
            )
            self._region_update_mode = "full"

        event_cfg = update_cfg.get("event_driven", {}) if isinstance(update_cfg, dict) else {}
        self._event_queue: List[Any] = []
        self._event_counter = 0
        self._sim_time_ms = 0.0

        try:
            self._event_input_epsilon = float(event_cfg.get("input_epsilon", 1e-3))
        except Exception:
            self._event_input_epsilon = 1e-3

        try:
            self._event_activation_threshold = float(event_cfg.get("activation_threshold", 0.05))
        except Exception:
            self._event_activation_threshold = 0.05

        try:
            self._event_max_pending = int(event_cfg.get("max_pending_events", 200_000))
        except Exception:
            self._event_max_pending = 200_000

        try:
            self._event_max_events_per_cycle = int(event_cfg.get("max_events_per_cycle", 50_000))
        except Exception:
            self._event_max_events_per_cycle = 50_000

        self._event_force_update_regions = set()
        try:
            raw_force = event_cfg.get("always_update", [])
            if isinstance(raw_force, (list, tuple, set)):
                name_to_region = {r.value: r for r in BrainRegion}
                for entry in raw_force:
                    key = str(entry).strip().lower()
                    if key in name_to_region:
                        self._event_force_update_regions.add(name_to_region[key])
        except Exception:
            self._event_force_update_regions = set()

        # Sleep / slow physiology hooks (optional; defaults keep behavior unchanged for short runs)
        sleep_cfg = runtime_cfg.get("sleep", {}) if isinstance(runtime_cfg, dict) else {}
        self._sleep_cfg = dict(sleep_cfg or {}) if isinstance(sleep_cfg, dict) else {}

        try:
            self._sleep_enabled = bool(self._sleep_cfg.get("enabled", False))
        except Exception:
            self._sleep_enabled = False

        try:
            self._sleep_auto = bool(self._sleep_cfg.get("auto", False))
        except Exception:
            self._sleep_auto = False

        self._sleep_force_stage: str | None = None
        forced = self._sleep_cfg.get("force_stage", None)
        if forced is not None:
            self._sleep_force_stage = str(forced).strip().lower() or None

        try:
            self._sleep_onset_ms = float(self._sleep_cfg.get("sleep_onset_ms", 5_000.0))
        except Exception:
            self._sleep_onset_ms = 5_000.0
        if not np.isfinite(self._sleep_onset_ms) or self._sleep_onset_ms < 0.0:
            self._sleep_onset_ms = 5_000.0

        try:
            self._sleep_cycle_ms = float(self._sleep_cfg.get("cycle_ms", 90_000.0))
        except Exception:
            self._sleep_cycle_ms = 90_000.0
        if not np.isfinite(self._sleep_cycle_ms) or self._sleep_cycle_ms <= 0.0:
            self._sleep_cycle_ms = 90_000.0

        stage_ratios = self._sleep_cfg.get("stage_ratios", {})
        if not isinstance(stage_ratios, dict):
            stage_ratios = {}
        try:
            self._sleep_ratio_n2 = float(stage_ratios.get("n2", 0.4))
        except Exception:
            self._sleep_ratio_n2 = 0.4
        try:
            self._sleep_ratio_n3 = float(stage_ratios.get("n3", 0.4))
        except Exception:
            self._sleep_ratio_n3 = 0.4
        try:
            self._sleep_ratio_rem = float(stage_ratios.get("rem", 0.2))
        except Exception:
            self._sleep_ratio_rem = 0.2

        ratios = np.array([self._sleep_ratio_n2, self._sleep_ratio_n3, self._sleep_ratio_rem], dtype=float)
        ratios = np.clip(ratios, 0.0, np.inf)
        total = float(np.sum(ratios))
        if not np.isfinite(total) or total <= 1e-9:
            ratios = np.array([0.4, 0.4, 0.2], dtype=float)
            total = float(np.sum(ratios))
        ratios = ratios / total
        self._sleep_ratio_n2, self._sleep_ratio_n3, self._sleep_ratio_rem = (float(r) for r in ratios)

        self._sleep_stage = "wake"
        self._sleep_stage_entered_ms = 0.0
        self._sleep_last_replay_ms = -1e12

        homeo_cfg = self._sleep_cfg.get("synaptic_homeostasis", {})
        if not isinstance(homeo_cfg, dict):
            homeo_cfg = {}
        try:
            self._sleep_downscale_rate_per_s = float(homeo_cfg.get("downscale_rate_per_s", 0.0))
        except Exception:
            self._sleep_downscale_rate_per_s = 0.0
        if not np.isfinite(self._sleep_downscale_rate_per_s) or self._sleep_downscale_rate_per_s < 0.0:
            self._sleep_downscale_rate_per_s = 0.0
        try:
            self._sleep_downscale_exc_only = bool(homeo_cfg.get("exc_only", True))
        except Exception:
            self._sleep_downscale_exc_only = True

        replay_cfg = self._sleep_cfg.get("replay", {})
        if not isinstance(replay_cfg, dict):
            replay_cfg = {}
        try:
            self._sleep_replay_enabled = bool(replay_cfg.get("enabled", True))
        except Exception:
            self._sleep_replay_enabled = True
        self._sleep_replay_mode = str(replay_cfg.get("mode", "interval") or "interval").strip().lower()
        try:
            self._sleep_replay_interval_ms = float(replay_cfg.get("interval_ms", 250.0))
        except Exception:
            self._sleep_replay_interval_ms = 250.0
        if not np.isfinite(self._sleep_replay_interval_ms) or self._sleep_replay_interval_ms < 0.0:
            self._sleep_replay_interval_ms = 250.0
        try:
            self._sleep_replay_probability_per_s = float(replay_cfg.get("probability_per_s", 0.0))
        except Exception:
            self._sleep_replay_probability_per_s = 0.0
        if not np.isfinite(self._sleep_replay_probability_per_s) or self._sleep_replay_probability_per_s < 0.0:
            self._sleep_replay_probability_per_s = 0.0
        try:
            self._sleep_replay_input_strength = float(replay_cfg.get("input_strength", 0.75))
        except Exception:
            self._sleep_replay_input_strength = 0.75
        if not np.isfinite(self._sleep_replay_input_strength) or self._sleep_replay_input_strength < 0.0:
            self._sleep_replay_input_strength = 0.75
        
        # 初始化脑区
        self._initialize_brain_regions()
        self._establish_connections()

        # Optional cross-system links (wired by the complete brain system)
        self.memory_system = None

    def set_memory_system(self, memory_system: Any) -> None:
        """Attach a memory system implementation for downstream use."""
        self.memory_system = memory_system

    def set_module_bus(self, bus: ModuleBus, *, manage_cycle: Optional[bool] = None) -> None:
        """Attach an external ModuleBus managed by a higher-level orchestrator."""

        if bus is None:
            return
        self.module_bus = bus
        for topic in ModuleTopic:
            try:
                self.module_bus.register_topic(topic)
            except Exception:
                continue
        if manage_cycle is None:
            manage_cycle = False
        try:
            self._module_bus_manage_cycle = bool(manage_cycle)
        except Exception:
            self._module_bus_manage_cycle = False

    def queue_module_signal(self, signal: ModuleSignal) -> None:
        """Queue a signal to be delivered on the next cognitive cycle."""

        if not getattr(self, "_module_bus_enabled", True):
            return
        inbox = getattr(self, "_module_bus_inbox", None)
        if inbox is None:
            self._module_bus_inbox = []
            inbox = self._module_bus_inbox
        try:
            inbox.append(signal)
        except Exception:
            return

    def publish_module_signal(
        self,
        topic: ModuleTopic,
        payload: Dict[str, Any],
        *,
        source: str = "cognitive_architecture",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModuleSignal]:
        if not getattr(self, "_module_bus_enabled", True):
            return None
        try:
            signal = ModuleSignal(topic=topic, payload=dict(payload or {}), source=str(source), metadata=metadata or {})
        except Exception:
            return None
        try:
            self.module_bus.publish(signal)
        except Exception:
            return None
        return signal

    def set_sleep_stage(self, stage: Any) -> None:
        """Force a sleep stage ("wake"/"n2"/"n3"/"rem") for downstream coupling."""

        if stage is None:
            self._sleep_force_stage = None
            return

        if hasattr(stage, "value"):
            try:
                stage = stage.value
            except Exception:
                pass

        value = str(stage).strip().lower()
        if not value:
            self._sleep_force_stage = None
            return

        aliases = {
            "awake": "wake",
            "w": "wake",
            "nrem": "n3",
            "sws": "n3",
        }
        value = aliases.get(value, value)
        if value not in {"wake", "n1", "n2", "n3", "rem"}:
            raise ValueError(f"Unsupported sleep stage: {value}")
        self._sleep_force_stage = value

    def _update_sleep_stage(self, current_time_ms: float) -> str:
        if not bool(getattr(self, "_sleep_enabled", False)):
            return "wake"

        forced = getattr(self, "_sleep_force_stage", None)
        if isinstance(forced, str) and forced:
            return forced

        if not bool(getattr(self, "_sleep_auto", False)):
            return str(getattr(self, "_sleep_stage", "wake") or "wake")

        onset_ms = float(getattr(self, "_sleep_onset_ms", 0.0) or 0.0)
        if float(current_time_ms) < onset_ms:
            return "wake"

        cycle_ms = float(getattr(self, "_sleep_cycle_ms", 90_000.0) or 90_000.0)
        if not np.isfinite(cycle_ms) or cycle_ms <= 0.0:
            cycle_ms = 90_000.0

        t = (float(current_time_ms) - onset_ms) % cycle_ms
        n2_end = float(getattr(self, "_sleep_ratio_n2", 0.4)) * cycle_ms
        n3_end = (float(getattr(self, "_sleep_ratio_n2", 0.4)) + float(getattr(self, "_sleep_ratio_n3", 0.4))) * cycle_ms

        if t < n2_end:
            return "n2"
        if t < n3_end:
            return "n3"
        return "rem"

    def _propagate_sleep_stage(self, stage: str) -> None:
        memory_system = getattr(self, "memory_system", None)
        if memory_system is None:
            return

        stage_value: Any = stage
        if ConsolidationSleepStage is not None:
            mapping = {
                "wake": ConsolidationSleepStage.WAKE,
                "n1": ConsolidationSleepStage.N1,
                "n2": ConsolidationSleepStage.N2,
                "n3": ConsolidationSleepStage.N3,
                "rem": ConsolidationSleepStage.REM,
            }
            stage_value = mapping.get(stage, ConsolidationSleepStage.WAKE)

        for candidate in (memory_system, getattr(memory_system, "consolidation_manager", None)):
            setter = getattr(candidate, "set_sleep_stage", None)
            if callable(setter):
                try:
                    setter(stage_value)
                except Exception:
                    continue
                break

    def _inject_sleep_replay_inputs(
        self, region_inputs: Dict[BrainRegion, Dict[str, float]], current_time_ms: float, dt_ms: float
    ) -> Dict[str, Any]:
        """Optionally inject sleep replay drive into hippocampus/PFC inputs."""

        stage = str(getattr(self, "_sleep_stage", "wake") or "wake")
        if stage not in {"n3", "rem"}:
            return {"enabled": False}
        if not bool(getattr(self, "_sleep_replay_enabled", True)):
            return {"enabled": False}

        mode = str(getattr(self, "_sleep_replay_mode", "interval") or "interval").strip().lower()
        strength = float(getattr(self, "_sleep_replay_input_strength", 0.75) or 0.75)
        if not np.isfinite(strength) or strength <= 0.0:
            strength = 0.0

        should_trigger = False
        if mode == "interval":
            interval_ms = float(getattr(self, "_sleep_replay_interval_ms", 250.0) or 250.0)
            if interval_ms <= 0.0:
                should_trigger = True
            else:
                last = float(getattr(self, "_sleep_last_replay_ms", -1e12) or -1e12)
                if float(current_time_ms) - last >= interval_ms:
                    should_trigger = True
        else:
            try:
                p = float(getattr(self, "_sleep_replay_probability_per_s", 0.0) or 0.0) * (float(dt_ms) / 1000.0)
            except Exception:
                p = 0.0
            if np.isfinite(p) and p > 0.0 and np.random.random() < min(p, 1.0):
                should_trigger = True

        if not should_trigger or strength <= 0.0:
            return {"enabled": True, "triggered": False, "stage": stage}

        region_inputs.setdefault(BrainRegion.HIPPOCAMPUS, {})
        region_inputs.setdefault(BrainRegion.PREFRONTAL_CORTEX, {})
        region_inputs[BrainRegion.HIPPOCAMPUS]["sleep_replay"] = float(region_inputs[BrainRegion.HIPPOCAMPUS].get("sleep_replay", 0.0)) + strength
        region_inputs[BrainRegion.PREFRONTAL_CORTEX]["sleep_replay"] = float(
            region_inputs[BrainRegion.PREFRONTAL_CORTEX].get("sleep_replay", 0.0)
        ) + (0.7 * strength)

        self._sleep_last_replay_ms = float(current_time_ms)
        return {"enabled": True, "triggered": True, "stage": stage, "strength": float(strength)}

    def _apply_sleep_synaptic_homeostasis(self, dt_ms: float) -> Dict[str, Any]:
        stage = str(getattr(self, "_sleep_stage", "wake") or "wake")
        if stage != "n3":
            return {"enabled": False, "stage": stage}

        rate = float(getattr(self, "_sleep_downscale_rate_per_s", 0.0) or 0.0)
        if not np.isfinite(rate) or rate <= 0.0:
            return {"enabled": False, "stage": stage}

        dt_s = max(float(dt_ms), 0.0) / 1000.0
        factor = float(np.exp(-rate * dt_s))
        exc_only = bool(getattr(self, "_sleep_downscale_exc_only", True))

        scaled_regions = 0
        scaled_synapses = 0
        for region in self.brain_regions.values():
            micro = getattr(region, "microcircuit", None)
            if micro is None:
                continue
            scaler = getattr(micro, "scale_synapses", None)
            if not callable(scaler):
                continue
            try:
                summary = scaler(factor, exc_only=exc_only, inh_only=not exc_only) or {}
            except Exception:
                continue
            scaled_regions += 1
            try:
                scaled_synapses += int(summary.get("scaled", 0) or 0)
            except Exception:
                pass

        return {
            "enabled": True,
            "stage": stage,
            "factor": float(factor),
            "scaled_regions": int(scaled_regions),
            "scaled_synapses": int(scaled_synapses),
        }

    def _step_external_microcircuit_engines(
        self, dt_ms: float, region_inputs: Dict[BrainRegion, Dict[str, float]]
    ) -> tuple[set[BrainRegion], Dict[str, Any]]:
        """Step process-global microcircuit engines (e.g., NEST) once per cognitive cycle."""

        engines: Dict[int, Any] = {}
        forced_regions: set[BrainRegion] = set()
        prepared = 0

        for region_type, region in self.brain_regions.items():
            micro = getattr(region, "microcircuit", None)
            if micro is None:
                continue
            if not bool(getattr(micro, "requires_global_step", False)):
                continue

            prepare = getattr(micro, "prepare_step_inputs", None)
            if callable(prepare):
                try:
                    prepare(dt_ms, region_inputs.get(region_type, {}) or {}, self.neuromodulation)
                    prepared += 1
                except Exception as exc:
                    self.logger.warning("Microcircuit prepare failed for %s: %s", region_type.value, exc)

            added = False
            micro_engines = getattr(micro, "engines", None)
            if micro_engines is not None and not isinstance(micro_engines, (str, bytes)):
                try:
                    for candidate in list(micro_engines):
                        if candidate is None:
                            continue
                        engines[id(candidate)] = candidate
                        added = True
                except Exception:
                    added = False

            if not added:
                engine = getattr(micro, "engine", None)
                if engine is None:
                    engine = getattr(micro, "_engine", None)
                if engine is not None:
                    engines[id(engine)] = engine

            if bool(getattr(micro, "force_update_each_cycle", False)):
                forced_regions.add(region_type)

        summaries: list[Dict[str, Any]] = []
        for engine in engines.values():
            stepper = getattr(engine, "step", None)
            if not callable(stepper):
                continue
            try:
                summary = stepper(dt_ms)
            except Exception as exc:
                self.logger.warning("External microcircuit engine step failed: %s", exc)
                continue
            if isinstance(summary, dict):
                summaries.append(summary)

        info = {
            "engines": int(len(engines)),
            "prepared_microcircuits": int(prepared),
            "summaries": summaries,
        }
        return forced_regions, info
    
    def _initialize_brain_regions(self):
        """初始化脑区"""
        
        region_configs = self.config.get('brain_regions', {})
        
        # 创建主要脑区
        major_regions = [
            BrainRegion.PREFRONTAL_CORTEX,
            BrainRegion.MOTOR_CORTEX,
            BrainRegion.SOMATOSENSORY_CORTEX,
            BrainRegion.VISUAL_CORTEX,
            BrainRegion.AUDITORY_CORTEX,
            BrainRegion.PARIETAL_CORTEX,
            BrainRegion.TEMPORAL_CORTEX,
            BrainRegion.HIPPOCAMPUS,
            BrainRegion.AMYGDALA,
            BrainRegion.THALAMUS,
            BrainRegion.BASAL_GANGLIA,
            BrainRegion.CEREBELLUM
        ]
        
        for region_type in major_regions:
            region_config = region_configs.get(region_type.value, {})
            
            # 设置默认参数
            default_config = self._get_default_region_config(region_type)
            default_config.update(region_config)

            # Optional: apply module->microcircuit parameter mapping before instantiation.
            mapper = getattr(self, "_module_parameter_mapper", None)
            if mapper is not None:
                try:
                    default_config, applied = mapper.apply_to_region_config(
                        region_type, default_config, global_config=self.config
                    )
                    if isinstance(applied, dict) and applied:
                        self._parameter_mapping_applied[region_type.value] = dict(applied)
                except Exception as exc:
                    self.logger.warning("Parameter mapping failed for %s: %s", region_type.value, exc)
            
            # 创建脑区
            brain_region = PhysiologicalBrainRegion(region_type, default_config)
            self.brain_regions[region_type] = brain_region
            
            # 添加到图
            try:
                self.region_graph.add_node(region_type, region=brain_region)
            except TypeError:
                self.region_graph.add_node(region_type)
    
    def _get_default_region_config(self, region_type: BrainRegion) -> Dict[str, Any]:
        """获取默认脑区配置"""
        
        default_configs = {
            BrainRegion.PREFRONTAL_CORTEX: {
                'volume': 15000.0,
                'surface_area': 2000.0,
                'thickness': 3.0,
                'neuron_density': 80000
            },
            BrainRegion.HIPPOCAMPUS: {
                'volume': 4000.0,
                'surface_area': 500.0,
                'thickness': 1.0,
                'neuron_density': 150000
            },
            BrainRegion.VISUAL_CORTEX: {
                'volume': 20000.0,
                'surface_area': 3000.0,
                'thickness': 2.0,
                'neuron_density': 120000
            },
            BrainRegion.MOTOR_CORTEX: {
                'volume': 8000.0,
                'surface_area': 1200.0,
                'thickness': 4.0,
                'neuron_density': 100000
            },
            BrainRegion.THALAMUS: {
                'volume': 6000.0,
                'surface_area': 800.0,
                'thickness': 0.0,  # 不是皮层结构
                'neuron_density': 200000
            },
            BrainRegion.CEREBELLUM: {
                'volume': 150000.0,
                'surface_area': 10000.0,
                'thickness': 1.5,
                'neuron_density': 300000  # 小脑颗粒细胞密度很高
            }
        }
        
        return default_configs.get(region_type, {
            'volume': 5000.0,
            'surface_area': 1000.0,
            'thickness': 2.5,
            'neuron_density': 100000
        })
    
    def _establish_connections(self):
        """建立脑区连接"""
        
        # 主要连接路径
        connections = [
            # 皮层-皮层连接
            (BrainRegion.VISUAL_CORTEX, BrainRegion.PARIETAL_CORTEX, 0.8),
            (BrainRegion.VISUAL_CORTEX, BrainRegion.TEMPORAL_CORTEX, 0.7),
            (BrainRegion.PARIETAL_CORTEX, BrainRegion.PREFRONTAL_CORTEX, 0.9),
            (BrainRegion.TEMPORAL_CORTEX, BrainRegion.PREFRONTAL_CORTEX, 0.8),
            (BrainRegion.PREFRONTAL_CORTEX, BrainRegion.MOTOR_CORTEX, 0.7),
            
            # 皮层-皮层下连接
            (BrainRegion.PREFRONTAL_CORTEX, BrainRegion.BASAL_GANGLIA, 0.8),
            (BrainRegion.MOTOR_CORTEX, BrainRegion.BASAL_GANGLIA, 0.9),
            (BrainRegion.MOTOR_CORTEX, BrainRegion.CEREBELLUM, 0.8),
            
            # 丘脑连接
            (BrainRegion.THALAMUS, BrainRegion.PREFRONTAL_CORTEX, 0.9),
            (BrainRegion.THALAMUS, BrainRegion.MOTOR_CORTEX, 0.8),
            (BrainRegion.THALAMUS, BrainRegion.SOMATOSENSORY_CORTEX, 0.9),
            (BrainRegion.THALAMUS, BrainRegion.VISUAL_CORTEX, 0.9),
            
            # 海马连接
            (BrainRegion.HIPPOCAMPUS, BrainRegion.PREFRONTAL_CORTEX, 0.7),
            (BrainRegion.TEMPORAL_CORTEX, BrainRegion.HIPPOCAMPUS, 0.8),
            
            # 杏仁核连接
            (BrainRegion.AMYGDALA, BrainRegion.PREFRONTAL_CORTEX, 0.6),
            (BrainRegion.AMYGDALA, BrainRegion.HIPPOCAMPUS, 0.5)
        ]
        
        for source, target, strength in connections:
            if source in self.brain_regions and target in self.brain_regions:
                # 添加连接
                self.brain_regions[source].add_connection(
                    self.brain_regions[target], strength
                )
                
                # 添加到图
                try:
                    self.region_graph.add_edge(source, target, weight=strength)
                except TypeError:
                    self.region_graph.add_edge(source, target)
    
    def _initialize_function_mapping(self) -> Dict[CognitiveFunction, List[BrainRegion]]:
        """初始化认知功能-脑区映射"""
        
        return {
            CognitiveFunction.ATTENTION: [
                BrainRegion.PREFRONTAL_CORTEX,
                BrainRegion.PARIETAL_CORTEX,
                BrainRegion.THALAMUS
            ],
            CognitiveFunction.WORKING_MEMORY: [
                BrainRegion.PREFRONTAL_CORTEX,
                BrainRegion.PARIETAL_CORTEX
            ],
            CognitiveFunction.LONG_TERM_MEMORY: [
                BrainRegion.HIPPOCAMPUS,
                BrainRegion.TEMPORAL_CORTEX
            ],
            CognitiveFunction.EXECUTIVE_CONTROL: [
                BrainRegion.PREFRONTAL_CORTEX,
                BrainRegion.BASAL_GANGLIA
            ],
            CognitiveFunction.PERCEPTION: [
                BrainRegion.VISUAL_CORTEX,
                BrainRegion.AUDITORY_CORTEX,
                BrainRegion.SOMATOSENSORY_CORTEX
            ],
            CognitiveFunction.MOTOR_CONTROL: [
                BrainRegion.MOTOR_CORTEX,
                BrainRegion.BASAL_GANGLIA,
                BrainRegion.CEREBELLUM
            ],
            CognitiveFunction.EMOTION: [
                BrainRegion.AMYGDALA,
                BrainRegion.PREFRONTAL_CORTEX
            ]
        }
    
    async def process_cognitive_cycle(self, dt: float, 
                                    sensory_inputs: Dict[str, float],
                                    task_demands: Dict[str, float]) -> Dict[str, Any]:
        """处理认知周期"""
        
        results = {
            'region_activities': {},
            'attention_state': {},
            'consciousness_state': {},
            'cognitive_load': 0.0,
            'neuromodulation': {},
            'oscillation_synchrony': {},
            'sleep': {},
        }

        try:
            dt_ms = float(dt)
        except Exception:
            dt_ms = 0.0
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            dt_ms = 1.0

        current_time_ms = float(getattr(self, "_sim_time_ms", 0.0) or 0.0)

        if not isinstance(sensory_inputs, dict):
            sensory_inputs = {}
        else:
            sensory_inputs = dict(sensory_inputs)

        if not isinstance(task_demands, dict):
            task_demands = {}
        else:
            task_demands = dict(task_demands)

        bus_region_inputs: Dict[BrainRegion, Dict[str, float]] = {}
        bus_control_reports: List[Dict[str, Any]] = []

        def _resolve_region(identifier: Any) -> Optional[BrainRegion]:
            if isinstance(identifier, BrainRegion):
                return identifier
            if isinstance(identifier, str):
                token = identifier.strip().lower()
                for region in BrainRegion:
                    if token in (region.name.lower(), region.value.lower()):
                        return region
            return None

        def _coerce_float(value: Any) -> Optional[float]:
            try:
                out = float(value)
            except Exception:
                return None
            if not np.isfinite(out):
                return None
            return float(out)

        if bool(getattr(self, "_module_bus_enabled", True)):
            if bool(getattr(self, "_module_bus_manage_cycle", True)):
                try:
                    self.module_bus.reset_cycle(float(current_time_ms) / 1000.0)
                except Exception:
                    pass

            inbox = list(getattr(self, "_module_bus_inbox", []) or [])
            try:
                self._module_bus_inbox.clear()
            except Exception:
                self._module_bus_inbox = []
            for sig in inbox:
                try:
                    self.module_bus.publish(sig)
                except Exception:
                    continue

            sensory_topics = {
                "visual": ModuleTopic.SENSORY_VISUAL,
                "auditory": ModuleTopic.SENSORY_AUDITORY,
                "somatosensory": ModuleTopic.SENSORY_SOMATOSENSORY,
            }
            for modality, topic in sensory_topics.items():
                val = _coerce_float(sensory_inputs.get(modality))
                if val is None:
                    continue
                self.publish_module_signal(topic, {"modality": modality, "intensity": float(val)}, source="sensory")

            command_signals: List[ModuleSignal] = []
            try:
                command_signals.extend(self.module_bus.get_signals(ModuleTopic.CONTROL_TOP_DOWN))
            except Exception:
                pass
            try:
                command_signals.extend(self.module_bus.get_signals(ModuleTopic.MICROCIRCUIT_COMMAND))
            except Exception:
                pass

            neuromod_overrides: Dict[str, float] = {}
            module_activity_overrides: Dict[str, float] = {}
            task_overrides: Dict[str, float] = {}

            for sig in command_signals:
                payload = getattr(sig, "payload", None)
                if not isinstance(payload, dict):
                    continue

                task_payload = payload.get("task_demands")
                if isinstance(task_payload, dict):
                    for key, value in task_payload.items():
                        val = _coerce_float(value)
                        if val is None:
                            continue
                        task_overrides[str(key)] = float(val)

                mods = payload.get("neuromodulation")
                if isinstance(mods, dict):
                    for key, value in mods.items():
                        val = _coerce_float(value)
                        if val is None:
                            continue
                        neuromod_overrides[str(key)] = float(val)
                for key in ("dopamine", "serotonin", "acetylcholine", "norepinephrine"):
                    if key in payload:
                        val = _coerce_float(payload.get(key))
                        if val is not None:
                            neuromod_overrides[key] = float(val)

                act = payload.get("module_activity")
                if isinstance(act, dict):
                    for key, value in act.items():
                        val = _coerce_float(value)
                        if val is None:
                            continue
                        module_activity_overrides[str(key)] = module_activity_overrides.get(str(key), 0.0) + float(val)

                injected = payload.get("region_inputs")
                if isinstance(injected, dict):
                    for region_key, region_payload in injected.items():
                        region = _resolve_region(region_key)
                        if region is None or not isinstance(region_payload, dict):
                            continue
                        bucket = bus_region_inputs.setdefault(region, {})
                        for key, value in region_payload.items():
                            val = _coerce_float(value)
                            if val is None:
                                continue
                            bucket[str(key)] = float(bucket.get(str(key), 0.0)) + float(val)

                control = payload.get("control")
                if control is None:
                    control = payload.get("microcircuit_control")
                if control is None:
                    control = payload.get("microcircuit")
                if isinstance(control, dict):
                    targets = payload.get("target_regions")
                    if targets is None:
                        targets = payload.get("target_region")
                    if targets is None:
                        targets = payload.get("region")
                    if targets is None:
                        targets = payload.get("regions")

                    resolved: List[BrainRegion] = []
                    if isinstance(targets, (list, tuple, set)):
                        for entry in targets:
                            region = _resolve_region(entry)
                            if region is not None:
                                resolved.append(region)
                    elif targets is not None:
                        region = _resolve_region(targets)
                        if region is not None:
                            resolved.append(region)

                    for region in resolved:
                        region_obj = self.brain_regions.get(region)
                        if region_obj is None:
                            continue
                        report = region_obj.apply_microcircuit_control(control)
                        bus_control_reports.append({"region": region.value, "result": report})

            if task_overrides:
                for key, val in task_overrides.items():
                    task_demands[str(key)] = float(val)

            if neuromod_overrides:
                existing = task_demands.get("neuromodulation")
                merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
                for key, val in neuromod_overrides.items():
                    merged[str(key)] = float(val)
                task_demands["neuromodulation"] = merged

            if module_activity_overrides:
                existing = task_demands.get("module_activity")
                merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
                for key, val in module_activity_overrides.items():
                    merged[str(key)] = float(val)
                task_demands["module_activity"] = merged

            if bus_control_reports:
                results["module_bus_control"] = {"microcircuit": list(bus_control_reports)}
        
        # 1. 更新注意力状态
        top_down_control = task_demands.get('attention_control', 0.5)
        attention_weights = self.attention_state.update_attention(
            sensory_inputs, top_down_control
        )
        results['attention_state'] = {
            'focus_strength': self.attention_state.focus_strength,
            'attention_weights': attention_weights
        }
        
        # 2. 计算脑区输入
        update_mode = getattr(self, "_region_update_mode", "full")
        current_time_ms = float(getattr(self, "_sim_time_ms", 0.0) or 0.0)

        # Sleep state update (optional) + propagation to memory consolidation (if attached)
        stage = self._update_sleep_stage(current_time_ms)
        if stage != str(getattr(self, "_sleep_stage", "wake") or "wake"):
            self._sleep_stage = stage
            self._sleep_stage_entered_ms = float(current_time_ms)
            self._propagate_sleep_stage(stage)

        try:
            stage_entered = float(getattr(self, "_sleep_stage_entered_ms", 0.0) or 0.0)
        except Exception:
            stage_entered = 0.0
        results["sleep"] = {
            "stage": stage,
            "stage_duration_ms": float(current_time_ms) - float(stage_entered),
        }

        if update_mode == "event_driven":
            region_inputs = self._calculate_external_inputs(sensory_inputs, attention_weights, task_demands)
            due_inputs = self._drain_due_events(current_time_ms)
            for region_type, payload in due_inputs.items():
                bucket = region_inputs.setdefault(region_type, {})
                for key, value in (payload or {}).items():
                    try:
                        bucket[key] = float(bucket.get(key, 0.0)) + float(value)
                    except Exception:
                        continue
        else:
            region_inputs = self._calculate_region_inputs(sensory_inputs, attention_weights, task_demands)

        # Apply optional region input injections from the module bus.
        if bus_region_inputs:
            for region_type, payload in bus_region_inputs.items():
                bucket = region_inputs.setdefault(region_type, {})
                for key, value in payload.items():
                    try:
                        bucket[str(key)] = float(bucket.get(str(key), 0.0)) + float(value)
                    except Exception:
                        continue

        # Inject sleep replay drives as additional region inputs (when enabled).
        try:
            results["sleep"]["replay"] = self._inject_sleep_replay_inputs(region_inputs, current_time_ms, dt_ms)
        except Exception:
            results["sleep"]["replay"] = {"enabled": False}

        # Optional: route module-level activity (from task_demands/sensory_inputs) into region inputs.
        topology_snapshot = None
        router = getattr(self, "_functional_topology_router", None)
        if router is not None:
            try:
                topology_snapshot = router.apply_to_region_inputs(
                    region_inputs,
                    sensory_inputs=dict(sensory_inputs or {}),
                    task_demands=dict(task_demands or {}),
                    available_regions=list(self.brain_regions.keys()),
                )
                if getattr(topology_snapshot, "module_activity_in", None):
                    results["functional_topology"] = {
                        "enabled": True,
                        "module_activity_in": dict(topology_snapshot.module_activity_in),
                        "module_to_regions": dict(topology_snapshot.module_to_regions),
                        "region_drive_total": dict(topology_snapshot.region_drive_total),
                    }
            except Exception as exc:
                self.logger.warning("Functional topology routing failed: %s", exc)
                topology_snapshot = None

        # Step external/process-global microcircuit engines (e.g., NEST) once per cycle.
        forced_engine_regions: set[BrainRegion] = set()
        try:
            forced_engine_regions, engine_info = self._step_external_microcircuit_engines(dt_ms, region_inputs)
            if isinstance(engine_info, dict) and int(engine_info.get("engines", 0) or 0) > 0:
                results["external_microcircuit_engines"] = engine_info
        except Exception:
            forced_engine_regions = set()
        
        # 3. 并行更新所有脑区
        
        # 等待所有脑区更新完成
        regions_to_update = list(self.brain_regions.keys())
        if update_mode == "event_driven":
            regions_to_update = [
                region_type
                for region_type, inputs in region_inputs.items()
                if region_type in self.brain_regions and isinstance(inputs, dict) and inputs
            ]
            for forced in getattr(self, "_event_force_update_regions", set()) or set():
                if forced in self.brain_regions and forced not in regions_to_update:
                    regions_to_update.append(forced)
            for forced in forced_engine_regions:
                if forced in self.brain_regions and forced not in regions_to_update:
                    regions_to_update.append(forced)

        updated_payloads: Dict[BrainRegion, Dict[str, Any]] = {}
        if getattr(self, "_region_executor", None) is not None and regions_to_update:
            tasks: List[RegionUpdateTask] = []
            for region_type in regions_to_update:
                brain_region = self.brain_regions[region_type]
                inputs = region_inputs.get(region_type, {})

                def _runner(dt_value, input_payload, *, _region=brain_region, _mods=self.neuromodulation):
                    return _region.update(dt_value, input_payload, _mods)

                tasks.append(
                    RegionUpdateTask(
                        name=region_type,
                        runner=_runner,
                        dt=dt,
                        inputs=inputs,
                        mode="cognitive_region_update",
                    )
                )

            updated_payloads = self._region_executor.run(tasks)  # type: ignore[attr-defined]
        else:
            for region_type in regions_to_update:
                brain_region = self.brain_regions[region_type]
                inputs = region_inputs.get(region_type, {})
                updated_payloads[region_type] = brain_region.update(dt, inputs, self.neuromodulation)

        region_activities: Dict[str, Dict[str, Any]] = {}
        for region_type, brain_region in self.brain_regions.items():
            if region_type in updated_payloads:
                region_result = updated_payloads[region_type]
            else:
                region_result = {"activation": float(brain_region.activation_level), "skipped": True}
            region_activities[region_type.value] = region_result
            results["region_activities"][region_type.value] = region_result

        # Compute module-level readouts after regions have advanced.
        if topology_snapshot is not None and "functional_topology" in results:
            try:
                topology_snapshot = FunctionalTopologyRouter.compute_module_readout(topology_snapshot, region_activities)
                results["functional_topology"]["module_activity_out"] = dict(topology_snapshot.module_activity_out)
            except Exception:
                pass

        if update_mode == "event_driven":
            for source_region in regions_to_update:
                src_region_obj = self.brain_regions.get(source_region)
                if src_region_obj is None:
                    continue
                src_result = updated_payloads.get(source_region, {})
                activation = src_result.get("activation", src_region_obj.activation_level)
                self._schedule_connection_events(source_region, activation, current_time_ms)

            results["event_driven"] = {
                "time_ms": float(current_time_ms),
                "queue_depth": int(len(getattr(self, "_event_queue", []) or [])),
                "regions_updated": int(len(regions_to_update)),
            }
        
        # 4. 更新意识状态
        neural_activities = {
            str(region_key): float(result.get('activation', 0.0) or 0.0)
            for region_key, result in region_activities.items()
            if isinstance(result, dict)
        }
        consciousness_result = self.consciousness_state.update_consciousness(
            neural_activities, self.attention_state
        )
        results['consciousness_state'] = consciousness_result
        
        # 5. 更新神经调节
        self._update_neuromodulation(region_activities, task_demands)
        results['neuromodulation'] = self.neuromodulation.copy()
        
        # 6. 计算振荡同步
        synchrony_result = self._calculate_oscillation_synchrony(region_activities)
        results['oscillation_synchrony'] = synchrony_result
        
        # 7. 更新认知负荷
        self.cognitive_load = self._calculate_cognitive_load(
            region_activities, task_demands
        )
        results['cognitive_load'] = self.cognitive_load

        # Sleep synaptic homeostasis (e.g., global downscaling during N3).
        try:
            results["sleep"]["synaptic_homeostasis"] = self._apply_sleep_synaptic_homeostasis(dt_ms)
        except Exception:
            results["sleep"]["synaptic_homeostasis"] = {"enabled": False}

        # Advance the internal cognitive-layer clock (milliseconds).
        self._sim_time_ms = float(current_time_ms) + float(dt_ms)

        if bool(getattr(self, "_module_bus_enabled", True)):
            # Publish summary state + bottom-up microcircuit events.
            if bool(getattr(self, "_bus_publish_cognitive_state", True)):
                self.publish_module_signal(
                    ModuleTopic.COGNITIVE_STATE,
                    {
                        "time_ms": float(self._sim_time_ms),
                        "cognitive_load": float(self.cognitive_load),
                        "neuromodulation": dict(self.neuromodulation),
                        "attention": dict(results.get("attention_state", {}) or {}),
                        "sleep": dict(results.get("sleep", {}) or {}),
                    },
                    source="cognitive_architecture",
                )

            if bool(getattr(self, "_bus_publish_micro_events", True)):
                try:
                    activities = results.get("region_activities", {}) or {}
                    for region_key, activity in activities.items():
                        if not isinstance(activity, dict):
                            continue
                        mc = activity.get("microcircuit")
                        if not isinstance(mc, dict):
                            continue

                        region_rates = mc.get("region_rates_hz")
                        if not isinstance(region_rates, dict):
                            region_rates = {}

                        try:
                            rate = float(mc.get("rate_hz_smooth", mc.get("rate_hz", 0.0)) or 0.0)
                        except Exception:
                            rate = 0.0

                        if np.isfinite(rate) and float(rate) >= float(getattr(self, "_bus_high_activity_rate_hz", 25.0)):
                            self.publish_module_signal(
                                ModuleTopic.MICROCIRCUIT_EVENT,
                                {
                                    "event": "high_activity",
                                    "region": str(region_key),
                                    "rate_hz": float(rate),
                                },
                                source=str(region_key),
                            )

                        if str(region_key) == BrainRegion.VISUAL_CORTEX.value:
                            v1_rate = region_rates.get("V1")
                            try:
                                v1_rate_f = float(v1_rate) if v1_rate is not None else None
                            except Exception:
                                v1_rate_f = None
                            if (
                                v1_rate_f is not None
                                and np.isfinite(v1_rate_f)
                                and float(v1_rate_f) >= float(getattr(self, "_bus_v1_salience_rate_hz", 20.0))
                            ):
                                self.publish_module_signal(
                                    ModuleTopic.MICROCIRCUIT_EVENT,
                                    {
                                        "event": "salient_visual",
                                        "region": str(region_key),
                                        "subregion": "V1",
                                        "rate_hz": float(v1_rate_f),
                                    },
                                    source=str(region_key),
                                )

                        if str(region_key) == BrainRegion.HIPPOCAMPUS.value:
                            ca1_rate = region_rates.get("CA1")
                            if ca1_rate is None:
                                ca1_rate = region_rates.get("REGION")
                            try:
                                ca1_rate_f = float(ca1_rate) if ca1_rate is not None else None
                            except Exception:
                                ca1_rate_f = None
                            demand = _coerce_float(task_demands.get("memory_retrieval"))
                            if (
                                ca1_rate_f is not None
                                and np.isfinite(ca1_rate_f)
                                and float(ca1_rate_f) >= float(getattr(self, "_bus_ca1_retrieval_rate_hz", 12.0))
                                and (demand is None or float(demand) > 0.0)
                            ):
                                self.publish_module_signal(
                                    ModuleTopic.MEMORY_EVENT,
                                    {
                                        "event": "retrieval",
                                        "region": str(region_key),
                                        "subregion": "CA1" if "CA1" in region_rates else "REGION",
                                        "rate_hz": float(ca1_rate_f),
                                        "demand": float(demand) if demand is not None else None,
                                        "success": True,
                                    },
                                    source=str(region_key),
                                )
                except Exception:
                    pass

            if bool(getattr(self, "_module_bus_export", True)):
                try:
                    results["module_bus"] = self.module_bus.export_cycle()
                except Exception:
                    pass
        
        return results
    
    def _calculate_region_inputs(self, sensory_inputs: Dict[str, float],
                               attention_weights: Dict[str, float],
                               task_demands: Dict[str, float]) -> Dict[BrainRegion, Dict[str, float]]:
        """计算各脑区的输入"""
        
        region_inputs: Dict[BrainRegion, Dict[str, float]] = {}
        
        # 感觉输入分配
        sensory_mapping = {
            'visual': BrainRegion.VISUAL_CORTEX,
            'auditory': BrainRegion.AUDITORY_CORTEX,
            'somatosensory': BrainRegion.SOMATOSENSORY_CORTEX
        }
        
        for modality, intensity in sensory_inputs.items():
            if modality in sensory_mapping:
                target_region = sensory_mapping[modality]
                if target_region not in region_inputs:
                    region_inputs[target_region] = {}
                
                # 应用注意力权重
                attention_weight = attention_weights.get(modality, 1.0)
                weighted_input = intensity * attention_weight
                region_inputs[target_region][f'sensory_{modality}'] = weighted_input
        
        # 任务需求分配
        task_mapping = {
            'working_memory': BrainRegion.PREFRONTAL_CORTEX,
            'motor_planning': BrainRegion.MOTOR_CORTEX,
            'memory_retrieval': BrainRegion.HIPPOCAMPUS,
            'emotional_processing': BrainRegion.AMYGDALA
        }
        
        for task, demand in task_demands.items():
            if task in task_mapping:
                target_region = task_mapping[task]
                if target_region not in region_inputs:
                    region_inputs[target_region] = {}
                region_inputs[target_region][f'task_{task}'] = demand
        
        # 脑区间连接输入
        for source_region, brain_region in self.brain_regions.items():
            source_activation = brain_region.activation_level
            
            for target_region_type, connection_info in brain_region.connection_strengths.items():
                if target_region_type not in region_inputs:
                    region_inputs[target_region_type] = {}
                
                connection_strength = connection_info['strength']
                connection_input = source_activation * connection_strength
                
                input_key = f'connection_{source_region.value}'
                region_inputs[target_region_type][input_key] = connection_input
        
        return region_inputs
    
    def _calculate_external_inputs(
        self,
        sensory_inputs: Dict[str, float],
        attention_weights: Dict[str, float],
        task_demands: Dict[str, float],
    ) -> Dict[BrainRegion, Dict[str, float]]:
        """Compute region inputs excluding inter-region connections (event-driven uses a queue)."""

        region_inputs: Dict[BrainRegion, Dict[str, float]] = {}

        sensory_mapping = {
            "visual": BrainRegion.VISUAL_CORTEX,
            "auditory": BrainRegion.AUDITORY_CORTEX,
            "somatosensory": BrainRegion.SOMATOSENSORY_CORTEX,
        }

        for modality, intensity in (sensory_inputs or {}).items():
            if modality in sensory_mapping:
                target_region = sensory_mapping[modality]
                region_inputs.setdefault(target_region, {})
                attention_weight = attention_weights.get(modality, 1.0) if isinstance(attention_weights, dict) else 1.0
                region_inputs[target_region][f"sensory_{modality}"] = float(intensity) * float(attention_weight)

        task_mapping = {
            "working_memory": BrainRegion.PREFRONTAL_CORTEX,
            "motor_planning": BrainRegion.MOTOR_CORTEX,
            "memory_retrieval": BrainRegion.HIPPOCAMPUS,
            "emotional_processing": BrainRegion.AMYGDALA,
        }

        for task, demand in (task_demands or {}).items():
            if task in task_mapping:
                target_region = task_mapping[task]
                region_inputs.setdefault(target_region, {})
                region_inputs[target_region][f"task_{task}"] = float(demand)

        return region_inputs

    def _enqueue_event(self, time_ms: float, target_region: BrainRegion, key: str, value: float) -> None:
        if not np.isfinite(time_ms):
            return

        try:
            value_f = float(value)
        except Exception:
            return

        if abs(value_f) < float(getattr(self, "_event_input_epsilon", 1e-3) or 1e-3):
            return

        queue = getattr(self, "_event_queue", None)
        if queue is None:
            return

        max_pending = int(getattr(self, "_event_max_pending", 200_000) or 200_000)
        if max_pending > 0 and len(queue) >= max_pending:
            return

        self._event_counter = int(getattr(self, "_event_counter", 0)) + 1
        heapq.heappush(queue, (float(time_ms), int(self._event_counter), target_region, str(key), value_f))

    def _drain_due_events(self, current_time_ms: float) -> Dict[BrainRegion, Dict[str, float]]:
        delivered: Dict[BrainRegion, Dict[str, float]] = {}

        queue = getattr(self, "_event_queue", None)
        if not queue:
            return delivered

        max_events = int(getattr(self, "_event_max_events_per_cycle", 50_000) or 50_000)
        if max_events <= 0:
            max_events = 50_000

        processed = 0
        while queue and processed < max_events:
            time_ms, _, target_region, key, value = queue[0]
            if float(time_ms) > float(current_time_ms):
                break
            heapq.heappop(queue)

            if target_region not in self.brain_regions:
                processed += 1
                continue

            region_bucket = delivered.setdefault(target_region, {})
            region_bucket[key] = float(region_bucket.get(key, 0.0)) + float(value)
            processed += 1

        return delivered

    def _schedule_connection_events(self, source_region: BrainRegion, activation: float, current_time_ms: float) -> None:
        if source_region not in self.brain_regions:
            return

        try:
            activation_f = float(activation)
        except Exception:
            activation_f = 0.0

        if activation_f < float(getattr(self, "_event_activation_threshold", 0.05) or 0.05):
            return

        brain_region = self.brain_regions[source_region]
        for target_region, connection_info in getattr(brain_region, "connection_strengths", {}).items():
            try:
                strength = float(connection_info.get("strength", 0.0))
            except Exception:
                strength = 0.0

            if abs(strength) < 1e-9:
                continue

            try:
                delay_ms = float(connection_info.get("delay", 0.0))
            except Exception:
                delay_ms = 0.0

            try:
                conn_type = str(connection_info.get("type", "excitatory") or "excitatory").strip().lower()
            except Exception:
                conn_type = "excitatory"

            sign = -1.0 if conn_type == "inhibitory" else 1.0
            value = sign * activation_f * strength

            self._enqueue_event(
                float(current_time_ms) + float(delay_ms),
                target_region,
                f"connection_{source_region.value}",
                value,
            )

    async def _update_brain_region_async(self, brain_region: PhysiologicalBrainRegion,
                                       dt: float, inputs: Dict[str, float]) -> Dict[str, Any]:
        """异步更新脑区"""
        
        return brain_region.update(dt, inputs, self.neuromodulation)
    
    def _update_neuromodulation(self, region_activities: Dict[str, Dict[str, Any]],
                              task_demands: Dict[str, float]):
        """更新神经调节水平"""
        
        # 多巴胺：基于奖励预测和目标达成
        reward_signal = task_demands.get('reward', 0.0)
        goal_achievement = task_demands.get('goal_achievement', 0.0)
        self.neuromodulation['dopamine'] = 0.3 + 0.4 * (reward_signal + goal_achievement) / 2.0
        
        # 乙酰胆碱：基于注意力需求和新奇性
        attention_demand = task_demands.get('attention_control', 0.0)
        novelty = task_demands.get('novelty', 0.0)
        self.neuromodulation['acetylcholine'] = 0.2 + 0.6 * (attention_demand + novelty) / 2.0
        
        # 去甲肾上腺素：基于觉醒和压力
        arousal = task_demands.get('arousal', 0.5)
        stress = task_demands.get('stress', 0.0)
        self.neuromodulation['norepinephrine'] = 0.3 + 0.5 * arousal + 0.2 * stress
        
        # 血清素：基于情绪状态和社会因素
        mood = task_demands.get('mood', 0.5)
        social_context = task_demands.get('social', 0.0)
        self.neuromodulation['serotonin'] = 0.4 + 0.3 * mood + 0.3 * social_context

        # Optional direct overrides from structured payloads (e.g., decision modules emitting dopamine_level).
        if isinstance(task_demands, dict):
            overrides = task_demands.get("neuromodulation")
            if isinstance(overrides, dict):
                for key, value in overrides.items():
                    if key in self.neuromodulation and isinstance(value, (int, float)):
                        self.neuromodulation[key] = float(value)

            aliases = {
                "dopamine_level": "dopamine",
                "dopamine": "dopamine",
                "serotonin_level": "serotonin",
                "acetylcholine_level": "acetylcholine",
                "norepinephrine_level": "norepinephrine",
            }
            for alias, dest in aliases.items():
                if alias in task_demands and isinstance(task_demands.get(alias), (int, float)):
                    self.neuromodulation[dest] = float(task_demands.get(alias))
        
        # 限制在生理范围
        for modulator in self.neuromodulation:
            upper = 2.0 if modulator == "dopamine" else 1.0
            self.neuromodulation[modulator] = np.clip(self.neuromodulation[modulator], 0.0, float(upper))
    
    def _calculate_oscillation_synchrony(self, region_activities: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算振荡同步性"""
        
        synchrony_results = {}
        
        # 提取各频段的相位信息
        phase_data: Dict[OscillationBand, list] = {}
        for band in OscillationBand:
            phase_data[band] = []
            for region_type, activity in region_activities.items():
                oscillations = activity.get('oscillations', {})
                band_data = oscillations.get(band.name, {})
                phase = band_data.get('phase', 0.0)
                phase_data[band].append(phase)
        
        # 计算相位锁定值（PLV）
        for band, phases in phase_data.items():
            if len(phases) > 1:
                phases_arr = np.array(phases)
                # 计算平均相位一致性
                complex_phases = np.exp(1j * phases_arr)
                plv = np.abs(np.mean(complex_phases))
                synchrony_results[band.name] = plv
            else:
                synchrony_results[band.name] = 0.0
        
        return synchrony_results
    
    def _calculate_cognitive_load(self, region_activities: Dict[str, Dict[str, Any]],
                                task_demands: Dict[str, float]) -> float:
        """计算认知负荷"""
        
        # 基于脑区激活水平
        total_activation = sum(
            activity['activation'] for activity in region_activities.values()
        )
        activation_load = total_activation / len(region_activities) if region_activities else 0.0
        
        # 基于任务需求（忽略非数值型负载，如 module_activity 等结构化字段）
        task_values: List[float] = []
        if isinstance(task_demands, dict):
            for value in task_demands.values():
                if isinstance(value, (int, float)):
                    task_values.append(float(value))
                    continue
                try:
                    val_f = float(value)  # type: ignore[arg-type]
                except Exception:
                    continue
                if np.isfinite(val_f):
                    task_values.append(float(val_f))

        task_load = float(sum(task_values) / len(task_values)) if task_values else 0.0
        
        # 基于注意力分散
        attention_load = 1.0 - self.attention_state.focus_strength
        
        # 综合认知负荷
        cognitive_load = (activation_load + task_load + attention_load) / 3.0
        
        return np.clip(cognitive_load, 0.0, 1.0)
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """获取认知状态"""
        
        state = {
            'attention_state': {
                'focus_strength': self.attention_state.focus_strength,
                'attention_span': self.attention_state.attention_span,
                'distraction_level': self.attention_state.distraction_level
            },
            'consciousness_state': {
                'awareness_level': self.consciousness_state.awareness_level,
                'global_workspace_activity': self.consciousness_state.global_workspace_activity,
                'integration_level': self.consciousness_state.integration_level
            },
            'neuromodulation': self.neuromodulation.copy(),
            'cognitive_load': self.cognitive_load,
            'processing_efficiency': self.processing_efficiency,
            'active_regions': len(self.brain_regions),
            'total_connections': self._safe_region_graph_edge_count()
        }

        try:
            applied = getattr(self, "_parameter_mapping_applied", {}) or {}
            if isinstance(applied, dict) and applied:
                state["parameter_mapping"] = {
                    "enabled": bool(getattr(self, "_module_parameter_mapper", None) is not None),
                    "applied": dict(applied),
                }
        except Exception:
            pass

        return state

    def _safe_region_graph_edge_count(self) -> int:
        try:
            return int(self.region_graph.number_of_edges())
        except Exception:
            pass

        try:
            edges = getattr(self.region_graph, "edges", None)
            if edges is None:
                return 0
            return len(edges)
        except Exception:
            return 0
