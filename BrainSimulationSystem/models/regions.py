"""
生理脑区模型
Physiological Brain Region Model

定义了 PhysiologicalBrainRegion 类，用于模拟大脑中特定区域的生理和功能特性。
"""
import logging
import numpy as np
from typing import Dict, Any

from .enums import BrainRegion, OscillationBand
from .states import NeuralOscillation
from .physiology import BloodFlowModel, MetabolismModel

class PhysiologicalBrainRegion:
    """生理脑区模型"""
    
    def __init__(self, region_type: BrainRegion, config: Dict[str, Any]):
        self.region_type = region_type
        self.config = config
        self.logger = logging.getLogger(f"BrainRegion_{region_type.value}")
        
        # 解剖参数
        self.volume = config.get('volume', 1000.0)  # mm³
        self.surface_area = config.get('surface_area', 100.0)  # mm²
        self.cortical_thickness = config.get('thickness', 2.5)  # mm
        
        # 细胞组成
        self.neuron_density = config.get('neuron_density', 100000)  # neurons/mm³
        self.total_neurons = int(self.volume * self.neuron_density)
        
        # 层状结构（皮层区域）
        self.layers = self._initialize_layers() if self._is_cortical() else {}
        
        # 神经振荡
        self.oscillations = self._initialize_oscillations()
        
        # 血流动力学
        self.blood_flow = BloodFlowModel(self.volume)
        
        # 代谢
        self.metabolism = MetabolismModel(self.total_neurons)

        # Neurovascular/glia coarse coupling state (region-level)
        physiology_cfg = config.get("physiology") if isinstance(config, dict) else None
        self.physiology_cfg = dict(physiology_cfg or {}) if isinstance(physiology_cfg, dict) else {}
        try:
            self.glia_tau_ms = float(self.physiology_cfg.get("glia_tau_ms", 5000.0))
        except Exception:
            self.glia_tau_ms = 5000.0
        if not np.isfinite(self.glia_tau_ms) or self.glia_tau_ms <= 0.0:
            self.glia_tau_ms = 5000.0
        self.glia_activation = 0.0

        # Optional microcircuit (hierarchical hybrid modeling)
        self.microcircuit = None
        micro_cfg = config.get("microcircuit")
        if micro_cfg is None:
            micro_cfg = {"enabled": True, "model": "biophysical", "preset": "auto"}
        elif isinstance(micro_cfg, bool):
            micro_cfg = {"enabled": bool(micro_cfg), "model": "biophysical", "preset": "auto"}
        elif isinstance(micro_cfg, dict):
            micro_cfg = dict(micro_cfg)
            micro_cfg.setdefault("enabled", True)
            micro_cfg.setdefault("model", "biophysical")
            micro_cfg.setdefault("preset", "auto")

        if isinstance(micro_cfg, dict) and bool(micro_cfg.get("enabled", False)):
            try:
                from .microcircuits import create_microcircuit_for_region

                self.microcircuit = create_microcircuit_for_region(region_type, micro_cfg)
            except Exception as exc:
                self.logger.warning("Microcircuit initialization failed for %s: %s", region_type.value, exc)
                self.microcircuit = None
        
        # 连接性
        self.connections = {}
        self.connection_strengths = {}
        
        # 功能状态
        self.activation_level = 0.0
        self.fatigue_level = 0.0
        self.plasticity_state = 0.5
        
    
    def _is_cortical(self) -> bool:
        """判断是否为皮层区域"""
        cortical_regions = [
            BrainRegion.PREFRONTAL_CORTEX, BrainRegion.MOTOR_CORTEX,
            BrainRegion.SOMATOSENSORY_CORTEX, BrainRegion.VISUAL_CORTEX,
            BrainRegion.AUDITORY_CORTEX, BrainRegion.PARIETAL_CORTEX,
            BrainRegion.TEMPORAL_CORTEX, BrainRegion.OCCIPITAL_CORTEX,
            BrainRegion.CINGULATE_CORTEX, BrainRegion.INSULAR_CORTEX
        ]
        return self.region_type in cortical_regions
    
    def _initialize_layers(self) -> Dict[str, Dict[str, Any]]:
        """初始化皮层层状结构"""
        
        if not self._is_cortical():
            return {}
        
        layers = {}
        layer_thicknesses = {
            'L1': 0.15,  # 分子层
            'L2/3': 0.35,  # 外颗粒层/外锥体层
            'L4': 0.15,  # 内颗粒层
            'L5': 0.20,  # 内锥体层
            'L6': 0.15   # 多形层
        }
        
        cumulative_depth = 0.0
        for layer_name, relative_thickness in layer_thicknesses.items():
            thickness = relative_thickness * self.cortical_thickness
            
            layers[layer_name] = {
                'thickness': thickness,
                'depth_range': (cumulative_depth, cumulative_depth + thickness),
                'neuron_density': self._get_layer_neuron_density(layer_name),
                'cell_types': self._get_layer_cell_types(layer_name),
                'connectivity': self._get_layer_connectivity(layer_name)
            }
            
            cumulative_depth += thickness
        
        return layers
    
    def _get_layer_neuron_density(self, layer_name: str) -> float:
        """获取层特异性神经元密度"""
        
        densities = {
            'L1': 0.1,    # 主要是神经纤维
            'L2/3': 1.2,  # 高密度
            'L4': 1.5,    # 最高密度（感觉皮层）
            'L5': 0.8,    # 中等密度
            'L6': 0.6     # 较低密度
        }
        
        base_density = self.neuron_density
        return base_density * densities.get(layer_name, 1.0)
    
    def _get_layer_cell_types(self, layer_name: str) -> Dict[str, float]:
        """获取层特异性细胞类型分布"""
        
        cell_type_distributions = {
            'L1': {'interneurons': 0.9, 'pyramidal': 0.1},
            'L2/3': {'pyramidal': 0.8, 'interneurons': 0.2},
            'L4': {'stellate': 0.6, 'pyramidal': 0.3, 'interneurons': 0.1},
            'L5': {'pyramidal': 0.85, 'interneurons': 0.15},
            'L6': {'pyramidal': 0.75, 'interneurons': 0.25}
        }
        
        return cell_type_distributions.get(layer_name, {'pyramidal': 0.8, 'interneurons': 0.2})
    
    def _get_layer_connectivity(self, layer_name: str) -> Dict[str, Any]:
        """获取层特异性连接模式"""
        
        connectivity_patterns = {
            'L1': {
                'local_connectivity': 0.1,
                'long_range_input': 0.8,
                'feedback_strength': 0.9
            },
            'L2/3': {
                'local_connectivity': 0.3,
                'horizontal_connections': 0.7,
                'cortico_cortical_output': 0.8
            },
            'L4': {
                'local_connectivity': 0.4,
                'thalamic_input': 0.9,
                'vertical_connections': 0.6
            },
            'L5': {
                'local_connectivity': 0.2,
                'subcortical_output': 0.9,
                'cortico_cortical_output': 0.5
            },
            'L6': {
                'local_connectivity': 0.3,
                'thalamic_feedback': 0.8,
                'cortico_thalamic_output': 0.9
            }
        }
        
        return connectivity_patterns.get(layer_name, {})
    
    def _initialize_oscillations(self) -> Dict[OscillationBand, NeuralOscillation]:
        """初始化神经振荡"""
        
        oscillations = {}
        
        for band in OscillationBand:
            freq_range = band.value
            base_freq = (freq_range[0] + freq_range[1]) / 2.0
            
            # 区域特异性频率调节
            region_freq_modifier = self._get_region_frequency_modifier()
            frequency = base_freq * region_freq_modifier
            
            # 初始幅度基于区域类型
            amplitude = self._get_region_oscillation_amplitude(band)
            
            oscillations[band] = NeuralOscillation(
                frequency=frequency,
                amplitude=amplitude,
                phase=np.random.uniform(0, 2*np.pi)
            )
        
        return oscillations
    
    def _get_region_frequency_modifier(self) -> float:
        """获取区域特异性频率调节因子"""
        
        modifiers = {
            BrainRegion.PREFRONTAL_CORTEX: 0.95,
            BrainRegion.MOTOR_CORTEX: 1.1,
            BrainRegion.VISUAL_CORTEX: 1.05,
            BrainRegion.HIPPOCAMPUS: 0.8,
            BrainRegion.THALAMUS: 1.2,
            BrainRegion.CEREBELLUM: 1.5
        }
        
        return modifiers.get(self.region_type, 1.0)
    
    def _get_region_oscillation_amplitude(self, band: OscillationBand) -> float:
        """获取区域和频段特异性振荡幅度"""
        
        # 基础幅度矩阵
        base_amplitudes = {
            OscillationBand.DELTA: 0.3,
            OscillationBand.THETA: 0.4,
            OscillationBand.ALPHA: 0.6,
            OscillationBand.BETA: 0.5,
            OscillationBand.GAMMA: 0.3,
            OscillationBand.HIGH_GAMMA: 0.2
        }
        
        # 区域特异性调节
        region_modifiers = {
            BrainRegion.PREFRONTAL_CORTEX: {
                OscillationBand.THETA: 1.2,
                OscillationBand.GAMMA: 1.3
            },
            BrainRegion.HIPPOCAMPUS: {
                OscillationBand.THETA: 2.0,
                OscillationBand.GAMMA: 1.5
            },
            BrainRegion.VISUAL_CORTEX: {
                OscillationBand.ALPHA: 1.5,
                OscillationBand.GAMMA: 1.4
            },
            BrainRegion.MOTOR_CORTEX: {
                OscillationBand.BETA: 1.8,
                OscillationBand.GAMMA: 1.2
            }
        }
        
        base_amp = base_amplitudes.get(band, 0.5)
        region_mod = region_modifiers.get(self.region_type, {}).get(band, 1.0)
        
        return base_amp * region_mod
    
    def update(self, dt: float, inputs: Dict[str, float], 
               neuromodulation: Dict[str, float]) -> Dict[str, Any]:
        """更新脑区状态"""
        
        results = {
            'activation': 0.0,
            'microcircuit': {},
            'oscillations': {},
            'blood_flow': {},
            'metabolism': {},
            'plasticity_changes': 0.0
        }
        
        # 计算总输入
        total_input = sum(inputs.values()) if inputs else 0.0
        
        # 更新激活水平
        if self.microcircuit is not None:
            try:
                readout = self.microcircuit.step(float(dt), inputs or {}, neuromodulation or {})
                self.activation_level = float(getattr(readout, "activation", 0.0) or 0.0)
                results["microcircuit"] = {
                    "activation": float(getattr(readout, "activation", 0.0) or 0.0),
                    "rate_hz": float(getattr(readout, "rate_hz", 0.0) or 0.0),
                    "rate_hz_smooth": float(getattr(readout, "rate_hz_smooth", 0.0) or 0.0),
                    "region_rates_hz": dict(getattr(readout, "region_rates_hz", {}) or {}),
                    "state": dict(getattr(readout, "state", {}) or {}),
                }
            except Exception as exc:
                self.logger.warning("Microcircuit step failed for %s: %s", self.region_type.value, exc)
                self.microcircuit = None
                self.activation_level = self._update_activation(total_input, dt)
        else:
            self.activation_level = self._update_activation(total_input, dt)

        results['activation'] = self.activation_level
        
        # 更新神经振荡
        oscillation_results = {}
        for band, oscillation in self.oscillations.items():
            # 计算耦合强度
            coupling = self._calculate_oscillation_coupling(band, inputs)
            
            # 外部驱动
            external_drive = self.activation_level * 0.1
            
            # 更新振荡
            osc_value = oscillation.update(dt, coupling, external_drive)
            
            oscillation_results[band.name] = {
                'value': osc_value,
                'frequency': oscillation.frequency,
                'amplitude': oscillation.amplitude,
                'phase': oscillation.phase,
                'power': oscillation.power
            }
        
        results['oscillations'] = oscillation_results

        # Coarse glia drive: low-pass filtered activity proxy (optionally can be overridden by config)
        try:
            tau = float(getattr(self, "glia_tau_ms", 5000.0))
        except Exception:
            tau = 5000.0
        alpha = float(np.clip(float(dt) / max(tau, 1e-6), 0.0, 1.0))
        self.glia_activation = (1.0 - alpha) * float(self.glia_activation) + alpha * float(self.activation_level)
        self.glia_activation = float(np.clip(self.glia_activation, 0.0, 1.0))

        # Use microcircuit firing-rate telemetry (if present) as an activity proxy for metabolism.
        activity_rate_hz = None
        mc = results.get("microcircuit", {})
        if isinstance(mc, dict):
            rate = mc.get("rate_hz_smooth")
            if rate is None:
                rate = mc.get("rate_hz")
            try:
                rate_val = float(rate)
            except Exception:
                rate_val = None
            if rate_val is not None and np.isfinite(rate_val):
                activity_rate_hz = float(max(rate_val, 0.0))

        # 更新血流动力学（加入代谢需求/胶质驱动）
        blood_flow_result = self.blood_flow.update(
            dt,
            self.activation_level,
            metabolic_demand=self.activation_level,
            vasoactive_signal=self.glia_activation,
        )
        results['blood_flow'] = blood_flow_result
        
        # 更新代谢（闭环：血流供给 -> ATP/能量池）
        metabolism_result = self.metabolism.update(
            dt,
            self.activation_level,
            activity_rate_hz=activity_rate_hz,
            oxygen_delivery=blood_flow_result.get("oxygen_delivery") if isinstance(blood_flow_result, dict) else None,
            glucose_delivery=blood_flow_result.get("glucose_delivery") if isinstance(blood_flow_result, dict) else None,
        )
        results['metabolism'] = metabolism_result
        
        # 更新可塑性
        plasticity_change = self._update_plasticity(dt, total_input, neuromodulation)
        results['plasticity_changes'] = plasticity_change
        
        # 更新疲劳
        atp_ratio = None
        if isinstance(metabolism_result, dict):
            try:
                atp_ratio = float(metabolism_result.get("atp_ratio"))
            except Exception:
                atp_ratio = None
        self._update_fatigue(dt, self.activation_level, atp_ratio=atp_ratio)
        
        return results

    def apply_microcircuit_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optional top-down control to the embedded microcircuit (if present)."""

        micro = getattr(self, "microcircuit", None)
        if micro is None:
            return {"applied": False, "reason": "microcircuit_unavailable"}

        applier = getattr(micro, "apply_control", None)
        if not callable(applier):
            return {"applied": False, "reason": "unsupported"}

        try:
            applied = applier(control or {})
        except Exception as exc:
            self.logger.warning("Microcircuit control failed for %s: %s", self.region_type.value, exc)
            return {"applied": False, "error": str(exc)}

        if isinstance(applied, dict):
            return {"applied": True, "result": dict(applied)}
        return {"applied": True, "result": applied}
    
    def _update_activation(self, total_input: float, dt: float) -> float:
        """更新激活水平"""
        
        # Sigmoid激活函数
        target_activation = 1.0 / (1.0 + np.exp(-(total_input - 5.0)))
        
        # 时间常数
        tau = 10.0  # ms
        
        # 指数衰减到目标值
        self.activation_level += (target_activation - self.activation_level) / tau * dt
        
        # 疲劳影响
        fatigue_factor = 1.0 - self.fatigue_level * 0.3
        self.activation_level *= fatigue_factor
        
        return np.clip(self.activation_level, 0.0, 1.0)
    
    def _calculate_oscillation_coupling(self, band: OscillationBand, 
                                      inputs: Dict[str, float]) -> float:
        """计算振荡耦合强度"""
        
        # 基于输入和区域特性的耦合
        coupling_strength = 0.0
        
        for source_region, input_strength in inputs.items():
            # 频段特异性耦合
            if band == OscillationBand.GAMMA:
                coupling_strength += input_strength * 0.3
            elif band == OscillationBand.THETA:
                coupling_strength += input_strength * 0.2
            elif band == OscillationBand.ALPHA:
                coupling_strength += input_strength * 0.1
        
        return coupling_strength
    
    def _update_plasticity(self, dt: float, input_strength: float, 
                          neuromodulation: Dict[str, float]) -> float:
        """更新可塑性状态"""
        
        # 基础可塑性变化
        base_change = input_strength * 0.001 * dt
        
        # 神经调节影响
        modulation_factor = 1.0
        if 'dopamine' in neuromodulation:
            modulation_factor += neuromodulation['dopamine'] * 0.5
        if 'acetylcholine' in neuromodulation:
            modulation_factor += neuromodulation['acetylcholine'] * 0.3
        
        plasticity_change = base_change * modulation_factor
        
        # 更新可塑性状态
        self.plasticity_state += plasticity_change
        self.plasticity_state = np.clip(self.plasticity_state, 0.0, 1.0)
        
        return plasticity_change
    
    def _update_fatigue(self, dt: float, activation: float, *, atp_ratio: float | None = None):
        """更新疲劳水平（可选代谢反馈）"""

        # 基于活动的疲劳累积
        fatigue_accumulation = float(activation) * 0.0001 * float(dt)

        # 低能量状态会加速疲劳累积并降低恢复速度
        if atp_ratio is not None:
            try:
                ratio = float(atp_ratio)
            except Exception:
                ratio = None
            if ratio is not None and np.isfinite(ratio):
                ratio = float(np.clip(ratio, 0.0, 1.5))
                fatigue_accumulation *= (1.0 + max(0.0, 1.0 - ratio) * 2.0)
                recovery_multiplier = 0.5 + 0.5 * min(ratio, 1.0)
            else:
                recovery_multiplier = 1.0
        else:
            recovery_multiplier = 1.0

        self.fatigue_level += fatigue_accumulation

        # 疲劳恢复
        recovery_rate = 0.00005 * float(dt) * float(recovery_multiplier)
        self.fatigue_level -= recovery_rate

        self.fatigue_level = float(np.clip(self.fatigue_level, 0.0, 1.0))
    
    def add_connection(self, target_region: 'PhysiologicalBrainRegion', 
                      strength: float, connection_type: str = 'excitatory'):
        """添加到其他脑区的连接"""
        
        self.connections[target_region.region_type] = target_region
        self.connection_strengths[target_region.region_type] = {
            'strength': strength,
            'type': connection_type,
            'delay': self._calculate_connection_delay(target_region)
        }
    
    def _calculate_connection_delay(self, target_region: 'PhysiologicalBrainRegion') -> float:
        """计算连接延迟"""
        
        # 简化的距离-延迟关系
        base_delays = {
            (BrainRegion.PREFRONTAL_CORTEX, BrainRegion.HIPPOCAMPUS): 15.0,
            (BrainRegion.VISUAL_CORTEX, BrainRegion.PREFRONTAL_CORTEX): 25.0,
            (BrainRegion.MOTOR_CORTEX, BrainRegion.CEREBELLUM): 10.0,
            (BrainRegion.THALAMUS, BrainRegion.PREFRONTAL_CORTEX): 8.0
        }
        
        connection_key = (self.region_type, target_region.region_type)
        reverse_key = (target_region.region_type, self.region_type)
        
        return base_delays.get(connection_key, 
                              base_delays.get(reverse_key, 20.0))
    
    def get_region_state(self) -> Dict[str, Any]:
        """获取脑区状态"""
        
        return {
            'region_type': self.region_type.value,
            'activation_level': self.activation_level,
            'fatigue_level': self.fatigue_level,
            'plasticity_state': self.plasticity_state,
            'total_neurons': self.total_neurons,
            'volume': self.volume,
            'oscillation_powers': {
                band.name: osc.power for band, osc in self.oscillations.items()
            },
            'connections': list(self.connections.keys()),
            'layers': list(self.layers.keys()) if self.layers else []
        }
