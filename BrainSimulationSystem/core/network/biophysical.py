"""Biophysical spiking network backend (human-brain inspired, downscaled).

This module provides a lightweight, dependency-free spiking simulation that is:
- Whole-brain in *structure* (multi-region connectome + delays)
- Biophysical in *mechanism* (E/I populations, conductance synapses, Izhikevich dynamics)

It is intentionally downscaled: you configure neurons-per-region rather than attempting
to simulate 8.6e10 neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import NeuralNetwork


def _default_regions() -> List[str]:
    return [
        "THALAMUS",
        "VISUAL_CORTEX",
        "AUDITORY_CORTEX",
        "SOMATOSENSORY_CORTEX",
        "MOTOR_CORTEX",
        "PREFRONTAL_CORTEX",
        "HIPPOCAMPUS",
        "BASAL_GANGLIA",
    ]


def _default_connectome(regions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (weights, delays_ms) for a small canonical connectome."""

    n = len(regions)
    weights = np.zeros((n, n), dtype=np.float32)
    delays = np.full((n, n), 10.0, dtype=np.float32)

    idx = {name: i for i, name in enumerate(regions)}

    th = idx.get("THALAMUS")
    v1 = idx.get("VISUAL_CORTEX")
    a1 = idx.get("AUDITORY_CORTEX")
    s1 = idx.get("SOMATOSENSORY_CORTEX")
    m1 = idx.get("MOTOR_CORTEX")
    pfc = idx.get("PREFRONTAL_CORTEX")
    hpc = idx.get("HIPPOCAMPUS")
    bg = idx.get("BASAL_GANGLIA")

    # Thalamus -> primary cortices
    if th is not None and v1 is not None:
        weights[th, v1] = 0.9
        delays[th, v1] = 6.0
    if th is not None and a1 is not None:
        weights[th, a1] = 0.8
        delays[th, a1] = 7.0
    if th is not None and s1 is not None:
        weights[th, s1] = 0.9
        delays[th, s1] = 6.5

    # Cortico-cortical (sensory -> association/executive)
    if v1 is not None and pfc is not None:
        weights[v1, pfc] = 0.35
        delays[v1, pfc] = 12.0
    if a1 is not None and pfc is not None:
        weights[a1, pfc] = 0.30
        delays[a1, pfc] = 12.0
    if s1 is not None and pfc is not None:
        weights[s1, pfc] = 0.25
        delays[s1, pfc] = 11.0
    if s1 is not None and m1 is not None:
        weights[s1, m1] = 0.40
        delays[s1, m1] = 8.0

    # Hippocampus <-> PFC loop
    if pfc is not None and hpc is not None:
        weights[pfc, hpc] = 0.25
        weights[hpc, pfc] = 0.35
        delays[pfc, hpc] = 15.0
        delays[hpc, pfc] = 15.0

    # PFC/M1 -> basal ganglia, BG -> thalamus (action gating loop)
    if pfc is not None and bg is not None:
        weights[pfc, bg] = 0.40
        delays[pfc, bg] = 10.0
    if m1 is not None and bg is not None:
        weights[m1, bg] = 0.35
        delays[m1, bg] = 9.0
    if bg is not None and th is not None:
        weights[bg, th] = 0.30
        delays[bg, th] = 8.0

    return weights, delays


def _default_cell_type_params_izh() -> Dict[str, Dict[str, float]]:
    """Small built-in Izhikevich parameter library.

    These are phenomenological (Izhikevich-style) cell-type presets used to add
    layer/region heterogeneity without requiring full multi-compartment models.
    Users can override entries via ``simulation.biophysical.cell_type_params``.
    """

    return {
        # Cortical excitatory
        "cortex_rs": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "v_peak": 30.0},
        "cortex_ib": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0, "v_peak": 30.0},
        # Thalamocortical relay-like (burst-prone)
        "thalamus_tc": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 0.05, "v_peak": 30.0},
        # Hippocampal pyramidal approximations
        "hippocampus_rs": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 6.0, "v_peak": 30.0},
        "hippocampus_ib": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0, "v_peak": 30.0},
        # Basal ganglia medium spiny-like (slower recovery)
        "bg_msn": {"a": 0.01, "b": 0.2, "c": -65.0, "d": 6.0, "v_peak": 30.0},
        # Inhibitory subtypes
        "pv_fs": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0, "v_peak": 30.0},
        "sst_lts": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0, "v_peak": 30.0},
        "vip_fs": {"a": 0.12, "b": 0.2, "c": -65.0, "d": 2.0, "v_peak": 30.0},
    }


def _default_cell_type_params_adex() -> Dict[str, Dict[str, float]]:
    """Small built-in AdEx parameter library for excitatory-like presets.

    The biophysical backend uses *scaled* AdEx parameters so that typical drive
    values (tens of units) can elicit spiking, similar to the Izhikevich mode.
    """

    return {
        "cortex_rs": {"b": 8.0, "tau_w_ms": 150.0, "a": 1.0},
        "cortex_ib": {"b": 4.0, "tau_w_ms": 120.0, "a": 0.8},
        "hippocampus_rs": {"b": 7.0, "tau_w_ms": 180.0, "a": 1.0},
        "hippocampus_ib": {"b": 3.0, "tau_w_ms": 110.0, "a": 0.7},
        "thalamus_tc": {"b": 6.0, "tau_w_ms": 220.0, "a": 1.2},
        "bg_msn": {"b": 10.0, "tau_w_ms": 300.0, "a": 1.5},
    }


@dataclass
class BiophysicalSpikingConfig:
    """Configuration for :class:`BiophysicalSpikingNetwork`."""

    seed: int = 42
    regions: List[str] = field(default_factory=_default_regions)
    neurons_per_region: int = 120
    excitatory_ratio: float = 0.8

    # Optional cell-type heterogeneity (primarily affects Izhikevich parameters).
    cell_types_enabled: bool = False
    cell_type_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    inhibitory_subtype_distribution: Dict[str, float] = field(
        default_factory=lambda: {"pv_fs": 0.7, "sst_lts": 0.3}
    )

    # Neuron dynamics model (global for this network)
    # - "izhikevich" (default): fast spiking phenomenological model
    # - "adex": Adaptive Exponential IF (scaled units; more biophysical than LIF/IZH)
    # - "hybrid": excitatory=AdEx, inhibitory=Izhikevich (cell-type aware)
    # - "hh": Hodgkin-Huxley single-compartment conductance-based model (slower, more biophysical)
    # - "lif": leaky integrate-and-fire (fast, phenomenological)
    # - "mc": two-compartment model (HH soma + passive dendrite)
    neuron_model: str = "izhikevich"
    v_init_mV: float = -65.0

    # Connectivity
    intra_connection_prob: float = 0.08
    inter_connection_prob: float = 0.01
    max_delay_ms: float = 25.0

    # Optional inline connectome (useful for embedded microcircuits / programmatic wiring).
    # If provided, these take precedence over ``connectome_npz_path``.
    connectome_weights: Optional[Any] = None
    connectome_delays_ms: Optional[Any] = None
    connectome_region_names: Optional[Any] = None
    connectome_coords_mm: Optional[Any] = None

    connectome_npz_path: Optional[str] = None
    connectome_synaptic_delay_ms: float = 1.0
    axonal_velocity_m_s: float = 5.0
    estimate_axonal_velocity: bool = False

    # Cortical laminar scaffold (applied to regions whose name contains "CORTEX")
    cortical_layers_enabled: bool = True
    cortical_layer_proportions: Dict[str, float] = field(
        default_factory=lambda: {
            "L2/3": 0.35,
            "L4": 0.25,
            "L5": 0.25,
            "L6": 0.15,
        }
    )
    cortical_layer_connectivity: Dict[str, float] = field(
        default_factory=lambda: {
            "L4->L2/3": 1.6,
            "L2/3->L5": 1.3,
            "L4->L5": 1.1,
            "L5->L6": 1.2,
            "L6->L4": 1.0,
            "L2/3->L2/3": 0.9,
            "L4->L4": 0.8,
            "L5->L5": 0.8,
            "L6->L6": 0.8,
        }
    )
    thalamus_target_layers: Tuple[str, ...] = ("L4", "L6")
    cortex_feedback_layer: str = "L6"

    # Plasticity (pair-based STDP; applied to excitatory synapses)
    stdp_enabled: bool = False
    stdp_tau_plus_ms: float = 20.0
    stdp_tau_minus_ms: float = 20.0
    stdp_A_plus: float = 0.005
    stdp_A_minus: float = 0.006
    stdp_weight_min: float = 0.0
    stdp_weight_max: float = 1.0
    dopamine_stdp_gain: float = 0.5

    # Synapse dynamics (conductance based)
    synapse_model: str = "exp"  # "exp" (single E/I), or "receptor" (AMPA/NMDA/GABA)
    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    e_exc_mV: float = 0.0
    e_inh_mV: float = -70.0

    # Receptor-based conductances (used when synapse_model == "receptor")
    ampa_fraction: float = 0.7
    nmda_fraction: float = 0.3
    gabab_fraction: float = 0.0
    tau_ampa_ms: float = 5.0
    tau_nmda_ms: float = 80.0
    tau_gabaa_ms: float = 10.0
    tau_gabab_ms: float = 150.0
    nmda_mg_mM: float = 1.0

    # Short-term plasticity (Tsodyks-Markram; applied to synaptic release)
    stp_enabled: bool = False
    stp_apply_to: str = "exc"  # "exc" or "all"
    stp_U: float = 0.2
    stp_tau_rec_ms: float = 800.0
    stp_tau_facil_ms: float = 50.0

    # Izhikevich parameters
    izh_exc: Dict[str, float] = field(
        default_factory=lambda: {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "v_peak": 30.0}
    )
    izh_inh: Dict[str, float] = field(
        default_factory=lambda: {"a": 0.10, "b": 0.2, "c": -65.0, "d": 2.0, "v_peak": 30.0}
    )

    # Hodgkin-Huxley parameters (classic HH, ms/mV/mA·cm^-2)
    hh_C_m: float = 1.0
    hh_g_Na: float = 120.0
    hh_g_K: float = 36.0
    hh_g_L: float = 0.3
    hh_E_Na_mV: float = 50.0
    hh_E_K_mV: float = -77.0
    hh_E_L_mV: float = -54.4
    hh_internal_dt_ms: float = 0.05
    hh_spike_threshold_mV: float = 0.0

    # Two-compartment "mc" parameters (HH soma + passive dendrite)
    mc_v_dend_init_mV: float = -65.0
    mc_C_soma: float = 1.0
    mc_C_dend: float = 2.0
    mc_g_L_dend: float = 0.1
    mc_E_L_dend_mV: float = -65.0
    mc_g_couple: float = 0.1
    mc_dendrite_drive_fraction: float = 0.7

    # Optional active dendrite approximation (adds plateau/spike-like current)
    mc_dendrite_active: bool = False
    mc_dend_spike_threshold_mV: float = -20.0
    mc_dend_spike_refractory_ms: float = 5.0
    mc_dend_plateau_tau_ms: float = 40.0
    mc_dend_plateau_current: float = 10.0
    mc_dend_plateau_to_soma_fraction: float = 0.7

    # LIF parameters (ms/mV)
    lif_tau_m_ms: float = 20.0
    lif_v_rest_mV: float = -65.0
    lif_v_reset_mV: float = -65.0
    lif_v_thresh_mV: float = -50.0
    lif_refractory_ms: float = 2.0
    lif_R_m: float = 10.0

    # AdEx parameters (used when neuron_model == "adex" or "hybrid")
    adex_C: float = 30.0
    adex_g_L: float = 3.0
    adex_E_L_mV: float = -65.0
    adex_V_T_mV: float = -50.0
    adex_Delta_T_mV: float = 2.0
    adex_a: float = 1.0
    adex_tau_w_ms: float = 150.0
    adex_b: float = 8.0
    adex_V_thresh_mV: float = 0.0
    adex_V_reset_mV: float = -65.0
    adex_cell_type_params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Drive / noise
    baseline_current_mean: float = 6.0
    baseline_current_std: float = 2.0
    noise_std: float = 1.5
    input_scale: float = 1.0

    # Physiology coupling (glia / neurovascular / metabolism; optional)
    physiology_enabled: bool = False
    physiology_feedback: bool = False
    physiology_target_rate_hz: float = 5.0

    glia_tau_ms: float = 5000.0

    bloodflow_baseline: float = 1.0
    bloodflow_tau_ms: float = 2000.0
    bloodflow_gain: float = 0.5
    bold_gain: float = 0.3

    metabolism_atp_baseline: float = 2.5
    metabolism_atp_min: float = 0.1
    metabolism_atp_consumption_base: float = 0.01
    metabolism_atp_consumption_activity: float = 0.05
    metabolism_atp_recovery_gain: float = 0.03

    # Telemetry
    max_sample_neurons: int = 512


class BiophysicalSpikingNetwork(NeuralNetwork):
    """Downscaled whole-brain spiking network with biophysical primitives."""

    def __init__(self, config: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.params = dict(params or {})
        allowed = {f.name for f in dataclass_fields(BiophysicalSpikingConfig)}
        filtered = {k: v for k, v in self.params.items() if k in allowed}
        self.bio = BiophysicalSpikingConfig(**filtered)

        self._rng = np.random.default_rng(int(self.bio.seed))
        self.regions: List[str] = list(self.bio.regions)
        self.region_count = len(self.regions)
        self._region_is_cortex = np.asarray(
            [("CORTEX" in str(name).upper()) for name in self.regions], dtype=bool
        )
        self._layer_names: List[str] = ["L1", "L2/3", "L4", "L5", "L6"]
        self._layer_index = {name: idx for idx, name in enumerate(self._layer_names)}

        # Populations / neuron indexing
        self._region_slices: List[slice] = []
        self._neuron_region: np.ndarray
        self._is_inhibitory: np.ndarray
        self._neuron_layer: np.ndarray
        self._build_populations()

        self.n_neurons = int(self._neuron_region.size)
        self.global_step = 0
        self._region_sizes = np.asarray(
            [int(sl.stop) - int(sl.start) for sl in self._region_slices], dtype=np.int32
        )

        self._cell_model = self._normalize_neuron_model(self.bio.neuron_model)
        self._synapse_model = self._normalize_synapse_model(self.bio.synapse_model)

        self._neuron_cell_type = np.full(self.n_neurons, "", dtype="<U32")
        self._assign_cell_types()

        baseline_flow = float(getattr(self.bio, "bloodflow_baseline", 1.0))
        if not np.isfinite(baseline_flow) or baseline_flow <= 0.0:
            baseline_flow = 1.0
        atp_baseline = float(getattr(self.bio, "metabolism_atp_baseline", 2.5))
        if not np.isfinite(atp_baseline) or atp_baseline <= 0.0:
            atp_baseline = 2.5

        self._region_glia = np.zeros(self.region_count, dtype=np.float32)
        self._region_flow = np.full(self.region_count, baseline_flow, dtype=np.float32)
        self._region_atp = np.full(self.region_count, atp_baseline, dtype=np.float32)
        self._region_bold = np.zeros(self.region_count, dtype=np.float32)

        # Shared voltage state
        self.v = np.full(self.n_neurons, float(self.bio.v_init_mV), dtype=np.float32)
        self._mc_v_dend = np.full(self.n_neurons, float(self.bio.mc_v_dend_init_mV), dtype=np.float32)
        self._mc_plateau = np.zeros(self.n_neurons, dtype=np.float32)
        self._mc_dend_refractory_remaining = np.zeros(self.n_neurons, dtype=np.float32)

        # Izhikevich state vectors (used when neuron_model == "izhikevich")
        self.u = np.zeros(self.n_neurons, dtype=np.float32)
        self._a = np.zeros(self.n_neurons, dtype=np.float32)
        self._b = np.zeros(self.n_neurons, dtype=np.float32)
        self._c = np.zeros(self.n_neurons, dtype=np.float32)
        self._d = np.zeros(self.n_neurons, dtype=np.float32)
        self._v_peak = np.zeros(self.n_neurons, dtype=np.float32)
        if self._cell_model in {"izhikevich", "hybrid"}:
            self._init_izhikevich_parameters()

        # AdEx state vectors (used when neuron_model == "adex" or "hybrid")
        self._adex_w = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_C = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_g_L = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_E_L = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_V_T = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_Delta_T = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_a = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_tau_w = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_b = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_V_thresh = np.zeros(self.n_neurons, dtype=np.float32)
        self._adex_V_reset = np.zeros(self.n_neurons, dtype=np.float32)
        if self._cell_model in {"adex", "hybrid"}:
            self._init_adex_parameters()

        # Hodgkin-Huxley state vectors (used when neuron_model == "hh")
        self._hh_m = np.zeros(self.n_neurons, dtype=np.float32)
        self._hh_h = np.zeros(self.n_neurons, dtype=np.float32)
        self._hh_n = np.zeros(self.n_neurons, dtype=np.float32)
        if self._cell_model in {"hh", "mc"}:
            self._init_hh_state()

        # LIF refractory state (used when neuron_model == "lif")
        self._lif_refractory_remaining = np.zeros(self.n_neurons, dtype=np.float32)

        # Constant baseline drive
        self._baseline_current = self._rng.normal(
            self.bio.baseline_current_mean, self.bio.baseline_current_std, size=self.n_neurons
        ).astype(np.float32)

        # Plasticity and neuromodulator state (optional)
        self._pre_trace = np.zeros(self.n_neurons, dtype=np.float32)
        self._post_trace = np.zeros(self.n_neurons, dtype=np.float32)
        self._neuromodulators: Dict[str, float] = {}

        # Connectome + synapse list
        self._region_coords_mm: Optional[np.ndarray] = None
        self._estimated_axonal_velocity_m_s: Optional[float] = None
        self._connectome_w, self._connectome_delay_ms = self._load_connectome()
        self._syn_pre: np.ndarray = np.zeros(0, dtype=np.int32)
        self._syn_post: np.ndarray = np.zeros(0, dtype=np.int32)
        self._syn_weight: np.ndarray = np.zeros(0, dtype=np.float32)
        self._syn_delay_steps: np.ndarray = np.zeros(0, dtype=np.int16)
        self._syn_is_inh: np.ndarray = np.zeros(0, dtype=bool)

        # Runtime buffers initialised on first step (dt-dependent)
        self._dt_ms: Optional[float] = None
        self._queue_ptr = 0
        self._queue_len = 0
        self._queue_exc: Optional[np.ndarray] = None
        self._queue_inh: Optional[np.ndarray] = None
        self._g_exc: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)
        self._g_inh: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)

        # Receptor-based synapse state (optional; used when synapse_model == "receptor")
        self._queue_ampa: Optional[np.ndarray] = None
        self._queue_nmda: Optional[np.ndarray] = None
        self._queue_gabaa: Optional[np.ndarray] = None
        self._queue_gabab: Optional[np.ndarray] = None
        self._g_ampa: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)
        self._g_nmda: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)
        self._g_gabaa: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)
        self._g_gabab: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)

        # Short-term plasticity state (optional; allocated after synapses are built)
        self._stp_u: Optional[np.ndarray] = None
        self._stp_x: Optional[np.ndarray] = None
        self._stp_last_t_ms: Optional[np.ndarray] = None
        self._stp_mask: Optional[np.ndarray] = None

        # Export minimal neuron metadata for higher-level tooling
        for nid in range(self.n_neurons):
            layer_idx = int(self._neuron_layer[nid])
            self.neurons[nid] = {
                "neuron_id": int(nid),
                "region": self.regions[int(self._neuron_region[nid])],
                "population": "inhibitory" if bool(self._is_inhibitory[nid]) else "excitatory",
            }
            if layer_idx >= 0:
                self.neurons[nid]["layer"] = self._layer_names[layer_idx]
            cell_type = str(self._neuron_cell_type[nid])
            if cell_type:
                self.neurons[nid]["cell_type"] = cell_type

    @staticmethod
    def _normalize_neuron_model(model: Any) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {
            "hybrid",
            "mixed",
            "adex_izh",
            "adex+izh",
            "adex-izh",
            "adex_izhikevich",
            "adex-izhikevich",
        }:
            return "hybrid"
        if normalized in {
            "adex",
            "adaptive_exponential",
            "adaptive_exponential_if",
            "adaptive-exponential",
            "adaptive-exponential-if",
        }:
            return "adex"
        if normalized in {
            "mc",
            "multi_compartment",
            "multicompartment",
            "multi-compartment",
            "two_compartment",
            "2c",
        }:
            return "mc"
        if normalized in {"hh", "hodgkin_huxley", "hodgkin-huxley"}:
            return "hh"
        if normalized in {"lif", "leaky_integrate_and_fire", "leaky-integrate-and-fire"}:
            return "lif"
        return "izhikevich"

    @staticmethod
    def _normalize_synapse_model(model: Any) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {"receptor", "receptors", "ampa_nmda_gaba", "ampa/nmda/gaba"}:
            return "receptor"
        return "exp"

    @staticmethod
    def _exp_clip(x: np.ndarray, clip: float = 50.0) -> np.ndarray:
        return np.exp(np.clip(x, -float(clip), float(clip)))

    def _nmda_mg_block(self, v_mV: np.ndarray) -> np.ndarray:
        mg = float(getattr(self.bio, "nmda_mg_mM", 1.0))
        if not np.isfinite(mg) or mg <= 0.0:
            return np.ones_like(v_mV, dtype=np.float32)
        # Jahr & Stevens-style magnesium block approximation.
        denom = 1.0 + (mg / 3.57) * self._exp_clip(-0.062 * np.asarray(v_mV, dtype=np.float32))
        return (1.0 / denom).astype(np.float32)

    def _hh_rates(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = np.asarray(v, dtype=np.float32)
        x_m = v + 40.0
        x_n = v + 55.0

        denom_m = -np.expm1(-x_m / 10.0)
        denom_n = -np.expm1(-x_n / 10.0)

        alpha_m = np.empty_like(v)
        alpha_n = np.empty_like(v)

        valid_m = np.abs(x_m) >= 1e-6
        valid_n = np.abs(x_n) >= 1e-6
        np.divide(0.1 * x_m, denom_m, out=alpha_m, where=valid_m)
        np.divide(0.01 * x_n, denom_n, out=alpha_n, where=valid_n)
        alpha_m[~valid_m] = 1.0
        alpha_n[~valid_n] = 0.1

        beta_m = 4.0 * self._exp_clip(-(v + 65.0) / 18.0)
        alpha_h = 0.07 * self._exp_clip(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + self._exp_clip(-(v + 35.0) / 10.0))
        beta_n = 0.125 * self._exp_clip(-(v + 65.0) / 80.0)

        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    def _init_hh_state(self) -> None:
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._hh_rates(self.v)
        self._hh_m = np.clip(alpha_m / (alpha_m + beta_m + 1e-12), 0.0, 1.0).astype(np.float32)
        self._hh_h = np.clip(alpha_h / (alpha_h + beta_h + 1e-12), 0.0, 1.0).astype(np.float32)
        self._hh_n = np.clip(alpha_n / (alpha_n + beta_n + 1e-12), 0.0, 1.0).astype(np.float32)

    def _load_connectome(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load connectome weights/delays from config or fall back to a default scaffold."""

        weights = getattr(self.bio, "connectome_weights", None)
        delays = getattr(self.bio, "connectome_delays_ms", None)
        coords = getattr(self.bio, "connectome_coords_mm", None)
        region_names = getattr(self.bio, "connectome_region_names", None)

        if weights is None:
            path = self.bio.connectome_npz_path
            if not path:
                return _default_connectome(self.regions)

            npz_path = Path(path)
            if not npz_path.is_file():
                raise FileNotFoundError(f"Connectome npz file not found: {npz_path}")

            payload = np.load(npz_path, allow_pickle=True)
            weights = payload.get("weights")
            delays = payload.get("delays_ms", payload.get("delays"))
            coords = payload.get("coords_mm", payload.get("coords"))
            region_names = payload.get("region_names", payload.get("regions"))
            if weights is None:
                raise ValueError("Connectome npz must contain a 'weights' array")

        weights = np.asarray(weights, dtype=np.float32)
        if delays is not None:
            delays = np.asarray(delays, dtype=np.float32)
        else:
            delays = None

        if region_names is not None:
            names = [str(x) for x in np.asarray(region_names).reshape(-1).tolist()]
            if len(names) != self.region_count:
                raise ValueError(
                    f"Connectome region_names length mismatch: expected {self.region_count}, got {len(names)}"
                )
            mapping = {name: idx for idx, name in enumerate(names)}
            try:
                order = [mapping[str(name)] for name in self.regions]
            except KeyError as exc:
                raise ValueError(
                    f"Connectome region_names does not include required region '{exc.args[0]}'"
                ) from exc
            weights = weights[np.ix_(order, order)]
            if delays is not None:
                delays = delays[np.ix_(order, order)]
            if coords is not None:
                coords = np.asarray(coords, dtype=np.float32)
                coords = coords[order]

        expected = (self.region_count, self.region_count)
        if weights.shape != expected or (delays is not None and delays.shape != expected):
            raise ValueError(
                f"Connectome shape mismatch: expected {expected}, got weights={weights.shape}, delays={None if delays is None else delays.shape}"
            )

        # Clamp to a safe numeric range; these are dimensionless scaffold weights.
        weights = np.clip(weights, 0.0, 1.0)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.shape != (self.region_count, 3):
                raise ValueError(
                    f"Connectome coords_mm shape mismatch: expected {(self.region_count, 3)}, got {coords.shape}"
                )
            self._region_coords_mm = coords

        syn_delay = float(self.bio.connectome_synaptic_delay_ms)
        if delays is None:
            if self._region_coords_mm is None:
                raise ValueError(
                    "Connectome npz must contain 'delays_ms' (or 'delays') or provide 'coords_mm' to derive delays"
                )
            dist = self._region_coords_mm[:, None, :] - self._region_coords_mm[None, :, :]
            dist_mm = np.linalg.norm(dist, axis=-1).astype(np.float32)
            vel = float(self.bio.axonal_velocity_m_s)
            if not np.isfinite(vel) or vel <= 0.0:
                vel = 5.0
            delays = dist_mm / vel + syn_delay
        else:
            delays = np.asarray(delays, dtype=np.float32)
            if bool(self.bio.estimate_axonal_velocity) and self._region_coords_mm is not None:
                dist = self._region_coords_mm[:, None, :] - self._region_coords_mm[None, :, :]
                dist_mm = np.linalg.norm(dist, axis=-1).astype(np.float32)
                mask = (weights > 0.0) & (delays > syn_delay + 1e-3)
                if np.any(mask):
                    estimates = dist_mm[mask] / np.maximum(delays[mask] - syn_delay, 1e-3)
                    if estimates.size:
                        self._estimated_axonal_velocity_m_s = float(np.median(estimates))

        delays = np.clip(delays, 0.1, float(self.bio.max_delay_ms))
        return weights, delays

    # ----------------------------------------------------------------- build
    def _build_populations(self) -> None:
        region_slices: List[slice] = []
        neuron_region: List[int] = []
        inhibitory: List[bool] = []
        neuron_layer: List[int] = []

        proportions_raw = self.bio.cortical_layer_proportions or {}
        proportions: Dict[str, float] = {}
        if isinstance(proportions_raw, dict):
            for key, value in proportions_raw.items():
                if not key:
                    continue
                try:
                    proportions[str(key)] = float(value)
                except Exception:
                    continue

        def _allocate_layers(count: int) -> List[int]:
            if count <= 0:
                return []

            layer_names: List[str] = []
            weights: List[float] = []
            for lname in self._layer_names:
                w = float(proportions.get(lname, 0.0))
                if w > 0.0:
                    layer_names.append(lname)
                    weights.append(w)

            if not weights:
                fallback = int(self._layer_index.get("L2/3", 0))
                return [fallback] * int(count)

            total = float(sum(weights))
            weights = [w / total for w in weights]

            raw = [float(count) * w for w in weights]
            counts = [int(np.floor(x)) for x in raw]
            remainder = int(count) - int(sum(counts))
            if remainder > 0:
                frac = np.asarray([x - np.floor(x) for x in raw], dtype=np.float64)
                order = np.argsort(-frac)  # descending
                for k in range(remainder):
                    counts[int(order[k % int(order.size)])] += 1

            assigned: List[int] = []
            for lname, n_alloc in zip(layer_names, counts):
                assigned.extend([int(self._layer_index[lname])] * int(n_alloc))
            self._rng.shuffle(assigned)
            return assigned

        next_id = 0
        for ridx in range(self.region_count):
            n = int(self.bio.neurons_per_region)
            start = next_id
            end = start + n
            next_id = end
            region_slices.append(slice(start, end))

            n_exc = max(1, int(round(n * float(np.clip(self.bio.excitatory_ratio, 0.0, 1.0)))))
            n_inh = max(0, n - n_exc)
            neuron_region.extend([ridx] * n)
            inhibitory.extend([False] * n_exc + [True] * n_inh)
            if bool(self.bio.cortical_layers_enabled) and bool(self._region_is_cortex[ridx]):
                neuron_layer.extend(_allocate_layers(n_exc))
                neuron_layer.extend(_allocate_layers(n_inh))
            else:
                neuron_layer.extend([-1] * n)

        self._region_slices = region_slices
        self._neuron_region = np.asarray(neuron_region, dtype=np.int16)
        self._is_inhibitory = np.asarray(inhibitory, dtype=bool)
        self._neuron_layer = np.asarray(neuron_layer, dtype=np.int8)

    def _assign_cell_types(self) -> None:
        if not bool(getattr(self.bio, "cell_types_enabled", False)):
            self._neuron_cell_type[:] = np.where(self._is_inhibitory, "inhibitory", "excitatory")
            return

        dist_raw = getattr(self.bio, "inhibitory_subtype_distribution", None) or {}
        names: List[str] = []
        probs: List[float] = []
        if isinstance(dist_raw, dict):
            for key, value in dist_raw.items():
                if not key:
                    continue
                try:
                    prob = float(value)
                except Exception:
                    continue
                if prob <= 0.0 or not np.isfinite(prob):
                    continue
                names.append(str(key))
                probs.append(prob)
        if not names:
            names = ["pv_fs", "sst_lts"]
            probs = [0.7, 0.3]
        total = float(sum(probs))
        probs = [p / total for p in probs]

        for ridx, sl in enumerate(self._region_slices):
            region_name = str(self.regions[ridx]).upper()
            ids = np.arange(sl.start, sl.stop, dtype=np.int32)
            if ids.size == 0:
                continue

            exc_ids = ids[~self._is_inhibitory[ids]]
            inh_ids = ids[self._is_inhibitory[ids]]

            # Excitatory types (layer/region specific).
            if "THAL" in region_name:
                self._neuron_cell_type[exc_ids] = "thalamus_tc"
            elif (
                "HIPPO" in region_name
                or region_name in {"DG", "CA1", "CA2", "CA3", "DENTATE", "DENTATE_GYRUS", "SUBICULUM"}
                or region_name.startswith("CA")
            ):
                if exc_ids.size:
                    choose_ib = self._rng.random(size=exc_ids.size) < 0.3
                    self._neuron_cell_type[exc_ids[choose_ib]] = "hippocampus_ib"
                    self._neuron_cell_type[exc_ids[~choose_ib]] = "hippocampus_rs"
            elif "BASAL" in region_name or "GANGLIA" in region_name or "STRIAT" in region_name:
                self._neuron_cell_type[exc_ids] = "bg_msn"
            elif "CORTEX" in region_name and bool(self.bio.cortical_layers_enabled):
                for layer_idx, lname in enumerate(self._layer_names):
                    layer_exc = exc_ids[self._neuron_layer[exc_ids] == layer_idx]
                    if layer_exc.size == 0:
                        continue
                    self._neuron_cell_type[layer_exc] = "cortex_ib" if lname == "L5" else "cortex_rs"
            else:
                self._neuron_cell_type[exc_ids] = "cortex_rs"

            # Inhibitory subtype distribution (region-agnostic by default).
            if inh_ids.size:
                chosen = self._rng.choice(
                    np.asarray(names, dtype="<U32"),
                    size=inh_ids.size,
                    replace=True,
                    p=np.asarray(probs, dtype=np.float64),
                )
                self._neuron_cell_type[inh_ids] = chosen.astype("<U32")

    def _init_izhikevich_parameters(self) -> None:
        exc = self.bio.izh_exc
        inh = self.bio.izh_inh

        exc_mask = ~self._is_inhibitory
        inh_mask = self._is_inhibitory

        self._a[exc_mask] = float(exc.get("a", 0.02))
        self._b[exc_mask] = float(exc.get("b", 0.2))
        self._c[exc_mask] = float(exc.get("c", -65.0))
        self._d[exc_mask] = float(exc.get("d", 8.0))
        self._v_peak[exc_mask] = float(exc.get("v_peak", 30.0))

        self._a[inh_mask] = float(inh.get("a", 0.1))
        self._b[inh_mask] = float(inh.get("b", 0.2))
        self._c[inh_mask] = float(inh.get("c", -65.0))
        self._d[inh_mask] = float(inh.get("d", 2.0))
        self._v_peak[inh_mask] = float(inh.get("v_peak", 30.0))

        if bool(getattr(self.bio, "cell_types_enabled", False)):
            library = _default_cell_type_params_izh()
            overrides = getattr(self.bio, "cell_type_params", None)
            if isinstance(overrides, dict):
                for key, params in overrides.items():
                    if not key or not isinstance(params, dict):
                        continue
                    merged = dict(library.get(str(key), {}))
                    merged.update(params)
                    library[str(key)] = merged

            for cell_type in np.unique(self._neuron_cell_type):
                ct = str(cell_type)
                if not ct:
                    continue
                params = library.get(ct)
                if not isinstance(params, dict) or not params:
                    continue
                mask = self._neuron_cell_type == ct
                if not np.any(mask):
                    continue
                if "a" in params:
                    self._a[mask] = float(params["a"])
                if "b" in params:
                    self._b[mask] = float(params["b"])
                if "c" in params:
                    self._c[mask] = float(params["c"])
                if "d" in params:
                    self._d[mask] = float(params["d"])
                if "v_peak" in params:
                    self._v_peak[mask] = float(params["v_peak"])

        self.u = self._b * self.v

    def _init_adex_parameters(self) -> None:
        """Initialise AdEx parameters (scaled units) with optional cell-type overrides."""

        def _safe(value: Any, default: float, *, minimum: Optional[float] = None) -> float:
            try:
                x = float(value)
            except Exception:
                x = float(default)
            if not np.isfinite(x):
                x = float(default)
            if minimum is not None:
                x = max(float(minimum), float(x))
            return float(x)

        C = _safe(getattr(self.bio, "adex_C", 30.0), 30.0, minimum=1e-6)
        g_L = _safe(getattr(self.bio, "adex_g_L", 3.0), 3.0, minimum=1e-6)
        E_L = _safe(getattr(self.bio, "adex_E_L_mV", -65.0), -65.0)
        V_T = _safe(getattr(self.bio, "adex_V_T_mV", -50.0), -50.0)
        Delta_T = _safe(getattr(self.bio, "adex_Delta_T_mV", 2.0), 2.0, minimum=1e-6)
        a = _safe(getattr(self.bio, "adex_a", 1.0), 1.0)
        tau_w = _safe(getattr(self.bio, "adex_tau_w_ms", 150.0), 150.0, minimum=1e-6)
        b = _safe(getattr(self.bio, "adex_b", 8.0), 8.0)
        V_thresh = _safe(getattr(self.bio, "adex_V_thresh_mV", 0.0), 0.0)
        V_reset = _safe(getattr(self.bio, "adex_V_reset_mV", -65.0), -65.0)

        self._adex_C.fill(C)
        self._adex_g_L.fill(g_L)
        self._adex_E_L.fill(E_L)
        self._adex_V_T.fill(V_T)
        self._adex_Delta_T.fill(Delta_T)
        self._adex_a.fill(a)
        self._adex_tau_w.fill(tau_w)
        self._adex_b.fill(b)
        self._adex_V_thresh.fill(V_thresh)
        self._adex_V_reset.fill(V_reset)
        self._adex_w.fill(0.0)

        if not bool(getattr(self.bio, "cell_types_enabled", False)):
            return

        library = _default_cell_type_params_adex()
        overrides = getattr(self.bio, "adex_cell_type_params", None)
        if isinstance(overrides, dict):
            for key, params in overrides.items():
                if not key or not isinstance(params, dict):
                    continue
                merged = dict(library.get(str(key), {}))
                merged.update(params)
                library[str(key)] = merged

        for cell_type in np.unique(self._neuron_cell_type):
            ct = str(cell_type)
            if not ct:
                continue
            params = library.get(ct)
            if not isinstance(params, dict) or not params:
                continue
            mask = self._neuron_cell_type == ct
            if self._cell_model == "hybrid":
                mask = mask & (~self._is_inhibitory)
            if not np.any(mask):
                continue

            if "C" in params:
                self._adex_C[mask] = _safe(params["C"], C, minimum=1e-6)
            if "g_L" in params:
                self._adex_g_L[mask] = _safe(params["g_L"], g_L, minimum=1e-6)
            if "E_L" in params or "E_L_mV" in params:
                self._adex_E_L[mask] = _safe(params.get("E_L_mV", params.get("E_L")), E_L)
            if "V_T" in params or "V_T_mV" in params:
                self._adex_V_T[mask] = _safe(params.get("V_T_mV", params.get("V_T")), V_T)
            if "Delta_T" in params or "Delta_T_mV" in params:
                self._adex_Delta_T[mask] = _safe(
                    params.get("Delta_T_mV", params.get("Delta_T")), Delta_T, minimum=1e-6
                )
            if "a" in params:
                self._adex_a[mask] = _safe(params["a"], a)
            if "tau_w" in params or "tau_w_ms" in params:
                self._adex_tau_w[mask] = _safe(
                    params.get("tau_w_ms", params.get("tau_w")), tau_w, minimum=1e-6
                )
            if "b" in params:
                self._adex_b[mask] = _safe(params["b"], b)
            if "V_thresh" in params or "V_thresh_mV" in params:
                self._adex_V_thresh[mask] = _safe(params.get("V_thresh_mV", params.get("V_thresh")), V_thresh)
            if "V_reset" in params or "V_reset_mV" in params:
                self._adex_V_reset[mask] = _safe(params.get("V_reset_mV", params.get("V_reset")), V_reset)

    def _ensure_synapses(self, dt_ms: float) -> None:
        if self._dt_ms is not None and abs(self._dt_ms - dt_ms) < 1e-9:
            if self._synapse_model == "exp" and self._queue_exc is not None and self._queue_inh is not None:
                return
            if (
                self._synapse_model == "receptor"
                and self._queue_ampa is not None
                and self._queue_nmda is not None
                and self._queue_gabaa is not None
                and self._queue_gabab is not None
            ):
                return

        self._dt_ms = float(dt_ms)
        max_delay_steps = max(1, int(np.ceil(float(self.bio.max_delay_ms) / max(self._dt_ms, 1e-6))))
        queue_len = max_delay_steps + 1
        self._queue_ptr = 0
        self._queue_len = int(queue_len)
        if self._synapse_model == "receptor":
            self._queue_exc = None
            self._queue_inh = None
            self._queue_ampa = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._queue_nmda = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._queue_gabaa = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._queue_gabab = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._g_ampa.fill(0.0)
            self._g_nmda.fill(0.0)
            self._g_gabaa.fill(0.0)
            self._g_gabab.fill(0.0)
        else:
            self._queue_ampa = None
            self._queue_nmda = None
            self._queue_gabaa = None
            self._queue_gabab = None
            self._queue_exc = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._queue_inh = np.zeros((queue_len, self.n_neurons), dtype=np.float32)
            self._g_exc.fill(0.0)
            self._g_inh.fill(0.0)

        self._build_synapses(max_delay_steps=max_delay_steps)

    def _build_synapses(self, *, max_delay_steps: int) -> None:
        pre: List[int] = []
        post: List[int] = []
        weight: List[float] = []
        delay_steps: List[int] = []
        is_inh: List[bool] = []

        dt_ms = float(self._dt_ms or 1.0)

        raw_layer_connectivity = self.bio.cortical_layer_connectivity
        layer_connectivity: Dict[str, float] = {}
        if isinstance(raw_layer_connectivity, dict):
            for key, value in raw_layer_connectivity.items():
                if not key:
                    continue
                try:
                    layer_connectivity[str(key).strip()] = float(value)
                except Exception:
                    continue

        def _layer_multiplier(pre_layer: int, post_layer: int) -> float:
            if pre_layer < 0 or post_layer < 0:
                return 1.0
            if pre_layer >= len(self._layer_names) or post_layer >= len(self._layer_names):
                return 1.0
            key = f"{self._layer_names[int(pre_layer)]}->{self._layer_names[int(post_layer)]}"
            return float(layer_connectivity.get(key, 1.0))

        # Intra-region microcircuit
        p_intra = float(np.clip(self.bio.intra_connection_prob, 0.0, 1.0))
        for ridx, sl in enumerate(self._region_slices):
            ids = np.arange(sl.start, sl.stop, dtype=np.int32)
            if ids.size <= 1 or p_intra <= 0.0:
                continue

            exc_ids = ids[~self._is_inhibitory[ids]]
            inh_ids = ids[self._is_inhibitory[ids]]

            def _connect_block(
                pre_ids: np.ndarray, post_ids: np.ndarray, *, inhibitory: bool, prob: float
            ) -> None:
                if pre_ids.size == 0 or post_ids.size == 0:
                    return
                prob_f = float(np.clip(float(prob), 0.0, 1.0))
                expected = int(round(prob_f * pre_ids.size * post_ids.size))
                if expected <= 0:
                    return
                # Sample pairs with replacement (sufficient for downscaled sims)
                pre_choice = self._rng.choice(pre_ids, size=expected, replace=True)
                post_choice = self._rng.choice(post_ids, size=expected, replace=True)
                w_mean = 0.08 if not inhibitory else 0.12
                w_std = 0.02 if not inhibitory else 0.03
                w = np.clip(self._rng.normal(w_mean, w_std, size=expected), 0.005, 1.0)
                d_ms = float(self._rng.uniform(1.0, min(self.bio.max_delay_ms, 6.0)))
                d_steps = int(np.clip(int(round(d_ms / dt_ms)), 0, max_delay_steps))

                pre.extend(int(x) for x in pre_choice)
                post.extend(int(x) for x in post_choice)
                weight.extend(float(x) for x in w)
                delay_steps.extend([d_steps] * expected)
                is_inh.extend([bool(inhibitory)] * expected)

            use_layers = bool(self.bio.cortical_layers_enabled) and bool(self._region_is_cortex[ridx])
            if use_layers:
                ids_by_layer: Dict[int, np.ndarray] = {}
                for layer_idx in range(len(self._layer_names)):
                    layer_mask = self._neuron_layer[ids] == layer_idx
                    layer_ids = ids[layer_mask]
                    if layer_ids.size:
                        ids_by_layer[int(layer_idx)] = layer_ids

                exc_by_layer: Dict[int, np.ndarray] = {
                    layer_idx: layer_ids[~self._is_inhibitory[layer_ids]]
                    for layer_idx, layer_ids in ids_by_layer.items()
                }
                post_by_layer: Dict[int, np.ndarray] = ids_by_layer

                # Excitatory projections (local, layer-specific)
                for pre_layer, pre_ids in exc_by_layer.items():
                    if pre_ids.size == 0:
                        continue
                    for post_layer, post_ids in post_by_layer.items():
                        if post_ids.size == 0:
                            continue
                        prob = p_intra * _layer_multiplier(int(pre_layer), int(post_layer))
                        _connect_block(pre_ids, post_ids, inhibitory=False, prob=prob)
            else:
                # Excitatory projections (local)
                _connect_block(exc_ids, ids, inhibitory=False, prob=p_intra)
            # Inhibitory projections (local)
            _connect_block(inh_ids, ids, inhibitory=True, prob=p_intra)

        # Inter-region long-range excitatory projections from connectome
        p_inter = float(np.clip(self.bio.inter_connection_prob, 0.0, 1.0))
        if p_inter > 0.0:
            for src in range(self.region_count):
                src_sl = self._region_slices[src]
                src_ids = np.arange(src_sl.start, src_sl.stop, dtype=np.int32)
                src_exc = src_ids[~self._is_inhibitory[src_ids]]
                if src_exc.size == 0:
                    continue
                src_name = str(self.regions[src]).upper()
                src_is_thalamus = "THALAMUS" in src_name
                src_is_cortex = bool(self._region_is_cortex[src])
                for dst in range(self.region_count):
                    if src == dst:
                        continue
                    w_ij = float(self._connectome_w[src, dst])
                    if w_ij <= 0.0:
                        continue
                    dst_sl = self._region_slices[dst]
                    dst_ids = np.arange(dst_sl.start, dst_sl.stop, dtype=np.int32)
                    if dst_ids.size == 0:
                        continue
                    dst_name = str(self.regions[dst]).upper()
                    dst_is_thalamus = "THALAMUS" in dst_name
                    dst_is_cortex = bool(self._region_is_cortex[dst])

                    pre_candidates = src_exc
                    post_candidates = dst_ids

                    # Thalamocortical targeting: thalamus -> cortex tends to target L4/L6.
                    if (
                        bool(self.bio.cortical_layers_enabled)
                        and src_is_thalamus
                        and dst_is_cortex
                        and self.bio.thalamus_target_layers
                    ):
                        target_layers = [
                            int(self._layer_index[layer])
                            for layer in self.bio.thalamus_target_layers
                            if layer in self._layer_index
                        ]
                        if target_layers:
                            mask = np.isin(self._neuron_layer[dst_ids], target_layers)
                            subset = dst_ids[mask]
                            if subset.size:
                                post_candidates = subset

                    # Corticothalamic feedback: cortex -> thalamus tends to originate in L6.
                    if (
                        bool(self.bio.cortical_layers_enabled)
                        and dst_is_thalamus
                        and src_is_cortex
                        and self.bio.cortex_feedback_layer
                        and self.bio.cortex_feedback_layer in self._layer_index
                    ):
                        fb_layer = int(self._layer_index[self.bio.cortex_feedback_layer])
                        mask = self._neuron_layer[src_exc] == fb_layer
                        subset = src_exc[mask]
                        if subset.size:
                            pre_candidates = subset

                    if pre_candidates.size == 0 or post_candidates.size == 0:
                        continue

                    prob_base = float(np.clip(p_inter * w_ij, 0.0, 1.0))
                    d_ms = float(self._connectome_delay_ms[src, dst])
                    d_steps = int(np.clip(int(round(d_ms / dt_ms)), 0, max_delay_steps))

                    use_layers = (
                        bool(self.bio.cortical_layers_enabled) and src_is_cortex and dst_is_cortex
                    )
                    if use_layers:
                        pre_by_layer: Dict[int, np.ndarray] = {}
                        for layer_idx in range(len(self._layer_names)):
                            mask = self._neuron_layer[pre_candidates] == layer_idx
                            layer_ids = pre_candidates[mask]
                            if layer_ids.size:
                                pre_by_layer[int(layer_idx)] = layer_ids
                        post_by_layer: Dict[int, np.ndarray] = {}
                        for layer_idx in range(len(self._layer_names)):
                            mask = self._neuron_layer[post_candidates] == layer_idx
                            layer_ids = post_candidates[mask]
                            if layer_ids.size:
                                post_by_layer[int(layer_idx)] = layer_ids

                        for pre_layer, pre_ids in pre_by_layer.items():
                            for post_layer, post_ids in post_by_layer.items():
                                prob = float(
                                    np.clip(
                                        prob_base * _layer_multiplier(int(pre_layer), int(post_layer)),
                                        0.0,
                                        1.0,
                                    )
                                )
                                expected = int(round(prob * pre_ids.size * post_ids.size))
                                if expected <= 0:
                                    continue
                                pre_choice = self._rng.choice(pre_ids, size=expected, replace=True)
                                post_choice = self._rng.choice(post_ids, size=expected, replace=True)
                                w_mean = 0.05 * w_ij
                                w_std = 0.02 * w_ij
                                w = np.clip(self._rng.normal(w_mean, w_std, size=expected), 0.002, 1.0)

                                pre.extend(int(x) for x in pre_choice)
                                post.extend(int(x) for x in post_choice)
                                weight.extend(float(x) for x in w)
                                delay_steps.extend([d_steps] * expected)
                                is_inh.extend([False] * expected)  # long-range is excitatory here
                    else:
                        expected = int(round(prob_base * pre_candidates.size * post_candidates.size))
                        if expected <= 0:
                            continue

                        pre_choice = self._rng.choice(pre_candidates, size=expected, replace=True)
                        post_choice = self._rng.choice(post_candidates, size=expected, replace=True)
                        w_mean = 0.05 * w_ij
                        w_std = 0.02 * w_ij
                        w = np.clip(self._rng.normal(w_mean, w_std, size=expected), 0.002, 1.0)

                        pre.extend(int(x) for x in pre_choice)
                        post.extend(int(x) for x in post_choice)
                        weight.extend(float(x) for x in w)
                        delay_steps.extend([d_steps] * expected)
                        is_inh.extend([False] * expected)  # long-range is excitatory here

        self._syn_pre = np.asarray(pre, dtype=np.int32)
        self._syn_post = np.asarray(post, dtype=np.int32)
        self._syn_weight = np.asarray(weight, dtype=np.float32)
        self._syn_delay_steps = np.asarray(delay_steps, dtype=np.int16)
        self._syn_is_inh = np.asarray(is_inh, dtype=bool)

        # Keep a small synapse view for optional introspection tools.
        self.synapses = {
            idx: {
                "pre": int(self._syn_pre[idx]),
                "post": int(self._syn_post[idx]),
                "weight": float(self._syn_weight[idx]),
                "delay_steps": int(self._syn_delay_steps[idx]),
                "type": "inhibitory" if bool(self._syn_is_inh[idx]) else "excitatory",
            }
            for idx in range(min(2000, int(self._syn_pre.size)))
        }

        self._init_short_term_plasticity()

    def _init_short_term_plasticity(self) -> None:
        self._stp_u = None
        self._stp_x = None
        self._stp_last_t_ms = None
        self._stp_mask = None

        if not bool(getattr(self.bio, "stp_enabled", False)):
            return
        syn_count = int(self._syn_pre.size)
        if syn_count <= 0:
            return

        U = float(np.clip(float(getattr(self.bio, "stp_U", 0.2)), 1e-6, 1.0))
        self._stp_u = np.full(syn_count, U, dtype=np.float32)
        self._stp_x = np.ones(syn_count, dtype=np.float32)
        self._stp_last_t_ms = np.full(syn_count, -1e9, dtype=np.float32)

        apply_to = str(getattr(self.bio, "stp_apply_to", "exc") or "exc").strip().lower()
        if apply_to == "all":
            self._stp_mask = np.ones(syn_count, dtype=bool)
        else:
            self._stp_mask = (~self._syn_is_inh).copy()

    def _stp_release_scale(self, synapse_indices: np.ndarray, *, time_ms: float) -> np.ndarray:
        scale = np.ones(int(np.asarray(synapse_indices).size), dtype=np.float32)
        if scale.size == 0:
            return scale
        if not bool(getattr(self.bio, "stp_enabled", False)):
            return scale
        if self._stp_u is None or self._stp_x is None or self._stp_last_t_ms is None or self._stp_mask is None:
            return scale

        synapse_indices = np.asarray(synapse_indices, dtype=np.int64).reshape(-1)
        mask = self._stp_mask[synapse_indices]
        if not np.any(mask):
            return scale

        idxs = synapse_indices[mask].astype(np.int64, copy=False)
        last = self._stp_last_t_ms[idxs]
        dt = np.maximum(float(time_ms) - last, 0.0).astype(np.float32)

        U = float(np.clip(float(getattr(self.bio, "stp_U", 0.2)), 1e-6, 1.0))
        tau_rec = max(float(getattr(self.bio, "stp_tau_rec_ms", 800.0)), 1e-6)
        tau_facil = float(getattr(self.bio, "stp_tau_facil_ms", 50.0))

        x_prev = self._stp_x[idxs]
        x_t = 1.0 + (x_prev - 1.0) * np.exp(-dt / tau_rec)
        if np.isfinite(tau_facil) and tau_facil > 1e-6:
            u_prev = self._stp_u[idxs]
            u_t = U + (u_prev - U) * np.exp(-dt / tau_facil)
            u_t = u_t + U * (1.0 - u_t)
        else:
            u_t = np.full_like(x_t, U, dtype=np.float32)

        release = u_t * x_t
        x_next = np.clip(x_t - release, 0.0, 1.0).astype(np.float32)
        u_next = np.clip(u_t, 0.0, 1.0).astype(np.float32)

        self._stp_x[idxs] = x_next
        self._stp_u[idxs] = u_next
        self._stp_last_t_ms[idxs] = float(time_ms)

        # Keep baseline amplitude compatible by normalizing relative to U at rest.
        scale_vals = np.clip(release / max(U, 1e-6), 0.0, 5.0).astype(np.float32)
        scale[np.nonzero(mask)[0]] = scale_vals
        return scale

    # ---------------------------------------------------------------- runtime
    def reset(self) -> None:
        super().reset()
        self.global_step = 0

        baseline_flow = float(getattr(self.bio, "bloodflow_baseline", 1.0))
        if not np.isfinite(baseline_flow) or baseline_flow <= 0.0:
            baseline_flow = 1.0
        atp_baseline = float(getattr(self.bio, "metabolism_atp_baseline", 2.5))
        if not np.isfinite(atp_baseline) or atp_baseline <= 0.0:
            atp_baseline = 2.5
        self._region_glia.fill(0.0)
        self._region_flow.fill(baseline_flow)
        self._region_atp.fill(atp_baseline)
        self._region_bold.fill(0.0)

        if self._cell_model == "lif":
            self.v.fill(float(self.bio.lif_v_rest_mV))
        else:
            self.v.fill(float(self.bio.v_init_mV))

        if self._cell_model == "mc":
            self._mc_v_dend.fill(float(self.bio.mc_v_dend_init_mV))
            self._mc_plateau.fill(0.0)
            self._mc_dend_refractory_remaining.fill(0.0)

        if self._cell_model in {"izhikevich", "hybrid"}:
            self.u = (self._b * self.v).astype(np.float32, copy=False)
        else:
            self.u.fill(0.0)

        if self._cell_model in {"adex", "hybrid"}:
            self._adex_w.fill(0.0)

        if self._cell_model in {"hh", "mc"}:
            self._init_hh_state()
        if self._cell_model == "lif":
            self._lif_refractory_remaining.fill(0.0)
        self._g_exc.fill(0.0)
        self._g_inh.fill(0.0)
        self._g_ampa.fill(0.0)
        self._g_nmda.fill(0.0)
        self._g_gabaa.fill(0.0)
        self._g_gabab.fill(0.0)
        if hasattr(self, "_pre_trace"):
            self._pre_trace.fill(0.0)
        if hasattr(self, "_post_trace"):
            self._post_trace.fill(0.0)
        self._queue_ptr = 0
        if self._queue_exc is not None:
            self._queue_exc.fill(0.0)
        if self._queue_inh is not None:
            self._queue_inh.fill(0.0)
        if self._queue_ampa is not None:
            self._queue_ampa.fill(0.0)
        if self._queue_nmda is not None:
            self._queue_nmda.fill(0.0)
        if self._queue_gabaa is not None:
            self._queue_gabaa.fill(0.0)
        if self._queue_gabab is not None:
            self._queue_gabab.fill(0.0)

        if self._stp_u is not None and self._stp_x is not None and self._stp_last_t_ms is not None:
            U = float(np.clip(float(getattr(self.bio, "stp_U", 0.2)), 1e-6, 1.0))
            self._stp_u.fill(U)
            self._stp_x.fill(1.0)
            self._stp_last_t_ms.fill(-1e9)

    def set_neuromodulators(self, neuromodulators: Dict[str, float]) -> None:
        if not isinstance(neuromodulators, dict):
            return
        cleaned: Dict[str, float] = {}
        for key, value in neuromodulators.items():
            if not key:
                continue
            try:
                cleaned[str(key)] = float(value)
            except Exception:
                continue
        self._neuromodulators = cleaned

    def scale_synaptic_weights(
        self,
        factor: float,
        *,
        exc_only: bool = False,
        inh_only: bool = False,
    ) -> Dict[str, Any]:
        """Scale synaptic weights in-place.

        This is a lightweight hook intended for slow processes such as sleep
        synaptic homeostasis / global downscaling. It is safe to call even when
        the network has no synapses yet (a no-op).
        """

        try:
            factor_f = float(factor)
        except Exception:
            factor_f = 1.0
        if not np.isfinite(factor_f) or factor_f < 0.0:
            factor_f = 1.0

        if self._syn_weight.size == 0:
            return {"scaled": 0, "factor": float(factor_f)}

        exc_mask = ~self._syn_is_inh
        inh_mask = self._syn_is_inh

        if bool(exc_only) and not bool(inh_only):
            mask = exc_mask
        elif bool(inh_only) and not bool(exc_only):
            mask = inh_mask
        else:
            mask = np.ones_like(self._syn_weight, dtype=bool)

        scaled_count = int(np.count_nonzero(mask))
        if scaled_count == 0:
            return {"scaled": 0, "factor": float(factor_f)}

        before_mean = float(np.mean(self._syn_weight[mask])) if scaled_count else 0.0
        self._syn_weight[mask] = (self._syn_weight[mask] * float(factor_f)).astype(np.float32, copy=False)
        np.clip(self._syn_weight, 0.0, np.inf, out=self._syn_weight)

        if bool(getattr(self.bio, "stdp_enabled", False)):
            exc_scaled = mask & exc_mask
            if np.any(exc_scaled):
                self._syn_weight[exc_scaled] = np.clip(
                    self._syn_weight[exc_scaled],
                    float(getattr(self.bio, "stdp_weight_min", 0.0)),
                    float(getattr(self.bio, "stdp_weight_max", 10.0)),
                )

        after_mean = float(np.mean(self._syn_weight[mask])) if scaled_count else 0.0
        return {
            "scaled": scaled_count,
            "factor": float(factor_f),
            "before_mean": before_mean,
            "after_mean": after_mean,
        }

    def step(self, dt: float) -> Dict[str, Any]:
        dt_ms = float(dt)
        if dt_ms <= 0:
            raise ValueError("dt must be positive (milliseconds)")

        self._ensure_synapses(dt_ms)
        if self._synapse_model == "exp":
            assert self._queue_exc is not None and self._queue_inh is not None
        else:
            assert (
                self._queue_ampa is not None
                and self._queue_nmda is not None
                and self._queue_gabaa is not None
                and self._queue_gabab is not None
            )

        current_time = self.global_step * dt_ms

        if bool(self.bio.stdp_enabled):
            pre_decay = float(np.exp(-dt_ms / max(float(self.bio.stdp_tau_plus_ms), 1e-6)))
            post_decay = float(np.exp(-dt_ms / max(float(self.bio.stdp_tau_minus_ms), 1e-6)))
            self._pre_trace *= pre_decay
            self._post_trace *= post_decay

        if self._synapse_model == "exp":
            # Deliver scheduled conductance increments
            self._g_exc += self._queue_exc[self._queue_ptr]
            self._g_inh += self._queue_inh[self._queue_ptr]
            self._queue_exc[self._queue_ptr].fill(0.0)
            self._queue_inh[self._queue_ptr].fill(0.0)

            # Decay conductances
            exc_decay = float(np.exp(-dt_ms / max(self.bio.tau_exc_ms, 1e-6)))
            inh_decay = float(np.exp(-dt_ms / max(self.bio.tau_inh_ms, 1e-6)))
            self._g_exc *= exc_decay
            self._g_inh *= inh_decay
        else:
            # Deliver scheduled receptor conductance increments
            self._g_ampa += self._queue_ampa[self._queue_ptr]
            self._g_nmda += self._queue_nmda[self._queue_ptr]
            self._g_gabaa += self._queue_gabaa[self._queue_ptr]
            self._g_gabab += self._queue_gabab[self._queue_ptr]
            self._queue_ampa[self._queue_ptr].fill(0.0)
            self._queue_nmda[self._queue_ptr].fill(0.0)
            self._queue_gabaa[self._queue_ptr].fill(0.0)
            self._queue_gabab[self._queue_ptr].fill(0.0)

            ampa_decay = float(np.exp(-dt_ms / max(float(self.bio.tau_ampa_ms), 1e-6)))
            nmda_decay = float(np.exp(-dt_ms / max(float(self.bio.tau_nmda_ms), 1e-6)))
            gabaa_decay = float(np.exp(-dt_ms / max(float(self.bio.tau_gabaa_ms), 1e-6)))
            gabab_decay = float(np.exp(-dt_ms / max(float(self.bio.tau_gabab_ms), 1e-6)))
            self._g_ampa *= ampa_decay
            self._g_nmda *= nmda_decay
            self._g_gabaa *= gabaa_decay
            self._g_gabab *= gabab_decay

        # External drive from the input buffer
        external = np.zeros(self.n_neurons, dtype=np.float32)
        if self._input_buffer:
            inp = np.asarray(self._input_buffer, dtype=np.float32).reshape(-1)
            if inp.size == self.region_count:
                for ridx, sl in enumerate(self._region_slices):
                    external[sl] = float(inp[ridx])
            elif inp.size == self.n_neurons:
                external[:] = inp
            else:
                external.fill(float(np.mean(inp)))
        external *= float(self.bio.input_scale)

        # Baseline + noise (applied current drive)
        I_drive = external + self._baseline_current
        if self.bio.noise_std > 0.0:
            I_drive += self._rng.normal(0.0, float(self.bio.noise_std), size=self.n_neurons).astype(np.float32)

        if bool(getattr(self.bio, "physiology_enabled", False)) and bool(
            getattr(self.bio, "physiology_feedback", False)
        ):
            atp_baseline = max(float(getattr(self.bio, "metabolism_atp_baseline", 2.5)), 1e-6)
            atp_scale = np.clip(self._region_atp / atp_baseline, 0.2, 1.0).astype(np.float32)
            I_drive *= atp_scale[self._neuron_region]

        e_exc = float(self.bio.e_exc_mV)
        e_inh = float(self.bio.e_inh_mV)

        spiked = np.zeros(self.n_neurons, dtype=bool)

        if self._cell_model == "mc":
            internal_dt = float(self.bio.hh_internal_dt_ms)
            if not np.isfinite(internal_dt) or internal_dt <= 0.0:
                internal_dt = dt_ms
            n_substeps = max(1, int(np.ceil(dt_ms / max(internal_dt, 1e-6))))
            dt_sub = float(dt_ms) / float(n_substeps)

            v_soma = self.v
            v_dend = self._mc_v_dend
            m = self._hh_m
            h = self._hh_h
            n = self._hh_n
            prev_v = np.empty_like(v_soma)
            prev_v_dend = np.empty_like(v_dend)

            C_soma = float(self.bio.mc_C_soma)
            C_dend = float(self.bio.mc_C_dend)
            g_L_soma = float(self.bio.hh_g_L)
            E_L_soma = float(self.bio.hh_E_L_mV)
            g_L_dend = float(self.bio.mc_g_L_dend)
            E_L_dend = float(self.bio.mc_E_L_dend_mV)
            g_c = float(self.bio.mc_g_couple)
            dend_frac = float(np.clip(float(self.bio.mc_dendrite_drive_fraction), 0.0, 1.0))

            g_Na = float(self.bio.hh_g_Na)
            g_K = float(self.bio.hh_g_K)
            E_Na = float(self.bio.hh_E_Na_mV)
            E_K = float(self.bio.hh_E_K_mV)
            spike_thresh = float(self.bio.hh_spike_threshold_mV)

            I_soma_drive = I_drive * (1.0 - dend_frac)
            I_dend_drive = I_drive * dend_frac

            dend_active = bool(getattr(self.bio, "mc_dendrite_active", False))
            plateau = self._mc_plateau
            dend_refr = self._mc_dend_refractory_remaining
            dend_thr = float(getattr(self.bio, "mc_dend_spike_threshold_mV", -20.0))
            dend_refr_ms = float(getattr(self.bio, "mc_dend_spike_refractory_ms", 5.0))
            plateau_tau_ms = max(float(getattr(self.bio, "mc_dend_plateau_tau_ms", 40.0)), 1e-6)
            plateau_current = float(getattr(self.bio, "mc_dend_plateau_current", 10.0))
            plateau_to_soma = float(
                np.clip(float(getattr(self.bio, "mc_dend_plateau_to_soma_fraction", 0.7)), 0.0, 1.0)
            )
            plateau_decay = float(np.exp(-dt_sub / plateau_tau_ms)) if dend_active else 0.0

            for _ in range(n_substeps):
                prev_v[:] = v_soma
                prev_v_dend[:] = v_dend

                if dend_active:
                    dend_refr -= dt_sub
                    np.clip(dend_refr, 0.0, np.inf, out=dend_refr)
                    plateau *= plateau_decay

                if self._synapse_model == "exp":
                    I_syn_soma = self._g_inh * (e_inh - v_soma)
                    I_syn_dend = self._g_exc * (e_exc - v_dend)
                else:
                    block_d = self._nmda_mg_block(v_dend)
                    I_syn_soma = (self._g_gabaa + self._g_gabab) * (e_inh - v_soma)
                    I_syn_dend = self._g_ampa * (e_exc - v_dend) + self._g_nmda * block_d * (e_exc - v_dend)

                I_c = g_c * (v_dend - v_soma)

                alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._hh_rates(v_soma)
                m += (alpha_m * (1.0 - m) - beta_m * m) * dt_sub
                h += (alpha_h * (1.0 - h) - beta_h * h) * dt_sub
                n += (alpha_n * (1.0 - n) - beta_n * n) * dt_sub
                np.clip(m, 0.0, 1.0, out=m)
                np.clip(h, 0.0, 1.0, out=h)
                np.clip(n, 0.0, 1.0, out=n)

                I_Na = g_Na * (m**3) * h * (v_soma - E_Na)
                I_K = g_K * (n**4) * (v_soma - E_K)
                I_L_s = g_L_soma * (v_soma - E_L_soma)
                I_L_d = g_L_dend * (v_dend - E_L_dend)

                I_plateau_soma = 0.0
                I_plateau_dend = 0.0
                if dend_active:
                    I_plateau_soma = plateau_current * plateau_to_soma * plateau
                    I_plateau_dend = plateau_current * (1.0 - plateau_to_soma) * plateau

                dv_s = (I_soma_drive + I_plateau_soma + I_syn_soma + I_c - I_Na - I_K - I_L_s) / max(C_soma, 1e-6)
                dv_d = (I_dend_drive + I_plateau_dend + I_syn_dend - I_L_d - I_c) / max(C_dend, 1e-6)

                v_soma += dv_s * dt_sub
                v_dend += dv_d * dt_sub

                spiked |= (prev_v < spike_thresh) & (v_soma >= spike_thresh)

                if dend_active:
                    crossed = (prev_v_dend < dend_thr) & (v_dend >= dend_thr) & (dend_refr <= 0.0)
                    if np.any(crossed):
                        plateau[crossed] = 1.0
                        dend_refr[crossed] = dend_refr_ms

            self.v = v_soma
            self._mc_v_dend = v_dend
            self._hh_m = m
            self._hh_h = h
            self._hh_n = n
        elif self._cell_model == "hh":
            internal_dt = float(self.bio.hh_internal_dt_ms)
            if not np.isfinite(internal_dt) or internal_dt <= 0.0:
                internal_dt = dt_ms
            n_substeps = max(1, int(np.ceil(dt_ms / max(internal_dt, 1e-6))))
            dt_sub = float(dt_ms) / float(n_substeps)

            v = self.v
            m = self._hh_m
            h = self._hh_h
            n = self._hh_n
            prev_v = np.empty_like(v)

            C_m = float(self.bio.hh_C_m)
            g_Na = float(self.bio.hh_g_Na)
            g_K = float(self.bio.hh_g_K)
            g_L = float(self.bio.hh_g_L)
            E_Na = float(self.bio.hh_E_Na_mV)
            E_K = float(self.bio.hh_E_K_mV)
            E_L = float(self.bio.hh_E_L_mV)
            spike_thresh = float(self.bio.hh_spike_threshold_mV)

            for _ in range(n_substeps):
                prev_v[:] = v
                if self._synapse_model == "exp":
                    I_syn = self._g_exc * (e_exc - v) + self._g_inh * (e_inh - v)
                else:
                    block = self._nmda_mg_block(v)
                    I_syn = (
                        self._g_ampa * (e_exc - v)
                        + self._g_nmda * block * (e_exc - v)
                        + (self._g_gabaa + self._g_gabab) * (e_inh - v)
                    )
                I_app = I_drive + I_syn

                alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._hh_rates(v)
                m += (alpha_m * (1.0 - m) - beta_m * m) * dt_sub
                h += (alpha_h * (1.0 - h) - beta_h * h) * dt_sub
                n += (alpha_n * (1.0 - n) - beta_n * n) * dt_sub
                np.clip(m, 0.0, 1.0, out=m)
                np.clip(h, 0.0, 1.0, out=h)
                np.clip(n, 0.0, 1.0, out=n)

                I_Na = g_Na * (m**3) * h * (v - E_Na)
                I_K = g_K * (n**4) * (v - E_K)
                I_L = g_L * (v - E_L)
                dv = (I_app - I_Na - I_K - I_L) / max(C_m, 1e-6)
                v += dv * dt_sub

                spiked |= (prev_v < spike_thresh) & (v >= spike_thresh)

            self.v = v
            self._hh_m = m
            self._hh_h = h
            self._hh_n = n
        elif self._cell_model == "lif":
            v = self.v
            if self._synapse_model == "exp":
                I_syn = self._g_exc * (e_exc - v) + self._g_inh * (e_inh - v)
            else:
                block = self._nmda_mg_block(v)
                I_syn = (
                    self._g_ampa * (e_exc - v)
                    + self._g_nmda * block * (e_exc - v)
                    + (self._g_gabaa + self._g_gabab) * (e_inh - v)
                )
            I_total = I_drive + I_syn

            v_rest = float(self.bio.lif_v_rest_mV)
            v_reset = float(self.bio.lif_v_reset_mV)
            v_thresh = float(self.bio.lif_v_thresh_mV)
            tau_m = max(float(self.bio.lif_tau_m_ms), 1e-6)
            r_m = float(self.bio.lif_R_m)
            refractory_ms = max(float(self.bio.lif_refractory_ms), 0.0)

            refr = self._lif_refractory_remaining
            refr -= dt_ms
            np.clip(refr, 0.0, np.inf, out=refr)

            active = refr <= 0.0
            if np.any(~active):
                v[~active] = v_reset

            dv = (v_rest - v + r_m * I_total) / tau_m
            v[active] = v[active] + dv[active] * dt_ms

            spiked = active & (v >= v_thresh)
            if np.any(spiked):
                v[spiked] = v_reset
                refr[spiked] = refractory_ms

            self.v = v
            self._lif_refractory_remaining = refr
        elif self._cell_model == "adex":
            v = self.v
            w = self._adex_w
            if self._synapse_model == "exp":
                I = I_drive + self._g_exc * (e_exc - v) + self._g_inh * (e_inh - v)
            else:
                block = self._nmda_mg_block(v)
                I = I_drive + (
                    self._g_ampa * (e_exc - v)
                    + self._g_nmda * block * (e_exc - v)
                    + (self._g_gabaa + self._g_gabab) * (e_inh - v)
                )

            C = self._adex_C
            g_L = self._adex_g_L
            E_L = self._adex_E_L
            V_T = self._adex_V_T
            Delta_T = np.maximum(self._adex_Delta_T, 1e-6).astype(np.float32, copy=False)
            a = self._adex_a
            tau_w = np.maximum(self._adex_tau_w, 1e-6).astype(np.float32, copy=False)
            b = self._adex_b
            V_thresh = self._adex_V_thresh
            V_reset = self._adex_V_reset

            exp_term = Delta_T * self._exp_clip((v - V_T) / Delta_T)
            dv = (-g_L * (v - E_L) + g_L * exp_term - w + I) / np.maximum(C, 1e-6)
            v = v + dv * dt_ms

            dw = (a * (v - E_L) - w) / tau_w
            w = w + dw * dt_ms

            spiked = v >= V_thresh
            if np.any(spiked):
                v[spiked] = V_reset[spiked]
                w[spiked] = w[spiked] + b[spiked]

            self.v = v
            self._adex_w = w
        elif self._cell_model == "hybrid":
            v = self.v
            u = self.u
            w = self._adex_w

            if self._synapse_model == "exp":
                I = I_drive + self._g_exc * (e_exc - v) + self._g_inh * (e_inh - v)
            else:
                block = self._nmda_mg_block(v)
                I = I_drive + (
                    self._g_ampa * (e_exc - v)
                    + self._g_nmda * block * (e_exc - v)
                    + (self._g_gabaa + self._g_gabab) * (e_inh - v)
                )

            exc_mask = ~self._is_inhibitory
            inh_mask = self._is_inhibitory

            spiked = np.zeros(self.n_neurons, dtype=bool)

            if np.any(exc_mask):
                v_exc = v[exc_mask]
                w_exc = w[exc_mask]
                I_exc = I[exc_mask]

                C = self._adex_C[exc_mask]
                g_L = self._adex_g_L[exc_mask]
                E_L = self._adex_E_L[exc_mask]
                V_T = self._adex_V_T[exc_mask]
                Delta_T = np.maximum(self._adex_Delta_T[exc_mask], 1e-6).astype(np.float32, copy=False)
                a = self._adex_a[exc_mask]
                tau_w = np.maximum(self._adex_tau_w[exc_mask], 1e-6).astype(np.float32, copy=False)
                b = self._adex_b[exc_mask]
                V_thresh = self._adex_V_thresh[exc_mask]
                V_reset = self._adex_V_reset[exc_mask]

                exp_term = Delta_T * self._exp_clip((v_exc - V_T) / Delta_T)
                dv = (-g_L * (v_exc - E_L) + g_L * exp_term - w_exc + I_exc) / np.maximum(C, 1e-6)
                v_exc = v_exc + dv * dt_ms

                dw = (a * (v_exc - E_L) - w_exc) / tau_w
                w_exc = w_exc + dw * dt_ms

                sp_exc = v_exc >= V_thresh
                if np.any(sp_exc):
                    v_exc[sp_exc] = V_reset[sp_exc]
                    w_exc[sp_exc] = w_exc[sp_exc] + b[sp_exc]

                v[exc_mask] = v_exc
                w[exc_mask] = w_exc
                spiked[exc_mask] = sp_exc

            if np.any(inh_mask):
                v_inh = v[inh_mask]
                u_inh = u[inh_mask]
                I_inh = I[inh_mask]

                a_inh = self._a[inh_mask]
                b_inh = self._b[inh_mask]
                c_inh = self._c[inh_mask]
                d_inh = self._d[inh_mask]
                v_peak = self._v_peak[inh_mask]

                dv = 0.04 * v_inh * v_inh + 5.0 * v_inh + 140.0 - u_inh + I_inh
                v_inh = v_inh + dv * dt_ms
                u_inh = u_inh + a_inh * (b_inh * v_inh - u_inh) * dt_ms

                sp_inh = v_inh >= v_peak
                if np.any(sp_inh):
                    v_inh[sp_inh] = c_inh[sp_inh]
                    u_inh[sp_inh] = u_inh[sp_inh] + d_inh[sp_inh]

                v[inh_mask] = v_inh
                u[inh_mask] = u_inh
                spiked[inh_mask] = sp_inh

            self.v = v
            self.u = u
            self._adex_w = w
        else:
            # Izhikevich integration (Euler)
            v = self.v
            u = self.u
            if self._synapse_model == "exp":
                I = I_drive + self._g_exc * (e_exc - v) + self._g_inh * (e_inh - v)
            else:
                block = self._nmda_mg_block(v)
                I = I_drive + (
                    self._g_ampa * (e_exc - v)
                    + self._g_nmda * block * (e_exc - v)
                    + (self._g_gabaa + self._g_gabab) * (e_inh - v)
                )

            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            v = v + dv * dt_ms
            u = u + self._a * (self._b * v - u) * dt_ms

            spiked = v >= self._v_peak
            spike_ids = np.nonzero(spiked)[0].astype(np.int32)
            if spike_ids.size:
                v[spiked] = self._c[spiked]
                u[spiked] = u[spiked] + self._d[spiked]

            self.v = v
            self.u = u

        spike_ids = np.nonzero(spiked)[0].astype(np.int32)

        stdp_modulation = 1.0
        if bool(self.bio.stdp_enabled) and self._neuromodulators:
            try:
                dopamine = float(self._neuromodulators.get("dopamine", 1.0))
            except Exception:
                dopamine = 1.0
            stdp_modulation = float(
                np.clip(
                    1.0 + float(self.bio.dopamine_stdp_gain) * (dopamine - 1.0),
                    0.0,
                    5.0,
                )
            )

        # Schedule outgoing synapses for presynaptic spikes
        if spike_ids.size and self._syn_pre.size:
            active_mask = spiked[self._syn_pre]
            active_idx = np.nonzero(active_mask)[0]
            if active_idx.size:
                if bool(self.bio.stdp_enabled):
                    exc_idx = active_idx[~self._syn_is_inh[active_idx]]
                    if exc_idx.size:
                        dw = -float(self.bio.stdp_A_minus) * self._post_trace[self._syn_post[exc_idx]]
                        if abs(stdp_modulation - 1.0) > 1e-9:
                            dw = dw * stdp_modulation
                        self._syn_weight[exc_idx] = np.clip(
                            self._syn_weight[exc_idx] + dw,
                            float(self.bio.stdp_weight_min),
                            float(self.bio.stdp_weight_max),
                        )

                effective_weights = self._syn_weight[active_idx].astype(np.float32, copy=False)
                if bool(getattr(self.bio, "stp_enabled", False)):
                    effective_weights = effective_weights * self._stp_release_scale(active_idx, time_ms=float(current_time))

                slots = (self._queue_ptr + self._syn_delay_steps[active_idx]) % int(self._queue_len)

                ampa_frac = float(np.clip(float(getattr(self.bio, "ampa_fraction", 0.7)), 0.0, 1.0))
                nmda_frac = float(np.clip(float(getattr(self.bio, "nmda_fraction", 0.3)), 0.0, 1.0))
                total_exc = ampa_frac + nmda_frac
                if total_exc <= 1e-6:
                    ampa_frac, nmda_frac = 1.0, 0.0
                else:
                    ampa_frac /= total_exc
                    nmda_frac /= total_exc

                gabab_frac = float(np.clip(float(getattr(self.bio, "gabab_fraction", 0.0)), 0.0, 1.0))
                gabaa_frac = 1.0 - gabab_frac

                for slot in np.unique(slots):
                    idxs = active_idx[slots == slot]
                    posts = self._syn_post[idxs]
                    weights = effective_weights[slots == slot]
                    inh = self._syn_is_inh[idxs]
                    if self._synapse_model == "exp":
                        if np.any(~inh):
                            np.add.at(self._queue_exc[slot], posts[~inh], weights[~inh])
                        if np.any(inh):
                            np.add.at(self._queue_inh[slot], posts[inh], weights[inh])
                    else:
                        if np.any(~inh):
                            post_exc = posts[~inh]
                            w_exc = weights[~inh]
                            if ampa_frac > 0.0:
                                np.add.at(self._queue_ampa[slot], post_exc, w_exc * ampa_frac)
                            if nmda_frac > 0.0:
                                np.add.at(self._queue_nmda[slot], post_exc, w_exc * nmda_frac)
                        if np.any(inh):
                            post_inh = posts[inh]
                            w_inh = weights[inh]
                            if gabaa_frac > 0.0:
                                np.add.at(self._queue_gabaa[slot], post_inh, w_inh * gabaa_frac)
                            if gabab_frac > 0.0:
                                np.add.at(self._queue_gabab[slot], post_inh, w_inh * gabab_frac)

        if bool(self.bio.stdp_enabled) and spike_ids.size:
            self._pre_trace[spike_ids] += 1.0

        if bool(self.bio.stdp_enabled) and spike_ids.size and self._syn_pre.size:
            active_post_mask = spiked[self._syn_post]
            active_post_idx = np.nonzero(active_post_mask)[0]
            if active_post_idx.size:
                exc_idx = active_post_idx[~self._syn_is_inh[active_post_idx]]
                if exc_idx.size:
                    dw = float(self.bio.stdp_A_plus) * self._pre_trace[self._syn_pre[exc_idx]]
                    if abs(stdp_modulation - 1.0) > 1e-9:
                        dw = dw * stdp_modulation
                    self._syn_weight[exc_idx] = np.clip(
                        self._syn_weight[exc_idx] + dw,
                        float(self.bio.stdp_weight_min),
                        float(self.bio.stdp_weight_max),
                    )

        if bool(self.bio.stdp_enabled) and spike_ids.size:
            self._post_trace[spike_ids] += 1.0

        # Advance time pointer
        self.global_step += 1
        self._queue_ptr = (self._queue_ptr + 1) % int(self._queue_len)

        # Telemetry (sample to keep payload small)
        sample_n = min(int(self.bio.max_sample_neurons), self.n_neurons)
        sample_ids = np.arange(sample_n, dtype=np.int32)
        voltages = {int(nid): float(self.v[nid]) for nid in sample_ids}
        voltages_dend = None
        dendrite_plateau = None
        if self._cell_model == "mc":
            voltages_dend = {int(nid): float(self._mc_v_dend[nid]) for nid in sample_ids}
            if bool(getattr(self.bio, "mc_dendrite_active", False)):
                dendrite_plateau = {int(nid): float(self._mc_plateau[nid]) for nid in sample_ids}

        spikes = [{"neuron": int(nid), "time_ms": float(current_time + dt_ms)} for nid in spike_ids]

        if spike_ids.size:
            region_counts = np.bincount(
                self._neuron_region[spike_ids].astype(np.int64, copy=False),
                minlength=self.region_count,
            ).astype(np.int32, copy=False)
        else:
            region_counts = np.zeros(self.region_count, dtype=np.int32)

        region_spike_counts = {self.regions[ridx]: int(region_counts[ridx]) for ridx in range(self.region_count)}

        physiology: Dict[str, Any] = {}
        if bool(getattr(self.bio, "physiology_enabled", False)):
            dt_s = float(dt_ms) / 1000.0
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = 1e-3
            denom = np.maximum(self._region_sizes.astype(np.float32), 1.0) * float(dt_s)
            rates_hz = region_counts.astype(np.float32) / denom

            target_rate = float(getattr(self.bio, "physiology_target_rate_hz", 5.0))
            if not np.isfinite(target_rate) or target_rate <= 0.0:
                target_rate = 5.0
            activity = np.clip(rates_hz / target_rate, 0.0, 5.0).astype(np.float32)

            glia_tau = max(float(getattr(self.bio, "glia_tau_ms", 5000.0)), 1e-6)
            alpha_glia = float(np.clip(dt_ms / glia_tau, 0.0, 1.0))
            self._region_glia += (activity - self._region_glia) * alpha_glia

            baseline_flow = float(getattr(self.bio, "bloodflow_baseline", 1.0))
            if not np.isfinite(baseline_flow) or baseline_flow <= 0.0:
                baseline_flow = 1.0
            flow_tau = max(float(getattr(self.bio, "bloodflow_tau_ms", 2000.0)), 1e-6)
            alpha_flow = float(np.clip(dt_ms / flow_tau, 0.0, 1.0))
            flow_gain = float(getattr(self.bio, "bloodflow_gain", 0.5))
            if not np.isfinite(flow_gain):
                flow_gain = 0.5
            target_flow = baseline_flow * (1.0 + flow_gain * self._region_glia)
            self._region_flow += (target_flow - self._region_flow) * alpha_flow
            np.clip(self._region_flow, 0.0, np.inf, out=self._region_flow)

            flow_ratio = self._region_flow / baseline_flow
            bold_gain = float(getattr(self.bio, "bold_gain", 0.3))
            if not np.isfinite(bold_gain):
                bold_gain = 0.3
            self._region_bold = (bold_gain * (flow_ratio - 1.0)).astype(np.float32)

            atp_baseline = float(getattr(self.bio, "metabolism_atp_baseline", 2.5))
            if not np.isfinite(atp_baseline) or atp_baseline <= 0.0:
                atp_baseline = 2.5
            atp_min = float(getattr(self.bio, "metabolism_atp_min", 0.1))
            if not np.isfinite(atp_min) or atp_min < 0.0:
                atp_min = 0.1

            cons_base = float(getattr(self.bio, "metabolism_atp_consumption_base", 0.01))
            cons_activity = float(getattr(self.bio, "metabolism_atp_consumption_activity", 0.05))
            recovery_gain = float(getattr(self.bio, "metabolism_atp_recovery_gain", 0.03))
            if not np.isfinite(cons_base):
                cons_base = 0.01
            if not np.isfinite(cons_activity):
                cons_activity = 0.05
            if not np.isfinite(recovery_gain):
                recovery_gain = 0.03

            consumption = (cons_base + cons_activity * activity) * float(dt_s)
            recovery = (recovery_gain * (flow_ratio - 1.0)) * float(dt_s)
            self._region_atp += (recovery - consumption).astype(np.float32)
            np.clip(self._region_atp, atp_min, atp_baseline, out=self._region_atp)

            for ridx in range(self.region_count):
                physiology[self.regions[ridx]] = {
                    "rate_hz": float(rates_hz[ridx]),
                    "activity": float(activity[ridx]),
                    "glia": float(self._region_glia[ridx]),
                    "blood_flow": float(self._region_flow[ridx]),
                    "bold": float(self._region_bold[ridx]),
                    "atp": float(self._region_atp[ridx]),
                }

        weights_summary: Dict[str, Any] = {"synapse_count": int(self._syn_weight.size)}
        if self._syn_weight.size:
            exc_mask = ~self._syn_is_inh
            inh_mask = self._syn_is_inh
            if np.any(exc_mask):
                weights_summary["exc_mean"] = float(np.mean(self._syn_weight[exc_mask]))
            if np.any(inh_mask):
                weights_summary["inh_mean"] = float(np.mean(self._syn_weight[inh_mask]))
            if bool(self.bio.stdp_enabled):
                weights_summary["stdp_modulation"] = float(stdp_modulation)

        connectome_info: Dict[str, Any] = {}
        if self._estimated_axonal_velocity_m_s is not None:
            connectome_info["estimated_axonal_velocity_m_s"] = float(self._estimated_axonal_velocity_m_s)
        if self._region_coords_mm is not None:
            connectome_info["has_coords_mm"] = True
        if connectome_info:
            connectome_info["synaptic_delay_ms"] = float(self.bio.connectome_synaptic_delay_ms)

        return {
            "time_ms": float(current_time + dt_ms),
            "spikes": spikes,
            "spike_count": int(spike_ids.size),
            "region_spike_counts": region_spike_counts,
            "voltages": voltages,
            "voltages_dend": voltages_dend,
            "dendrite_plateau": dendrite_plateau,
            "physiology": physiology,
            "weights": weights_summary,
            "neuromodulators": dict(self._neuromodulators) if self._neuromodulators else {},
            "connectome": connectome_info,
        }


def create_biophysical_network(config: Dict[str, Any]) -> BiophysicalSpikingNetwork:
    """Factory for the biophysical backend.

    Configuration sources (merged, later wins):
    - ``config['simulation']['biophysical']``
    - ``config['biophysical']`` (convenience)
    """

    params: Dict[str, Any] = {}
    simulation = config.get("simulation")
    if isinstance(simulation, dict) and isinstance(simulation.get("biophysical"), dict):
        params.update(simulation["biophysical"])
    if isinstance(config.get("biophysical"), dict):
        params.update(config["biophysical"])
    return BiophysicalSpikingNetwork(config, params=params)


__all__ = ["BiophysicalSpikingNetwork", "BiophysicalSpikingConfig", "create_biophysical_network"]
