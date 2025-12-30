"""Synapse parameter data structure used by detailed synapse models."""
from dataclasses import dataclass
from typing import Dict, Optional

from .enums import CellType


@dataclass
class SynapseParameters:
    """Container describing the configurable properties of a synapse."""

    pre_cell_type: CellType
    post_cell_type: CellType

    # Basic parameters
    weight: float  # nS
    delay: float  # ms

    # Neurotransmitter type
    neurotransmitter: str  # "glutamate", "gaba", "dopamine", etc.

    # Receptor configuration (e.g. {"ampa": 100.0, "nmda": 20.0})
    receptor_types: Dict[str, float]

    # Plasticity toggles
    stp_enabled: bool
    ltp_enabled: bool
    ltd_enabled: bool

    # Short-term plasticity parameters
    tau_rec: float  # ms
    tau_fac: float  # ms
    U: float  # utilization

    # Long-term plasticity baseline parameters
    ltp_threshold: float  # mV
    ltd_threshold: float  # mV
    learning_rate: float

    # Stability & metaplasticity (optional, defaults keep legacy behaviour)
    weight_min: float = 0.0
    weight_max: float = 10.0
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_A_plus: float = 0.01
    stdp_A_minus: float = 0.012
    weight_normalization_target: Optional[float] = None
    weight_normalization_interval: float = 50.0  # ms
    weight_normalization_rate: float = 0.1
    inactivity_threshold: float = 2000.0  # ms
    pruning_rate: float = 0.0
    metaplasticity_tau: float = 500.0  # ms
    metaplasticity_target: float = 0.2
    metaplasticity_beta: float = 0.05
