"""Production-grade neuromorphic computing framework.

This module provides a comprehensive neuromorphic computing platform with
hardware drivers, event-driven processing, power optimization, advanced
learning algorithms, and production deployment capabilities.
"""

from .spiking_network import (
    SpikingNetworkConfig,
    SpikingNeuralNetwork,
    AdExNeuronModel,
    LIFNeuronModel,
    DenseSynapseModel,
    NeuromorphicBackend,
    NeuromorphicRunResult,
)

from .tuning import random_search, TuningResult
from .evaluate import evaluate, EvaluationMetrics
from .data import DatasetLoader

from .temporal_encoding import (
    latency_encode, 
    rate_encode, 
    decode_spike_counts, 
    decode_average_rate
)

from .advanced_core import AdvancedNeuromorphicCore

# Production-grade components
from .hardware_drivers import (
    HardwareStatus,
    HardwareCapabilities,
    PowerMetrics as HardwarePowerMetrics,
    AEREvent,
    HardwareDriver,
    LoihiDriver,
    BrainScaleSDriver,
    SpiNNakerDriver,
    PowerMonitor,
    HardwareDriverFactory,
    NeuromorphicHAL
)

from .event_driven_core import (
    EventType,
    NeuromorphicEvent,
    SpikeEvent,
    PlasticityEvent,
    EventProcessor,
    SpikeProcessor,
    EventDrivenCore,
    NetworkTopology,
    EventDrivenNetwork
)

from .power_optimization import (
    PowerState,
    PowerProfile,
    EnergyBudget,
    PowerMetrics,
    PowerModel,
    DetailedPowerModel,
    PowerOptimizer,
    AdaptivePowerManager
)

from .advanced_learning import (
    LearningMode,
    LearningParameters,
    SynapticTrace,
    PlasticityRule,
    STDPRule,
    OnlineLearningEngine
)

from .deployment_system import (
    DeploymentStatus,
    DeploymentConfig,
    PerformanceMetrics,
    SystemHealth,
    NeuromorphicDeployment
)

__all__ = [
    # Legacy components
    "DatasetLoader",
    "evaluate",
    "EvaluationMetrics",
    "SpikingNetworkConfig",
    "random_search",
    "TuningResult",
    "AdExNeuronModel",
    "LIFNeuronModel", 
    "DenseSynapseModel",
    "SpikingNeuralNetwork",
    "NeuromorphicBackend",
    "NeuromorphicRunResult",
    "latency_encode",
    "rate_encode",
    "decode_spike_counts",
    "decode_average_rate",
    "AdvancedNeuromorphicCore",
    
    # Production-grade hardware drivers
    "HardwareStatus",
    "HardwareCapabilities",
    "HardwarePowerMetrics",
    "AEREvent",
    "HardwareDriver",
    "LoihiDriver",
    "BrainScaleSDriver",
    "SpiNNakerDriver",
    "PowerMonitor",
    "HardwareDriverFactory",
    "NeuromorphicHAL",
    
    # Event-driven processing
    "EventType",
    "NeuromorphicEvent",
    "SpikeEvent",
    "PlasticityEvent",
    "EventProcessor",
    "SpikeProcessor",
    "EventDrivenCore",
    "NetworkTopology",
    "EventDrivenNetwork",
    
    # Power optimization
    "PowerState",
    "PowerProfile",
    "EnergyBudget",
    "PowerMetrics",
    "PowerModel",
    "DetailedPowerModel",
    "PowerOptimizer",
    "AdaptivePowerManager",
    
    # Advanced learning
    "LearningMode",
    "LearningParameters",
    "SynapticTrace",
    "PlasticityRule",
    "STDPRule",
    "OnlineLearningEngine",
    
    # Deployment system
    "DeploymentStatus",
    "DeploymentConfig",
    "PerformanceMetrics",
    "SystemHealth",
    "NeuromorphicDeployment"
]

# Version information
__version__ = "2.0.0"
__author__ = "Neuromorphic Computing Team"
__description__ = "Production-grade neuromorphic computing framework"