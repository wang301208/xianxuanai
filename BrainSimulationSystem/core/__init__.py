"""Core module exports for the BrainSimulationSystem package."""

__all__ = []

try:  # pragma: no cover - import-time wiring only
    from .neuron_base import (
        NeuronBase,
        SynapseBase,
        NeuronType,
        SynapseType,
    )
except Exception:  # pragma: no cover - gracefully degrade when optional deps fail
    NeuronBase = None  # type: ignore[assignment]
    SynapseBase = None  # type: ignore[assignment]
    NeuronType = None  # type: ignore[assignment]
    SynapseType = None  # type: ignore[assignment]
else:  # pragma: no cover - executed during import wiring
    __all__.extend([
        "NeuronBase",
        "SynapseBase",
        "NeuronType",
        "SynapseType",
    ])

try:  # pragma: no cover - import-time wiring only
    from .multi_neuron_models import (
        LIFNeuron,
        IzhikevichNeuron,
        AdExNeuron,
        HodgkinHuxleyNeuron,
        MultiCompartmentNeuron,
        create_neuron,
        get_default_parameters,
    )
except Exception:  # pragma: no cover - gracefully degrade
    LIFNeuron = None  # type: ignore[assignment]
    IzhikevichNeuron = None  # type: ignore[assignment]
    AdExNeuron = None  # type: ignore[assignment]
    HodgkinHuxleyNeuron = None  # type: ignore[assignment]
    MultiCompartmentNeuron = None  # type: ignore[assignment]
    create_neuron = None  # type: ignore[assignment]
    get_default_parameters = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.extend([
        "LIFNeuron",
        "IzhikevichNeuron",
        "AdExNeuron",
        "HodgkinHuxleyNeuron",
        "MultiCompartmentNeuron",
        "create_neuron",
        "get_default_parameters",
    ])

    if 'NeuronBase' in globals():  # pragma: no cover
        Neuron = NeuronBase
        __all__.append("Neuron")
    if 'MultiCompartmentNeuron' in globals():  # pragma: no cover
        PyramidalNeuron = MultiCompartmentNeuron
        __all__.append("PyramidalNeuron")
    if 'LIFNeuron' in globals():  # pragma: no cover
        Interneuron = LIFNeuron
        __all__.append("Interneuron")

try:  # pragma: no cover
    from .synapse_models import (
        StaticSynapse,
        DynamicSynapse,
    )
except Exception:
    StaticSynapse = None  # type: ignore[assignment]
    DynamicSynapse = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.extend([
        "StaticSynapse",
        "DynamicSynapse",
    ])

try:  # pragma: no cover
    from .stp_synapse import STPSynapse
except Exception:
    STPSynapse = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("STPSynapse")

try:  # pragma: no cover
    from .network import NeuralNetwork
except Exception:
    NeuralNetwork = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("NeuralNetwork")

try:  # pragma: no cover
    from .network_models import (
        StructuredNetwork,
        FeedForwardNetwork,
        RecurrentNetwork,
        ReservoirNetwork,
        ModularNetwork,
        BrainInspiredNetwork,
    )
except Exception:
    StructuredNetwork = None  # type: ignore[assignment]
    FeedForwardNetwork = None  # type: ignore[assignment]
    RecurrentNetwork = None  # type: ignore[assignment]
    ReservoirNetwork = None  # type: ignore[assignment]
    ModularNetwork = None  # type: ignore[assignment]
    BrainInspiredNetwork = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.extend([
        "StructuredNetwork",
        "FeedForwardNetwork",
        "RecurrentNetwork",
        "ReservoirNetwork",
        "ModularNetwork",
        "BrainInspiredNetwork",
    ])

try:  # pragma: no cover
    from .enhanced_configs import (
        EnhancedSynapseConfig,
        GlialConfig,
        VolumeTransmissionConfig,
        SynapseState,
        PlasticityType,
        NeuromodulatorType,
    )
except Exception:
    EnhancedSynapseConfig = None  # type: ignore[assignment]
    GlialConfig = None  # type: ignore[assignment]
    VolumeTransmissionConfig = None  # type: ignore[assignment]
    SynapseState = None  # type: ignore[assignment]
    PlasticityType = None  # type: ignore[assignment]
    NeuromodulatorType = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.extend([
        "EnhancedSynapseConfig",
        "GlialConfig",
        "VolumeTransmissionConfig",
        "SynapseState",
        "PlasticityType",
        "NeuromodulatorType",
    ])

try:  # pragma: no cover
    from .glia_system import GlialSystem
except Exception:
    GlialSystem = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("GlialSystem")

try:  # pragma: no cover
    from .volume_transmission import VolumeTransmissionSystem
except Exception:
    VolumeTransmissionSystem = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("VolumeTransmissionSystem")

try:  # pragma: no cover
    from .enhanced_synapse import EnhancedSynapse
except Exception:
    EnhancedSynapse = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("EnhancedSynapse")

try:  # pragma: no cover
    from .enhanced_synapse_manager import EnhancedSynapseManager
except Exception:
    EnhancedSynapseManager = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("EnhancedSynapseManager")

from .automation_check import CodeAnalyzer, run_automated_checks

__all__.extend([
    "CodeAnalyzer",
    "run_automated_checks",
])
