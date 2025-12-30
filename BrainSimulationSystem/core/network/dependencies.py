"""Shared dependency imports for brain network modules."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import pickle
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, DefaultDict, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional numba support (legacy compatibility)
try:  # pragma: no cover - optional dependency
    from numba import cuda, jit  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    def jit(*args: Any, **kwargs: Any):
        def decorator(func):
            return func
        return decorator

    class cuda:  # type: ignore
        """Placeholder when numba.cuda is not available."""

        @staticmethod
        def is_available() -> bool:
            return False

# Optional h5py persistence support
try:  # pragma: no cover - optional dependency
    import h5py  # type: ignore
    H5PY_AVAILABLE = True
except Exception:  # pragma: no cover - optional fallback
    h5py = None  # type: ignore
    H5PY_AVAILABLE = False

from ..enums import BrainRegion, CellType
from ..synapses import (
    SynapseManager,
    create_gaba_synapse_config,
    create_glutamate_synapse_config,
    create_synapse_manager,
)

try:  # pragma: no cover - optional dependency
    from ..integration.neuromorphic_bridge import (
        NeuromorphicBridge,
        NeuromorphicIntegrationConfig,
        get_default_integration_config,
    )
except Exception:  # pragma: no cover - optional fallback
    NeuromorphicBridge = None  # type: ignore
    NeuromorphicIntegrationConfig = None  # type: ignore

    def get_default_integration_config() -> Optional[Any]:
        return None

try:  # pragma: no cover - optional dependency
    from ..backends import (
        NeuromorphicBackendManager,
        create_neuromorphic_backend_manager,
    )
except Exception:  # pragma: no cover - optional fallback
    NeuromorphicBackendManager = None  # type: ignore
    create_neuromorphic_backend_manager = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..partition import PartitionManager
except Exception:  # pragma: no cover - optional fallback
    PartitionManager = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..thalamocortical import ThalamicRelay, connect_thalamus_to_cortex
except Exception:  # pragma: no cover - optional fallback
    ThalamicRelay = None  # type: ignore

    def connect_thalamus_to_cortex(*_args: Any, **_kwargs: Any) -> List[Any]:
        return []

try:  # pragma: no cover - optional dependency
    from ..hippocampus_pfc_loop import (
        connect_hippocampus_to_pfc,
        initialize_hippocampus_pfc,
    )
except Exception:  # pragma: no cover - optional fallback
    initialize_hippocampus_pfc = None  # type: ignore

    def connect_hippocampus_to_pfc(*_args: Any, **_kwargs: Any) -> List[Any]:
        return []

try:  # pragma: no cover - optional dependency
    from ..anatomy import anatomy_metadata
except Exception:  # pragma: no cover - optional fallback
    def anatomy_metadata() -> Dict[str, Any]:
        return {}

try:  # pragma: no cover - optional dependency
    from ..cognition_binding import binding_metadata, suggest_bindings_for_config
except Exception:  # pragma: no cover - optional fallback
    def binding_metadata() -> Dict[str, Any]:
        return {}

    def suggest_bindings_for_config(_scope_regions: Any) -> Dict[str, Any]:
        return {}

from ..parameters import NeuronParameters, get_cell_parameters
from ..synapse_parameters import SynapseParameters
from ..detailed_neuron import DetailedNeuron
from ..detailed_synapse import DetailedSynapse

try:  # pragma: no cover - optional dependency
    import nengo  # type: ignore
    import nengo_loihi  # type: ignore
    LOIHI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    nengo = None  # type: ignore
    nengo_loihi = None  # type: ignore
    LOIHI_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import spynnaker8 as sim  # type: ignore
    SPINNAKER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    sim = None  # type: ignore
    SPINNAKER_AVAILABLE = False

__all__ = [
    'Any',
    'Optional',
    'Tuple',
    'Union',
    'Dict',
    'List',
    'DefaultDict',
    'Deque',
    'BrainRegion',
    'CellType',
    'SynapseManager',
    'create_synapse_manager',
    'create_glutamate_synapse_config',
    'create_gaba_synapse_config',
    'NeuromorphicBridge',
    'NeuromorphicIntegrationConfig',
    'get_default_integration_config',
    'NeuromorphicBackendManager',
    'create_neuromorphic_backend_manager',
    'PartitionManager',
    'ThalamicRelay',
    'connect_thalamus_to_cortex',
    'initialize_hippocampus_pfc',
    'connect_hippocampus_to_pfc',
    'anatomy_metadata',
    'binding_metadata',
    'suggest_bindings_for_config',
    'NeuronParameters',
    'get_cell_parameters',
    'SynapseParameters',
    'DetailedNeuron',
    'DetailedSynapse',
    'LOIHI_AVAILABLE',
    'SPINNAKER_AVAILABLE',
    'nengo',
    'nengo_loihi',
    'sim',
    'np',
    'asyncio',
    'logging',
    'mp',
    'pickle',
    'defaultdict',
    'deque',
    'ThreadPoolExecutor',
    'ProcessPoolExecutor',
    'H5PY_AVAILABLE',
    'h5py',
]
