"""
Enhanced Cortical Column Stack with 6-layer architecture, thalamic input, 
synaptic delays, short-term plasticity, and Brian2 backend validation.

This module implements a biologically realistic cortical column with:
- 6 cortical layers (L1, L2/3, L4, L5, L6) with layer-specific connectivity
- Thalamic input integration with realistic delay distributions
- Short-term synaptic plasticity (STP) mechanisms
- Brian2 backend for validation and detailed biophysical modeling
- Real sensory data processing pipeline
"""

from __future__ import annotations

import numpy as np
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .network import NeuralNetwork
from .neurons import Neuron, create_neuron
from .synapses import Synapse, create_synapse, STPSynapse

try:
    import brian2 as b2
    BRIAN2_AVAILABLE = True
except ImportError:
    BRIAN2_AVAILABLE = False
    b2 = None

try:
    import nengo
    NENGO_AVAILABLE = True
except ImportError:
    NENGO_AVAILABLE = False
    nengo = None


class ThalamicNucleus:
    """Thalamic nucleus providing input to cortical layers."""
    
    def __init__(self, name: str, size: int, params: Dict[str, Any]):
        self.name = name
        self.size = size
        self.params = params
        self.neurons: List[Neuron] = []
        self.activity_pattern: Optional[np.ndarray] = None
        self.baseline_rate: float = params.get('baseline_rate', 5.0)  # Hz
        self.burst_probability: float = params.get('burst_probability', 0.1)
        self.burst_duration: float = params.get('burst_duration', 50.0)  # ms
        self._current_time = 0.0
        
        # Create thalamic neurons
        for i in range(size):
            neuron_params = {
                'threshold': -50.0,
                'reset': -70.0,
                'tau_m': 20.0,
                'baseline_current': self.baseline_rate * 0.1
            }
            neuron = create_neuron('lif', i + 10000, neuron_params)  # Offset IDs
            self.neurons.append(neuron)
    
    def set_sensory_input(self, sensory_data: np.ndarray):
        """Process sensory input and generate thalamic activity."""
        if len(sensory_data) != self.size:
            # Resize sensory data to match thalamic size
            if len(sensory_data) > self.size:
                sensory_data = sensory_data[:self.size]
            else:
                sensory_data = np.pad(sensory_data, (0, self.size - len(sensory_data)))
        
        # Normalize and scale sensory input
        normalized_input = (sensory_data - np.mean(sensory_data)) / (np.std(sensory_data) + 1e-6)
        self.activity_pattern = self.baseline_rate + normalized_input * 10.0
        
        # Apply activity to neurons
        for i, neuron in enumerate(self.neurons):
            input_current = self.activity_pattern[i] * 0.1
            # Add burst activity
            if random.random() < self.burst_probability:
                input_current *= 3.0  # Burst amplification
            setattr(neuron, '_input_current', input_current)
    
    def step(self, dt: float) -> List[int]:
        """Update thalamic neurons and return spike list."""
        spikes = []
        current_time = float(getattr(self, "_current_time", 0.0))
        for neuron in self.neurons:
            input_current = getattr(neuron, '_input_current', 0.0)
            step_fn = getattr(neuron, "step", None)
            if callable(step_fn):
                if step_fn(float(dt), float(input_current), current_time=current_time):
                    spikes.append(neuron.id)
            else:
                if neuron.update(input_current, dt):
                    spikes.append(neuron.id)
            # Reset input current
            setattr(neuron, '_input_current', 0.0)
        self._current_time = current_time + float(dt)
        return spikes


class EnhancedCorticalColumn(NeuralNetwork):
    """Enhanced 6-layer cortical column with thalamic input and realistic connectivity."""
    
    def __init__(self, config: Dict[str, Any]):
        # Default cortical column configuration
        default_config = {
            'total_neurons': 2000,
            'layer_proportions': {
                'L1': 0.05,    # Molecular layer - sparse
                'L2/3': 0.35,  # Supragranular layers - dense
                'L4': 0.25,    # Granular layer - input layer
                'L5': 0.25,    # Infragranular layer - output
                'L6': 0.10     # Deep layer - feedback
            },
            'neuron_types': {
                'L1': 'lif',        # Simple interneurons
                'L2/3': 'adex',     # Complex pyramidal cells
                'L4': 'lif',        # Spiny stellate cells
                'L5': 'hh',         # Large pyramidal cells
                'L6': 'izhikevich'  # Corticothalamic cells
            },
            'excitatory_ratio': 0.8,  # 80% excitatory, 20% inhibitory
            'connection_probabilities': {
                # Feedforward connections
                'L4->L2/3': 0.4,
                'L2/3->L5': 0.3,
                'L4->L5': 0.2,
                'L5->L6': 0.3,
                
                # Feedback connections
                'L6->L4': 0.2,
                'L5->L2/3': 0.2,
                'L2/3->L1': 0.3,
                
                # Lateral connections
                'L2/3->L2/3': 0.1,
                'L5->L5': 0.1,
                
                # Thalamic input
                'Thalamus->L4': 0.3,
                'Thalamus->L6': 0.2
            },
            'delay_ranges': {
                'local': [0.5, 2.0],      # Local connections (ms)
                'feedforward': [1.0, 3.0], # Feedforward (ms)
                'feedback': [2.0, 5.0],    # Feedback (ms)
                'thalamic': [1.0, 4.0]     # Thalamic input (ms)
            },
            'stp_enabled': True,
            'brian2_validation': False
        }
        
        # Merge with user config
        self.column_config = {**default_config, **config}
        
        # Initialize thalamic input (region-specific nucleus identifier if provided).
        thalamic_size = config.get('thalamic_size', 200)
        nucleus = config.get('thalamic_nucleus', config.get('thalamic_name', 'VPL'))
        self.thalamus = ThalamicNucleus(str(nucleus), thalamic_size, config.get('thalamic_params', {}))
        
        # Create network configuration
        network_config = self._build_network_config()
        super().__init__(network_config)
        
        # Create cortical layers and connections
        self._create_cortical_architecture()
        self._create_thalamic_connections()
        
        # Initialize Brian2 backend if requested
        self.brian2_net = None
        if self.column_config.get('brian2_validation') and BRIAN2_AVAILABLE:
            self._initialize_brian2_backend()
        
        # Sensory processing pipeline
        self.sensory_buffer = deque(maxlen=100)
        self.current_time = 0.0
    
    def _build_network_config(self) -> Dict[str, Any]:
        """Build network configuration from cortical column parameters."""
        layers = []
        total_neurons = self.column_config['total_neurons']
        
        for layer_name, proportion in self.column_config['layer_proportions'].items():
            size = max(1, int(total_neurons * proportion))
            
            # Split into excitatory and inhibitory populations
            exc_size = int(size * self.column_config['excitatory_ratio'])
            inh_size = size - exc_size
            
            # Excitatory population
            layers.append({
                'name': f'{layer_name}_exc',
                'size': exc_size,
                'type': 'hidden',
                'neuron_type': self.column_config['neuron_types'][layer_name],
                'population_type': 'excitatory'
            })
            
            # Inhibitory population
            if inh_size > 0:
                layers.append({
                    'name': f'{layer_name}_inh',
                    'size': inh_size,
                    'type': 'hidden',
                    'neuron_type': 'lif',  # Inhibitory interneurons
                    'population_type': 'inhibitory'
                })
        
        return {
            'layers': layers,
            'neuron_params': self._get_neuron_parameters(),
            'synapse_params': self._get_synapse_parameters()
        }
    
    def _get_neuron_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define layer-specific neuron parameters."""
        return {
            'L1': {'tau_m': 15.0, 'threshold': -55.0, 'reset': -70.0},
            'L2/3': {'tau_m': 20.0, 'threshold': -50.0, 'reset': -70.0, 'adaptation': 0.1},
            'L4': {'tau_m': 10.0, 'threshold': -52.0, 'reset': -65.0},
            'L5': {'tau_m': 25.0, 'threshold': -45.0, 'reset': -70.0, 'spike_height': 30.0},
            'L6': {'tau_m': 20.0, 'threshold': -50.0, 'reset': -70.0, 'recovery': 0.02}
        }
    
    def _get_synapse_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define synapse-specific parameters including STP."""
        base_params = {
            'excitatory': {
                'tau_depression': 200.0,
                'tau_facilitation': 50.0,
                'use_baseline': 0.2,
                'use_facilitation': 0.1
            },
            'inhibitory': {
                'tau_depression': 100.0,
                'tau_facilitation': 20.0,
                'use_baseline': 0.4,
                'use_facilitation': 0.05
            }
        }
        return base_params
    
    def _create_cortical_architecture(self):
        """Create realistic cortical layer architecture."""
        # Clear existing connections
        self.synapses.clear()
        self.pre_synapses = defaultdict(list)
        self.post_synapses = defaultdict(list)
        
        # Create layer-to-layer connections
        for connection, probability in self.column_config['connection_probabilities'].items():
            if '->' not in connection or connection.startswith('Thalamus'):
                continue
                
            pre_layer, post_layer = connection.split('->')
            self._connect_cortical_layers(pre_layer, post_layer, probability)
        
        # Add lateral inhibition within layers
        self._add_lateral_inhibition()
    
    def _connect_cortical_layers(self, pre_layer: str, post_layer: str, probability: float):
        """Connect two cortical layers with realistic parameters."""
        # Get layer populations
        pre_exc = self.layers.get(f'{pre_layer}_exc')
        pre_inh = self.layers.get(f'{pre_layer}_inh')
        post_exc = self.layers.get(f'{post_layer}_exc')
        post_inh = self.layers.get(f'{post_layer}_inh')
        
        if not all([pre_exc, post_exc]):
            return
        
        # Determine connection type for delay selection
        connection_type = self._classify_connection(pre_layer, post_layer)
        delay_range = self.column_config['delay_ranges'][connection_type]
        
        # Excitatory to excitatory connections
        self._connect_populations(pre_exc, post_exc, probability, 'excitatory', delay_range)
        
        # Excitatory to inhibitory connections
        if post_inh:
            self._connect_populations(pre_exc, post_inh, probability * 0.8, 'excitatory', delay_range)
        
        # Inhibitory to excitatory connections
        if pre_inh:
            self._connect_populations(pre_inh, post_exc, probability * 0.6, 'inhibitory', delay_range)
    
    def _classify_connection(self, pre_layer: str, post_layer: str) -> str:
        """Classify connection type for delay assignment."""
        layer_order = ['L1', 'L2/3', 'L4', 'L5', 'L6']
        
        try:
            pre_idx = layer_order.index(pre_layer)
            post_idx = layer_order.index(post_layer)
            
            if pre_idx == post_idx:
                return 'local'
            elif pre_idx < post_idx:
                return 'feedforward'
            else:
                return 'feedback'
        except ValueError:
            return 'local'
    
    def _connect_populations(self, pre_layer: Dict[str, Any], post_layer: Dict[str, Any], 
                           probability: float, synapse_type: str, delay_range: List[float]):
        """Connect two neuron populations."""
        for pre_neuron in pre_layer.neurons:
            for post_neuron in post_layer.neurons:
                if random.random() < probability:
                    # Determine synapse parameters
                    weight = self._sample_weight(synapse_type)
                    delay = random.uniform(delay_range[0], delay_range[1])
                    
                    # Create synapse with STP if enabled
                    if self.column_config.get('stp_enabled', True):
                        synapse_class = 'stp'
                        params = {
                            'weight': weight,
                            'delay': delay,
                            **self.config['synapse_params'][synapse_type]
                        }
                    else:
                        synapse_class = 'static'
                        params = {'weight': weight, 'delay': delay}
                    
                    self.add_synapse(pre_neuron.id, post_neuron.id, synapse_class, params)
    
    def _sample_weight(self, synapse_type: str) -> float:
        """Sample synaptic weight based on connection type."""
        if synapse_type == 'excitatory':
            # Python's stdlib `random` uses `lognormvariate(mu, sigma)`.
            return float(random.lognormvariate(-1.0, 0.5))  # Positive weights
        else:  # inhibitory
            return -float(random.lognormvariate(-1.0, 0.3))  # Negative weights
    
    def _add_lateral_inhibition(self):
        """Add lateral inhibition within layers."""
        for layer_name in ['L2/3', 'L4', 'L5']:
            exc_layer = self.layers.get(f'{layer_name}_exc')
            inh_layer = self.layers.get(f'{layer_name}_inh')
            
            if exc_layer and inh_layer:
                # Inhibitory neurons inhibit nearby excitatory neurons
                for inh_neuron in inh_layer.neurons:
                    # Select subset of excitatory neurons to inhibit
                    targets = random.sample(exc_layer.neurons, 
                                          min(10, len(exc_layer.neurons)))
                    for target in targets:
                        weight = -random.uniform(0.5, 2.0)
                        delay = random.uniform(0.5, 1.5)
                        self.add_synapse(inh_neuron.id, target.id, 'static', 
                                       {'weight': weight, 'delay': delay})
    
    def _create_thalamic_connections(self):
        """Create connections from thalamus to cortical layers."""
        thalamic_connections = {
            'Thalamus->L4': 0.3,
            'Thalamus->L6': 0.2
        }
        
        for connection, probability in thalamic_connections.items():
            _, target_layer = connection.split('->')
            target_exc = self.layers.get(f'{target_layer}_exc')
            
            if target_exc:
                delay_range = self.column_config['delay_ranges']['thalamic']
                
                for thalamic_neuron in self.thalamus.neurons:
                    for cortical_neuron in target_exc.neurons:
                        if random.random() < probability:
                            weight = random.uniform(0.3, 1.0)
                            delay = random.uniform(delay_range[0], delay_range[1])
                            self.add_synapse(thalamic_neuron.id, cortical_neuron.id, 
                                           'static', {'weight': weight, 'delay': delay})
    
    def process_sensory_input(self, sensory_data: np.ndarray):
        """Process sensory input through thalamic relay."""
        # Store sensory data
        self.sensory_buffer.append(sensory_data.copy())
        
        # Send to thalamus
        self.thalamus.set_sensory_input(sensory_data)
    
    def step(self, dt: float) -> Dict[str, Any]:
        """Enhanced step function with thalamic processing."""
        self.current_time += dt
        
        # Update thalamic neurons
        thalamic_spikes = self.thalamus.step(dt)
        
        # Update cortical network
        cortical_state = super().step(dt)
        
        # Combine results
        all_spikes = list(set(thalamic_spikes + cortical_state['spikes']))
        
        # Add thalamic voltages
        thalamic_voltages = {n.id: n.voltage for n in self.thalamus.neurons}
        all_voltages = {**cortical_state['voltages'], **thalamic_voltages}
        
        return {
            'spikes': all_spikes,
            'voltages': all_voltages,
            'weights': cortical_state['weights'],
            'thalamic_spikes': thalamic_spikes,
            'cortical_spikes': cortical_state['spikes'],
            'thalamic_activity': self.thalamus.activity_pattern
        }
    
    def _initialize_brian2_backend(self):
        """Initialize Brian2 backend for validation."""
        if not BRIAN2_AVAILABLE:
            print("Warning: Brian2 not available, skipping backend initialization")
            return
        
        # This would implement Brian2 network creation
        # For now, just set up the framework
        b2.start_scope()
        
        # Create neuron groups for each layer
        self.brian2_groups = {}
        for layer_name, layer in self.layers.items():
            # Define neuron model based on layer type
            if 'exc' in layer_name:
                model = '''
                dv/dt = (I - v) / tau : volt
                I : amp
                tau : second
                '''
            else:
                model = '''
                dv/dt = (I - v) / tau : volt
                I : amp  
                tau : second
                '''
            
            group = b2.NeuronGroup(layer.size, model, 
                                 threshold='v > -50*mV', 
                                 reset='v = -70*mV')
            group.v = -70 * b2.mV
            group.tau = 20 * b2.ms
            
            self.brian2_groups[layer_name] = group
        
        # 创建突触连接
        self.brian2_synapses = {}
        # Implementation would create Brian2 Synapses objects here
        
        self.brian2_net = b2.Network(list(self.brian2_groups.values()))
        print("Brian2 backend initialized for validation")
    
    def validate_with_brian2(self, duration: float = 100.0) -> Dict[str, Any]:
        """Run validation simulation with Brian2 backend."""
        if not self.brian2_net:
            return {'error': 'Brian2 backend not initialized'}
        
        # Run Brian2 simulation
        self.brian2_net.run(duration * b2.ms)
        
        # Collect results
        results = {}
        for layer_name, group in self.brian2_groups.items():
            results[layer_name] = {
                'voltages': np.array(group.v / b2.mV),
                'spike_count': len(group.spike_trains())
            }
        
        return results
    
    def get_layer_activity(self, layer_name: str) -> Dict[str, Any]:
        """Get activity statistics for a specific layer."""
        exc_layer = self.layers.get(f'{layer_name}_exc')
        inh_layer = self.layers.get(f'{layer_name}_inh')
        
        activity = {}
        
        if exc_layer:
            exc_voltages = [self.neurons[nid].voltage for nid in exc_layer.neuron_ids]
            activity['excitatory'] = {
                'mean_voltage': np.mean(exc_voltages),
                'std_voltage': np.std(exc_voltages),
                'active_neurons': sum(1 for v in exc_voltages if v > -60.0)
            }
        
        if inh_layer:
            inh_voltages = [self.neurons[nid].voltage for nid in inh_layer.neuron_ids]
            activity['inhibitory'] = {
                'mean_voltage': np.mean(inh_voltages),
                'std_voltage': np.std(inh_voltages),
                'active_neurons': sum(1 for v in inh_voltages if v > -60.0)
            }
        
        return activity
    
    def get_connectivity_matrix(self) -> np.ndarray:
        """Return connectivity matrix for analysis."""
        n_neurons = len(self.neurons)
        connectivity = np.zeros((n_neurons, n_neurons))
        
        for (pre_id, post_id), synapse in self.synapses.items():
            # Map neuron IDs to matrix indices
            pre_idx = list(self.neurons.keys()).index(pre_id)
            post_idx = list(self.neurons.keys()).index(post_id)
            connectivity[pre_idx, post_idx] = synapse.weight
        
        return connectivity

class CorticalColumn(EnhancedCorticalColumn):
    """Compatibility wrapper expected by legacy orchestration code."""

    def __init__(self, column_id: int, position: Tuple[float, float], config: Dict[str, Any]):
        column_cfg = dict(config.get('cortical_column', {}))
        # Keep unit tests and lightweight runs fast: default to a small column.
        column_cfg.setdefault('total_neurons', int(config.get('scope', {}).get('column_total_neurons', 120)))
        column_cfg.setdefault('thalamic_size', config.get('thalamic_size', 200))
        if 'thalamic_nucleus' in config and 'thalamic_nucleus' not in column_cfg:
            column_cfg['thalamic_nucleus'] = config['thalamic_nucleus']
        if 'thalamic_params' in config and 'thalamic_params' not in column_cfg:
            column_cfg['thalamic_params'] = config['thalamic_params']
        column_cfg.setdefault('column_id', column_id)
        column_cfg.setdefault('position', position)
        self.synapses = {}
        self.pre_synapses = defaultdict(list)
        self.post_synapses = defaultdict(list)
        super().__init__(column_cfg)
        self.column_id = column_id
        self.position = position
        self.source_config = config
