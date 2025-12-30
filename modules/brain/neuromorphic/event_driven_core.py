"""Event-driven neuromorphic computing core with AER protocol support.

This module implements a production-grade event-driven architecture for
neuromorphic computing, featuring Address Event Representation (AER) protocol,
asynchronous spike processing, and ultra-low power event handling.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import struct
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of neuromorphic events."""
    SPIKE = "spike"
    INHIBITION = "inhibition"
    PLASTICITY = "plasticity"
    NEUROMODULATION = "neuromodulation"
    RESET = "reset"
    CONFIGURATION = "configuration"


@dataclass
class NeuromorphicEvent:
    """Base neuromorphic event with AER encoding."""
    timestamp_ns: int
    event_type: EventType
    source_id: int
    target_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    
    def __lt__(self, other):
        """Priority queue ordering."""
        if self.timestamp_ns != other.timestamp_ns:
            return self.timestamp_ns < other.timestamp_ns
        return self.priority < other.priority
    
    def to_aer_packet(self) -> bytes:
        """Convert to AER packet format."""
        # AER packet: [timestamp(8)] [type(1)] [source(4)] [target(4)] [payload_len(2)] [payload]
        packet = struct.pack(
            '>QBI I H',
            self.timestamp_ns,
            self.event_type.value.encode('ascii')[0],
            self.source_id,
            self.target_id or 0,
            len(self.payload)
        )
        
        if self.payload:
            payload_bytes = str(self.payload).encode('utf-8')
            packet += payload_bytes
        
        return packet
    
    @classmethod
    def from_aer_packet(cls, packet: bytes) -> 'NeuromorphicEvent':
        """Create event from AER packet."""
        if len(packet) < 19:  # Minimum packet size
            raise ValueError("Invalid AER packet size")
        
        timestamp_ns, event_type_byte, source_id, target_id, payload_len = struct.unpack(
            '>QBI I H', packet[:19]
        )
        
        event_type = EventType.SPIKE  # Default
        for et in EventType:
            if et.value.encode('ascii')[0] == event_type_byte:
                event_type = et
                break
        
        payload = {}
        if payload_len > 0 and len(packet) >= 19 + payload_len:
            payload_bytes = packet[19:19 + payload_len]
            try:
                payload = eval(payload_bytes.decode('utf-8'))
            except:
                payload = {"raw": payload_bytes}
        
        return cls(
            timestamp_ns=timestamp_ns,
            event_type=event_type,
            source_id=source_id,
            target_id=target_id if target_id != 0 else None,
            payload=payload
        )


@dataclass
class SpikeEvent(NeuromorphicEvent):
    """Spike event with neuromorphic-specific data."""
    amplitude: float = 1.0
    duration_ns: int = 1000
    
    def __post_init__(self):
        self.event_type = EventType.SPIKE
        self.payload.update({
            "amplitude": self.amplitude,
            "duration_ns": self.duration_ns
        })


@dataclass
class PlasticityEvent(NeuromorphicEvent):
    """Synaptic plasticity event."""
    weight_delta: float = 0.0
    learning_rate: float = 0.01
    
    def __post_init__(self):
        self.event_type = EventType.PLASTICITY
        self.payload.update({
            "weight_delta": self.weight_delta,
            "learning_rate": self.learning_rate
        })


class EventProcessor(ABC):
    """Abstract event processor interface."""
    
    @abstractmethod
    async def process_event(self, event: NeuromorphicEvent) -> List[NeuromorphicEvent]:
        """Process an event and return generated events."""
        pass
    
    @abstractmethod
    def get_processor_id(self) -> str:
        """Get unique processor identifier."""
        pass


class SpikeProcessor(EventProcessor):
    """Processes spike events and generates responses."""
    
    def __init__(self, neuron_id: int, threshold: float = 1.0, 
                 refractory_period_ns: int = 1000000):
        self.neuron_id = neuron_id
        self.threshold = threshold
        self.refractory_period_ns = refractory_period_ns
        self.membrane_potential = 0.0
        self.last_spike_time = 0
        self.synaptic_weights: Dict[int, float] = {}
        self.decay_rate = 0.95
    
    async def process_event(self, event: NeuromorphicEvent) -> List[NeuromorphicEvent]:
        """Process incoming spike and update membrane potential."""
        generated_events = []
        
        if event.event_type == EventType.SPIKE:
            # Check refractory period
            if event.timestamp_ns - self.last_spike_time < self.refractory_period_ns:
                return generated_events
            
            # Update membrane potential
            weight = self.synaptic_weights.get(event.source_id, 0.1)
            amplitude = event.payload.get("amplitude", 1.0)
            self.membrane_potential += weight * amplitude
            
            # Apply decay
            time_diff = event.timestamp_ns - self.last_spike_time
            decay_factor = np.exp(-time_diff / 1e9 * (1 - self.decay_rate))
            self.membrane_potential *= decay_factor
            
            # Check for spike generation
            if self.membrane_potential >= self.threshold:
                # Generate output spike
                spike = SpikeEvent(
                    timestamp_ns=event.timestamp_ns + 100000,  # 100μs delay
                    source_id=self.neuron_id,
                    amplitude=self.membrane_potential
                )
                generated_events.append(spike)
                
                # Reset membrane potential
                self.membrane_potential = 0.0
                self.last_spike_time = event.timestamp_ns
                
                # Generate plasticity events for active synapses
                for source_id, weight in self.synaptic_weights.items():
                    if abs(event.timestamp_ns - self.last_spike_time) < 20000000:  # 20ms window
                        plasticity = PlasticityEvent(
                            timestamp_ns=event.timestamp_ns + 200000,  # 200μs delay
                            source_id=source_id,
                            target_id=self.neuron_id,
                            weight_delta=0.01 * np.exp(-(event.timestamp_ns - self.last_spike_time) / 20000000)
                        )
                        generated_events.append(plasticity)
        
        elif event.event_type == EventType.PLASTICITY:
            # Update synaptic weight
            if event.target_id == self.neuron_id:
                source_id = event.source_id
                delta = event.payload.get("weight_delta", 0.0)
                current_weight = self.synaptic_weights.get(source_id, 0.1)
                new_weight = np.clip(current_weight + delta, 0.0, 2.0)
                self.synaptic_weights[source_id] = new_weight
        
        return generated_events
    
    def get_processor_id(self) -> str:
        return f"spike_processor_{self.neuron_id}"


class EventDrivenCore:
    """Event-driven neuromorphic computing core."""
    
    def __init__(self, max_events: int = 1000000):
        self.event_queue: List[NeuromorphicEvent] = []
        self.processors: Dict[str, EventProcessor] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.max_events = max_events
        self.current_time_ns = 0
        self.total_events_processed = 0
        self.power_consumption_nj = 0.0
        self.running = False
        
        # Performance metrics
        self.event_latencies: deque = deque(maxlen=1000)
        self.processing_times: deque = deque(maxlen=1000)
        self.queue_sizes: deque = deque(maxlen=1000)
        
        # AER interface
        self.aer_input_buffer: deque = deque(maxlen=10000)
        self.aer_output_buffer: deque = deque(maxlen=10000)
    
    def add_processor(self, processor: EventProcessor):
        """Add an event processor."""
        self.processors[processor.get_processor_id()] = processor
        logger.debug(f"Added processor: {processor.get_processor_id()}")
    
    def remove_processor(self, processor_id: str):
        """Remove an event processor."""
        if processor_id in self.processors:
            del self.processors[processor_id]
            logger.debug(f"Removed processor: {processor_id}")
    
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """Add event handler for specific event type."""
        self.event_handlers[event_type].append(handler)
    
    def inject_event(self, event: NeuromorphicEvent):
        """Inject event into the processing queue."""
        if len(self.event_queue) >= self.max_events:
            # Remove oldest event to prevent overflow
            heapq.heappop(self.event_queue)
        
        heapq.heappush(self.event_queue, event)
        
        # Add to AER output buffer
        self.aer_output_buffer.append(event.to_aer_packet())
    
    def inject_aer_packet(self, packet: bytes):
        """Inject AER packet for processing."""
        try:
            event = NeuromorphicEvent.from_aer_packet(packet)
            self.inject_event(event)
            self.aer_input_buffer.append(packet)
        except Exception as e:
            logger.error(f"Failed to parse AER packet: {e}")
    
    async def process_events(self, duration_ns: int) -> Dict[str, Any]:
        """Process events for specified duration."""
        start_time = time.time_ns()
        end_time_ns = self.current_time_ns + duration_ns
        events_processed = 0
        
        self.running = True
        
        try:
            while self.running and self.event_queue and self.current_time_ns < end_time_ns:
                # Get next event
                if not self.event_queue:
                    break
                
                event = heapq.heappop(self.event_queue)
                
                # Update current time
                self.current_time_ns = max(self.current_time_ns, event.timestamp_ns)
                
                # Skip events beyond duration
                if event.timestamp_ns > end_time_ns:
                    heapq.heappush(self.event_queue, event)
                    break
                
                # Process event
                process_start = time.time_ns()
                generated_events = await self._process_single_event(event)
                process_end = time.time_ns()
                
                # Add generated events to queue
                for gen_event in generated_events:
                    self.inject_event(gen_event)
                
                # Update metrics
                events_processed += 1
                self.total_events_processed += 1
                
                processing_time = process_end - process_start
                self.processing_times.append(processing_time)
                
                event_latency = process_start - event.timestamp_ns
                self.event_latencies.append(event_latency)
                
                self.queue_sizes.append(len(self.event_queue))
                
                # Update power consumption (simplified model)
                self.power_consumption_nj += self._calculate_event_power(event, processing_time)
                
                # Yield control periodically
                if events_processed % 100 == 0:
                    await asyncio.sleep(0)
        
        finally:
            self.running = False
        
        processing_duration = time.time_ns() - start_time
        
        return {
            "events_processed": events_processed,
            "processing_duration_ns": processing_duration,
            "final_time_ns": self.current_time_ns,
            "queue_size": len(self.event_queue),
            "power_consumption_nj": self.power_consumption_nj,
            "avg_latency_ns": np.mean(self.event_latencies) if self.event_latencies else 0,
            "avg_processing_time_ns": np.mean(self.processing_times) if self.processing_times else 0
        }
    
    async def _process_single_event(self, event: NeuromorphicEvent) -> List[NeuromorphicEvent]:
        """Process a single event through all relevant processors."""
        generated_events = []
        
        # Call event handlers
        for handler in self.event_handlers[event.event_type]:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        # Process through relevant processors
        for processor in self.processors.values():
            try:
                proc_events = await processor.process_event(event)
                generated_events.extend(proc_events)
            except Exception as e:
                logger.error(f"Processor error in {processor.get_processor_id()}: {e}")
        
        return generated_events
    
    def _calculate_event_power(self, event: NeuromorphicEvent, processing_time_ns: int) -> float:
        """Calculate power consumption for event processing."""
        # Simplified power model based on event type and processing time
        base_power_nj = {
            EventType.SPIKE: 0.1,
            EventType.INHIBITION: 0.05,
            EventType.PLASTICITY: 0.2,
            EventType.NEUROMODULATION: 0.15,
            EventType.RESET: 0.01,
            EventType.CONFIGURATION: 0.5
        }
        
        base = base_power_nj.get(event.event_type, 0.1)
        processing_factor = processing_time_ns / 1000.0  # Scale by processing time
        
        return base + processing_factor * 0.001
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and power metrics."""
        return {
            "total_events_processed": self.total_events_processed,
            "current_time_ns": self.current_time_ns,
            "queue_size": len(self.event_queue),
            "power_consumption_nj": self.power_consumption_nj,
            "avg_latency_ns": np.mean(self.event_latencies) if self.event_latencies else 0,
            "max_latency_ns": np.max(self.event_latencies) if self.event_latencies else 0,
            "avg_processing_time_ns": np.mean(self.processing_times) if self.processing_times else 0,
            "max_processing_time_ns": np.max(self.processing_times) if self.processing_times else 0,
            "avg_queue_size": np.mean(self.queue_sizes) if self.queue_sizes else 0,
            "max_queue_size": np.max(self.queue_sizes) if self.queue_sizes else 0,
            "aer_input_buffer_size": len(self.aer_input_buffer),
            "aer_output_buffer_size": len(self.aer_output_buffer)
        }
    
    def reset(self):
        """Reset the event-driven core."""
        self.event_queue.clear()
        self.current_time_ns = 0
        self.total_events_processed = 0
        self.power_consumption_nj = 0.0
        self.event_latencies.clear()
        self.processing_times.clear()
        self.queue_sizes.clear()
        self.aer_input_buffer.clear()
        self.aer_output_buffer.clear()
        
        # Reset all processors
        for processor in self.processors.values():
            if hasattr(processor, 'reset'):
                processor.reset()
    
    def stop(self):
        """Stop event processing."""
        self.running = False
    
    def get_aer_packets(self) -> List[bytes]:
        """Get AER packets from output buffer."""
        packets = list(self.aer_output_buffer)
        self.aer_output_buffer.clear()
        return packets
    
    def inject_aer_packets(self, packets: List[bytes]):
        """Inject multiple AER packets."""
        for packet in packets:
            self.inject_aer_packet(packet)


class NetworkTopology:
    """Manages network topology for event-driven processing."""
    
    def __init__(self):
        self.connections: Dict[int, Set[int]] = defaultdict(set)
        self.weights: Dict[Tuple[int, int], float] = {}
        self.delays: Dict[Tuple[int, int], int] = {}  # in nanoseconds
    
    def add_connection(self, source: int, target: int, weight: float = 1.0, delay_ns: int = 1000):
        """Add synaptic connection."""
        self.connections[source].add(target)
        self.weights[(source, target)] = weight
        self.delays[(source, target)] = delay_ns
    
    def remove_connection(self, source: int, target: int):
        """Remove synaptic connection."""
        if target in self.connections[source]:
            self.connections[source].remove(target)
        self.weights.pop((source, target), None)
        self.delays.pop((source, target), None)
    
    def get_targets(self, source: int) -> Set[int]:
        """Get target neurons for source neuron."""
        return self.connections[source].copy()
    
    def get_weight(self, source: int, target: int) -> float:
        """Get synaptic weight."""
        return self.weights.get((source, target), 0.0)
    
    def get_delay(self, source: int, target: int) -> int:
        """Get synaptic delay in nanoseconds."""
        return self.delays.get((source, target), 1000)
    
    def update_weight(self, source: int, target: int, new_weight: float):
        """Update synaptic weight."""
        if (source, target) in self.weights:
            self.weights[(source, target)] = new_weight
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get topology statistics."""
        total_connections = sum(len(targets) for targets in self.connections.values())
        total_neurons = len(self.connections)
        
        if total_connections > 0:
            avg_weight = np.mean(list(self.weights.values()))
            avg_delay = np.mean(list(self.delays.values()))
        else:
            avg_weight = 0.0
            avg_delay = 0.0
        
        return {
            "total_neurons": total_neurons,
            "total_connections": total_connections,
            "avg_connections_per_neuron": total_connections / max(total_neurons, 1),
            "avg_weight": avg_weight,
            "avg_delay_ns": avg_delay
        }


class EventDrivenNetwork:
    """Complete event-driven neuromorphic network."""
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.core = EventDrivenCore()
        self.topology = NetworkTopology()
        
        # Create spike processors for each neuron
        for i in range(n_neurons):
            processor = SpikeProcessor(i)
            self.core.add_processor(processor)
    
    def add_connection(self, source: int, target: int, weight: float = 1.0, delay_ns: int = 1000):
        """Add synaptic connection."""
        self.topology.add_connection(source, target, weight, delay_ns)
        
        # Update processor weights
        processor_id = f"spike_processor_{target}"
        if processor_id in self.core.processors:
            processor = self.core.processors[processor_id]
            if hasattr(processor, 'synaptic_weights'):
                processor.synaptic_weights[source] = weight
    
    def inject_spike(self, neuron_id: int, timestamp_ns: int, amplitude: float = 1.0):
        """Inject spike into network."""
        spike = SpikeEvent(
            timestamp_ns=timestamp_ns,
            source_id=neuron_id,
            amplitude=amplitude
        )
        self.core.inject_event(spike)
    
    def inject_spike_train(self, neuron_id: int, spike_times_ns: List[int], amplitude: float = 1.0):
        """Inject spike train into network."""
        for timestamp_ns in spike_times_ns:
            self.inject_spike(neuron_id, timestamp_ns, amplitude)
    
    async def run(self, duration_ns: int) -> Dict[str, Any]:
        """Run network simulation."""
        return await self.core.process_events(duration_ns)
    
    def get_output_spikes(self) -> List[Tuple[int, int]]:  # (timestamp_ns, neuron_id)
        """Get output spikes from AER packets."""
        spikes = []
        packets = self.core.get_aer_packets()
        
        for packet in packets:
            try:
                event = NeuromorphicEvent.from_aer_packet(packet)
                if event.event_type == EventType.SPIKE:
                    spikes.append((event.timestamp_ns, event.source_id))
            except:
                continue
        
        return sorted(spikes)
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive network metrics."""
        core_metrics = self.core.get_metrics()
        topology_metrics = self.topology.get_statistics()
        
        return {
            "core": core_metrics,
            "topology": topology_metrics,
            "n_neurons": self.n_neurons
        }
    
    def reset(self):
        """Reset network state."""
        self.core.reset()
    
    def save_aer_stream(self, filename: str):
        """Save AER event stream to file."""
        packets = self.core.get_aer_packets()
        with open(filename, 'wb') as f:
            for packet in packets:
                f.write(len(packet).to_bytes(4, 'big'))  # Packet length
                f.write(packet)
    
    def load_aer_stream(self, filename: str):
        """Load AER event stream from file."""
        packets = []
        with open(filename, 'rb') as f:
            while True:
                length_bytes = f.read(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                packet = f.read(length)
                if len(packet) == length:
                    packets.append(packet)
        
        self.core.inject_aer_packets(packets)


__all__ = [
    "EventType",
    "NeuromorphicEvent", 
    "SpikeEvent",
    "PlasticityEvent",
    "EventProcessor",
    "SpikeProcessor",
    "EventDrivenCore",
    "NetworkTopology",
    "EventDrivenNetwork"
]