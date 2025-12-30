"""Power optimization and energy-aware computing for neuromorphic systems.

This module provides comprehensive power modeling, optimization algorithms,
and energy-efficient computing strategies for neuromorphic hardware platforms.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """Power management states."""
    ACTIVE = "active"
    IDLE = "idle"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"
    OFF = "off"


@dataclass
class PowerProfile:
    """Power consumption profile for different operations."""
    spike_processing_nj: float = 0.1
    synapse_update_nj: float = 0.05
    memory_access_nj: float = 0.02
    communication_nj: float = 0.15
    idle_power_mw: float = 0.5
    sleep_power_mw: float = 0.01
    leakage_power_mw: float = 0.001
    voltage_v: float = 1.0
    frequency_mhz: float = 1000.0


@dataclass
class EnergyBudget:
    """Energy budget constraints."""
    total_budget_mj: float
    time_window_s: float
    max_power_mw: float
    critical_threshold: float = 0.1  # 10% remaining
    warning_threshold: float = 0.2   # 20% remaining


@dataclass
class PowerMetrics:
    """Real-time power consumption metrics."""
    current_power_mw: float
    average_power_mw: float
    peak_power_mw: float
    total_energy_mj: float
    efficiency_ops_per_mj: float
    temperature_celsius: float
    voltage_v: float
    frequency_mhz: float
    utilization_percent: float


class PowerModel(ABC):
    """Abstract power model interface."""
    
    @abstractmethod
    def calculate_spike_power(self, spike_count: int, neuron_complexity: float) -> float:
        """Calculate power for spike processing."""
        pass
    
    @abstractmethod
    def calculate_synapse_power(self, synapse_updates: int, weight_precision: int) -> float:
        """Calculate power for synapse updates."""
        pass
    
    @abstractmethod
    def calculate_memory_power(self, memory_accesses: int, data_size: int) -> float:
        """Calculate power for memory operations."""
        pass
    
    @abstractmethod
    def calculate_communication_power(self, data_transmitted: int) -> float:
        """Calculate power for communication."""
        pass


class DetailedPowerModel(PowerModel):
    """Detailed power model based on hardware characteristics."""
    
    def __init__(self, profile: PowerProfile):
        self.profile = profile
        self.voltage_scaling_factor = 1.0
        self.frequency_scaling_factor = 1.0
        self.temperature_factor = 1.0
    
    def calculate_spike_power(self, spike_count: int, neuron_complexity: float = 1.0) -> float:
        """Calculate power for spike processing."""
        base_power = self.profile.spike_processing_nj * spike_count
        complexity_factor = 1.0 + (neuron_complexity - 1.0) * 0.5
        voltage_factor = self.voltage_scaling_factor ** 2
        frequency_factor = self.frequency_scaling_factor
        temp_factor = self.temperature_factor
        
        return base_power * complexity_factor * voltage_factor * frequency_factor * temp_factor
    
    def calculate_synapse_power(self, synapse_updates: int, weight_precision: int = 8) -> float:
        """Calculate power for synapse updates."""
        base_power = self.profile.synapse_update_nj * synapse_updates
        precision_factor = (weight_precision / 8.0) ** 1.5  # Non-linear scaling
        voltage_factor = self.voltage_scaling_factor ** 2
        
        return base_power * precision_factor * voltage_factor
    
    def calculate_memory_power(self, memory_accesses: int, data_size: int) -> float:
        """Calculate power for memory operations."""
        base_power = self.profile.memory_access_nj * memory_accesses
        size_factor = math.log2(max(1, data_size / 32))  # Logarithmic scaling
        voltage_factor = self.voltage_scaling_factor ** 2
        
        return base_power * (1.0 + size_factor * 0.1) * voltage_factor
    
    def calculate_communication_power(self, data_transmitted: int) -> float:
        """Calculate power for communication."""
        base_power = self.profile.communication_nj * data_transmitted
        voltage_factor = self.voltage_scaling_factor ** 2
        
        return base_power * voltage_factor
    
    def update_operating_conditions(self, voltage_v: float, frequency_mhz: float, 
                                  temperature_celsius: float):
        """Update operating conditions affecting power consumption."""
        self.voltage_scaling_factor = voltage_v / self.profile.voltage_v
        self.frequency_scaling_factor = frequency_mhz / self.profile.frequency_mhz
        
        # Temperature effect on leakage (exponential relationship)
        temp_diff = temperature_celsius - 25.0  # Reference temperature
        self.temperature_factor = 1.0 + temp_diff * 0.02  # 2% per degree


class PowerOptimizer:
    """Power optimization engine for neuromorphic systems."""
    
    def __init__(self, power_model: PowerModel, energy_budget: EnergyBudget):
        self.power_model = power_model
        self.energy_budget = energy_budget
        self.current_state = PowerState.ACTIVE
        self.power_history: deque = deque(maxlen=1000)
        self.energy_consumed = 0.0
        self.optimization_strategies: List[Callable] = []
        
        # Adaptive parameters
        self.voltage_levels = [0.8, 0.9, 1.0, 1.1]  # Available voltage levels
        self.frequency_levels = [500, 750, 1000, 1250]  # Available frequency levels (MHz)
        self.current_voltage_idx = 2  # Start at nominal
        self.current_frequency_idx = 2  # Start at nominal
        
        # Activity monitoring
        self.activity_window = deque(maxlen=100)
        self.idle_threshold = 0.1  # 10% activity threshold for sleep
        self.sleep_delay_ms = 10  # Delay before entering sleep
        self.last_activity_time = time.time()
    
    def add_optimization_strategy(self, strategy: Callable):
        """Add custom optimization strategy."""
        self.optimization_strategies.append(strategy)
    
    def monitor_activity(self, spike_count: int, timestamp_s: float):
        """Monitor system activity for power management."""
        self.activity_window.append((timestamp_s, spike_count))
        if spike_count > 0:
            self.last_activity_time = timestamp_s
    
    def get_current_activity_level(self) -> float:
        """Get current activity level (0.0 to 1.0)."""
        if not self.activity_window:
            return 0.0
        
        recent_spikes = sum(count for _, count in self.activity_window[-10:])
        max_possible = len(self.activity_window[-10:]) * 100  # Assume max 100 spikes per window
        
        return min(1.0, recent_spikes / max(1, max_possible))
    
    def should_enter_sleep(self, current_time_s: float) -> bool:
        """Determine if system should enter sleep mode."""
        if self.current_state != PowerState.ACTIVE:
            return False
        
        activity_level = self.get_current_activity_level()
        time_since_activity = current_time_s - self.last_activity_time
        
        return (activity_level < self.idle_threshold and 
                time_since_activity > self.sleep_delay_ms / 1000.0)
    
    def should_wake_up(self, spike_count: int) -> bool:
        """Determine if system should wake up from sleep."""
        return spike_count > 0 and self.current_state in [PowerState.IDLE, PowerState.SLEEP]
    
    def optimize_voltage_frequency(self, target_performance: float, 
                                 current_workload: float) -> Tuple[float, float]:
        """Optimize voltage and frequency based on workload."""
        # Calculate required performance level
        performance_ratio = current_workload / max(target_performance, 0.1)
        
        # Select appropriate voltage and frequency
        if performance_ratio < 0.3:
            # Low workload - reduce voltage and frequency
            self.current_voltage_idx = max(0, self.current_voltage_idx - 1)
            self.current_frequency_idx = max(0, self.current_frequency_idx - 1)
        elif performance_ratio > 0.8:
            # High workload - increase voltage and frequency
            self.current_voltage_idx = min(len(self.voltage_levels) - 1, 
                                         self.current_voltage_idx + 1)
            self.current_frequency_idx = min(len(self.frequency_levels) - 1, 
                                           self.current_frequency_idx + 1)
        
        voltage = self.voltage_levels[self.current_voltage_idx]
        frequency = self.frequency_levels[self.current_frequency_idx]
        
        return voltage, frequency
    
    def calculate_power_consumption(self, operations: Dict[str, Any]) -> float:
        """Calculate total power consumption for given operations."""
        spike_power = self.power_model.calculate_spike_power(
            operations.get('spikes', 0),
            operations.get('neuron_complexity', 1.0)
        )
        
        synapse_power = self.power_model.calculate_synapse_power(
            operations.get('synapse_updates', 0),
            operations.get('weight_precision', 8)
        )
        
        memory_power = self.power_model.calculate_memory_power(
            operations.get('memory_accesses', 0),
            operations.get('data_size', 32)
        )
        
        comm_power = self.power_model.calculate_communication_power(
            operations.get('data_transmitted', 0)
        )
        
        # Convert from nJ to mW (assuming 1ms time window)
        dynamic_power = (spike_power + synapse_power + memory_power + comm_power) / 1e6
        
        # Add static power based on current state
        static_power = self._get_static_power()
        
        total_power = dynamic_power + static_power
        
        # Record power consumption
        self.power_history.append((time.time(), total_power))
        self.energy_consumed += total_power * 0.001  # Convert to mJ (1ms window)
        
        return total_power
    
    def _get_static_power(self) -> float:
        """Get static power consumption based on current state."""
        if hasattr(self.power_model, 'profile'):
            profile = self.power_model.profile
            if self.current_state == PowerState.ACTIVE:
                return profile.idle_power_mw
            elif self.current_state == PowerState.IDLE:
                return profile.idle_power_mw * 0.5
            elif self.current_state == PowerState.SLEEP:
                return profile.sleep_power_mw
            elif self.current_state == PowerState.DEEP_SLEEP:
                return profile.leakage_power_mw
            else:
                return 0.0
        return 0.5  # Default idle power
    
    def optimize_for_energy_budget(self, remaining_time_s: float) -> Dict[str, Any]:
        """Optimize system configuration for remaining energy budget."""
        remaining_energy = self.energy_budget.total_budget_mj - self.energy_consumed
        remaining_ratio = remaining_energy / self.energy_budget.total_budget_mj
        
        recommendations = {
            "voltage_scaling": 1.0,
            "frequency_scaling": 1.0,
            "sleep_threshold": self.idle_threshold,
            "precision_reduction": 1.0,
            "communication_throttling": 1.0
        }
        
        if remaining_ratio < self.energy_budget.critical_threshold:
            # Critical energy level - aggressive optimization
            recommendations.update({
                "voltage_scaling": 0.8,
                "frequency_scaling": 0.6,
                "sleep_threshold": 0.05,  # More aggressive sleep
                "precision_reduction": 0.5,  # Reduce weight precision
                "communication_throttling": 0.3
            })
            logger.warning("Critical energy level - applying aggressive power optimization")
            
        elif remaining_ratio < self.energy_budget.warning_threshold:
            # Warning level - moderate optimization
            recommendations.update({
                "voltage_scaling": 0.9,
                "frequency_scaling": 0.8,
                "sleep_threshold": 0.08,
                "precision_reduction": 0.75,
                "communication_throttling": 0.6
            })
            logger.info("Low energy level - applying moderate power optimization")
        
        # Apply custom optimization strategies
        for strategy in self.optimization_strategies:
            try:
                strategy_recommendations = strategy(remaining_ratio, remaining_time_s)
                if isinstance(strategy_recommendations, dict):
                    recommendations.update(strategy_recommendations)
            except Exception as e:
                logger.error(f"Optimization strategy failed: {e}")
        
        return recommendations
    
    def transition_power_state(self, new_state: PowerState) -> bool:
        """Transition to new power state."""
        if new_state == self.current_state:
            return True
        
        # Validate state transition
        valid_transitions = {
            PowerState.ACTIVE: [PowerState.IDLE, PowerState.SLEEP],
            PowerState.IDLE: [PowerState.ACTIVE, PowerState.SLEEP],
            PowerState.SLEEP: [PowerState.ACTIVE, PowerState.DEEP_SLEEP],
            PowerState.DEEP_SLEEP: [PowerState.SLEEP, PowerState.ACTIVE],
            PowerState.OFF: [PowerState.ACTIVE]
        }
        
        if new_state not in valid_transitions.get(self.current_state, []):
            logger.warning(f"Invalid power state transition: {self.current_state} -> {new_state}")
            return False
        
        old_state = self.current_state
        self.current_state = new_state
        
        logger.info(f"Power state transition: {old_state.value} -> {new_state.value}")
        return True
    
    def get_power_metrics(self) -> PowerMetrics:
        """Get comprehensive power metrics."""
        if not self.power_history:
            return PowerMetrics(0, 0, 0, 0, 0, 25, 1.0, 1000, 0)
        
        recent_power = [power for _, power in self.power_history[-100:]]
        current_power = recent_power[-1] if recent_power else 0
        average_power = np.mean(recent_power) if recent_power else 0
        peak_power = np.max(recent_power) if recent_power else 0
        
        # Calculate efficiency (operations per mJ)
        total_operations = sum(1 for _, _ in self.power_history)  # Simplified
        efficiency = total_operations / max(self.energy_consumed, 0.001)
        
        # Get current operating conditions
        voltage = self.voltage_levels[self.current_voltage_idx]
        frequency = self.frequency_levels[self.current_frequency_idx]
        
        # Estimate utilization based on activity
        utilization = self.get_current_activity_level() * 100
        
        return PowerMetrics(
            current_power_mw=current_power,
            average_power_mw=average_power,
            peak_power_mw=peak_power,
            total_energy_mj=self.energy_consumed,
            efficiency_ops_per_mj=efficiency,
            temperature_celsius=25.0,  # Would be measured in real system
            voltage_v=voltage,
            frequency_mhz=frequency,
            utilization_percent=utilization
        )
    
    def get_energy_status(self) -> Dict[str, Any]:
        """Get energy budget status."""
        remaining_energy = self.energy_budget.total_budget_mj - self.energy_consumed
        remaining_ratio = remaining_energy / self.energy_budget.total_budget_mj
        
        status = "normal"
        if remaining_ratio < self.energy_budget.critical_threshold:
            status = "critical"
        elif remaining_ratio < self.energy_budget.warning_threshold:
            status = "warning"
        
        return {
            "total_budget_mj": self.energy_budget.total_budget_mj,
            "consumed_mj": self.energy_consumed,
            "remaining_mj": remaining_energy,
            "remaining_ratio": remaining_ratio,
            "status": status,
            "estimated_runtime_s": remaining_energy / max(self.get_power_metrics().average_power_mw, 0.001) * 1000
        }


class AdaptivePowerManager:
    """Adaptive power management system."""
    
    def __init__(self, power_optimizer: PowerOptimizer):
        self.optimizer = power_optimizer
        self.adaptation_history: deque = deque(maxlen=100)
        self.performance_targets = {
            "latency_ms": 1.0,
            "throughput_ops_per_s": 1000,
            "accuracy_percent": 95.0
        }
        self.current_performance = {
            "latency_ms": 1.0,
            "throughput_ops_per_s": 1000,
            "accuracy_percent": 95.0
        }
    
    def set_performance_targets(self, targets: Dict[str, float]):
        """Set performance targets for adaptation."""
        self.performance_targets.update(targets)
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update current performance metrics."""
        self.current_performance.update(metrics)
        self.adaptation_history.append((time.time(), dict(metrics)))
    
    def adapt_power_settings(self) -> Dict[str, Any]:
        """Adapt power settings based on performance feedback."""
        # Calculate performance gaps
        latency_gap = (self.current_performance["latency_ms"] - 
                      self.performance_targets["latency_ms"]) / self.performance_targets["latency_ms"]
        
        throughput_gap = (self.performance_targets["throughput_ops_per_s"] - 
                         self.current_performance["throughput_ops_per_s"]) / self.performance_targets["throughput_ops_per_s"]
        
        accuracy_gap = (self.performance_targets["accuracy_percent"] - 
                       self.current_performance["accuracy_percent"]) / self.performance_targets["accuracy_percent"]
        
        # Determine adaptation strategy
        adaptations = {}
        
        if latency_gap > 0.1:  # Latency too high
            adaptations["increase_frequency"] = True
            adaptations["reduce_sleep_threshold"] = True
        elif latency_gap < -0.1:  # Latency too low (can reduce power)
            adaptations["decrease_frequency"] = True
            adaptations["increase_sleep_threshold"] = True
        
        if throughput_gap > 0.1:  # Throughput too low
            adaptations["increase_voltage"] = True
            adaptations["increase_parallelism"] = True
        elif throughput_gap < -0.1:  # Throughput too high (can reduce power)
            adaptations["decrease_voltage"] = True
        
        if accuracy_gap > 0.05:  # Accuracy too low
            adaptations["increase_precision"] = True
            adaptations["reduce_approximations"] = True
        elif accuracy_gap < -0.05:  # Accuracy too high (can reduce precision)
            adaptations["decrease_precision"] = True
            adaptations["increase_approximations"] = True
        
        return adaptations
    
    def execute_adaptations(self, adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """Execute power adaptations."""
        results = {}
        
        # Voltage/frequency adaptations
        if adaptations.get("increase_frequency"):
            self.optimizer.current_frequency_idx = min(
                len(self.optimizer.frequency_levels) - 1,
                self.optimizer.current_frequency_idx + 1
            )
            results["frequency_increased"] = True
        
        if adaptations.get("decrease_frequency"):
            self.optimizer.current_frequency_idx = max(
                0, self.optimizer.current_frequency_idx - 1
            )
            results["frequency_decreased"] = True
        
        if adaptations.get("increase_voltage"):
            self.optimizer.current_voltage_idx = min(
                len(self.optimizer.voltage_levels) - 1,
                self.optimizer.current_voltage_idx + 1
            )
            results["voltage_increased"] = True
        
        if adaptations.get("decrease_voltage"):
            self.optimizer.current_voltage_idx = max(
                0, self.optimizer.current_voltage_idx - 1
            )
            results["voltage_decreased"] = True
        
        # Sleep threshold adaptations
        if adaptations.get("reduce_sleep_threshold"):
            self.optimizer.idle_threshold = max(0.01, self.optimizer.idle_threshold * 0.8)
            results["sleep_threshold_reduced"] = True
        
        if adaptations.get("increase_sleep_threshold"):
            self.optimizer.idle_threshold = min(0.5, self.optimizer.idle_threshold * 1.2)
            results["sleep_threshold_increased"] = True
        
        return results


__all__ = [
    "PowerState",
    "PowerProfile",
    "EnergyBudget", 
    "PowerMetrics",
    "PowerModel",
    "DetailedPowerModel",
    "PowerOptimizer",
    "AdaptivePowerManager"
]