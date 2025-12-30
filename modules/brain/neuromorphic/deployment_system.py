"""Production deployment and monitoring system for neuromorphic computing.

This module provides comprehensive deployment, monitoring, and management
capabilities for neuromorphic systems in production environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .hardware_drivers import NeuromorphicHAL, HardwareStatus, PowerMetrics
from .event_driven_core import EventDrivenNetwork, NeuromorphicEvent
from .power_optimization import PowerOptimizer, AdaptivePowerManager

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states."""
    INITIALIZING = "initializing"
    DEPLOYED = "deployed"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    version: str
    hardware_platform: str
    network_config: Dict[str, Any]
    power_budget_mw: float = 1000.0
    max_runtime_hours: float = 24.0
    monitoring_interval_ms: int = 100
    auto_restart: bool = True
    backup_enabled: bool = True
    logging_level: str = "INFO"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    throughput_events_per_sec: float
    latency_ms: float
    accuracy_percent: float
    power_efficiency_ops_per_mw: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate_percent: float
    uptime_hours: float


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str
    hardware_status: str
    software_status: str
    power_status: str
    temperature_celsius: float
    error_count: int
    warning_count: int
    last_error: Optional[str] = None
    last_warning: Optional[str] = None


class NeuromorphicDeployment:
    """Production neuromorphic system deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.status = DeploymentStatus.INITIALIZING
        self.hal = NeuromorphicHAL()
        self.network: Optional[EventDrivenNetwork] = None
        self.power_optimizer: Optional[PowerOptimizer] = None
        self.power_manager: Optional[AdaptivePowerManager] = None
        
        # Monitoring
        self.metrics_history: List[Tuple[float, PerformanceMetrics]] = []
        self.health_history: List[Tuple[float, SystemHealth]] = []
        self.event_log: List[Dict[str, Any]] = []
        
        # Runtime tracking
        self.start_time = 0.0
        self.total_events_processed = 0
        self.total_errors = 0
        self.total_warnings = 0
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup deployment logging."""
        log_level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format=f'%(asctime)s - {self.config.name} - %(levelname)s - %(message)s'
        )
    
    async def initialize(self) -> bool:
        """Initialize the deployment."""
        try:
            logger.info(f"Initializing deployment: {self.config.name} v{self.config.version}")
            
            # Add hardware platform
            success = await self.hal.add_platform(
                "primary", 
                self.config.hardware_platform
            )
            
            if not success:
                logger.error("Failed to initialize hardware platform")
                self.status = DeploymentStatus.ERROR
                return False
            
            # Set active platform
            self.hal.set_active_platform("primary")
            
            # Create network
            n_neurons = self.config.network_config.get("n_neurons", 100)
            self.network = EventDrivenNetwork(n_neurons)
            
            # Configure network connections
            self._configure_network()
            
            # Initialize power management
            self._initialize_power_management()
            
            self.status = DeploymentStatus.DEPLOYED
            self.start_time = time.time()
            
            logger.info("Deployment initialized successfully")
            self._notify_status_change()
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment initialization failed: {e}")
            self.status = DeploymentStatus.ERROR
            self._notify_error(str(e))
            return False
    
    def _configure_network(self):
        """Configure network topology and connections."""
        if not self.network:
            return
        
        # Configure connections from config
        connections = self.config.network_config.get("connections", [])
        for conn in connections:
            self.network.add_connection(
                conn["source"],
                conn["target"], 
                conn.get("weight", 1.0),
                conn.get("delay_ns", 1000)
            )
        
        logger.info(f"Configured {len(connections)} network connections")
    
    def _initialize_power_management(self):
        """Initialize power optimization and management."""
        from .power_optimization import PowerProfile, EnergyBudget, DetailedPowerModel
        
        # Create power profile
        profile = PowerProfile(
            spike_processing_nj=0.1,
            synapse_update_nj=0.05,
            memory_access_nj=0.02,
            communication_nj=0.15,
            idle_power_mw=self.config.power_budget_mw * 0.1
        )
        
        # Create energy budget
        budget = EnergyBudget(
            total_budget_mj=self.config.power_budget_mw * self.config.max_runtime_hours * 3.6,
            time_window_s=3600.0,  # 1 hour
            max_power_mw=self.config.power_budget_mw
        )
        
        # Create power model and optimizer
        power_model = DetailedPowerModel(profile)
        self.power_optimizer = PowerOptimizer(power_model, budget)
        self.power_manager = AdaptivePowerManager(self.power_optimizer)
        
        logger.info("Power management initialized")
    
    async def start(self) -> bool:
        """Start the deployment."""
        if self.status != DeploymentStatus.DEPLOYED:
            logger.error("Cannot start - deployment not properly initialized")
            return False
        
        try:
            self.status = DeploymentStatus.RUNNING
            logger.info("Starting deployment")
            
            # Start monitoring
            asyncio.create_task(self._monitoring_loop())
            
            # Start power management
            if self.power_optimizer:
                asyncio.create_task(self._power_management_loop())
            
            self._notify_status_change()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start deployment: {e}")
            self.status = DeploymentStatus.ERROR
            self._notify_error(str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop the deployment."""
        try:
            logger.info("Stopping deployment")
            self.status = DeploymentStatus.SHUTDOWN
            
            # Save final metrics
            await self._save_metrics()
            
            # Cleanup hardware
            for platform in list(self.hal._drivers.keys()):
                await self.hal.remove_platform(platform)
            
            self._notify_status_change()
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    async def process_events(self, events: List[NeuromorphicEvent], 
                           duration_ms: int = 1000) -> Dict[str, Any]:
        """Process neuromorphic events."""
        if self.status != DeploymentStatus.RUNNING or not self.network:
            raise RuntimeError("Deployment not running")
        
        try:
            # Inject events into network
            for event in events:
                self.network.core.inject_event(event)
            
            # Run network
            start_time = time.time()
            results = await self.network.run(duration_ms * 1_000_000)  # Convert to ns
            processing_time = time.time() - start_time
            
            # Update metrics
            self.total_events_processed += len(events)
            
            # Calculate performance metrics
            throughput = len(events) / max(processing_time, 0.001)
            latency = processing_time * 1000  # Convert to ms
            
            # Update power consumption
            if self.power_optimizer:
                power_ops = {
                    'spikes': results.get('events_processed', 0),
                    'synapse_updates': results.get('events_processed', 0) // 2,
                    'memory_accesses': results.get('events_processed', 0),
                    'data_transmitted': len(events)
                }
                power_consumption = self.power_optimizer.calculate_power_consumption(power_ops)
            
            return {
                'events_processed': len(events),
                'processing_time_ms': processing_time * 1000,
                'throughput_events_per_sec': throughput,
                'latency_ms': latency,
                'network_results': results
            }
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Event processing failed: {e}")
            self._notify_error(str(e))
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.status == DeploymentStatus.RUNNING:
            try:
                # Collect metrics
                metrics = self._collect_performance_metrics()
                health = self._assess_system_health()
                
                # Store metrics
                timestamp = time.time()
                self.metrics_history.append((timestamp, metrics))
                self.health_history.append((timestamp, health))
                
                # Limit history size
                if len(self.metrics_history) > 10000:
                    self.metrics_history = self.metrics_history[-5000:]
                if len(self.health_history) > 10000:
                    self.health_history = self.health_history[-5000:]
                
                # Notify callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics, health)
                    except Exception as e:
                        logger.warning(f"Metrics callback failed: {e}")
                
                # Check for issues
                if health.overall_status == "error":
                    self._handle_system_error(health)
                elif health.overall_status == "warning":
                    self._handle_system_warning(health)
                
                await asyncio.sleep(self.config.monitoring_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        current_time = time.time()
        uptime = (current_time - self.start_time) / 3600.0  # Convert to hours
        
        # Calculate throughput from recent history
        recent_events = sum(1 for t, _ in self.metrics_history[-10:] if current_time - t < 10)
        throughput = recent_events / 10.0  # Events per second over last 10 seconds
        
        # Get power metrics
        power_metrics = self.hal.get_power_metrics() or PowerMetrics(0, 0, 0, 0, 0, 25, 1.0, 1000, 0)
        
        # Calculate efficiency
        efficiency = throughput / max(power_metrics.current_power_mw, 0.001)
        
        return PerformanceMetrics(
            throughput_events_per_sec=throughput,
            latency_ms=1.0,  # Would be measured from actual processing
            accuracy_percent=95.0,  # Would be calculated from validation
            power_efficiency_ops_per_mw=efficiency,
            memory_usage_mb=50.0,  # Would be measured from system
            cpu_usage_percent=power_metrics.utilization_percent,
            error_rate_percent=(self.total_errors / max(self.total_events_processed, 1)) * 100,
            uptime_hours=uptime
        )
    
    def _assess_system_health(self) -> SystemHealth:
        """Assess overall system health."""
        # Get hardware status
        platforms = self.hal.list_platforms()
        hardware_status = "healthy"
        for platform_info in platforms.values():
            if platform_info["status"] == "error":
                hardware_status = "error"
                break
            elif platform_info["status"] in ["connecting", "maintenance"]:
                hardware_status = "warning"
        
        # Assess software status
        software_status = "healthy"
        if self.status == DeploymentStatus.ERROR:
            software_status = "error"
        elif self.status in [DeploymentStatus.PAUSED, DeploymentStatus.MAINTENANCE]:
            software_status = "warning"
        
        # Assess power status
        power_status = "healthy"
        if self.power_optimizer:
            energy_status = self.power_optimizer.get_energy_status()
            if energy_status["status"] == "critical":
                power_status = "error"
            elif energy_status["status"] == "warning":
                power_status = "warning"
        
        # Overall status
        if any(status == "error" for status in [hardware_status, software_status, power_status]):
            overall_status = "error"
        elif any(status == "warning" for status in [hardware_status, software_status, power_status]):
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Get temperature
        power_metrics = self.hal.get_power_metrics()
        temperature = power_metrics.temperature_celsius if power_metrics else 25.0
        
        return SystemHealth(
            overall_status=overall_status,
            hardware_status=hardware_status,
            software_status=software_status,
            power_status=power_status,
            temperature_celsius=temperature,
            error_count=self.total_errors,
            warning_count=self.total_warnings
        )
    
    async def _power_management_loop(self):
        """Power management loop."""
        while self.status == DeploymentStatus.RUNNING and self.power_manager:
            try:
                # Get current performance
                metrics = self._collect_performance_metrics()
                
                # Update performance metrics for adaptation
                self.power_manager.update_performance_metrics({
                    "latency_ms": metrics.latency_ms,
                    "throughput_ops_per_s": metrics.throughput_events_per_sec,
                    "accuracy_percent": metrics.accuracy_percent
                })
                
                # Get adaptation recommendations
                adaptations = self.power_manager.adapt_power_settings()
                
                # Execute adaptations
                if adaptations:
                    results = self.power_manager.execute_adaptations(adaptations)
                    if results:
                        logger.info(f"Applied power adaptations: {results}")
                
                await asyncio.sleep(5.0)  # Adapt every 5 seconds
                
            except Exception as e:
                logger.error(f"Power management error: {e}")
                await asyncio.sleep(1.0)
    
    def _handle_system_error(self, health: SystemHealth):
        """Handle system error condition."""
        logger.error(f"System error detected: {health.last_error}")
        
        if self.config.auto_restart:
            logger.info("Attempting automatic restart")
            asyncio.create_task(self._restart_system())
    
    def _handle_system_warning(self, health: SystemHealth):
        """Handle system warning condition."""
        self.total_warnings += 1
        logger.warning(f"System warning: {health.last_warning}")
    
    async def _restart_system(self):
        """Restart the system."""
        try:
            logger.info("Restarting system")
            await self.stop()
            await asyncio.sleep(2.0)
            await self.initialize()
            await self.start()
            logger.info("System restart completed")
        except Exception as e:
            logger.error(f"System restart failed: {e}")
    
    async def _save_metrics(self):
        """Save metrics to file."""
        if not self.config.backup_enabled:
            return
        
        try:
            metrics_file = Path(f"{self.config.name}_metrics.json")
            metrics_data = {
                "config": self.config.__dict__,
                "metrics_history": [(t, m.__dict__) for t, m in self.metrics_history],
                "health_history": [(t, h.__dict__) for t, h in self.health_history],
                "event_log": self.event_log,
                "summary": {
                    "total_events_processed": self.total_events_processed,
                    "total_errors": self.total_errors,
                    "total_warnings": self.total_warnings,
                    "uptime_hours": (time.time() - self.start_time) / 3600.0
                }
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def add_status_callback(self, callback: Callable):
        """Add status change callback."""
        self.status_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add metrics callback."""
        self.metrics_callbacks.append(callback)
    
    def _notify_status_change(self):
        """Notify status change callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")
    
    def _notify_error(self, error_msg: str):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_msg)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "status": self.status.value,
            "uptime_hours": (time.time() - self.start_time) / 3600.0 if self.start_time else 0,
            "total_events_processed": self.total_events_processed,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings
        }


__all__ = [
    "DeploymentStatus",
    "DeploymentConfig",
    "PerformanceMetrics", 
    "SystemHealth",
    "NeuromorphicDeployment"
]