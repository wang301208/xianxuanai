"""Comprehensive test suite for production neuromorphic system.

This module provides unit tests, integration tests, and system tests
for the production-grade neuromorphic computing framework.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from ..hardware_drivers import NeuromorphicHAL, HardwareStatus, PowerMetrics
from ..event_driven_core import EventDrivenNetwork, NeuromorphicEvent, EventType
from ..power_optimization import PowerOptimizer, PowerProfile, EnergyBudget, DetailedPowerModel
from ..advanced_learning import OnlineLearningEngine, STDPRule, LearningParameters
from ..deployment_system import NeuromorphicDeployment, DeploymentConfig, DeploymentStatus
from ..benchmarks import NeuromorphicBenchmarkSuite, BenchmarkConfig, BenchmarkType


class TestHardwareDrivers:
    """Test hardware driver functionality."""
    
    def test_hal_initialization(self):
        """Test HAL initialization."""
        hal = NeuromorphicHAL()
        assert hal is not None
        assert len(hal._drivers) == 0
        assert hal._active_platform is None
    
    @pytest.mark.asyncio
    async def test_platform_management(self):
        """Test platform addition and removal."""
        hal = NeuromorphicHAL()
        
        # Add platform
        success = await hal.add_platform("test_platform", "simulation")
        assert success
        assert "test_platform" in hal._drivers
        
        # Set active platform
        hal.set_active_platform("test_platform")
        assert hal._active_platform == "test_platform"
        
        # Remove platform
        success = await hal.remove_platform("test_platform")
        assert success
        assert "test_platform" not in hal._drivers
    
    def test_power_metrics(self):
        """Test power metrics collection."""
        hal = NeuromorphicHAL()
        
        # Should return None when no active platform
        metrics = hal.get_power_metrics()
        assert metrics is None
        
        # Mock active platform
        hal._active_platform = "test"
        mock_driver = Mock()
        mock_driver.get_power_metrics.return_value = PowerMetrics(
            current_power_mw=100.0,
            average_power_mw=95.0,
            peak_power_mw=120.0,
            energy_consumed_mj=50.0,
            utilization_percent=75.0,
            temperature_celsius=45.0,
            voltage_v=3.3,
            frequency_mhz=1000,
            efficiency_percent=85.0
        )
        hal._drivers["test"] = mock_driver
        
        metrics = hal.get_power_metrics()
        assert metrics is not None
        assert metrics.current_power_mw == 100.0
        assert metrics.temperature_celsius == 45.0


class TestEventDrivenCore:
    """Test event-driven processing core."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        network = EventDrivenNetwork(100)
        assert network.n_neurons == 100
        assert network.core is not None
        assert len(network.topology.connections) == 0
    
    def test_connection_management(self):
        """Test connection addition and management."""
        network = EventDrivenNetwork(10)
        
        # Add connection
        network.add_connection(0, 1, 0.5, 1000)
        assert len(network.topology.connections) == 1
        
        connection = network.topology.connections[0]
        assert connection.source_id == 0
        assert connection.target_id == 1
        assert connection.weight == 0.5
        assert connection.delay_ns == 1000
    
    @pytest.mark.asyncio
    async def test_event_processing(self):
        """Test event processing."""
        network = EventDrivenNetwork(10)
        network.add_connection(0, 1, 0.5, 1000)
        
        # Create test event
        event = NeuromorphicEvent(
            event_type=EventType.SPIKE,
            timestamp_ns=1000000,
            neuron_id=0,
            data={"amplitude": 1.0}
        )
        
        # Inject event
        network.core.inject_event(event)
        
        # Run network
        results = await network.run(10_000_000)  # 10ms
        
        assert "events_processed" in results
        assert results["events_processed"] >= 1


class TestPowerOptimization:
    """Test power optimization functionality."""
    
    def test_power_profile_creation(self):
        """Test power profile creation."""
        profile = PowerProfile(
            spike_processing_nj=0.1,
            synapse_update_nj=0.05,
            memory_access_nj=0.02,
            communication_nj=0.15,
            idle_power_mw=10.0
        )
        
        assert profile.spike_processing_nj == 0.1
        assert profile.idle_power_mw == 10.0
    
    def test_energy_budget(self):
        """Test energy budget management."""
        budget = EnergyBudget(
            total_budget_mj=1000.0,
            time_window_s=3600.0,
            max_power_mw=500.0
        )
        
        assert budget.total_budget_mj == 1000.0
        assert budget.get_remaining_budget() == 1000.0
        
        # Consume energy
        budget.consume_energy(100.0)
        assert budget.get_remaining_budget() == 900.0
    
    def test_power_optimizer(self):
        """Test power optimizer."""
        profile = PowerProfile()
        budget = EnergyBudget(1000.0, 3600.0, 500.0)
        power_model = DetailedPowerModel(profile)
        optimizer = PowerOptimizer(power_model, budget)
        
        # Test power calculation
        operations = {
            'spikes': 1000,
            'synapse_updates': 500,
            'memory_accesses': 200,
            'data_transmitted': 100
        }
        
        power_consumption = optimizer.calculate_power_consumption(operations)
        assert power_consumption > 0
        
        # Test optimization
        optimizations = optimizer.optimize_power_consumption(operations)
        assert isinstance(optimizations, list)


class TestAdvancedLearning:
    """Test advanced learning algorithms."""
    
    def test_stdp_rule(self):
        """Test STDP plasticity rule."""
        stdp = STDPRule(a_plus=0.01, a_minus=0.012)
        
        # Test LTP (pre before post)
        from ..advanced_learning import SynapticTrace
        trace = SynapticTrace()
        
        new_weight, updated_trace = stdp.update_weight(
            pre_spike_time=10.0,
            post_spike_time=15.0,  # Post after pre
            current_weight=0.5,
            trace=trace
        )
        
        # Should increase weight (LTP)
        assert new_weight > 0.5
        assert updated_trace.last_update_time == 15.0
    
    def test_online_learning_engine(self):
        """Test online learning engine."""
        stdp = STDPRule()
        params = LearningParameters()
        engine = OnlineLearningEngine(stdp, params)
        
        # Add synapse
        engine.add_synapse(0, 1, 0.5)
        assert engine.get_weight(0, 1) == 0.5
        
        # Process spike pair
        new_weight = engine.process_spike_pair(0, 1, 10.0, 15.0)
        assert new_weight != 0.5  # Weight should change
        
        # Get statistics
        stats = engine.get_learning_statistics()
        assert stats["total_updates"] == 1
        assert stats["total_synapses"] == 1


class TestDeploymentSystem:
    """Test deployment system functionality."""
    
    def test_deployment_config(self):
        """Test deployment configuration."""
        config = DeploymentConfig(
            name="test_deployment",
            version="1.0.0",
            hardware_platform="simulation",
            network_config={"n_neurons": 100},
            power_budget_mw=500.0
        )
        
        assert config.name == "test_deployment"
        assert config.hardware_platform == "simulation"
        assert config.power_budget_mw == 500.0
    
    @pytest.mark.asyncio
    async def test_deployment_lifecycle(self):
        """Test deployment initialization and lifecycle."""
        config = DeploymentConfig(
            name="test_deployment",
            version="1.0.0",
            hardware_platform="simulation",
            network_config={"n_neurons": 10, "connections": []},
            power_budget_mw=100.0,
            max_runtime_hours=1.0
        )
        
        deployment = NeuromorphicDeployment(config)
        
        # Test initialization
        success = await deployment.initialize()
        assert success
        assert deployment.status == DeploymentStatus.DEPLOYED
        
        # Test start
        success = await deployment.start()
        assert success
        assert deployment.status == DeploymentStatus.RUNNING
        
        # Test event processing
        events = [
            NeuromorphicEvent(
                event_type=EventType.SPIKE,
                timestamp_ns=1000000,
                neuron_id=0,
                data={"amplitude": 1.0}
            )
        ]
        
        results = await deployment.process_events(events, duration_ms=10)
        assert "events_processed" in results
        assert results["events_processed"] == 1
        
        # Test stop
        success = await deployment.stop()
        assert success
        assert deployment.status == DeploymentStatus.SHUTDOWN
    
    def test_status_callbacks(self):
        """Test status change callbacks."""
        config = DeploymentConfig(
            name="test_deployment",
            version="1.0.0",
            hardware_platform="simulation",
            network_config={"n_neurons": 10},
            power_budget_mw=100.0
        )
        
        deployment = NeuromorphicDeployment(config)
        
        # Add callback
        callback_called = False
        def status_callback(status):
            nonlocal callback_called
            callback_called = True
        
        deployment.add_status_callback(status_callback)
        
        # Trigger status change
        deployment._notify_status_change()
        assert callback_called


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""
    
    def test_benchmark_config(self):
        """Test benchmark configuration."""
        config = BenchmarkConfig(
            name="test_benchmark",
            benchmark_type=BenchmarkType.PERFORMANCE,
            duration_seconds=10.0,
            n_trials=3
        )
        
        assert config.name == "test_benchmark"
        assert config.benchmark_type == BenchmarkType.PERFORMANCE
        assert config.duration_seconds == 10.0
        assert config.n_trials == 3
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        suite = NeuromorphicBenchmarkSuite()
        assert len(suite.results) == 0
        assert suite.deployment is None
    
    def test_test_deployment_creation(self):
        """Test test deployment creation."""
        suite = NeuromorphicBenchmarkSuite()
        deployment = suite.create_test_deployment(100)
        
        assert deployment is not None
        assert deployment.config.name == "benchmark_system"
        assert deployment.config.network_config["n_neurons"] == 100
    
    def test_event_generation(self):
        """Test test event generation."""
        suite = NeuromorphicBenchmarkSuite()
        events = suite._generate_test_events(10, 100)
        
        assert len(events) == 10
        for event in events:
            assert event.event_type == EventType.SPIKE
            assert 0 <= event.neuron_id < 100
            assert event.timestamp_ns > 0
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test performance benchmark execution."""
        suite = NeuromorphicBenchmarkSuite()
        
        config = BenchmarkConfig(
            name="test_performance",
            benchmark_type=BenchmarkType.PERFORMANCE,
            duration_seconds=1.0,  # Short duration for testing
            n_trials=1,
            warmup_seconds=0.1,
            parameters={"n_neurons": 10, "events_per_batch": 5}
        )
        
        result = await suite.run_benchmark(config)
        
        assert result.success
        assert result.benchmark_type == BenchmarkType.PERFORMANCE
        assert "avg_throughput_events_per_sec" in result.metrics
        assert result.metrics["avg_throughput_events_per_sec"] > 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_system_integration(self):
        """Test complete system integration."""
        # Create deployment
        config = DeploymentConfig(
            name="integration_test",
            version="1.0.0",
            hardware_platform="simulation",
            network_config={
                "n_neurons": 50,
                "connections": [
                    {"source": i, "target": (i + 1) % 50, "weight": 0.5, "delay_ns": 1000}
                    for i in range(50)
                ]
            },
            power_budget_mw=200.0,
            max_runtime_hours=1.0,
            monitoring_interval_ms=100
        )
        
        deployment = NeuromorphicDeployment(config)
        
        try:
            # Initialize
            success = await deployment.initialize()
            assert success
            
            # Start
            success = await deployment.start()
            assert success
            
            # Create learning engine
            stdp_rule = STDPRule(a_plus=0.01, a_minus=0.012)
            learning_params = LearningParameters(learning_rate=0.01)
            learning_engine = OnlineLearningEngine(stdp_rule, learning_params)
            
            # Add synapses
            for i in range(49):
                learning_engine.add_synapse(i, i + 1, 0.5)
            
            # Process events with learning
            events = []
            for i in range(10):
                event = NeuromorphicEvent(
                    event_type=EventType.SPIKE,
                    timestamp_ns=1000000 + i * 1000000,
                    neuron_id=i % 50,
                    data={"amplitude": 1.0}
                )
                events.append(event)
            
            # Process events
            results = await deployment.process_events(events, duration_ms=100)
            assert results["events_processed"] == 10
            
            # Update learning
            for i in range(9):
                pre_neuron = i % 50
                post_neuron = (i + 1) % 50
                pre_time = i * 1.0
                post_time = (i + 1) * 1.0
                
                new_weight = learning_engine.process_spike_pair(
                    pre_neuron, post_neuron, pre_time, post_time
                )
                assert new_weight > 0
            
            # Get system status
            status = deployment.get_status()
            assert status["name"] == "integration_test"
            assert status["status"] == "running"
            assert status["total_events_processed"] >= 10
            
            # Get learning statistics
            learning_stats = learning_engine.get_learning_statistics()
            assert learning_stats["total_updates"] == 9
            assert learning_stats["total_synapses"] == 49
            
        finally:
            # Cleanup
            await deployment.stop()
    
    @pytest.mark.asyncio
    async def test_benchmark_integration(self):
        """Test benchmark suite integration."""
        suite = NeuromorphicBenchmarkSuite()
        
        # Run a quick benchmark
        config = BenchmarkConfig(
            name="integration_benchmark",
            benchmark_type=BenchmarkType.PERFORMANCE,
            duration_seconds=0.5,  # Very short for testing
            n_trials=1,
            warmup_seconds=0.1,
            parameters={"n_neurons": 20, "events_per_batch": 10}
        )
        
        result = await suite.run_benchmark(config)
        
        assert result.success
        assert result.benchmark_name == "integration_benchmark"
        assert "avg_throughput_events_per_sec" in result.metrics
        
        # Generate report
        suite.results.append(result)
        report = suite.generate_benchmark_report()
        
        assert "Neuromorphic Computing Benchmark Report" in report
        assert "integration_benchmark" in report
        assert "Success" in report


# Pytest configuration
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])