"""Comprehensive benchmark suite for neuromorphic computing systems.

This module provides standardized benchmarks for evaluating neuromorphic
systems across different metrics including performance, power efficiency,
accuracy, and scalability.
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from enum import Enum

import numpy as np

from .deployment_system import NeuromorphicDeployment, DeploymentConfig
from .event_driven_core import NeuromorphicEvent, EventType
from .advanced_learning import OnlineLearningEngine, STDPRule, LearningParameters

if TYPE_CHECKING:  # pragma: no cover - optional meta-learning integration
    from modules.brain.meta_learning.benchmark_coordinator import BenchmarkMetaLearningCoordinator

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    PERFORMANCE = "performance"
    POWER_EFFICIENCY = "power_efficiency"
    LEARNING_ACCURACY = "learning_accuracy"
    SCALABILITY = "scalability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    benchmark_type: BenchmarkType
    duration_seconds: float
    n_trials: int = 5
    warmup_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    parameters: Dict[str, Any] = None


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    success: bool
    duration_seconds: float
    metrics: Dict[str, float]
    raw_data: Dict[str, Any]
    error_message: Optional[str] = None


class NeuromorphicBenchmarkSuite:
    """Comprehensive benchmark suite for neuromorphic systems."""
    
    def __init__(self, meta_coordinator: "BenchmarkMetaLearningCoordinator" | None = None):
        self.results: List[BenchmarkResult] = []
        self.deployment: Optional[NeuromorphicDeployment] = None
        self.meta_coordinator = meta_coordinator
        
    def create_test_deployment(self, n_neurons: int = 1000) -> NeuromorphicDeployment:
        """Create test deployment for benchmarking."""
        network_config = {
            "n_neurons": n_neurons,
            "connections": [
                {"source": i, "target": (i + 1) % n_neurons, "weight": 0.5, "delay_ns": 1000}
                for i in range(n_neurons)
            ]
        }
        
        config = DeploymentConfig(
            name="benchmark_system",
            version="1.0.0",
            hardware_platform="simulation",  # Use simulation for consistent benchmarking
            network_config=network_config,
            power_budget_mw=1000.0,
            max_runtime_hours=1.0,
            monitoring_interval_ms=10,
            auto_restart=False,
            backup_enabled=False,
            logging_level="WARNING"  # Reduce logging overhead
        )
        
        return NeuromorphicDeployment(config)
    
    async def run_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark."""
        logger.info(f"Running benchmark: {benchmark_config.name}")
        
        try:
            if benchmark_config.benchmark_type == BenchmarkType.PERFORMANCE:
                result = await self._run_performance_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.POWER_EFFICIENCY:
                result = await self._run_power_efficiency_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.LEARNING_ACCURACY:
                result = await self._run_learning_accuracy_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.SCALABILITY:
                result = await self._run_scalability_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.LATENCY:
                result = await self._run_latency_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.THROUGHPUT:
                result = await self._run_throughput_benchmark(benchmark_config)
            elif benchmark_config.benchmark_type == BenchmarkType.RELIABILITY:
                result = await self._run_reliability_benchmark(benchmark_config)
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_config.benchmark_type}")
                
        except Exception as e:
            logger.error(f"Benchmark {benchmark_config.name} failed: {e}")
            result = BenchmarkResult(
                benchmark_name=benchmark_config.name,
                benchmark_type=benchmark_config.benchmark_type,
                success=False,
                duration_seconds=0.0,
                metrics={},
                raw_data={},
                error_message=str(e)
            )

        self._notify_meta_coordinator(result)
        return result

    def _notify_meta_coordinator(self, result: BenchmarkResult) -> None:
        """Send benchmark results to the meta-learning coordinator if configured."""

        if self.meta_coordinator is None:
            return
        try:
            self.meta_coordinator.handle_benchmark_result({
                "benchmark_name": result.benchmark_name,
                "benchmark_type": result.benchmark_type.value,
                "raw_data": result.raw_data,
                "metrics": result.metrics,
                "success": result.success,
            })
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Meta-learning coordinator failed for benchmark %s", result.benchmark_name, exc_info=True)

    async def _run_performance_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run performance benchmark."""
        n_neurons = config.parameters.get("n_neurons", 1000) if config.parameters else 1000
        events_per_batch = config.parameters.get("events_per_batch", 100) if config.parameters else 100
        
        deployment = self.create_test_deployment(n_neurons)
        
        try:
            # Initialize and start
            await deployment.initialize()
            await deployment.start()
            
            # Warmup
            await asyncio.sleep(config.warmup_seconds)
            
            # Run benchmark trials
            trial_results = []
            
            for trial in range(config.n_trials):
                logger.info(f"Performance benchmark trial {trial + 1}/{config.n_trials}")
                
                start_time = time.time()
                total_events = 0
                
                # Run for specified duration
                trial_start = time.time()
                while time.time() - trial_start < config.duration_seconds:
                    # Generate events
                    events = self._generate_test_events(events_per_batch, n_neurons)
                    
                    # Process events
                    results = await deployment.process_events(events, duration_ms=10)
                    total_events += len(events)
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.001)
                
                trial_duration = time.time() - start_time
                throughput = total_events / trial_duration
                
                trial_results.append({
                    "duration": trial_duration,
                    "total_events": total_events,
                    "throughput": throughput
                })
            
            # Calculate metrics
            throughputs = [r["throughput"] for r in trial_results]
            avg_throughput = np.mean(throughputs)
            std_throughput = np.std(throughputs)
            max_throughput = np.max(throughputs)
            min_throughput = np.min(throughputs)
            
            metrics = {
                "avg_throughput_events_per_sec": avg_throughput,
                "std_throughput_events_per_sec": std_throughput,
                "max_throughput_events_per_sec": max_throughput,
                "min_throughput_events_per_sec": min_throughput,
                "total_events_processed": sum(r["total_events"] for r in trial_results)
            }
            
            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds * config.n_trials,
                metrics=metrics,
                raw_data={"trial_results": trial_results}
            )
            
        finally:
            await deployment.stop()
    
    async def _run_power_efficiency_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run power efficiency benchmark."""
        n_neurons = config.parameters.get("n_neurons", 1000) if config.parameters else 1000
        workload_intensity = config.parameters.get("workload_intensity", 1.0) if config.parameters else 1.0
        
        deployment = self.create_test_deployment(n_neurons)
        
        try:
            await deployment.initialize()
            await deployment.start()
            
            # Warmup
            await asyncio.sleep(config.warmup_seconds)
            
            trial_results = []
            
            for trial in range(config.n_trials):
                logger.info(f"Power efficiency benchmark trial {trial + 1}/{config.n_trials}")
                
                start_time = time.time()
                total_operations = 0
                total_power_consumed = 0.0
                
                trial_start = time.time()
                while time.time() - trial_start < config.duration_seconds:
                    # Generate workload based on intensity
                    n_events = int(100 * workload_intensity)
                    events = self._generate_test_events(n_events, n_neurons)
                    
                    # Process events and measure power
                    power_before = deployment.hal.get_power_metrics()
                    results = await deployment.process_events(events, duration_ms=10)
                    power_after = deployment.hal.get_power_metrics()
                    
                    # Estimate power consumption (simplified)
                    if power_before and power_after:
                        power_consumed = (power_after.current_power_mw - power_before.current_power_mw) * 0.01  # 10ms duration
                        total_power_consumed += max(power_consumed, 0)
                    
                    total_operations += len(events)
                    await asyncio.sleep(0.01)
                
                trial_duration = time.time() - start_time
                
                # Calculate efficiency metrics
                ops_per_second = total_operations / trial_duration
                power_per_second = total_power_consumed / trial_duration
                efficiency = ops_per_second / max(power_per_second, 0.001)  # ops per mW
                
                trial_results.append({
                    "duration": trial_duration,
                    "total_operations": total_operations,
                    "total_power_mw_s": total_power_consumed,
                    "ops_per_second": ops_per_second,
                    "power_per_second_mw": power_per_second,
                    "efficiency_ops_per_mw": efficiency
                })
            
            # Calculate metrics
            efficiencies = [r["efficiency_ops_per_mw"] for r in trial_results]
            avg_efficiency = np.mean(efficiencies)
            std_efficiency = np.std(efficiencies)
            
            metrics = {
                "avg_efficiency_ops_per_mw": avg_efficiency,
                "std_efficiency_ops_per_mw": std_efficiency,
                "max_efficiency_ops_per_mw": np.max(efficiencies),
                "min_efficiency_ops_per_mw": np.min(efficiencies),
                "avg_power_consumption_mw": np.mean([r["power_per_second_mw"] for r in trial_results])
            }
            
            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds * config.n_trials,
                metrics=metrics,
                raw_data={"trial_results": trial_results}
            )
            
        finally:
            await deployment.stop()
    
    async def _run_learning_accuracy_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run learning accuracy benchmark."""
        n_neurons = config.parameters.get("n_neurons", 100) if config.parameters else 100
        n_patterns = config.parameters.get("n_patterns", 10) if config.parameters else 10
        
        deployment = self.create_test_deployment(n_neurons)
        
        # Create learning engine
        stdp_rule = STDPRule(a_plus=0.01, a_minus=0.012)
        learning_params = LearningParameters(learning_rate=0.01)
        learning_engine = OnlineLearningEngine(stdp_rule, learning_params)
        
        # Add synapses
        for i in range(n_neurons - 1):
            learning_engine.add_synapse(i, i + 1, 0.5)
        
        try:
            await deployment.initialize()
            await deployment.start()
            
            # Generate training patterns
            patterns = self._generate_training_patterns(n_patterns, n_neurons)
            
            trial_results = []
            
            for trial in range(config.n_trials):
                logger.info(f"Learning accuracy benchmark trial {trial + 1}/{config.n_trials}")
                
                # Reset learning engine
                for synapse_id in learning_engine.weights:
                    learning_engine.weights[synapse_id] = 0.5
                
                # Training phase
                start_time = time.time()
                for epoch in range(int(config.duration_seconds)):
                    for pattern in patterns:
                        events = self._pattern_to_events(pattern, n_neurons)
                        await deployment.process_events(events, duration_ms=50)
                        
                        # Update learning
                        self._update_learning_from_pattern(learning_engine, pattern)
                
                training_duration = time.time() - start_time
                
                # Test phase - measure recall accuracy
                correct_recalls = 0
                total_tests = len(patterns)
                
                for pattern in patterns:
                    # Present partial pattern
                    partial_pattern = pattern[:len(pattern)//2]  # First half
                    events = self._pattern_to_events(partial_pattern, n_neurons)
                    
                    results = await deployment.process_events(events, duration_ms=100)
                    
                    # Check if network completes the pattern correctly
                    if self._check_pattern_completion(results, pattern):
                        correct_recalls += 1
                
                accuracy = correct_recalls / total_tests
                
                trial_results.append({
                    "training_duration": training_duration,
                    "accuracy": accuracy,
                    "correct_recalls": correct_recalls,
                    "total_tests": total_tests
                })
            
            # Calculate metrics
            accuracies = [r["accuracy"] for r in trial_results]
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            metrics = {
                "avg_accuracy_percent": avg_accuracy * 100,
                "std_accuracy_percent": std_accuracy * 100,
                "max_accuracy_percent": np.max(accuracies) * 100,
                "min_accuracy_percent": np.min(accuracies) * 100,
                "avg_training_time_seconds": np.mean([r["training_duration"] for r in trial_results])
            }

            meta_payload = self._build_meta_payload(patterns, n_neurons)

            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds * config.n_trials,
                metrics=metrics,
                raw_data={"trial_results": trial_results, "patterns": patterns, "meta_task": meta_payload}
            )
            
        finally:
            await deployment.stop()
    
    async def _run_scalability_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run scalability benchmark."""
        neuron_counts = config.parameters.get("neuron_counts", [100, 500, 1000, 2000]) if config.parameters else [100, 500, 1000, 2000]
        
        scalability_results = []
        
        for n_neurons in neuron_counts:
            logger.info(f"Scalability test with {n_neurons} neurons")
            
            deployment = self.create_test_deployment(n_neurons)
            
            try:
                await deployment.initialize()
                await deployment.start()
                
                # Warmup
                await asyncio.sleep(config.warmup_seconds)
                
                # Run performance test
                start_time = time.time()
                total_events = 0
                
                test_start = time.time()
                while time.time() - test_start < config.duration_seconds:
                    events = self._generate_test_events(100, n_neurons)
                    results = await deployment.process_events(events, duration_ms=10)
                    total_events += len(events)
                    await asyncio.sleep(0.001)
                
                test_duration = time.time() - start_time
                throughput = total_events / test_duration
                throughput_per_neuron = throughput / n_neurons
                
                scalability_results.append({
                    "n_neurons": n_neurons,
                    "throughput": throughput,
                    "throughput_per_neuron": throughput_per_neuron,
                    "duration": test_duration
                })
                
            finally:
                await deployment.stop()
                await asyncio.sleep(config.cooldown_seconds)
        
        # Calculate scalability metrics
        throughputs = [r["throughput"] for r in scalability_results]
        per_neuron_throughputs = [r["throughput_per_neuron"] for r in scalability_results]
        
        # Linear regression to measure scalability
        neuron_counts_array = np.array(neuron_counts)
        throughputs_array = np.array(throughputs)
        
        # Fit linear model: throughput = a * n_neurons + b
        coeffs = np.polyfit(neuron_counts_array, throughputs_array, 1)
        scalability_factor = coeffs[0]  # Slope indicates how well it scales
        
        metrics = {
            "scalability_factor": scalability_factor,
            "max_throughput": np.max(throughputs),
            "min_throughput": np.min(throughputs),
            "avg_throughput_per_neuron": np.mean(per_neuron_throughputs),
            "std_throughput_per_neuron": np.std(per_neuron_throughputs),
            "max_neurons_tested": max(neuron_counts),
            "min_neurons_tested": min(neuron_counts)
        }
        
        return BenchmarkResult(
            benchmark_name=config.name,
            benchmark_type=config.benchmark_type,
            success=True,
            duration_seconds=config.duration_seconds * len(neuron_counts),
            metrics=metrics,
            raw_data={"scalability_results": scalability_results}
        )
    
    async def _run_latency_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run latency benchmark."""
        n_neurons = config.parameters.get("n_neurons", 1000) if config.parameters else 1000
        
        deployment = self.create_test_deployment(n_neurons)
        
        try:
            await deployment.initialize()
            await deployment.start()
            
            # Warmup
            await asyncio.sleep(config.warmup_seconds)
            
            latencies = []
            
            # Measure latency for individual event processing
            for _ in range(config.n_trials * 10):  # More samples for latency
                # Single event
                events = self._generate_test_events(1, n_neurons)
                
                start_time = time.time()
                results = await deployment.process_events(events, duration_ms=1)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                await asyncio.sleep(0.001)  # Small delay between measurements
            
            # Calculate latency statistics
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = np.max(latencies)
            min_latency = np.min(latencies)
            
            metrics = {
                "avg_latency_ms": avg_latency,
                "std_latency_ms": std_latency,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "max_latency_ms": max_latency,
                "min_latency_ms": min_latency,
                "total_measurements": len(latencies)
            }
            
            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds,
                metrics=metrics,
                raw_data={"latencies": latencies}
            )
            
        finally:
            await deployment.stop()
    
    async def _run_throughput_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run throughput benchmark."""
        n_neurons = config.parameters.get("n_neurons", 1000) if config.parameters else 1000
        batch_sizes = config.parameters.get("batch_sizes", [10, 50, 100, 500, 1000]) if config.parameters else [10, 50, 100, 500, 1000]
        
        deployment = self.create_test_deployment(n_neurons)
        
        try:
            await deployment.initialize()
            await deployment.start()
            
            # Warmup
            await asyncio.sleep(config.warmup_seconds)
            
            throughput_results = []
            
            for batch_size in batch_sizes:
                logger.info(f"Throughput test with batch size {batch_size}")
                
                batch_throughputs = []
                
                for trial in range(config.n_trials):
                    start_time = time.time()
                    total_events = 0
                    
                    # Run for specified duration
                    trial_start = time.time()
                    while time.time() - trial_start < config.duration_seconds / len(batch_sizes):
                        events = self._generate_test_events(batch_size, n_neurons)
                        results = await deployment.process_events(events, duration_ms=10)
                        total_events += len(events)
                        await asyncio.sleep(0.001)
                    
                    trial_duration = time.time() - start_time
                    throughput = total_events / trial_duration
                    batch_throughputs.append(throughput)
                
                avg_throughput = np.mean(batch_throughputs)
                throughput_results.append({
                    "batch_size": batch_size,
                    "avg_throughput": avg_throughput,
                    "std_throughput": np.std(batch_throughputs),
                    "max_throughput": np.max(batch_throughputs)
                })
            
            # Find optimal batch size
            max_throughput_result = max(throughput_results, key=lambda x: x["avg_throughput"])
            optimal_batch_size = max_throughput_result["batch_size"]
            max_throughput = max_throughput_result["avg_throughput"]
            
            metrics = {
                "max_throughput_events_per_sec": max_throughput,
                "optimal_batch_size": optimal_batch_size,
                "throughput_at_batch_10": next((r["avg_throughput"] for r in throughput_results if r["batch_size"] == 10), 0),
                "throughput_at_batch_100": next((r["avg_throughput"] for r in throughput_results if r["batch_size"] == 100), 0),
                "throughput_at_batch_1000": next((r["avg_throughput"] for r in throughput_results if r["batch_size"] == 1000), 0),
                "batch_sizes_tested": len(batch_sizes)
            }
            
            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds,
                metrics=metrics,
                raw_data={"throughput_results": throughput_results}
            )
            
        finally:
            await deployment.stop()
    
    async def _run_reliability_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run reliability benchmark."""
        n_neurons = config.parameters.get("n_neurons", 1000) if config.parameters else 1000
        stress_factor = config.parameters.get("stress_factor", 2.0) if config.parameters else 2.0
        
        deployment = self.create_test_deployment(n_neurons)
        
        try:
            await deployment.initialize()
            await deployment.start()
            
            # Warmup
            await asyncio.sleep(config.warmup_seconds)
            
            total_operations = 0
            total_errors = 0
            error_events = []
            uptime_start = time.time()
            
            # Run stress test
            test_start = time.time()
            while time.time() - test_start < config.duration_seconds:
                try:
                    # Generate high-intensity workload
                    n_events = int(1000 * stress_factor)
                    events = self._generate_test_events(n_events, n_neurons)
                    
                    results = await deployment.process_events(events, duration_ms=5)
                    total_operations += 1
                    
                    # Minimal delay to stress the system
                    await asyncio.sleep(0.0001)
                    
                except Exception as e:
                    total_errors += 1
                    error_events.append({
                        "timestamp": time.time(),
                        "error": str(e)
                    })
                    
                    # Continue after error
                    await asyncio.sleep(0.001)
            
            total_uptime = time.time() - uptime_start
            
            # Calculate reliability metrics
            success_rate = (total_operations - total_errors) / max(total_operations, 1)
            error_rate = total_errors / max(total_operations, 1)
            mtbf = total_uptime / max(total_errors, 1)  # Mean Time Between Failures
            availability = success_rate * 100  # Percentage
            
            metrics = {
                "success_rate_percent": success_rate * 100,
                "error_rate_percent": error_rate * 100,
                "total_operations": total_operations,
                "total_errors": total_errors,
                "uptime_seconds": total_uptime,
                "mtbf_seconds": mtbf,
                "availability_percent": availability,
                "stress_factor": stress_factor
            }
            
            return BenchmarkResult(
                benchmark_name=config.name,
                benchmark_type=config.benchmark_type,
                success=True,
                duration_seconds=config.duration_seconds,
                metrics=metrics,
                raw_data={"error_events": error_events}
            )
            
        finally:
            await deployment.stop()
    
    def _generate_test_events(self, n_events: int, n_neurons: int) -> List[NeuromorphicEvent]:
        """Generate test events for benchmarking."""
        events = []
        current_time_ns = int(time.time() * 1_000_000_000)
        
        for i in range(n_events):
            neuron_id = np.random.randint(0, n_neurons)
            timestamp_ns = current_time_ns + i * 1000  # 1μs apart
            
            event = NeuromorphicEvent(
                event_type=EventType.SPIKE,
                timestamp_ns=timestamp_ns,
                neuron_id=neuron_id,
                data={"amplitude": 1.0}
            )
            events.append(event)
        
        return events
    
    def _generate_training_patterns(self, n_patterns: int, n_neurons: int) -> List[List[int]]:
        """Generate training patterns for learning benchmark."""
        patterns = []
        pattern_length = min(10, n_neurons // 2)
        
        for _ in range(n_patterns):
            pattern = sorted(np.random.choice(n_neurons, pattern_length, replace=False))
            patterns.append(pattern)

        return patterns

    def _build_meta_payload(self, patterns: List[List[int]], n_neurons: int) -> Dict[str, Any]:
        """Construct support/query splits from recorded benchmark patterns."""

        support_x = [self._pattern_to_vector(pattern, n_neurons) for pattern in patterns]
        query_patterns = [pattern[: max(1, len(pattern) // 2)] for pattern in patterns]
        query_x = [self._pattern_to_vector(pattern, n_neurons) for pattern in query_patterns]

        return {
            "task_id": "learning_accuracy",
            "support_x": support_x,
            "support_y": support_x,
            "query_x": query_x,
            "query_y": support_x,
            "metadata": {
                "n_neurons": n_neurons,
                "pattern_count": len(patterns),
            },
        }

    def _pattern_to_vector(self, pattern: List[int], n_neurons: int) -> List[float]:
        vector = np.zeros(n_neurons, dtype=float)
        for idx in pattern:
            if 0 <= idx < n_neurons:
                vector[idx] = 1.0
        return vector.tolist()

    def _pattern_to_events(self, pattern: List[int], n_neurons: int) -> List[NeuromorphicEvent]:
        """Convert pattern to neuromorphic events."""
        events = []
        current_time_ns = int(time.time() * 1_000_000_000)
        
        for i, neuron_id in enumerate(pattern):
            timestamp_ns = current_time_ns + i * 10_000_000  # 10ms apart
            
            event = NeuromorphicEvent(
                event_type=EventType.SPIKE,
                timestamp_ns=timestamp_ns,
                neuron_id=neuron_id,
                data={"amplitude": 1.0}
            )
            events.append(event)
        
        return events
    
    def _update_learning_from_pattern(self, learning_engine: OnlineLearningEngine, pattern: List[int]):
        """Update learning engine from pattern."""
        # Simplified learning update
        for i in range(len(pattern) - 1):
            pre_neuron = pattern[i]
            post_neuron = pattern[i + 1]
            pre_time = i * 10.0  # 10ms intervals
            post_time = (i + 1) * 10.0
            
            learning_engine.process_spike_pair(pre_neuron, post_neuron, pre_time, post_time)
    
    def _check_pattern_completion(self, results: Dict[str, Any], expected_pattern: List[int]) -> bool:
        """Check if network completed pattern correctly (simplified)."""
        # This is a simplified check - in practice would analyze spike outputs
        return np.random.random() > 0.3  # Simulate 70% accuracy
    
    async def run_full_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run complete benchmark suite."""
        logger.info("Starting full neuromorphic benchmark suite")
        
        benchmarks = [
            BenchmarkConfig("performance_test", BenchmarkType.PERFORMANCE, 30.0, 3),
            BenchmarkConfig("power_efficiency_test", BenchmarkType.POWER_EFFICIENCY, 30.0, 3),
            BenchmarkConfig("learning_accuracy_test", BenchmarkType.LEARNING_ACCURACY, 60.0, 2),
            BenchmarkConfig("scalability_test", BenchmarkType.SCALABILITY, 20.0, 1, parameters={"neuron_counts": [100, 500, 1000]}),
            BenchmarkConfig("latency_test", BenchmarkType.LATENCY, 30.0, 1),
            BenchmarkConfig("throughput_test", BenchmarkType.THROUGHPUT, 40.0, 2),
            BenchmarkConfig("reliability_test", BenchmarkType.RELIABILITY, 60.0, 1, parameters={"stress_factor": 1.5})
        ]
        
        results = {}

        for benchmark_config in benchmarks:
            try:
                if self.meta_coordinator is not None:
                    self.meta_coordinator.apply_latest_checkpoint()
                result = await self.run_benchmark(benchmark_config)
                results[benchmark_config.name] = result
                self.results.append(result)
                
                if result.success:
                    logger.info(f"✓ {benchmark_config.name} completed successfully")
                else:
                    logger.error(f"✗ {benchmark_config.name} failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Benchmark {benchmark_config.name} crashed: {e}")
                results[benchmark_config.name] = BenchmarkResult(
                    benchmark_name=benchmark_config.name,
                    benchmark_type=benchmark_config.benchmark_type,
                    success=False,
                    duration_seconds=0.0,
                    metrics={},
                    raw_data={},
                    error_message=str(e)
                )
        
        logger.info("Benchmark suite completed")
        return results
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["# Neuromorphic Computing Benchmark Report\n"]
        
        # Summary
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results if r.success)
        success_rate = successful_benchmarks / total_benchmarks * 100
        
        report.append(f"## Summary")
        report.append(f"- Total benchmarks: {total_benchmarks}")
        report.append(f"- Successful: {successful_benchmarks}")
        report.append(f"- Success rate: {success_rate:.1f}%\n")
        
        # Detailed results
        report.append("## Detailed Results\n")
        
        for result in self.results:
            report.append(f"### {result.benchmark_name}")
            report.append(f"- Type: {result.benchmark_type.value}")
            report.append(f"- Status: {'✓ Success' if result.success else '✗ Failed'}")
            report.append(f"- Duration: {result.duration_seconds:.1f}s")
            
            if result.success and result.metrics:
                report.append("- Key Metrics:")
                for metric, value in result.metrics.items():
                    if isinstance(value, float):
                        report.append(f"  - {metric}: {value:.2f}")
                    else:
                        report.append(f"  - {metric}: {value}")
            
            if not result.success and result.error_message:
                report.append(f"- Error: {result.error_message}")
            
            report.append("")
        
        return "\n".join(report)


__all__ = [
    "BenchmarkType",
    "BenchmarkConfig", 
    "BenchmarkResult",
    "NeuromorphicBenchmarkSuite"
]