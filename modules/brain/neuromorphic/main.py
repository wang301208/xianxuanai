"""Main entry point for production neuromorphic computing system.

This module provides a command-line interface and demonstration of the
complete production-grade neuromorphic computing framework.
"""

import asyncio
import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .deployment_system import NeuromorphicDeployment, DeploymentConfig, DeploymentStatus
from .benchmarks import NeuromorphicBenchmarkSuite, BenchmarkConfig, BenchmarkType
from .examples.production_deployment import ProductionNeuromorphicSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuromorphicCLI:
    """Command-line interface for neuromorphic system."""
    
    def __init__(self):
        self.system: Optional[ProductionNeuromorphicSystem] = None
        self.benchmark_suite: Optional[NeuromorphicBenchmarkSuite] = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Production Neuromorphic Computing System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run demo system
  python -m modules.brain.neuromorphic.main demo --duration 60

  # Run benchmarks
  python -m modules.brain.neuromorphic.main benchmark --suite full

  # Deploy from config
  python -m modules.brain.neuromorphic.main deploy --config config.yaml

  # Run specific benchmark
  python -m modules.brain.neuromorphic.main benchmark --type performance --duration 30
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Demo command
        demo_parser = subparsers.add_parser('demo', help='Run demonstration system')
        demo_parser.add_argument('--duration', type=int, default=30,
                               help='Demo duration in seconds (default: 30)')
        demo_parser.add_argument('--neurons', type=int, default=1000,
                               help='Number of neurons (default: 1000)')
        demo_parser.add_argument('--platform', type=str, default='simulation',
                               choices=['simulation', 'loihi', 'brainscales', 'spinnaker'],
                               help='Hardware platform (default: simulation)')
        demo_parser.add_argument('--verbose', action='store_true',
                               help='Enable verbose logging')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
        benchmark_parser.add_argument('--suite', type=str, choices=['full', 'quick'],
                                    help='Benchmark suite to run')
        benchmark_parser.add_argument('--type', type=str,
                                    choices=['performance', 'power', 'learning', 'scalability', 
                                           'latency', 'throughput', 'reliability'],
                                    help='Specific benchmark type')
        benchmark_parser.add_argument('--duration', type=int, default=30,
                                    help='Benchmark duration in seconds (default: 30)')
        benchmark_parser.add_argument('--trials', type=int, default=3,
                                    help='Number of trials (default: 3)')
        benchmark_parser.add_argument('--output', type=str,
                                    help='Output file for results')
        
        # Deploy command
        deploy_parser = subparsers.add_parser('deploy', help='Deploy system from config')
        deploy_parser.add_argument('--config', type=str, required=True,
                                 help='Configuration file path')
        deploy_parser.add_argument('--duration', type=int,
                                 help='Override runtime duration in seconds')
        deploy_parser.add_argument('--monitor', action='store_true',
                                 help='Enable monitoring dashboard')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Run system tests')
        test_parser.add_argument('--type', type=str, choices=['unit', 'integration', 'all'],
                               default='all', help='Test type to run')
        test_parser.add_argument('--verbose', action='store_true',
                               help='Enable verbose test output')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show system information')
        info_parser.add_argument('--hardware', action='store_true',
                               help='Show hardware information')
        info_parser.add_argument('--capabilities', action='store_true',
                               help='Show system capabilities')
        
        return parser
    
    async def run_demo(self, args) -> int:
        """Run demonstration system."""
        logger.info("Starting neuromorphic system demonstration")
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Create system
            self.system = ProductionNeuromorphicSystem(f"demo_system_{args.platform}")
            
            # Initialize
            success = await self.system.initialize()
            if not success:
                logger.error("Failed to initialize demo system")
                return 1
            
            # Start
            success = await self.system.start()
            if not success:
                logger.error("Failed to start demo system")
                return 1
            
            logger.info(f"Demo system running for {args.duration} seconds...")
            logger.info(f"Platform: {args.platform}, Neurons: {args.neurons}")
            
            # Run for specified duration
            await asyncio.sleep(args.duration)
            
            # Get final status
            status = self.system.get_system_status()
            logger.info("Demo completed successfully!")
            logger.info(f"Final status: {status}")
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return 1
        finally:
            if self.system:
                await self.system.stop()
    
    async def run_benchmark(self, args) -> int:
        """Run benchmark suite."""
        logger.info("Starting neuromorphic benchmarks")
        
        try:
            self.benchmark_suite = NeuromorphicBenchmarkSuite()
            
            if args.suite == 'full':
                # Run full benchmark suite
                results = await self.benchmark_suite.run_full_benchmark_suite()
                
            elif args.suite == 'quick':
                # Run quick benchmark suite
                quick_benchmarks = [
                    BenchmarkConfig("quick_performance", BenchmarkType.PERFORMANCE, 10.0, 1),
                    BenchmarkConfig("quick_latency", BenchmarkType.LATENCY, 10.0, 1),
                    BenchmarkConfig("quick_throughput", BenchmarkType.THROUGHPUT, 10.0, 1)
                ]
                
                results = {}
                for config in quick_benchmarks:
                    result = await self.benchmark_suite.run_benchmark(config)
                    results[config.name] = result
                    
            elif args.type:
                # Run specific benchmark type
                benchmark_type = getattr(BenchmarkType, args.type.upper())
                config = BenchmarkConfig(
                    f"{args.type}_benchmark",
                    benchmark_type,
                    args.duration,
                    args.trials
                )
                
                result = await self.benchmark_suite.run_benchmark(config)
                results = {config.name: result}
                
            else:
                logger.error("Must specify --suite or --type")
                return 1
            
            # Generate report
            report = self.benchmark_suite.generate_benchmark_report()
            print("\n" + report)
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                logger.info(f"Results saved to {args.output}")
            
            # Check if all benchmarks passed
            all_passed = all(result.success for result in results.values())
            return 0 if all_passed else 1
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return 1
    
    async def run_deploy(self, args) -> int:
        """Deploy system from configuration file."""
        logger.info(f"Deploying system from config: {args.config}")
        
        try:
            # Load configuration
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {args.config}")
                return 1
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    import json
                    config_data = json.load(f)
            
            # Create deployment config
            deployment_config = self._create_deployment_config_from_data(config_data)
            
            # Override duration if specified
            if args.duration:
                deployment_config.max_runtime_hours = args.duration / 3600.0
            
            # Create and run deployment
            deployment = NeuromorphicDeployment(deployment_config)
            
            # Initialize
            success = await deployment.initialize()
            if not success:
                logger.error("Failed to initialize deployment")
                return 1
            
            # Start
            success = await deployment.start()
            if not success:
                logger.error("Failed to start deployment")
                return 1
            
            logger.info("Deployment started successfully")
            
            # Run until completion or interruption
            try:
                runtime_seconds = deployment_config.max_runtime_hours * 3600
                await asyncio.sleep(runtime_seconds)
                logger.info("Deployment completed successfully")
                
            except KeyboardInterrupt:
                logger.info("Deployment interrupted by user")
            
            return 0
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return 1
        finally:
            if 'deployment' in locals():
                await deployment.stop()
    
    def _create_deployment_config_from_data(self, config_data: Dict[str, Any]) -> DeploymentConfig:
        """Create deployment config from loaded data."""
        system_config = config_data.get('system', {})
        deployment_config = config_data.get('deployment', {})
        network_config = config_data.get('network', {})
        
        return DeploymentConfig(
            name=system_config.get('name', 'configured_system'),
            version=system_config.get('version', '1.0.0'),
            hardware_platform=deployment_config.get('hardware', {}).get('primary_platform', 'simulation'),
            network_config=network_config,
            power_budget_mw=deployment_config.get('power', {}).get('budget_mw', 1000.0),
            max_runtime_hours=deployment_config.get('power', {}).get('max_runtime_hours', 24.0),
            monitoring_interval_ms=deployment_config.get('monitoring', {}).get('interval_ms', 100),
            auto_restart=deployment_config.get('monitoring', {}).get('auto_restart', True),
            backup_enabled=deployment_config.get('monitoring', {}).get('backup_enabled', True),
            logging_level=deployment_config.get('logging', {}).get('level', 'INFO')
        )
    
    async def run_test(self, args) -> int:
        """Run system tests."""
        logger.info(f"Running {args.type} tests")
        
        try:
            import pytest
            
            # Determine test files
            test_dir = Path(__file__).parent / "tests"
            
            if args.type == 'unit':
                test_files = [str(test_dir / "test_production_system.py::TestHardwareDrivers"),
                             str(test_dir / "test_production_system.py::TestEventDrivenCore"),
                             str(test_dir / "test_production_system.py::TestPowerOptimization"),
                             str(test_dir / "test_production_system.py::TestAdvancedLearning")]
            elif args.type == 'integration':
                test_files = [str(test_dir / "test_production_system.py::TestIntegration")]
            else:  # all
                test_files = [str(test_dir / "test_production_system.py")]
            
            # Run tests
            pytest_args = test_files + ["-v"] if args.verbose else test_files
            result = pytest.main(pytest_args)
            
            return result
            
        except ImportError:
            logger.error("pytest not available. Install with: pip install pytest")
            return 1
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return 1
    
    def show_info(self, args) -> int:
        """Show system information."""
        print("Production Neuromorphic Computing System")
        print("=" * 50)
        print(f"Version: 2.0.0")
        print(f"Framework: Event-driven neuromorphic computing")
        print()
        
        if args.hardware:
            print("Supported Hardware Platforms:")
            print("- Intel Loihi (neuromorphic chip)")
            print("- BrainScaleS (analog neuromorphic)")
            print("- SpiNNaker (digital neuromorphic)")
            print("- Simulation (software backend)")
            print()
        
        if args.capabilities:
            print("System Capabilities:")
            print("- Real-time event-driven processing")
            print("- Hardware-accelerated spike processing")
            print("- Online learning with STDP")
            print("- Power optimization and management")
            print("- Production deployment and monitoring")
            print("- Comprehensive benchmarking")
            print("- Multi-platform hardware support")
            print("- Fault tolerance and auto-recovery")
            print()
        
        print("Available Commands:")
        print("- demo: Run demonstration system")
        print("- benchmark: Execute performance benchmarks")
        print("- deploy: Deploy from configuration file")
        print("- test: Run system tests")
        print("- info: Show system information")
        
        return 0
    
    async def run(self, args) -> int:
        """Run the CLI command."""
        if args.command == 'demo':
            return await self.run_demo(args)
        elif args.command == 'benchmark':
            return await self.run_benchmark(args)
        elif args.command == 'deploy':
            return await self.run_deploy(args)
        elif args.command == 'test':
            return await self.run_test(args)
        elif args.command == 'info':
            return self.show_info(args)
        else:
            print("No command specified. Use --help for usage information.")
            return 1


async def main():
    """Main entry point."""
    cli = NeuromorphicCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    try:
        exit_code = await cli.run(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())