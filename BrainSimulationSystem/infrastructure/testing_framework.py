"""
大规模单元/集成测试框架

提供：
- 分层测试架构（单元测试、集成测试、系统测试）
- 自动化测试发现与执行
- 性能基准测试
- 回归测试
- 测试覆盖率分析
- 持续集成支持
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
import time
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import unittest
import pytest
import inspect
import traceback
import sys
import os
from pathlib import Path
import importlib
import pickle
import h5py
import yaml
from datetime import datetime, timedelta
import uuid
import subprocess
import coverage
import memory_profiler
import psutil

logger = logging.getLogger(__name__)

class TestLevel(Enum):
    """测试级别"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    REGRESSION = "regression"
    STRESS = "stress"

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestPriority(Enum):
    """测试优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestCase:
    """测试用例"""
    test_id: str
    name: str
    description: str
    level: TestLevel
    priority: TestPriority
    
    test_function: Callable
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    
    # 测试配置
    timeout: float = 300.0  # 秒
    retry_count: int = 0
    max_retries: int = 3
    
    # 依赖关系
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # 执行状态
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    
    # 结果
    result: Optional[Any] = None
    error_message: str = ""
    traceback_info: str = ""
    
    # 性能指标
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
@dataclass
class TestSuite:
    """测试套件"""
    suite_id: str
    name: str
    description: str
    
    test_cases: List[TestCase] = field(default_factory=list)
    setup_suite: Optional[Callable] = None
    teardown_suite: Optional[Callable] = None
    
    # 执行配置
    parallel_execution: bool = True
    max_parallel_tests: int = 4
    
    # 统计信息
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0

@dataclass
class TestResult:
    """测试结果"""
    suite_id: str
    test_results: List[TestCase]
    
    # 统计摘要
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    
    # 时间统计
    start_time: datetime
    end_time: datetime
    total_execution_time: float
    
    # 覆盖率信息
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    
    # 性能基准
    performance_benchmarks: Dict[str, Any] = field(default_factory=dict)

class BaseTestCase(ABC):
    """测试用例基类"""
    
    def __init__(self, test_id: str, name: str, description: str = ""):
        self.test_id = test_id
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Test.{test_id}")
    
    @abstractmethod
    def setup(self):
        """测试前置条件"""
        self.logger.info("设置测试前置条件")
    
    @abstractmethod
    def execute(self) -> Any:
        """执行测试"""
        self.logger.info("执行测试")
        return True
    
    @abstractmethod
    def teardown(self):
        """测试清理"""
        self.logger.info("清理测试环境")
    
    def assert_equal(self, actual, expected, message: str = ""):
        """断言相等"""
        if actual != expected:
            raise AssertionError(f"{message}: 期望 {expected}, 实际 {actual}")
    
    def assert_almost_equal(self, actual, expected, tolerance: float = 1e-6, message: str = ""):
        """断言近似相等"""
        if abs(actual - expected) > tolerance:
            raise AssertionError(f"{message}: 期望 {expected}, 实际 {actual}, 容差 {tolerance}")
    
    def assert_array_equal(self, actual, expected, tolerance: float = 1e-6, message: str = ""):
        """断言数组相等"""
        actual_array = np.array(actual)
        expected_array = np.array(expected)
        
        if actual_array.shape != expected_array.shape:
            raise AssertionError(f"{message}: 形状不匹配 {actual_array.shape} vs {expected_array.shape}")
        
        if not np.allclose(actual_array, expected_array, atol=tolerance):
            raise AssertionError(f"{message}: 数组不相等，最大差异 {np.max(np.abs(actual_array - expected_array))}")
    
    def assert_spike_train_similar(self, actual_spikes, expected_spikes, 
                                  time_tolerance: float = 1.0, message: str = ""):
        """断言尖峰序列相似"""
        actual_times = np.array(actual_spikes)
        expected_times = np.array(expected_spikes)
        
        # 检查尖峰数量
        if abs(len(actual_times) - len(expected_times)) > len(expected_times) * 0.1:
            raise AssertionError(f"{message}: 尖峰数量差异过大 {len(actual_times)} vs {len(expected_times)}")
        
        # 检查时间分布
        if len(actual_times) > 0 and len(expected_times) > 0:
            # 使用直方图比较
            bins = np.linspace(0, max(actual_times.max(), expected_times.max()), 50)
            actual_hist, _ = np.histogram(actual_times, bins)
            expected_hist, _ = np.histogram(expected_times, bins)
            
            correlation = np.corrcoef(actual_hist, expected_hist)[0, 1]
            if correlation < 0.8:  # 相关性阈值
                raise AssertionError(f"{message}: 尖峰时间分布相关性过低 {correlation:.3f}")

class NeuronModelUnitTest(BaseTestCase):
    """神经元模型单元测试"""
    
    def __init__(self):
        super().__init__("neuron_model_unit", "神经元模型单元测试", "测试单个神经元模型的基本功能")
        self.neuron = None
    
    def setup(self):
        """设置测试神经元"""
        # 这里应该导入实际的神经元模型
        # from BrainSimulationSystem.models.neurons import LIFNeuron
        # self.neuron = LIFNeuron(tau_m=20.0, v_rest=-70.0, v_threshold=-50.0)
        
        # 模拟神经元
        self.neuron = MockNeuron()
    
    def execute(self) -> Any:
        """执行神经元测试"""
        results = {}
        
        # 测试1: 静息状态
        self.neuron.reset()
        voltage = self.neuron.get_voltage()
        self.assert_almost_equal(voltage, -70.0, message="静息电位测试")
        results['resting_potential'] = voltage
        
        # 测试2: 电流注入响应
        self.neuron.inject_current(10.0, duration=50.0)
        voltages = []
        for t in range(100):
            self.neuron.step(dt=0.1)
            voltages.append(self.neuron.get_voltage())
        
        # 检查电压上升
        max_voltage = max(voltages)
        self.assert_equal(max_voltage > -70.0, True, message="电流注入响应测试")
        results['current_response'] = voltages
        
        # 测试3: 尖峰生成
        self.neuron.reset()
        self.neuron.inject_current(50.0, duration=10.0)  # 强电流
        spike_times = []
        
        for t in range(200):
            self.neuron.step(dt=0.1)
            if self.neuron.has_spiked():
                spike_times.append(t * 0.1)
        
        self.assert_equal(len(spike_times) > 0, True, message="尖峰生成测试")
        results['spike_times'] = spike_times
        
        return results
    
    def teardown(self):
        """清理测试"""
        self.neuron = None

class NetworkIntegrationTest(BaseTestCase):
    """网络集成测试"""
    
    def __init__(self):
        super().__init__("network_integration", "网络集成测试", "测试神经元网络的集成功能")
        self.network = None
    
    def setup(self):
        """设置测试网络"""
        # 创建小型测试网络
        self.network = MockNetwork(num_neurons=100)
    
    def execute(self) -> Any:
        """执行网络测试"""
        results = {}
        
        # 测试1: 网络连接
        connections = self.network.get_connections()
        self.assert_equal(len(connections) > 0, True, message="网络连接测试")
        results['num_connections'] = len(connections)
        
        # 测试2: 信号传播
        self.network.stimulate_neuron(0, current=20.0)
        
        spike_data = []
        for t in range(1000):  # 100ms仿真
            self.network.step(dt=0.1)
            spikes = self.network.get_spikes()
            if spikes:
                spike_data.extend([(t * 0.1, neuron_id) for neuron_id in spikes])
        
        # 检查信号传播
        unique_neurons = set(spike[1] for spike in spike_data)
        self.assert_equal(len(unique_neurons) > 1, True, message="信号传播测试")
        results['spike_data'] = spike_data
        
        # 测试3: 网络稳定性
        # 检查是否有异常的爆发性活动
        spike_counts = [0] * 100  # 每10ms的尖峰计数
        for spike_time, _ in spike_data:
            bin_idx = min(int(spike_time / 10.0), 99)
            spike_counts[bin_idx] += 1
        
        max_spikes_per_bin = max(spike_counts)
        self.assert_equal(max_spikes_per_bin < 50, True, message="网络稳定性测试")
        results['spike_counts'] = spike_counts
        
        return results
    
    def teardown(self):
        """清理测试"""
        self.network = None

class CorticalColumnSystemTest(BaseTestCase):
    """皮层柱系统测试"""
    
    def __init__(self):
        super().__init__("cortical_column_system", "皮层柱系统测试", "测试完整皮层柱系统功能")
        self.cortical_column = None
    
    def setup(self):
        """设置皮层柱"""
        # 这里应该导入实际的皮层柱模型
        self.cortical_column = MockCorticalColumn()
    
    def execute(self) -> Any:
        """执行系统测试"""
        results = {}
        
        # 测试1: 层间连接
        layer_activities = self.cortical_column.get_layer_activities()
        self.assert_equal(len(layer_activities), 6, message="皮层层数测试")
        results['layer_activities'] = layer_activities
        
        # 测试2: 感觉输入处理
        self.cortical_column.apply_sensory_input(input_pattern=[1, 0, 1, 0, 1])
        
        responses = []
        for t in range(500):  # 50ms仿真
            self.cortical_column.step(dt=0.1)
            response = self.cortical_column.get_output()
            responses.append(response)
        
        # 检查响应
        max_response = max(responses)
        self.assert_equal(max_response > 0, True, message="感觉输入处理测试")
        results['sensory_responses'] = responses
        
        # 测试3: 丘脑反馈
        self.cortical_column.enable_thalamic_feedback(True)
        
        feedback_responses = []
        for t in range(500):
            self.cortical_column.step(dt=0.1)
            response = self.cortical_column.get_output()
            feedback_responses.append(response)
        
        # 比较有无反馈的差异
        response_diff = np.mean(np.abs(np.array(feedback_responses) - np.array(responses)))
        self.assert_equal(response_diff > 0.01, True, message="丘脑反馈测试")
        results['feedback_responses'] = feedback_responses
        
        return results
    
    def teardown(self):
        """清理测试"""
        self.cortical_column = None

class PerformanceBenchmarkTest(BaseTestCase):
    """性能基准测试"""
    
    def __init__(self):
        super().__init__("performance_benchmark", "性能基准测试", "测试系统性能指标")
        self.benchmark_data = {}
    
    def setup(self):
        """设置性能测试"""
        self.benchmark_data = {
            'network_sizes': [100, 500, 1000, 5000],
            'simulation_times': [100, 500, 1000],  # ms
            'results': {}
        }
    
    def execute(self) -> Any:
        """执行性能测试"""
        results = {}
        
        for network_size in self.benchmark_data['network_sizes']:
            for sim_time in self.benchmark_data['simulation_times']:
                # 创建测试网络
                network = MockNetwork(num_neurons=network_size)
                
                # 测量执行时间
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # 运行仿真
                for t in range(int(sim_time / 0.1)):
                    network.step(dt=0.1)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # 记录性能指标
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                key = f"n{network_size}_t{sim_time}"
                results[key] = {
                    'network_size': network_size,
                    'simulation_time': sim_time,
                    'execution_time': execution_time,
                    'memory_usage': memory_usage,
                    'neurons_per_second': network_size / execution_time,
                    'real_time_factor': sim_time / (execution_time * 1000)
                }
                
                self.logger.info(f"基准测试 {key}: {execution_time:.3f}s, {memory_usage:.1f}MB")
        
        # 性能回归检查
        self._check_performance_regression(results)
        
        return results
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _check_performance_regression(self, results: Dict[str, Any]):
        """检查性能回归"""
        # 这里应该与历史基准数据比较
        # 简化版本：检查是否有异常慢的测试
        for key, result in results.items():
            neurons_per_second = result['neurons_per_second']
            
            # 简单的性能阈值检查
            min_expected_nps = 1000  # 每秒至少处理1000个神经元
            if neurons_per_second < min_expected_nps:
                self.logger.warning(f"性能警告 {key}: {neurons_per_second:.0f} neurons/s < {min_expected_nps}")
    
    def teardown(self):
        """清理测试"""
        pass

class TestDiscovery:
    """测试发现器"""
    
    def __init__(self, test_directories: List[str]):
        self.test_directories = test_directories
        self.logger = logging.getLogger("TestDiscovery")
    
    def discover_tests(self) -> List[TestCase]:
        """发现测试用例"""
        test_cases = []
        
        for test_dir in self.test_directories:
            test_cases.extend(self._discover_in_directory(test_dir))
        
        self.logger.info(f"发现 {len(test_cases)} 个测试用例")
        return test_cases
    
    def _discover_in_directory(self, directory: str) -> List[TestCase]:
        """在目录中发现测试"""
        test_cases = []
        test_dir = Path(directory)
        
        if not test_dir.exists():
            self.logger.warning(f"测试目录不存在: {directory}")
            return test_cases
        
        # 查找Python测试文件
        for test_file in test_dir.glob("test_*.py"):
            try:
                # 动态导入测试模块
                spec = importlib.util.spec_from_file_location("test_module", test_file)
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # 查找测试类
                for name, obj in inspect.getmembers(test_module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTestCase) and 
                        obj != BaseTestCase):
                        
                        # 创建测试实例
                        test_instance = obj()
                        
                        test_case = TestCase(
                            test_id=test_instance.test_id,
                            name=test_instance.name,
                            description=test_instance.description,
                            level=self._infer_test_level(test_instance.test_id),
                            priority=TestPriority.MEDIUM,
                            test_function=test_instance.execute,
                            setup_function=test_instance.setup,
                            teardown_function=test_instance.teardown
                        )
                        
                        test_cases.append(test_case)
                        
            except Exception as e:
                self.logger.error(f"导入测试文件失败 {test_file}: {str(e)}")
        
        return test_cases
    
    def _infer_test_level(self, test_id: str) -> TestLevel:
        """推断测试级别"""
        if "unit" in test_id.lower():
            return TestLevel.UNIT
        elif "integration" in test_id.lower():
            return TestLevel.INTEGRATION
        elif "system" in test_id.lower():
            return TestLevel.SYSTEM
        elif "performance" in test_id.lower() or "benchmark" in test_id.lower():
            return TestLevel.PERFORMANCE
        else:
            return TestLevel.UNIT

class TestExecutor:
    """测试执行器"""
    
    def __init__(self, max_parallel_tests: int = 4):
        self.max_parallel_tests = max_parallel_tests
        self.logger = logging.getLogger("TestExecutor")
        
        # 覆盖率分析
        self.coverage = coverage.Coverage()
        
    async def execute_test_suite(self, test_suite: TestSuite) -> TestResult:
        """执行测试套件"""
        self.logger.info(f"开始执行测试套件: {test_suite.name}")
        
        start_time = datetime.now()
        
        # 启动覆盖率分析
        self.coverage.start()
        
        try:
            # 执行套件设置
            if test_suite.setup_suite:
                test_suite.setup_suite()
            
            # 执行测试用例
            if test_suite.parallel_execution:
                results = await self._execute_tests_parallel(test_suite.test_cases)
            else:
                results = await self._execute_tests_sequential(test_suite.test_cases)
            
            # 执行套件清理
            if test_suite.teardown_suite:
                test_suite.teardown_suite()
            
        finally:
            # 停止覆盖率分析
            self.coverage.stop()
        
        end_time = datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()
        
        # 统计结果
        passed_tests = sum(1 for test in results if test.status == TestStatus.PASSED)
        failed_tests = sum(1 for test in results if test.status == TestStatus.FAILED)
        skipped_tests = sum(1 for test in results if test.status == TestStatus.SKIPPED)
        error_tests = sum(1 for test in results if test.status == TestStatus.ERROR)
        
        # 生成覆盖率报告
        coverage_report = self._generate_coverage_report()
        
        test_result = TestResult(
            suite_id=test_suite.suite_id,
            test_results=results,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            start_time=start_time,
            end_time=end_time,
            total_execution_time=total_execution_time,
            coverage_report=coverage_report
        )
        
        self.logger.info(f"测试套件完成: {passed_tests}/{len(results)} 通过")
        
        return test_result
    
    async def _execute_tests_parallel(self, test_cases: List[TestCase]) -> List[TestCase]:
        """并行执行测试"""
        semaphore = asyncio.Semaphore(self.max_parallel_tests)
        
        async def execute_with_semaphore(test_case):
            async with semaphore:
                return await self._execute_single_test(test_case)
        
        tasks = [execute_with_semaphore(test) for test in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_cases[i].status = TestStatus.ERROR
                test_cases[i].error_message = str(result)
                processed_results.append(test_cases[i])
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_tests_sequential(self, test_cases: List[TestCase]) -> List[TestCase]:
        """顺序执行测试"""
        results = []
        
        for test_case in test_cases:
            result = await self._execute_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _execute_single_test(self, test_case: TestCase) -> TestCase:
        """执行单个测试"""
        test_case.status = TestStatus.RUNNING
        test_case.start_time = datetime.now()
        
        try:
            # 执行设置
            if test_case.setup_function:
                test_case.setup_function()
            
            # 执行测试
            result = test_case.test_function()
            test_case.result = result
            test_case.status = TestStatus.PASSED
            
            self.logger.info(f"测试通过: {test_case.test_id}")
            
        except AssertionError as e:
            test_case.status = TestStatus.FAILED
            test_case.error_message = str(e)
            test_case.traceback_info = traceback.format_exc()
            
            self.logger.error(f"测试失败: {test_case.test_id} - {str(e)}")
            
        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.error_message = str(e)
            test_case.traceback_info = traceback.format_exc()
            
            self.logger.error(f"测试错误: {test_case.test_id} - {str(e)}")
            
        finally:
            # 执行清理
            try:
                if test_case.teardown_function:
                    test_case.teardown_function()
            except Exception as e:
                self.logger.error(f"测试清理失败: {test_case.test_id} - {str(e)}")
            
            test_case.end_time = datetime.now()
            test_case.execution_time = (test_case.end_time - test_case.start_time).total_seconds()
        
        return test_case
    
    def _generate_coverage_report(self) -> Dict[str, Any]:
        """生成覆盖率报告"""
        try:
            self.coverage.save()
            
            # 获取覆盖率数据
            coverage_data = self.coverage.get_data()
            
            # 计算覆盖率统计
            total_lines = 0
            covered_lines = 0
            
            for filename in coverage_data.measured_files():
                analysis = self.coverage.analysis2(filename)
                total_lines += len(analysis.statements)
                covered_lines += len(analysis.statements) - len(analysis.missing)
            
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            
            return {
                'total_lines': total_lines,
                'covered_lines': covered_lines,
                'coverage_percentage': coverage_percentage,
                'files': list(coverage_data.measured_files())
            }
            
        except Exception as e:
            self.logger.error(f"生成覆盖率报告失败: {str(e)}")
            return {}

class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, output_directory: str = "./test_reports"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("TestReporter")
    
    def generate_report(self, test_result: TestResult, format: str = "json"):
        """生成测试报告"""
        if format == "json":
            self._generate_json_report(test_result)
        elif format == "html":
            self._generate_html_report(test_result)
        elif format == "junit":
            self._generate_junit_report(test_result)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
    
    def _generate_json_report(self, test_result: TestResult):
        """生成JSON格式报告"""
        report_data = {
            'suite_id': test_result.suite_id,
            'summary': {
                'total_tests': test_result.total_tests,
                'passed_tests': test_result.passed_tests,
                'failed_tests': test_result.failed_tests,
                'skipped_tests': test_result.skipped_tests,
                'error_tests': test_result.error_tests,
                'success_rate': test_result.passed_tests / test_result.total_tests if test_result.total_tests > 0 else 0.0
            },
            'timing': {
                'start_time': test_result.start_time.isoformat(),
                'end_time': test_result.end_time.isoformat(),
                'total_execution_time': test_result.total_execution_time
            },
            'coverage': test_result.coverage_report,
            'test_cases': []
        }
        
        for test_case in test_result.test_results:
            test_data = {
                'test_id': test_case.test_id,
                'name': test_case.name,
                'status': test_case.status.value,
                'execution_time': test_case.execution_time,
                'error_message': test_case.error_message
            }
            report_data['test_cases'].append(test_data)
        
        report_file = self.output_directory / f"{test_result.suite_id}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"JSON报告已生成: {report_file}")
    
    def _generate_html_report(self, test_result: TestResult):
        """生成HTML格式报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>测试报告 - {test_result.suite_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>测试报告</h1>
            <div class="summary">
                <h2>摘要</h2>
                <p>总测试数: {test_result.total_tests}</p>
                <p class="passed">通过: {test_result.passed_tests}</p>
                <p class="failed">失败: {test_result.failed_tests}</p>
                <p class="error">错误: {test_result.error_tests}</p>
                <p class="skipped">跳过: {test_result.skipped_tests}</p>
                <p>成功率: {test_result.passed_tests / test_result.total_tests * 100:.1f}%</p>
                <p>执行时间: {test_result.total_execution_time:.2f} 秒</p>
            </div>
            
            <h2>测试用例详情</h2>
            <table>
                <tr>
                    <th>测试ID</th>
                    <th>名称</th>
                    <th>状态</th>
                    <th>执行时间</th>
                    <th>错误信息</th>
                </tr>
        """
        
        for test_case in test_result.test_results:
            status_class = test_case.status.value
            html_content += f"""
                <tr>
                    <td>{test_case.test_id}</td>
                    <td>{test_case.name}</td>
                    <td class="{status_class}">{test_case.status.value}</td>
                    <td>{test_case.execution_time:.3f}s</td>
                    <td>{test_case.error_message}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        report_file = self.output_directory / f"{test_result.suite_id}_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已生成: {report_file}")
    
    def _generate_junit_report(self, test_result: TestResult):
        """生成JUnit XML格式报告"""
        # 简化的JUnit XML生成
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="{test_result.suite_id}" 
           tests="{test_result.total_tests}"
           failures="{test_result.failed_tests}"
           errors="{test_result.error_tests}"
           skipped="{test_result.skipped_tests}"
           time="{test_result.total_execution_time}">
"""
        
        for test_case in test_result.test_results:
            xml_content += f'  <testcase name="{test_case.name}" classname="{test_case.test_id}" time="{test_case.execution_time}"'
            
            if test_case.status == TestStatus.PASSED:
                xml_content += ' />\n'
            elif test_case.status == TestStatus.FAILED:
                xml_content += f'>\n    <failure message="{test_case.error_message}"></failure>\n  </testcase>\n'
            elif test_case.status == TestStatus.ERROR:
                xml_content += f'>\n    <error message="{test_case.error_message}"></error>\n  </testcase>\n'
            elif test_case.status == TestStatus.SKIPPED:
                xml_content += f'>\n    <skipped></skipped>\n  </testcase>\n'
        
        xml_content += '</testsuite>\n'
        
        report_file = self.output_directory / f"{test_result.suite_id}_junit.xml"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        self.logger.info(f"JUnit报告已生成: {report_file}")

# 模拟类（用于测试）
class MockNeuron:
    """模拟神经元"""
    
    def __init__(self):
        self.voltage = -70.0
        self.current = 0.0
        self.spiked = False
    
    def reset(self):
        self.voltage = -70.0
        self.current = 0.0
        self.spiked = False
    
    def inject_current(self, current: float, duration: float):
        self.current = current
    
    def step(self, dt: float):
        # 简化的LIF模型
        tau_m = 20.0
        self.voltage += dt * (-self.voltage + self.current * 10) / tau_m
        
        self.spiked = False
        if self.voltage > -50.0:  # 阈值
            self.spiked = True
            self.voltage = -70.0  # 重置
    
    def get_voltage(self) -> float:
        return self.voltage
    
    def has_spiked(self) -> bool:
        return self.spiked

class MockNetwork:
    """模拟网络"""
    
    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.neurons = [MockNeuron() for _ in range(num_neurons)]
        self.connections = self._create_random_connections()
        self.current_spikes = []
    
    def _create_random_connections(self):
        connections = []
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < 0.1:  # 10%连接概率
                    connections.append((i, j, np.random.uniform(0.5, 2.0)))  # (源, 目标, 权重)
        return connections
    
    def get_connections(self):
        return self.connections
    
    def stimulate_neuron(self, neuron_id: int, current: float):
        if 0 <= neuron_id < self.num_neurons:
            self.neurons[neuron_id].inject_current(current, 10.0)
    
    def step(self, dt: float):
        # 更新所有神经元
        for neuron in self.neurons:
            neuron.step(dt)
        
        # 收集尖峰
        self.current_spikes = []
        for i, neuron in enumerate(self.neurons):
            if neuron.has_spiked():
                self.current_spikes.append(i)
        
        # 传播尖峰
        for source_id in self.current_spikes:
            for source, target, weight in self.connections:
                if source == source_id:
                    self.neurons[target].inject_current(weight, 1.0)
    
    def get_spikes(self):
        return self.current_spikes.copy()

class MockCorticalColumn:
    """模拟皮层柱"""
    
    def __init__(self):
        self.layers = [MockNetwork(100) for _ in range(6)]  # 6层
        self.thalamic_feedback = False
        self.output_activity = 0.0
    
    def get_layer_activities(self):
        return [len(layer.get_spikes()) for layer in self.layers]
    
    def apply_sensory_input(self, input_pattern):
        # 刺激第4层
        for i, value in enumerate(input_pattern):
            if value > 0 and i < self.layers[3].num_neurons:
                self.layers[3].stimulate_neuron(i, 10.0)
    
    def enable_thalamic_feedback(self, enabled: bool):
        self.thalamic_feedback = enabled
    
    def step(self, dt: float):
        # 更新所有层
        for layer in self.layers:
            layer.step(dt)
        
        # 计算输出活动
        total_spikes = sum(len(layer.get_spikes()) for layer in self.layers)
        self.output_activity = total_spikes / 600.0  # 归一化
        
        # 丘脑反馈
        if self.thalamic_feedback and self.output_activity > 0.1:
            self.layers[0].stimulate_neuron(0, 5.0)  # 反馈到第1层
    
    def get_output(self):
        return self.output_activity

async def run_comprehensive_tests():
    """运行综合测试"""
    # 创建测试套件
    test_suite = TestSuite(
        suite_id="brain_simulation_comprehensive",
        name="大脑仿真系统综合测试",
        description="包含单元测试、集成测试和系统测试的综合测试套件"
    )
    
    # 添加测试用例
    test_cases = [
        NeuronModelUnitTest(),
        NetworkIntegrationTest(),
        CorticalColumnSystemTest(),
        PerformanceBenchmarkTest()
    ]
    
    for test_instance in test_cases:
        test_case = TestCase(
            test_id=test_instance.test_id,
            name=test_instance.name,
            description=test_instance.description,
            level=TestLevel.UNIT if "unit" in test_instance.test_id else TestLevel.INTEGRATION,
            priority=TestPriority.HIGH,
            test_function=test_instance.execute,
            setup_function=test_instance.setup,
            teardown_function=test_instance.teardown
        )
        test_suite.test_cases.append(test_case)
    
    # 执行测试
    executor = TestExecutor(max_parallel_tests=2)
    result = await executor.execute_test_suite(test_suite)
    
    # 生成报告
    reporter = TestReporter()
    reporter.generate_report(result, format="json")
    reporter.generate_report(result, format="html")
    
    print(f"测试完成: {result.passed_tests}/{result.total_tests} 通过")
    print(f"覆盖率: {result.coverage_report.get('coverage_percentage', 0):.1f}%")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    asyncio.run(run_comprehensive_tests())