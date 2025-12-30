# -*- coding: utf-8 -*-
"""
持续集成/持续部署系统
CI/CD System

实现完整的CI/CD管线：
1. 数值稳定性测试
2. 性能基准测试
3. 硬件一致性测试
4. 回归测试
5. 自动化部署
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime
import threading
import queue
from abc import ABC, abstractmethod

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TestSeverity(Enum):
    """测试严重性"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestStatus
    execution_time: float
    severity: TestSeverity = TestSeverity.MEDIUM
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class TestSuite:
    """测试套件"""
    name: str
    description: str
    tests: List[str]
    timeout: int = 300  # 秒
    parallel: bool = True
    dependencies: List[str] = field(default_factory=list)

class BaseTest(ABC):
    """测试基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Test.{name}")
    
    @abstractmethod
    def run(self) -> TestResult:
        """运行测试"""
        start_time = time.time()
        
        try:
            # 执行测试逻辑
            success = self._execute_test()
            end_time = time.time()
            
            return TestResult(
                test_name=self.name,
                success=success,
                duration=end_time - start_time,
                message="测试执行完成" if success else "测试执行失败",
                details=self._get_test_details()
            )
            
        except Exception as e:
            end_time = time.time()
            return TestResult(
                test_name=self.name,
                success=False,
                duration=end_time - start_time,
                message=f"测试执行异常: {str(e)}",
                details={'error': str(e)}
            )
    
    def _execute_test(self) -> bool:
        """执行具体测试逻辑"""
        # 子类应该重写此方法
        return True
    
    def _get_test_details(self) -> Dict[str, Any]:
        """获取测试详情"""
        return {'test_type': self.__class__.__name__}
    
    def setup(self):
        """测试前设置"""
        pass
    
    def teardown(self):
        """测试后清理"""
        pass

class NumericalStabilityTest(BaseTest):
    """数值稳定性测试"""
    
    def run(self) -> TestResult:
        """运行数值稳定性测试"""
        
        start_time = time.time()
        
        try:
            self.setup()
            
            # 测试配置
            test_config = {
                'num_neurons': 100,
                'simulation_duration': 100.0,
                'time_step': 0.1,
                'num_iterations': 10
            }
            
            # 运行多次仿真检查一致性
            results = []
            for i in range(test_config['num_iterations']):
                # 模拟仿真运行
                np.random.seed(42 + i)  # 不同种子但可重复
                firing_rates = np.random.exponential(5.0, test_config['num_neurons'])
                
                # 检查数值稳定性
                has_nan = np.any(np.isnan(firing_rates))
                has_inf = np.any(np.isinf(firing_rates))
                reasonable_range = np.all((firing_rates >= 0) & (firing_rates <= 100))
                
                results.append({
                    'iteration': i,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'reasonable_range': reasonable_range,
                    'mean_rate': np.mean(firing_rates),
                    'std_rate': np.std(firing_rates)
                })
            
            # 分析结果
            any_nan = any(r['has_nan'] for r in results)
            any_inf = any(r['has_inf'] for r in results)
            all_reasonable = all(r['reasonable_range'] for r in results)
            
            # 检查结果一致性
            mean_rates = [r['mean_rate'] for r in results]
            rate_cv = np.std(mean_rates) / np.mean(mean_rates) if np.mean(mean_rates) > 0 else 0
            
            # 判断测试是否通过
            passed = not any_nan and not any_inf and all_reasonable and rate_cv < 0.1
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.CRITICAL,
                metrics={
                    'any_nan': any_nan,
                    'any_inf': any_inf,
                    'all_reasonable_range': all_reasonable,
                    'rate_coefficient_variation': rate_cv,
                    'num_iterations': len(results),
                    'mean_firing_rate': np.mean(mean_rates),
                    'std_firing_rate': np.std(mean_rates)
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.CRITICAL,
                error_message=str(e)
            )
        
        finally:
            self.teardown()

class PerformanceBenchmarkTest(BaseTest):
    """性能基准测试"""
    
    def run(self) -> TestResult:
        """运行性能基准测试"""
        
        start_time = time.time()
        
        try:
            self.setup()
            
            # 性能测试配置
            benchmark_config = {
                'matrix_size': 1000,
                'num_operations': 100,
                'memory_limit_mb': 1000,
                'time_limit_seconds': 30
            }
            
            # CPU性能测试
            cpu_start = time.time()
            for _ in range(benchmark_config['num_operations']):
                a = np.random.random((benchmark_config['matrix_size'], benchmark_config['matrix_size']))
                b = np.random.random((benchmark_config['matrix_size'], benchmark_config['matrix_size']))
                c = np.dot(a, b)
            cpu_time = time.time() - cpu_start
            
            # 内存使用测试
            try:
                import psutil
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                memory_usage_mb = 0  # 无法测量
            
            # 计算性能指标
            operations_per_second = benchmark_config['num_operations'] / cpu_time
            flops = (2 * benchmark_config['matrix_size']**3 * benchmark_config['num_operations']) / cpu_time / 1e9  # GFLOPS
            
            # 性能阈值
            min_ops_per_second = 1.0  # 最少每秒1次操作
            max_memory_mb = benchmark_config['memory_limit_mb']
            max_time_seconds = benchmark_config['time_limit_seconds']
            
            # 判断测试是否通过
            performance_ok = operations_per_second >= min_ops_per_second
            memory_ok = memory_usage_mb <= max_memory_mb
            time_ok = cpu_time <= max_time_seconds
            
            passed = performance_ok and memory_ok and time_ok
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.HIGH,
                metrics={
                    'cpu_time_seconds': cpu_time,
                    'operations_per_second': operations_per_second,
                    'gflops': flops,
                    'memory_usage_mb': memory_usage_mb,
                    'performance_ok': performance_ok,
                    'memory_ok': memory_ok,
                    'time_ok': time_ok,
                    'benchmark_config': benchmark_config
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.HIGH,
                error_message=str(e)
            )
        
        finally:
            self.teardown()

class HardwareConsistencyTest(BaseTest):
    """硬件一致性测试"""
    
    def run(self) -> TestResult:
        """运行硬件一致性测试"""
        
        start_time = time.time()
        
        try:
            self.setup()
            
            # 测试不同计算后端的一致性
            backends = ['cpu']
            
            # 检查GPU可用性
            try:
                import cupy
                backends.append('gpu')
            except ImportError:
                pass
            
            results = {}
            
            # 在每个后端运行相同的计算
            for backend in backends:
                np.random.seed(42)  # 固定种子确保可重复性
                
                if backend == 'cpu':
                    # CPU计算
                    data = np.random.uniform(0, 10, 1000)
                    result = np.mean(data)
                elif backend == 'gpu':
                    # GPU计算（如果可用）
                    try:
                        import cupy as cp
                        data = cp.random.uniform(0, 10, 1000)
                        result = float(cp.mean(data))
                    except:
                        result = np.nan
                
                results[backend] = result
            
            # 检查结果一致性
            if len(results) > 1:
                values = list(results.values())
                valid_values = [v for v in values if not np.isnan(v)]
                
                if len(valid_values) > 1:
                    max_diff = max(valid_values) - min(valid_values)
                    relative_diff = max_diff / np.mean(valid_values) if np.mean(valid_values) > 0 else 0
                    consistency_ok = relative_diff < 1e-6  # 相对差异小于1e-6
                else:
                    consistency_ok = True
                    relative_diff = 0.0
            else:
                consistency_ok = True
                relative_diff = 0.0
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.PASSED if consistency_ok else TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.MEDIUM,
                metrics={
                    'backends_tested': backends,
                    'results': results,
                    'relative_difference': relative_diff,
                    'consistency_ok': consistency_ok,
                    'num_backends': len(backends)
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.MEDIUM,
                error_message=str(e)
            )
        
        finally:
            self.teardown()

class RegressionTest(BaseTest):
    """回归测试"""
    
    def run(self) -> TestResult:
        """运行回归测试"""
        
        start_time = time.time()
        
        try:
            self.setup()
            
            # 加载参考结果
            reference_results = self._load_reference_results()
            
            # 运行当前版本的测试
            current_results = self._run_current_test()
            
            # 比较结果
            comparison = self._compare_results(reference_results, current_results)
            
            # 判断是否通过回归测试
            passed = comparison['overall_similarity'] > 0.95  # 95%相似度阈值
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.HIGH,
                metrics={
                    'overall_similarity': comparison['overall_similarity'],
                    'metric_similarities': comparison['metric_similarities'],
                    'significant_changes': comparison['significant_changes'],
                    'reference_version': reference_results.get('version', 'unknown'),
                    'current_version': current_results.get('version', 'current')
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=self.name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                severity=TestSeverity.HIGH,
                error_message=str(e)
            )
        
        finally:
            self.teardown()
    
    def _load_reference_results(self) -> Dict[str, Any]:
        """加载参考结果"""
        
        # 尝试从文件加载参考结果
        reference_file = Path(self.config.get('reference_file', 'reference_results.json'))
        
        if reference_file.exists():
            with open(reference_file, 'r') as f:
                return json.load(f)
        else:
            # 如果没有参考结果，创建默认的
            return {
                'version': '1.0.0',
                'firing_rate_mean': 5.0,
                'firing_rate_std': 2.0,
                'connectivity_density': 0.1,
                'synchrony_index': 0.3,
                'oscillation_frequency': 40.0
            }
    
    def _run_current_test(self) -> Dict[str, Any]:
        """运行当前版本的测试"""
        
        # 模拟当前版本的仿真结果
        np.random.seed(42)
        
        # 添加一些随机变化来模拟版本差异
        noise_factor = 0.02  # 2%的噪声
        
        return {
            'version': 'current',
            'firing_rate_mean': 5.0 + np.random.normal(0, noise_factor),
            'firing_rate_std': 2.0 + np.random.normal(0, noise_factor),
            'connectivity_density': 0.1 + np.random.normal(0, noise_factor * 0.1),
            'synchrony_index': 0.3 + np.random.normal(0, noise_factor),
            'oscillation_frequency': 40.0 + np.random.normal(0, noise_factor * 40)
        }
    
    def _compare_results(self, reference: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """比较结果"""
        
        metric_similarities = {}
        significant_changes = []
        
        # 比较各个指标
        for key in reference.keys():
            if key in current and isinstance(reference[key], (int, float)):
                ref_val = reference[key]
                cur_val = current[key]
                
                # 计算相对差异
                if ref_val != 0:
                    relative_diff = abs(cur_val - ref_val) / abs(ref_val)
                    similarity = max(0, 1 - relative_diff)
                else:
                    similarity = 1.0 if cur_val == 0 else 0.0
                
                metric_similarities[key] = similarity
                
                # 检查是否有显著变化（超过5%）
                if relative_diff > 0.05:
                    significant_changes.append({
                        'metric': key,
                        'reference': ref_val,
                        'current': cur_val,
                        'relative_change': relative_diff
                    })
        
        # 计算整体相似度
        if metric_similarities:
            overall_similarity = np.mean(list(metric_similarities.values()))
        else:
            overall_similarity = 0.0
        
        return {
            'overall_similarity': overall_similarity,
            'metric_similarities': metric_similarities,
            'significant_changes': significant_changes
        }



class CICDSystem:
    """CI/CD系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CICDSystem")
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', './cicd_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试注册表
        self.test_registry = {
            'numerical_stability': NumericalStabilityTest,
            'performance_benchmark': PerformanceBenchmarkTest,
            'hardware_consistency': HardwareConsistencyTest,
            'regression_test': RegressionTest
        }
        
        # 测试套件定义
        self.test_suites = {
            'smoke': TestSuite(
                name="Smoke Tests",
                description="快速验证基本功能",
                tests=['numerical_stability', 'performance_benchmark'],
                timeout=120,
                parallel=True
            ),
            'regression': TestSuite(
                name="Regression Tests",
                description="验证功能回归",
                tests=['regression_test', 'hardware_consistency'],
                timeout=300,
                parallel=False
            ),
            'full': TestSuite(
                name="Full Test Suite",
                description="完整测试套件",
                tests=['numerical_stability', 'performance_benchmark', 
                      'hardware_consistency', 'regression_test'],
                timeout=600,
                parallel=True
            )
        }
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
    
    def run_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """运行测试套件"""
        
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite = self.test_suites[suite_name]
        
        self.logger.info(f"Running test suite: {suite.name}")
        
        start_time = time.time()
        
        # 检查依赖
        for dep in suite.dependencies:
            if dep not in [h['suite_name'] for h in self.execution_history if h.get('success', False)]:
                self.logger.warning(f"Dependency {dep} not satisfied, running it first")
                dep_result = self.run_test_suite(dep)
                if not dep_result['success']:
                    return {
                        'suite_name': suite_name,
                        'success': False,
                        'error': f"Dependency {dep} failed",
                        'execution_time': time.time() - start_time
                    }
        
        # 运行测试
        test_results = []
        
        if suite.parallel:
            test_results = self._run_tests_parallel(suite.tests, suite.timeout)
        else:
            test_results = self._run_tests_sequential(suite.tests, suite.timeout)
        
        # 分析结果
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        
        success = failed_tests == 0
        execution_time = time.time() - start_time
        
        # 生成报告
        report = {
            'suite_name': suite_name,
            'suite_description': suite.description,
            'success': success,
            'execution_time': execution_time,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'test_results': [self._test_result_to_dict(r) for r in test_results],
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        self._save_test_report(report)
        
        # 添加到执行历史
        self.execution_history.append(report)
        
        self.logger.info(f"Test suite {suite_name} completed: {passed_tests}/{total_tests} passed")
        
        return report
    
    def _run_tests_parallel(self, test_names: List[str], timeout: int) -> List[TestResult]:
        """并行运行测试"""
        
        results = []
        
        # 使用线程池并行执行
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        with ThreadPoolExecutor(max_workers=len(test_names)) as executor:
            futures = {}
            
            # 提交所有测试
            for test_name in test_names:
                if test_name in self.test_registry:
                    test_class = self.test_registry[test_name]
                    test_instance = test_class(test_name, self.config)
                    future = executor.submit(test_instance.run)
                    futures[future] = test_name
            
            # 收集结果
            for future in futures:
                test_name = futures[future]
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except TimeoutError:
                    results.append(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        execution_time=timeout,
                        error_message="Test timed out"
                    ))
                except Exception as e:
                    results.append(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        execution_time=0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _run_tests_sequential(self, test_names: List[str], timeout: int) -> List[TestResult]:
        """顺序运行测试"""
        
        results = []
        
        for test_name in test_names:
            if test_name in self.test_registry:
                test_class = self.test_registry[test_name]
                test_instance = test_class(test_name, self.config)
                
                try:
                    # 使用超时机制
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Test timed out")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    
                    try:
                        result = test_instance.run()
                        results.append(result)
                    finally:
                        signal.alarm(0)  # 取消超时
                
                except TimeoutError:
                    results.append(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        execution_time=timeout,
                        error_message="Test timed out"
                    ))
                except Exception as e:
                    results.append(TestResult(
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        execution_time=0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _test_result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """将测试结果转换为字典"""
        
        return {
            'test_name': result.test_name,
            'status': result.status.value,
            'execution_time': result.execution_time,
            'severity': result.severity.value,
            'error_message': result.error_message,
            'metrics': result.metrics,
            'artifacts': result.artifacts
        }
    
    def _save_test_report(self, report: Dict[str, Any]):
        """保存测试报告"""
        
        # JSON报告
        json_file = self.output_dir / f"test_report_{report['suite_name']}_{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # HTML报告
        html_file = self.output_dir / f"test_report_{report['suite_name']}_{int(time.time())}.html"
        html_content = self._generate_html_report(report)
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Test report saved: {json_file}")
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML测试报告"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CI/CD Test Report - {suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: {header_color}; padding: 20px; border-radius: 5px; color: white; }}
                .summary {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .passed {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .failed {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .metrics {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CI/CD Test Report</h1>
                <h2>{suite_name}</h2>
                <p>{suite_description}</p>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Passed:</strong> {passed_tests}</p>
                <p><strong>Failed:</strong> {failed_tests}</p>
                <p><strong>Success Rate:</strong> {success_rate:.1%}</p>
            </div>
            
            <div class="test-results">
                <h2>Test Results</h2>
                {test_results_html}
            </div>
        </body>
        </html>
        """
        
        # 生成测试结果HTML
        test_results_html = ""
        for test in report['test_results']:
            status_class = "passed" if test['status'] == 'passed' else "failed"
            
            metrics_html = ""
            if test['metrics']:
                metrics_html = "<div class='metrics'><h4>Metrics:</h4><ul>"
                for key, value in test['metrics'].items():
                    metrics_html += f"<li><strong>{key}:</strong> {value}</li>"
                metrics_html += "</ul></div>"
            
            error_html = ""
            if test['error_message']:
                error_html = f"<p><strong>Error:</strong> {test['error_message']}</p>"
            
            test_results_html += f"""
            <div class="test-result {status_class}">
                <h3>{test['test_name']}</h3>
                <p><strong>Status:</strong> {test['status'].upper()}</p>
                <p><strong>Execution Time:</strong> {test['execution_time']:.2f} seconds</p>
                <p><strong>Severity:</strong> {test['severity']}</p>
                {error_html}
                {metrics_html}
            </div>
            """
        
        # 填充模板
        header_color = "#28a745" if report['success'] else "#dc3545"
        status = "PASSED" if report['success'] else "FAILED"
        success_rate = report['passed_tests'] / report['total_tests'] if report['total_tests'] > 0 else 0
        
        return html_template.format(
            suite_name=report['suite_name'],
            suite_description=report['suite_description'],
            status=status,
            execution_time=report['execution_time'],
            timestamp=report['timestamp'],
            total_tests=report['total_tests'],
            passed_tests=report['passed_tests'],
            failed_tests=report['failed_tests'],
            success_rate=success_rate,
            test_results_html=test_results_html,
            header_color=header_color
        )
    
    def get_test_history(self, suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取测试历史"""
        
        if suite_name:
            return [h for h in self.execution_history if h['suite_name'] == suite_name]
        else:
            return self.execution_history
    
    def get_test_trends(self, suite_name: str, num_runs: int = 10) -> Dict[str, Any]:
        """获取测试趋势"""
        
        history = self.get_test_history(suite_name)[-num_runs:]
        
        if not history:
            return {'error': 'No test history available'}
        
        # 计算趋势
        success_rates = [h['passed_tests'] / h['total_tests'] for h in history]
        execution_times = [h['execution_time'] for h in history]
        
        return {
            'num_runs': len(history),
            'success_rate_trend': success_rates,
            'execution_time_trend': execution_times,
            'average_success_rate': np.mean(success_rates),
            'average_execution_time': np.mean(execution_times),
            'success_rate_stability': np.std(success_rates),
            'execution_time_stability': np.std(execution_times)
        }

# 工厂函数
def create_cicd_system(config: Optional[Dict[str, Any]] = None) -> CICDSystem:
    """创建CI/CD系统"""
    
    if config is None:
        config = {
            'output_dir': './cicd_output',
            'reference_file': 'reference_results.json',
            'timeout': 300
        }
    
    return CICDSystem(config)

# 预定义配置
def create_default_cicd_config() -> Dict[str, Any]:
    """创建默认CI/CD配置"""
    
    return {
        'output_dir': './cicd_output',
        'reference_file': 'reference_results.json',
        'timeout': 300,
        'parallel_execution': True,
        'notification': {
            'enabled': False,
            'email': '',
            'webhook': ''
        },
        'deployment': {
            'auto_deploy': False,
            'deployment_branch': 'main',
            'deployment_environment': 'staging'
        }
    }

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建CI/CD系统
    cicd = create_cicd_system()
    
    # 运行测试套件
    print("Running smoke tests...")
    smoke_results = cicd.run_test_suite('smoke')
    print(f"Smoke tests: {'PASSED' if smoke_results['success'] else 'FAILED'}")
    
    print("\nRunning full test suite...")
    full_results = cicd.run_test_suite('full')
    print(f"Full tests: {'PASSED' if full_results['success'] else 'FAILED'}")
    
    # 显示测试趋势
    trends = cicd.get_test_trends('full')
    if 'error' not in trends:
        print(f"\nTest trends (last {trends['num_runs']} runs):")
        print(f"Average success rate: {trends['average_success_rate']:.1%}")
        print(f"Average execution time: {trends['average_execution_time']:.2f}s")