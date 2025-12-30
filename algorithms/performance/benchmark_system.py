"""
算法性能基准测试系统

生产级的算法性能测试和监控系统，提供全面的性能分析、
回归测试、性能优化验证等功能。
"""

import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum

import numpy as np


class BenchmarkStatus(Enum):
    """基准测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    algorithm_name: str
    test_sizes: List[int]
    iterations: int = 3
    timeout: float = 300.0  # 5分钟超时
    warmup_iterations: int = 1
    memory_profiling: bool = False
    cpu_profiling: bool = False
    parallel_execution: bool = False


@dataclass
class PerformanceMetrics:
    """性能指标"""
    algorithm_name: str
    input_size: int
    execution_time: float
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # 计算吞吐量（每秒处理的元素数）
        if self.throughput is None and self.execution_time > 0:
            self.throughput = self.input_size / self.execution_time


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    config: BenchmarkConfig
    metrics: List[PerformanceMetrics]
    status: BenchmarkStatus
    start_time: str
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取汇总统计信息"""
        if not self.metrics:
            return {}
        
        # 按输入大小分组
        size_groups = {}
        for metric in self.metrics:
            size = metric.input_size
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(metric)
        
        summary = {}
        for size, group_metrics in size_groups.items():
            execution_times = [m.execution_time for m in group_metrics]
            throughputs = [m.throughput for m in group_metrics if m.throughput]
            
            size_summary = {
                "input_size": size,
                "sample_count": len(execution_times),
                "execution_time": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    "min": min(execution_times),
                    "max": max(execution_times)
                }
            }
            
            if throughputs:
                size_summary["throughput"] = {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                    "min": min(throughputs),
                    "max": max(throughputs)
                }
            
            summary[f"size_{size}"] = size_summary
        
        return summary


class DataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_random_integers(size: int, min_val: int = 0, max_val: Optional[int] = None) -> List[int]:
        """生成随机整数列表"""
        if max_val is None:
            max_val = size * 2
        return np.random.randint(min_val, max_val, size).tolist()
    
    @staticmethod
    def generate_sorted_integers(size: int, reverse: bool = False) -> List[int]:
        """生成有序整数列表"""
        data = list(range(size))
        return data[::-1] if reverse else data
    
    @staticmethod
    def generate_nearly_sorted(size: int, disorder_ratio: float = 0.1) -> List[int]:
        """生成接近有序的数据"""
        data = list(range(size))
        disorder_count = int(size * disorder_ratio)
        
        for _ in range(disorder_count):
            i, j = np.random.choice(size, 2, replace=False)
            data[i], data[j] = data[j], data[i]
        
        return data
    
    @staticmethod
    def generate_duplicate_heavy(size: int, unique_ratio: float = 0.1) -> List[int]:
        """生成重复元素较多的数据"""
        unique_count = max(1, int(size * unique_ratio))
        unique_values = list(range(unique_count))
        return np.random.choice(unique_values, size).tolist()


class PerformanceBenchmark:
    """
    性能基准测试系统
    
    提供全面的算法性能测试、分析和监控功能。
    """
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        """
        初始化基准测试系统
        
        Args:
            results_dir: 结果存储目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.data_generator = DataGenerator()
        self._active_benchmarks: Dict[str, BenchmarkResult] = {}
    
    def _setup_logging(self) -> None:
        """设置日志记录"""
        log_file = self.results_dir / "benchmark.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def run_benchmark(self, algorithm_func: Callable, config: BenchmarkConfig) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            algorithm_func: 算法函数
            config: 测试配置
            
        Returns:
            测试结果
        """
        benchmark_id = f"{config.algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = BenchmarkResult(
            config=config,
            metrics=[],
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now().isoformat()
        )
        
        self._active_benchmarks[benchmark_id] = result
        
        try:
            self.logger.info(f"开始基准测试: {config.algorithm_name}")
            
            for size in config.test_sizes:
                self.logger.info(f"测试数据大小: {size}")
                
                # 生成测试数据
                test_data = self._generate_test_data(size)
                
                # 预热运行
                if config.warmup_iterations > 0:
                    self._run_warmup(algorithm_func, test_data, config.warmup_iterations)
                
                # 正式测试
                for iteration in range(config.iterations):
                    try:
                        metrics = self._measure_performance(
                            algorithm_func, test_data, config.algorithm_name, size
                        )
                        result.metrics.append(metrics)
                        
                    except Exception as e:
                        self.logger.error(f"测试迭代失败 (size={size}, iter={iteration}): {e}")
                        continue
            
            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.now().isoformat()
            
            self.logger.info(f"基准测试完成: {config.algorithm_name}")
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now().isoformat()
            
            self.logger.error(f"基准测试失败: {config.algorithm_name} - {e}")
        
        finally:
            # 保存结果
            self._save_result(benchmark_id, result)
            del self._active_benchmarks[benchmark_id]
        
        return result
    
    def run_comparative_benchmark(self, algorithms: Dict[str, Callable], 
                                 test_sizes: List[int], iterations: int = 3) -> Dict[str, BenchmarkResult]:
        """
        运行对比基准测试
        
        Args:
            algorithms: 算法字典 {名称: 函数}
            test_sizes: 测试数据大小列表
            iterations: 迭代次数
            
        Returns:
            测试结果字典
        """
        results = {}
        
        for name, algorithm_func in algorithms.items():
            config = BenchmarkConfig(
                algorithm_name=name,
                test_sizes=test_sizes,
                iterations=iterations
            )
            
            result = self.run_benchmark(algorithm_func, config)
            results[name] = result
        
        # 生成对比报告
        self._generate_comparative_report(results)
        
        return results
    
    def run_regression_test(self, algorithm_func: Callable, algorithm_name: str,
                           baseline_file: str, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        运行性能回归测试
        
        Args:
            algorithm_func: 算法函数
            algorithm_name: 算法名称
            baseline_file: 基线结果文件
            tolerance: 性能容忍度（百分比）
            
        Returns:
            回归测试结果
        """
        # 加载基线数据
        baseline_result = self._load_baseline(baseline_file)
        if not baseline_result:
            raise ValueError(f"无法加载基线数据: {baseline_file}")
        
        # 运行当前测试
        config = baseline_result.config
        current_result = self.run_benchmark(algorithm_func, config)
        
        # 比较性能
        regression_analysis = self._analyze_regression(
            baseline_result, current_result, tolerance
        )
        
        # 保存回归测试报告
        report_file = self.results_dir / f"regression_{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(regression_analysis, f, indent=2, ensure_ascii=False)
        
        return regression_analysis
    
    def _generate_test_data(self, size: int) -> List[int]:
        """生成测试数据"""
        # 使用多种数据模式进行测试
        return self.data_generator.generate_random_integers(size)
    
    def _run_warmup(self, algorithm_func: Callable, test_data: List[int], iterations: int) -> None:
        """运行预热"""
        for _ in range(iterations):
            try:
                algorithm_func(test_data.copy())
            except Exception:
                pass  # 预热阶段忽略错误
    
    def _measure_performance(self, algorithm_func: Callable, test_data: List[int],
                           algorithm_name: str, input_size: int) -> PerformanceMetrics:
        """测量性能指标"""
        # 复制数据以避免原地修改影响测试
        data_copy = test_data.copy()
        
        # 测量执行时间
        start_time = time.perf_counter()
        result = algorithm_func(data_copy)
        execution_time = time.perf_counter() - start_time
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            algorithm_name=algorithm_name,
            input_size=input_size,
            execution_time=execution_time
        )
        
        return metrics
    
    def _save_result(self, benchmark_id: str, result: BenchmarkResult) -> None:
        """保存测试结果"""
        result_file = self.results_dir / f"{benchmark_id}.json"
        
        # 转换为可序列化的字典
        result_dict = asdict(result)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    def _load_baseline(self, baseline_file: str) -> Optional[BenchmarkResult]:
        """加载基线结果"""
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重构BenchmarkResult对象
            config = BenchmarkConfig(**data['config'])
            metrics = [PerformanceMetrics(**m) for m in data['metrics']]
            
            return BenchmarkResult(
                config=config,
                metrics=metrics,
                status=BenchmarkStatus(data['status']),
                start_time=data['start_time'],
                end_time=data.get('end_time'),
                error_message=data.get('error_message')
            )
            
        except Exception as e:
            self.logger.error(f"加载基线数据失败: {e}")
            return None
    
    def _analyze_regression(self, baseline: BenchmarkResult, current: BenchmarkResult,
                          tolerance: float) -> Dict[str, Any]:
        """分析性能回归"""
        baseline_summary = baseline.get_summary_statistics()
        current_summary = current.get_summary_statistics()
        
        regression_results = {
            "test_timestamp": datetime.now().isoformat(),
            "algorithm_name": current.config.algorithm_name,
            "tolerance": tolerance,
            "overall_status": "PASS",
            "size_comparisons": {}
        }
        
        for size_key in baseline_summary.keys():
            if size_key not in current_summary:
                continue
            
            baseline_time = baseline_summary[size_key]["execution_time"]["mean"]
            current_time = current_summary[size_key]["execution_time"]["mean"]
            
            performance_change = (current_time - baseline_time) / baseline_time
            
            size_result = {
                "baseline_time": baseline_time,
                "current_time": current_time,
                "performance_change": performance_change,
                "status": "PASS" if abs(performance_change) <= tolerance else "FAIL"
            }
            
            if size_result["status"] == "FAIL":
                regression_results["overall_status"] = "FAIL"
            
            regression_results["size_comparisons"][size_key] = size_result
        
        return regression_results
    
    def _generate_comparative_report(self, results: Dict[str, BenchmarkResult]) -> None:
        """生成对比报告"""
        report_file = self.results_dir / f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "algorithms": list(results.keys()),
            "summary": {}
        }
        
        # 为每个算法生成摘要
        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                report["summary"][name] = result.get_summary_statistics()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"对比报告已生成: {report_file}")


def create_sorting_benchmark() -> PerformanceBenchmark:
    """创建排序算法基准测试实例"""
    return PerformanceBenchmark("benchmarks/sorting")


def create_search_benchmark() -> PerformanceBenchmark:
    """创建搜索算法基准测试实例"""
    return PerformanceBenchmark("benchmarks/searching")