"""
算法管理器 - 生产级算法执行和管理系统

提供统一的算法接口，支持算法注册、执行、监控和错误处理。
适用于生产环境的高性能算法管理系统。
"""

import logging
import time
from typing import Dict, Any, Optional, Type, List, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from enum import Enum

from .base import Algorithm
from .sorting.basic.bubble_sort import BubbleSort
from .sorting.basic.quick_sort import QuickSort
from .searching.basic.linear_search import LinearSearch
from .searching.basic.binary_search import BinarySearch
from .dynamic_programming.basic.fibonacci import Fibonacci
from .dynamic_programming.basic.lcs import LongestCommonSubsequence
from .graph.basic.bfs import BreadthFirstSearch
from .graph.basic.dfs import DepthFirstSearch
from .storage.basic.lru_cache import LRUCache
from .storage.basic.lfu_cache import LFUCache
from .storage.advanced.btree_index import BTreeIndex
from .causal.causal_graph import CausalGraph


class AlgorithmCategory(Enum):
    """算法分类枚举"""
    SORTING = "sorting"
    SEARCHING = "searching"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GRAPH = "graph"
    STORAGE = "storage"
    CAUSAL = "causal"


@dataclass
class AlgorithmMetrics:
    """算法执行指标"""
    execution_time: float
    memory_usage: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    input_size: Optional[int] = None


@dataclass
class AlgorithmConfig:
    """算法配置"""
    timeout: float = 30.0  # 超时时间（秒）
    max_retries: int = 3   # 最大重试次数
    enable_metrics: bool = True  # 是否启用指标收集
    parallel_execution: bool = False  # 是否支持并行执行


class AlgorithmRegistry:
    """算法注册表"""
    
    def __init__(self):
        self._algorithms: Dict[str, Type[Algorithm]] = {}
        self._categories: Dict[str, AlgorithmCategory] = {}
        self._configs: Dict[str, AlgorithmConfig] = {}
        self._register_default_algorithms()
    
    def _register_default_algorithms(self) -> None:
        """注册默认算法"""
        # 排序算法
        self.register("bubble_sort", BubbleSort, AlgorithmCategory.SORTING)
        self.register("quick_sort", QuickSort, AlgorithmCategory.SORTING)
        
        # 搜索算法
        self.register("linear_search", LinearSearch, AlgorithmCategory.SEARCHING)
        self.register("binary_search", BinarySearch, AlgorithmCategory.SEARCHING)
        
        # 动态规划算法
        self.register("fibonacci", Fibonacci, AlgorithmCategory.DYNAMIC_PROGRAMMING)
        self.register("lcs", LongestCommonSubsequence, AlgorithmCategory.DYNAMIC_PROGRAMMING)
        
        # 图算法
        self.register("bfs", BreadthFirstSearch, AlgorithmCategory.GRAPH)
        self.register("dfs", DepthFirstSearch, AlgorithmCategory.GRAPH)
        
        # 存储算法
        self.register("lru_cache", LRUCache, AlgorithmCategory.STORAGE)
        self.register("lfu_cache", LFUCache, AlgorithmCategory.STORAGE)
        self.register("btree_index", BTreeIndex, AlgorithmCategory.STORAGE)
        
        # 因果推理算法
        self.register("causal_graph", CausalGraph, AlgorithmCategory.CAUSAL)
    
    def register(self, name: str, algorithm_class: Type[Algorithm], 
                category: AlgorithmCategory, config: Optional[AlgorithmConfig] = None) -> None:
        """
        注册算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
            category: 算法分类
            config: 算法配置
        """
        if not issubclass(algorithm_class, Algorithm):
            raise ValueError(f"算法类 {algorithm_class} 必须继承自 Algorithm")
        
        self._algorithms[name] = algorithm_class
        self._categories[name] = category
        self._configs[name] = config or AlgorithmConfig()
    
    def get_algorithm(self, name: str) -> Type[Algorithm]:
        """获取算法类"""
        if name not in self._algorithms:
            raise KeyError(f"未找到算法: {name}")
        return self._algorithms[name]
    
    def get_category(self, name: str) -> AlgorithmCategory:
        """获取算法分类"""
        return self._categories.get(name)
    
    def get_config(self, name: str) -> AlgorithmConfig:
        """获取算法配置"""
        return self._configs.get(name, AlgorithmConfig())
    
    def list_algorithms(self, category: Optional[AlgorithmCategory] = None) -> List[str]:
        """列出算法"""
        if category is None:
            return list(self._algorithms.keys())
        return [name for name, cat in self._categories.items() if cat == category]


class AlgorithmManager:
    """
    算法管理器 - 生产级算法执行和管理系统
    
    提供统一的算法接口，支持算法注册、执行、监控和错误处理。
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.registry = AlgorithmRegistry()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._metrics_history: Dict[str, List[AlgorithmMetrics]] = {}
    
    def execute_algorithm(self, algorithm_name: str, *args, **kwargs) -> Any:
        """
        执行算法
        
        Args:
            algorithm_name: 算法名称
            *args: 算法参数
            **kwargs: 算法关键字参数
            
        Returns:
            算法执行结果
            
        Raises:
            KeyError: 算法不存在
            TimeoutError: 执行超时
            Exception: 算法执行错误
        """
        algorithm_class = self.registry.get_algorithm(algorithm_name)
        config = self.registry.get_config(algorithm_name)
        
        start_time = time.perf_counter()
        
        try:
            # 创建算法实例
            algorithm = algorithm_class()
            
            # 执行算法
            if config.parallel_execution and hasattr(algorithm, 'execute_parallel'):
                result = algorithm.execute_parallel(*args, **kwargs)
            else:
                result = algorithm.execute(*args, **kwargs)
            
            execution_time = time.perf_counter() - start_time
            
            # 记录指标
            if config.enable_metrics:
                metrics = AlgorithmMetrics(
                    execution_time=execution_time,
                    success=True,
                    input_size=self._estimate_input_size(args, kwargs)
                )
                self._record_metrics(algorithm_name, metrics)
            
            self.logger.info(f"算法 {algorithm_name} 执行成功，耗时: {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            # 记录错误指标
            if config.enable_metrics:
                metrics = AlgorithmMetrics(
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e),
                    input_size=self._estimate_input_size(args, kwargs)
                )
                self._record_metrics(algorithm_name, metrics)
            
            self.logger.error(f"算法 {algorithm_name} 执行失败: {e}")
            raise
    
    def execute_algorithm_async(self, algorithm_name: str, *args, **kwargs) -> Future:
        """
        异步执行算法
        
        Args:
            algorithm_name: 算法名称
            *args: 算法参数
            **kwargs: 算法关键字参数
            
        Returns:
            Future对象
        """
        return self.executor.submit(self.execute_algorithm, algorithm_name, *args, **kwargs)
    
    def batch_execute(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """
        批量执行算法
        
        Args:
            tasks: 任务列表，每个任务包含算法名称和参数
            
        Returns:
            执行结果列表
        """
        futures = []
        for task in tasks:
            algorithm_name = task['algorithm']
            args = task.get('args', [])
            kwargs = task.get('kwargs', {})
            future = self.execute_algorithm_async(algorithm_name, *args, **kwargs)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量执行任务失败: {e}")
                results.append(None)
        
        return results
    
    def get_metrics(self, algorithm_name: str) -> List[AlgorithmMetrics]:
        """获取算法执行指标"""
        return self._metrics_history.get(algorithm_name, [])
    
    def get_performance_summary(self, algorithm_name: str) -> Dict[str, Any]:
        """
        获取算法性能摘要
        
        Args:
            algorithm_name: 算法名称
            
        Returns:
            性能摘要字典
        """
        metrics = self.get_metrics(algorithm_name)
        if not metrics:
            return {}
        
        successful_metrics = [m for m in metrics if m.success]
        if not successful_metrics:
            return {"total_executions": len(metrics), "success_rate": 0.0}
        
        execution_times = [m.execution_time for m in successful_metrics]
        
        return {
            "total_executions": len(metrics),
            "successful_executions": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(metrics),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_execution_time": sum(execution_times)
        }
    
    def _estimate_input_size(self, args: tuple, kwargs: dict) -> Optional[int]:
        """估算输入数据大小"""
        try:
            total_size = 0
            for arg in args:
                if hasattr(arg, '__len__'):
                    total_size += len(arg)
            for value in kwargs.values():
                if hasattr(value, '__len__'):
                    total_size += len(value)
            return total_size if total_size > 0 else None
        except:
            return None
    
    def _record_metrics(self, algorithm_name: str, metrics: AlgorithmMetrics) -> None:
        """记录算法执行指标"""
        if algorithm_name not in self._metrics_history:
            self._metrics_history[algorithm_name] = []
        
        self._metrics_history[algorithm_name].append(metrics)
        
        # 限制历史记录数量
        max_history = 1000
        if len(self._metrics_history[algorithm_name]) > max_history:
            self._metrics_history[algorithm_name] = self._metrics_history[algorithm_name][-max_history:]
    
    def shutdown(self) -> None:
        """关闭算法管理器"""
        self.executor.shutdown(wait=True)


# 全局算法管理器实例
_algorithm_manager = None


def get_algorithm_manager() -> AlgorithmManager:
    """获取全局算法管理器实例"""
    global _algorithm_manager
    if _algorithm_manager is None:
        _algorithm_manager = AlgorithmManager()
    return _algorithm_manager


def execute_algorithm(algorithm_name: str, *args, **kwargs) -> Any:
    """便捷函数：执行算法"""
    return get_algorithm_manager().execute_algorithm(algorithm_name, *args, **kwargs)


def register_algorithm(name: str, algorithm_class: Type[Algorithm], 
                      category: AlgorithmCategory, config: Optional[AlgorithmConfig] = None) -> None:
    """便捷函数：注册算法"""
    get_algorithm_manager().registry.register(name, algorithm_class, category, config)