"""
生产级分布式大脑节点系统

提供高性能、容错的多节点任务执行能力，包含完整的监控、
负载均衡和故障恢复机制。
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import hashlib
import socket
from contextlib import contextmanager

try:
    from .distributed_brain_node import DistributedBrainNode
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from distributed_brain_node import DistributedBrainNode


@dataclass
class NodeMetrics:
    """节点性能指标"""
    node_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    average_task_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    is_healthy: bool = True


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    node_id: str
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DistributedSystemConfig:
    """分布式系统配置"""
    max_workers: int = mp.cpu_count()
    task_timeout: float = 30.0
    heartbeat_interval: float = 5.0
    max_retries: int = 3
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, random
    enable_monitoring: bool = True
    log_level: str = "INFO"
    backup_results: bool = True
    results_backup_path: Optional[Path] = None


class ProductionDistributedSystem:
    """生产级分布式系统管理器"""
    
    def __init__(self, config: Optional[DistributedSystemConfig] = None):
        """
        初始化分布式系统
        
        Args:
            config: 系统配置
        """
        self.config = config or DistributedSystemConfig()
        self.logger = self._setup_logging()
        
        # 系统状态
        self.is_running = False
        self.master_node: Optional[DistributedBrainNode] = None
        self.worker_processes: List[mp.Process] = []
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.task_queue: List[Any] = []
        self.completed_tasks: List[TaskResult] = []
        self.failed_tasks: List[TaskResult] = []
        
        # 同步原语
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # 网络配置
        self.base_port = self._find_available_port()
        self.authkey = self._generate_secure_authkey()
        
        # 性能统计
        self.system_start_time = time.time()
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _find_available_port(self, start_port: int = 50000) -> int:
        """查找可用端口"""
        for port in range(start_port, start_port + 1000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("无法找到可用端口")
    
    def _generate_secure_authkey(self) -> bytes:
        """生成安全的认证密钥"""
        timestamp = str(time.time()).encode()
        return hashlib.sha256(timestamp).digest()[:16]
    
    def start_system(self, tasks: List[Any], task_handler: Callable) -> bool:
        """
        启动分布式系统
        
        Args:
            tasks: 待处理任务列表
            task_handler: 任务处理函数
            
        Returns:
            启动是否成功
        """
        try:
            with self._lock:
                if self.is_running:
                    self.logger.warning("系统已在运行中")
                    return False
                
                self.logger.info(f"启动分布式系统，任务数量: {len(tasks)}")
                
                # 初始化任务队列
                self.task_queue = tasks.copy()
                self.completed_tasks.clear()
                self.failed_tasks.clear()
                
                # 启动主节点
                address = ("localhost", self.base_port)
                self.master_node = DistributedBrainNode(address, self.authkey)
                self.master_node.start_master(tasks)
                
                # 启动工作节点
                self._start_worker_nodes(task_handler)
                
                # 启动监控线程
                if self.config.enable_monitoring:
                    self._start_monitoring()
                
                self.is_running = True
                self.system_start_time = time.time()
                
                self.logger.info("分布式系统启动成功")
                return True
                
        except Exception as e:
            self.logger.error(f"启动系统失败: {e}")
            self.shutdown_system()
            return False
    
    def _start_worker_nodes(self, task_handler: Callable) -> None:
        """启动工作节点"""
        address = ("localhost", self.base_port)
        
        for i in range(self.config.max_workers):
            node_id = f"worker_{i}"
            
            # 创建增强的任务处理器
            enhanced_handler = self._create_enhanced_handler(task_handler, node_id)
            
            # 启动工作进程
            process = mp.Process(
                target=self._worker_process_target,
                args=(address, self.authkey, enhanced_handler, node_id)
            )
            process.start()
            self.worker_processes.append(process)
            
            # 初始化节点指标
            self.node_metrics[node_id] = NodeMetrics(node_id=node_id)
            
            self.logger.info(f"启动工作节点: {node_id}")
    
    def _create_enhanced_handler(self, original_handler: Callable, node_id: str) -> Callable:
        """创建增强的任务处理器，包含监控和错误处理"""
        def enhanced_handler(task: Any) -> Any:
            start_time = time.time()
            
            try:
                # 执行原始任务
                result = original_handler(task)
                
                # 记录成功指标
                execution_time = time.time() - start_time
                self._update_node_metrics(node_id, execution_time, success=True)
                
                return result
                
            except Exception as e:
                # 记录失败指标
                execution_time = time.time() - start_time
                self._update_node_metrics(node_id, execution_time, success=False)
                
                self.logger.error(f"节点 {node_id} 任务执行失败: {e}")
                raise
        
        return enhanced_handler
    
    def _worker_process_target(self, address: Tuple[str, int], authkey: bytes, 
                              handler: Callable, node_id: str) -> None:
        """工作进程目标函数"""
        try:
            node = DistributedBrainNode(address, authkey)
            node.run_worker(handler)
        except Exception as e:
            self.logger.error(f"工作节点 {node_id} 异常退出: {e}")
    
    def _update_node_metrics(self, node_id: str, execution_time: float, success: bool) -> None:
        """更新节点性能指标"""
        with self._lock:
            if node_id in self.node_metrics:
                metrics = self.node_metrics[node_id]
                
                if success:
                    metrics.tasks_completed += 1
                else:
                    metrics.tasks_failed += 1
                
                metrics.total_processing_time += execution_time
                total_tasks = metrics.tasks_completed + metrics.tasks_failed
                
                if total_tasks > 0:
                    metrics.average_task_time = metrics.total_processing_time / total_tasks
                
                metrics.last_heartbeat = time.time()
    
    def _start_monitoring(self) -> None:
        """启动监控线程"""
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while not self._shutdown_event.is_set():
            try:
                self._collect_system_metrics()
                self._check_node_health()
                self._log_system_status()
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
    
    def _heartbeat_loop(self) -> None:
        """心跳检测循环"""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                with self._lock:
                    for node_id, metrics in self.node_metrics.items():
                        # 检查心跳超时
                        if current_time - metrics.last_heartbeat > self.config.heartbeat_interval * 3:
                            if metrics.is_healthy:
                                self.logger.warning(f"节点 {node_id} 心跳超时")
                                metrics.is_healthy = False
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"心跳检测异常: {e}")
    
    def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        try:
            import psutil
            
            # 系统级指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.logger.debug(f"系统CPU使用率: {cpu_percent}%")
            self.logger.debug(f"系统内存使用率: {memory.percent}%")
            
        except ImportError:
            # psutil不可用时的简化指标收集
            pass
    
    def _check_node_health(self) -> None:
        """检查节点健康状态"""
        unhealthy_nodes = []
        
        with self._lock:
            for node_id, metrics in self.node_metrics.items():
                if not metrics.is_healthy:
                    unhealthy_nodes.append(node_id)
        
        if unhealthy_nodes:
            self.logger.warning(f"发现不健康节点: {unhealthy_nodes}")
    
    def _log_system_status(self) -> None:
        """记录系统状态"""
        with self._lock:
            total_completed = sum(m.tasks_completed for m in self.node_metrics.values())
            total_failed = sum(m.tasks_failed for m in self.node_metrics.values())
            healthy_nodes = sum(1 for m in self.node_metrics.values() if m.is_healthy)
            
            self.logger.info(
                f"系统状态 - 健康节点: {healthy_nodes}/{len(self.node_metrics)}, "
                f"已完成任务: {total_completed}, 失败任务: {total_failed}"
            )
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            任务结果列表
        """
        if not self.is_running or not self.master_node:
            raise RuntimeError("系统未运行")
        
        try:
            start_time = time.time()
            
            # 等待工作进程完成
            for process in self.worker_processes:
                remaining_timeout = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                
                process.join(timeout=remaining_timeout)
                
                if process.is_alive():
                    self.logger.warning(f"工作进程超时，强制终止: PID {process.pid}")
                    process.terminate()
                    process.join(timeout=5)
                    
                    if process.is_alive():
                        process.kill()
            
            # 收集结果
            results = self.master_node.gather_results(len(self.task_queue))
            
            # 转换为TaskResult对象
            task_results = []
            for i, result in enumerate(results):
                task_result = TaskResult(
                    task_id=f"task_{i}",
                    node_id="unknown",  # 实际实现中应该从结果中获取
                    result=result,
                    execution_time=0.0,  # 实际实现中应该记录执行时间
                    success=True
                )
                task_results.append(task_result)
            
            self.completed_tasks = task_results
            
            # 备份结果
            if self.config.backup_results:
                self._backup_results(task_results)
            
            return task_results
            
        except Exception as e:
            self.logger.error(f"等待任务完成时发生错误: {e}")
            raise
    
    def _backup_results(self, results: List[TaskResult]) -> None:
        """备份任务结果"""
        try:
            if self.config.results_backup_path:
                backup_path = self.config.results_backup_path
            else:
                backup_path = Path(f"results_backup_{int(time.time())}.json")
            
            backup_data = {
                'timestamp': time.time(),
                'system_config': {
                    'max_workers': self.config.max_workers,
                    'task_timeout': self.config.task_timeout
                },
                'results': [
                    {
                        'task_id': r.task_id,
                        'node_id': r.node_id,
                        'result': r.result,
                        'execution_time': r.execution_time,
                        'success': r.success,
                        'error_message': r.error_message,
                        'timestamp': r.timestamp
                    }
                    for r in results
                ]
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"结果已备份到: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"备份结果失败: {e}")
    
    def shutdown_system(self) -> None:
        """关闭分布式系统"""
        try:
            with self._lock:
                if not self.is_running:
                    return
                
                self.logger.info("开始关闭分布式系统")
                
                # 设置关闭标志
                self._shutdown_event.set()
                
                # 终止工作进程
                for process in self.worker_processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        
                        if process.is_alive():
                            process.kill()
                
                # 关闭主节点
                if self.master_node:
                    self.master_node.shutdown()
                
                # 等待监控线程结束
                if self._monitor_thread and self._monitor_thread.is_alive():
                    self._monitor_thread.join(timeout=5)
                
                if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                    self._heartbeat_thread.join(timeout=5)
                
                # 清理资源
                self.worker_processes.clear()
                self.master_node = None
                self.is_running = False
                
                self.logger.info("分布式系统已关闭")
                
        except Exception as e:
            self.logger.error(f"关闭系统时发生错误: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统性能指标"""
        with self._lock:
            uptime = time.time() - self.system_start_time
            
            total_completed = sum(m.tasks_completed for m in self.node_metrics.values())
            total_failed = sum(m.tasks_failed for m in self.node_metrics.values())
            healthy_nodes = sum(1 for m in self.node_metrics.values() if m.is_healthy)
            
            return {
                'uptime': uptime,
                'is_running': self.is_running,
                'total_nodes': len(self.node_metrics),
                'healthy_nodes': healthy_nodes,
                'tasks_completed': total_completed,
                'tasks_failed': total_failed,
                'tasks_in_queue': len(self.task_queue),
                'throughput': total_completed / uptime if uptime > 0 else 0,
                'node_metrics': {
                    node_id: {
                        'tasks_completed': m.tasks_completed,
                        'tasks_failed': m.tasks_failed,
                        'average_task_time': m.average_task_time,
                        'is_healthy': m.is_healthy,
                        'last_heartbeat': m.last_heartbeat
                    }
                    for node_id, m in self.node_metrics.items()
                }
            }
    
    @contextmanager
    def managed_execution(self, tasks: List[Any], task_handler: Callable):
        """上下文管理器，自动管理系统生命周期"""
        try:
            if not self.start_system(tasks, task_handler):
                raise RuntimeError("启动分布式系统失败")
            
            yield self
            
        finally:
            self.shutdown_system()


# 生产级任务处理器示例
class ProductionTaskProcessor:
    """生产级任务处理器"""
    
    def __init__(self, enable_caching: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_caching = enable_caching
        self.cache: Dict[str, Any] = {}
        self.processing_stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'processing_times': []
        }
    
    def process_computational_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理计算任务
        
        Args:
            task: 包含任务参数的字典
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        try:
            # 输入验证
            if not isinstance(task, dict):
                raise ValueError("任务必须是字典类型")
            
            if 'operation' not in task:
                raise ValueError("任务缺少operation字段")
            
            # 缓存检查
            cache_key = self._generate_cache_key(task)
            if self.enable_caching and cache_key in self.cache:
                self.processing_stats['cache_hits'] += 1
                return self.cache[cache_key]
            
            # 执行任务
            result = self._execute_task(task)
            
            # 缓存结果
            if self.enable_caching:
                self.cache[cache_key] = result
            
            # 更新统计
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['processing_times'].append(processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"任务处理失败: {e}")
            raise
    
    def _generate_cache_key(self, task: Dict[str, Any]) -> str:
        """生成缓存键"""
        task_str = json.dumps(task, sort_keys=True)
        return hashlib.md5(task_str.encode()).hexdigest()
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体任务"""
        operation = task['operation']
        
        if operation == 'square':
            value = task.get('value', 0)
            result = value * value
            
        elif operation == 'fibonacci':
            n = task.get('n', 0)
            result = self._fibonacci(n)
            
        elif operation == 'matrix_multiply':
            matrix_a = task.get('matrix_a', [[1]])
            matrix_b = task.get('matrix_b', [[1]])
            result = self._matrix_multiply(matrix_a, matrix_b)
            
        else:
            raise ValueError(f"不支持的操作: {operation}")
        
        return {
            'operation': operation,
            'result': result,
            'timestamp': time.time()
        }
    
    def _fibonacci(self, n: int) -> int:
        """计算斐波那契数列"""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    def _matrix_multiply(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """矩阵乘法"""
        if len(a[0]) != len(b):
            raise ValueError("矩阵维度不匹配")
        
        result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
        
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result


def main():
    """主函数 - 生产环境使用示例"""
    # 配置系统
    config = DistributedSystemConfig(
        max_workers=4,
        task_timeout=60.0,
        enable_monitoring=True,
        log_level="INFO",
        backup_results=True
    )
    
    # 创建任务处理器
    processor = ProductionTaskProcessor(enable_caching=True)
    
    # 准备任务
    tasks = [
        {'operation': 'square', 'value': i}
        for i in range(100)
    ]
    
    # 创建分布式系统
    system = ProductionDistributedSystem(config)
    
    try:
        # 使用上下文管理器执行任务
        with system.managed_execution(tasks, processor.process_computational_task) as sys:
            # 等待完成
            results = sys.wait_for_completion(timeout=300)
            
            # 输出结果统计
            print(f"处理完成 {len(results)} 个任务")
            
            # 获取系统指标
            metrics = sys.get_system_metrics()
            print(f"系统吞吐量: {metrics['throughput']:.2f} 任务/秒")
            print(f"健康节点: {metrics['healthy_nodes']}/{metrics['total_nodes']}")
            
    except Exception as e:
        logging.error(f"执行失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行主程序
    success = main()
    exit(0 if success else 1)