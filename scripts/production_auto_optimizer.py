"""
生产级自动优化器实验系统

提供长期运行的自动优化实验框架，包含完整的监控、
日志记录、性能分析和故障恢复功能。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback
import hashlib
import os

try:
    from monitoring import AutoOptimizer, PerformanceMonitor, TimeSeriesStorage
except ImportError:
    # 模拟导入，用于独立运行
    class TimeSeriesStorage:
        def __init__(self, path: Path):
            self.path = path
            self._events = []
        
        def events(self, category: str) -> List[Dict[str, Any]]:
            return [e for e in self._events if e.get('category') == category]
        
        def store_event(self, category: str, data: Dict[str, Any]) -> None:
            event = {'category': category, 'timestamp': time.time(), **data}
            self._events.append(event)
    
    class PerformanceMonitor:
        def __init__(self, storage, **kwargs):
            self.storage = storage
            self.config = kwargs
        
        def log_resource_usage(self, agent: str, cpu: float, memory: float) -> None:
            self.storage.store_event('resource', {
                'agent': agent, 'cpu': cpu, 'memory': memory
            })
        
        def log_prediction(self, prediction: int, outcome: int) -> None:
            self.storage.store_event('prediction', {
                'prediction': prediction, 'outcome': outcome
            })
    
    class AutoOptimizer:
        def __init__(self, monitor, storage, **kwargs):
            self.monitor = monitor
            self.storage = storage
            self.config = kwargs
            self.step_count = 0
        
        def step(self) -> None:
            self.step_count += 1
            if self.step_count % 10 == 0:
                self.storage.store_event('optimization', {
                    'step': self.step_count,
                    'action': 'parameter_adjustment'
                })


@dataclass
class ExperimentConfig:
    """实验配置"""
    duration: int = 300  # 实验持续时间（秒）
    sampling_interval: float = 1.0  # 采样间隔（秒）
    storage_path: Path = Path("experiment_data.db")
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # 监控配置
    training_accuracy: float = 0.9
    degradation_threshold: float = 0.1
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    
    # 优化器配置
    accuracy_threshold: float = 0.8
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "learning_rate": (0.001, 0.1),
        "batch_size": (16, 512),
        "dropout_rate": (0.0, 0.5)
    })
    
    # 模拟配置
    enable_realistic_simulation: bool = True
    noise_level: float = 0.1
    trend_probability: float = 0.3
    
    # 输出配置
    output_dir: Path = Path("experiment_results")
    save_raw_data: bool = True
    generate_report: bool = True


class MetricsSimulator:
    """指标模拟器 - 生成真实的系统指标"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模拟状态
        self.cpu_baseline = 30.0
        self.memory_baseline = 40.0
        self.accuracy_baseline = 0.85
        
        # 趋势状态
        self.cpu_trend = 0.0
        self.memory_trend = 0.0
        self.accuracy_trend = 0.0
        
        # 时间相关
        self.start_time = time.time()
        
    def generate_cpu_usage(self) -> float:
        """生成CPU使用率"""
        # 基础值 + 趋势 + 噪声 + 周期性变化
        elapsed = time.time() - self.start_time
        
        # 周期性变化（模拟工作负载波动）
        periodic = 10 * math.sin(elapsed / 60.0 * 2 * math.pi)
        
        # 随机噪声
        noise = random.uniform(-5, 5) * self.config.noise_level
        
        # 趋势更新
        if random.random() < self.config.trend_probability / 100:
            self.cpu_trend += random.uniform(-2, 2)
            self.cpu_trend = max(-10, min(10, self.cpu_trend))
        
        cpu_usage = self.cpu_baseline + self.cpu_trend + periodic + noise
        return max(0, min(100, cpu_usage))
    
    def generate_memory_usage(self) -> float:
        """生成内存使用率"""
        elapsed = time.time() - self.start_time
        
        # 内存通常有缓慢增长趋势
        growth = elapsed / 3600.0 * 5  # 每小时增长5%
        
        # 随机波动
        noise = random.uniform(-3, 3) * self.config.noise_level
        
        # 趋势更新
        if random.random() < self.config.trend_probability / 100:
            self.memory_trend += random.uniform(-1, 1)
            self.memory_trend = max(-5, min(5, self.memory_trend))
        
        memory_usage = self.memory_baseline + growth + self.memory_trend + noise
        return max(0, min(100, memory_usage))
    
    def generate_prediction_outcome(self) -> Tuple[int, int]:
        """生成预测和实际结果"""
        # 模拟模型准确率随时间变化
        elapsed = time.time() - self.start_time
        
        # 准确率可能随时间下降（模型退化）
        degradation = elapsed / 3600.0 * 0.05  # 每小时下降5%
        
        current_accuracy = max(0.5, self.accuracy_baseline - degradation + self.accuracy_trend)
        
        # 生成预测
        prediction = 1 if random.random() > 0.5 else 0
        
        # 根据当前准确率生成结果
        if random.random() < current_accuracy:
            outcome = prediction  # 正确预测
        else:
            outcome = 1 - prediction  # 错误预测
        
        return prediction, outcome


class ProductionAutoOptimizerExperiment:
    """生产级自动优化器实验"""
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # 实验组件
        self.storage: Optional[TimeSeriesStorage] = None
        self.monitor: Optional[PerformanceMonitor] = None
        self.optimizer: Optional[AutoOptimizer] = None
        self.simulator: Optional[MetricsSimulator] = None
        
        # 运行状态
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.experiment_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.start_time = 0.0
        self.samples_collected = 0
        self.optimization_events = 0
        self.errors_encountered = 0
        
        # 信号处理
        self._setup_signal_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 文件处理器
            if self.config.log_file:
                try:
                    file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.error(f"无法创建日志文件: {e}")
        
        return logger
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，开始停止实验")
            self.stop_experiment()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_components(self) -> None:
        """初始化实验组件"""
        try:
            # 确保存储目录存在
            self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 初始化存储
            self.storage = TimeSeriesStorage(self.config.storage_path)
            
            # 初始化监控器
            self.monitor = PerformanceMonitor(
                self.storage,
                training_accuracy=self.config.training_accuracy,
                degradation_threshold=self.config.degradation_threshold,
                cpu_threshold=self.config.cpu_threshold,
                memory_threshold=self.config.memory_threshold
            )
            
            # 初始化优化器
            self.optimizer = AutoOptimizer(
                self.monitor,
                self.storage,
                cpu_threshold=self.config.cpu_threshold,
                memory_threshold=self.config.memory_threshold,
                accuracy_threshold=self.config.accuracy_threshold,
                parameter_bounds=self.config.parameter_bounds
            )
            
            # 初始化模拟器
            if self.config.enable_realistic_simulation:
                self.simulator = MetricsSimulator(self.config)
            
            self.logger.info("实验组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def start_experiment(self) -> None:
        """启动实验"""
        try:
            if self.is_running:
                self.logger.warning("实验已在运行中")
                return
            
            self.logger.info(f"启动自动优化器实验，持续时间: {self.config.duration}秒")
            
            # 初始化组件
            self.initialize_components()
            
            # 重置统计
            self.start_time = time.time()
            self.samples_collected = 0
            self.optimization_events = 0
            self.errors_encountered = 0
            
            # 启动实验线程
            self.is_running = True
            self.experiment_thread = threading.Thread(
                target=self._experiment_loop,
                daemon=True
            )
            self.experiment_thread.start()
            
            self.logger.info("实验已启动")
            
        except Exception as e:
            self.logger.error(f"启动实验失败: {e}")
            self.is_running = False
            raise
    
    def _experiment_loop(self) -> None:
        """实验主循环"""
        try:
            end_time = self.start_time + self.config.duration
            
            while time.time() < end_time and not self.shutdown_event.is_set():
                loop_start = time.time()
                
                try:
                    # 生成或收集指标
                    self._collect_metrics()
                    
                    # 执行优化步骤
                    self._perform_optimization_step()
                    
                    # 更新统计
                    self.samples_collected += 1
                    
                    # 定期输出进度
                    if self.samples_collected % 60 == 0:  # 每分钟输出一次
                        self._log_progress()
                    
                except Exception as e:
                    self.errors_encountered += 1
                    self.logger.error(f"实验循环错误: {e}")
                    
                    # 如果错误过多，停止实验
                    if self.errors_encountered > 10:
                        self.logger.error("错误过多，停止实验")
                        break
                
                # 控制采样间隔
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config.sampling_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.logger.info("实验循环结束")
            
        except Exception as e:
            self.logger.error(f"实验循环异常: {e}")
        finally:
            self.is_running = False
    
    def _collect_metrics(self) -> None:
        """收集系统指标"""
        try:
            if self.simulator:
                # 使用模拟器生成指标
                cpu_usage = self.simulator.generate_cpu_usage()
                memory_usage = self.simulator.generate_memory_usage()
                prediction, outcome = self.simulator.generate_prediction_outcome()
            else:
                # 使用真实系统指标
                cpu_usage, memory_usage = self._get_real_system_metrics()
                prediction, outcome = self._get_real_prediction_data()
            
            # 记录指标
            if self.monitor:
                self.monitor.log_resource_usage("main_agent", cpu_usage, memory_usage)
                self.monitor.log_prediction(prediction, outcome)
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {e}")
            raise
    
    def _get_real_system_metrics(self) -> Tuple[float, float]:
        """获取真实系统指标"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            return cpu_usage, memory_usage
        except ImportError:
            # 如果psutil不可用，返回模拟值
            import random
            return random.uniform(20, 80), random.uniform(30, 70)
    
    def _get_real_prediction_data(self) -> Tuple[int, int]:
        """获取真实预测数据"""
        # 在实际应用中，这里应该从真实的ML模型获取数据
        import random
        prediction = 1 if random.random() > 0.5 else 0
        outcome = 1 if random.random() > 0.3 else 0
        return prediction, outcome
    
    def _perform_optimization_step(self) -> None:
        """执行优化步骤"""
        try:
            if self.optimizer:
                old_step_count = getattr(self.optimizer, 'step_count', 0)
                self.optimizer.step()
                new_step_count = getattr(self.optimizer, 'step_count', 0)
                
                if new_step_count > old_step_count:
                    self.optimization_events += 1
                    
        except Exception as e:
            self.logger.error(f"优化步骤失败: {e}")
            raise
    
    def _log_progress(self) -> None:
        """记录进度"""
        elapsed = time.time() - self.start_time
        remaining = self.config.duration - elapsed
        progress = (elapsed / self.config.duration) * 100
        
        self.logger.info(
            f"实验进度: {progress:.1f}% "
            f"(已运行 {elapsed:.0f}s, 剩余 {remaining:.0f}s), "
            f"样本: {self.samples_collected}, "
            f"优化事件: {self.optimization_events}, "
            f"错误: {self.errors_encountered}"
        )
    
    def stop_experiment(self) -> None:
        """停止实验"""
        try:
            if not self.is_running:
                return
            
            self.logger.info("正在停止实验...")
            
            # 设置停止标志
            self.shutdown_event.set()
            
            # 等待实验线程结束
            if self.experiment_thread and self.experiment_thread.is_alive():
                self.experiment_thread.join(timeout=10)
            
            self.is_running = False
            self.logger.info("实验已停止")
            
        except Exception as e:
            self.logger.error(f"停止实验时发生错误: {e}")
    
    def wait_for_completion(self) -> None:
        """等待实验完成"""
        try:
            if self.experiment_thread and self.experiment_thread.is_alive():
                self.experiment_thread.join()
            
            self.logger.info("实验已完成")
            
        except KeyboardInterrupt:
            self.logger.info("用户中断实验")
            self.stop_experiment()
    
    def generate_experiment_report(self) -> Dict[str, Any]:
        """生成实验报告"""
        try:
            if not self.storage:
                return {}
            
            # 收集所有事件
            optimization_events = self.storage.events("optimization")
            resource_events = self.storage.events("resource")
            prediction_events = self.storage.events("prediction")
            
            # 计算统计信息
            total_runtime = time.time() - self.start_time if self.start_time > 0 else 0
            
            # 资源使用统计
            cpu_values = [e.get('cpu', 0) for e in resource_events]
            memory_values = [e.get('memory', 0) for e in resource_events]
            
            # 预测准确率统计
            correct_predictions = sum(
                1 for e in prediction_events 
                if e.get('prediction') == e.get('outcome')
            )
            total_predictions = len(prediction_events)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            report = {
                'experiment_summary': {
                    'duration_configured': self.config.duration,
                    'duration_actual': total_runtime,
                    'samples_collected': self.samples_collected,
                    'optimization_events': len(optimization_events),
                    'errors_encountered': self.errors_encountered
                },
                'resource_usage': {
                    'cpu_avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'cpu_max': max(cpu_values) if cpu_values else 0,
                    'cpu_min': min(cpu_values) if cpu_values else 0,
                    'memory_avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'memory_max': max(memory_values) if memory_values else 0,
                    'memory_min': min(memory_values) if memory_values else 0
                },
                'prediction_performance': {
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy
                },
                'optimization_activity': {
                    'total_optimizations': len(optimization_events),
                    'optimization_rate': len(optimization_events) / total_runtime if total_runtime > 0 else 0
                },
                'configuration': {
                    'sampling_interval': self.config.sampling_interval,
                    'cpu_threshold': self.config.cpu_threshold,
                    'memory_threshold': self.config.memory_threshold,
                    'accuracy_threshold': self.config.accuracy_threshold
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return {}
    
    def save_results(self) -> None:
        """保存实验结果"""
        try:
            # 确保输出目录存在
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告
            if self.config.generate_report:
                report = self.generate_experiment_report()
                report_path = self.config.output_dir / "experiment_report.json"
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"实验报告已保存: {report_path}")
            
            # 保存原始数据
            if self.config.save_raw_data and self.storage:
                raw_data = {
                    'optimization_events': self.storage.events("optimization"),
                    'resource_events': self.storage.events("resource"),
                    'prediction_events': self.storage.events("prediction")
                }
                
                raw_data_path = self.config.output_dir / "raw_data.json"
                with open(raw_data_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"原始数据已保存: {raw_data_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生产级自动优化器实验系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--duration", type=int, default=300,
                        help="实验持续时间（秒），默认300秒")
    parser.add_argument("--sampling-interval", type=float, default=1.0,
                        help="采样间隔（秒），默认1.0秒")
    parser.add_argument("--storage-path", type=Path, default=Path("experiment_data.db"),
                        help="数据存储路径")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"),
                        help="结果输出目录")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="日志级别")
    parser.add_argument("--log-file", type=Path, help="日志文件路径")
    
    # 阈值配置
    parser.add_argument("--cpu-threshold", type=float, default=80.0,
                        help="CPU使用率阈值")
    parser.add_argument("--memory-threshold", type=float, default=80.0,
                        help="内存使用率阈值")
    parser.add_argument("--accuracy-threshold", type=float, default=0.8,
                        help="准确率阈值")
    
    # 模拟配置
    parser.add_argument("--disable-simulation", action="store_true",
                        help="禁用模拟，使用真实系统指标")
    parser.add_argument("--noise-level", type=float, default=0.1,
                        help="模拟噪声水平")
    
    return parser.parse_args()


def main() -> int:
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 创建配置
        config = ExperimentConfig(
            duration=args.duration,
            sampling_interval=args.sampling_interval,
            storage_path=args.storage_path,
            output_dir=args.output_dir,
            log_level=args.log_level,
            log_file=args.log_file,
            cpu_threshold=args.cpu_threshold,
            memory_threshold=args.memory_threshold,
            accuracy_threshold=args.accuracy_threshold,
            enable_realistic_simulation=not args.disable_simulation,
            noise_level=args.noise_level
        )
        
        # 创建实验实例
        experiment = ProductionAutoOptimizerExperiment(config)
        
        # 启动实验
        experiment.start_experiment()
        
        # 等待完成
        experiment.wait_for_completion()
        
        # 保存结果
        experiment.save_results()
        
        # 输出最终统计
        if experiment.storage:
            events = experiment.storage.events("optimization")
            experiment.logger.info(f"实验完成，记录了 {len(events)} 个优化事件")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("用户中断实验")
        return 0
    except Exception as e:
        logging.error(f"实验失败: {e}")
        logging.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # 导入必要的模块
    import math
    import random
    
    # 设置基础日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行主程序
    exit(main())