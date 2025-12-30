"""
大脑模拟系统生产级主入口

提供高性能、安全的命令行接口和API服务，包含完整的错误处理、
日志记录、监控和配置管理功能。
"""

import argparse
import json
import logging
import signal
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from third_party.autogpt.autogpt.core.brain.config import BrainSimulationConfig
from modules.brain.backends import BrainBackendInitError
from modules.brain.backends.brain_simulation_app import (
    BrainSimulationBootstrap,
    bootstrap_brain_simulation,
    resolve_brain_simulation_config,
)

try:
    from BrainSimulationSystem.api.brain_api import BrainAPI
    from BrainSimulationSystem.visualization.visualizer import BrainVisualizer
except ImportError as e:
    logging.error(f"导入模块失败: {e}")
    sys.exit(1)


@dataclass
class SystemConfig:
    """系统配置"""
    # 基本配置
    mode: str = "interactive"
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # API配置
    host: str = "127.0.0.1"  # 生产环境默认只监听本地
    port: int = 5000
    enable_cors: bool = False
    api_key: Optional[str] = None
    rate_limit: int = 100  # 每分钟请求数

    # 模拟配置
    duration: float = 1000.0
    input_file: Optional[Path] = None
    output_file: Optional[Path] = None
    
    # 可视化配置
    enable_visualization: bool = False
    enable_live_visualization: bool = False
    visualization_port: int = 8080
    
    # 性能配置
    max_workers: int = 4
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 80.0
    
    # 安全配置
    enable_security: bool = True
    max_request_size_mb: int = 10
    timeout_seconds: int = 300


class ProductionBrainSystem:
    """生产级大脑模拟系统"""

    def __init__(
        self,
        config: SystemConfig,
        brain_config: Optional[BrainSimulationConfig] = None,
    ):
        """
        初始化系统

        Args:
            config: 系统配置
            brain_config: ``BrainSimulationConfig`` 实例，用于启动大脑后端
        """
        self.config = config
        self.brain_config = brain_config or BrainSimulationConfig()
        self.logger = self._setup_logging()

        # 系统组件
        self.brain_bootstrap: Optional[BrainSimulationBootstrap] = None
        self.brain_simulation: Optional[Any] = None
        self.visualizer: Optional[BrainVisualizer] = None
        self.api_server: Optional[BrainAPI] = None
        
        # 运行状态
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # 性能监控
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # 信号处理
        self._setup_signal_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("BrainSystem")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.log_file:
            try:
                file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.error(f"无法创建日志文件处理器: {e}")
        
        return logger
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，开始优雅关闭")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_components(self, brain_config_overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化系统组件

        Args:
            brain_config_overrides: 可选的配置覆盖，会合并到 ``BrainSimulationConfig.overrides``
        """
        try:
            if brain_config_overrides:
                self.brain_config.overrides.update(brain_config_overrides)
            # 初始化大脑模拟后端
            self.brain_bootstrap = bootstrap_brain_simulation(
                self.brain_config, use_environment_defaults=False
            )
            self.brain_simulation = self.brain_bootstrap.brain
            self.logger.info(
                "大脑模拟组件初始化完成 (profile=%s, stage=%s)",
                self.brain_config.profile,
                self.brain_config.stage,
            )

            # 初始化可视化器（如果启用）
            if self.config.enable_visualization:
                self.visualizer = BrainVisualizer(self.brain_simulation)
                self.logger.info("Visualization subsystem initialized")
                if self.config.enable_live_visualization:
                    self.logger.info("Live visualization enabled")
            
            # 初始化API服务器（如果是API模式）
            if self.config.mode == "api":
                self.api_server = BrainAPI(
                    self.brain_simulation,
                    host=self.config.host,
                    port=self.config.port,
                    enable_cors=self.config.enable_cors
                )
                self.logger.info("API服务器初始化完成")
                
        except BrainBackendInitError as exc:
            self.logger.error(f"大脑后端初始化失败: {exc}")
            raise
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise

    def _brain_dt(self) -> float:
        """Resolve the simulation timestep in milliseconds."""

        return max(float(self.brain_config.timestep_ms), 1e-3)

    def run_interactive_mode(self) -> int:
        """
        运行交互模式
        
        Returns:
            退出代码
        """
        try:
            self.logger.info("启动交互模式")
            
            # 启动可视化（如果启用）
            if self.visualizer and self.config.enable_live_visualization:
                self.visualizer.start_live_visualization()
            
            # 交互式命令循环
            self._interactive_loop()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("用户中断操作")
            return 0
        except Exception as e:
            self.logger.error(f"交互模式运行失败: {e}")
            return 1
    
    def _interactive_loop(self) -> None:
        """交互式命令循环"""
        commands = {
            'start': self._cmd_start_simulation,
            'stop': self._cmd_stop_simulation,
            'status': self._cmd_show_status,
            'config': self._cmd_show_config,
            'metrics': self._cmd_show_metrics,
            'help': self._cmd_show_help,
            'quit': self._cmd_quit,
            'exit': self._cmd_quit
        }
        
        self.logger.info("进入交互模式，输入 'help' 查看可用命令")
        
        while not self.shutdown_event.is_set():
            try:
                command = input("BrainSystem> ").strip().lower()
                
                if not command:
                    continue
                
                if command in commands:
                    result = commands[command]()
                    if result == 'quit':
                        break
                else:
                    self.logger.warning(f"未知命令: {command}，输入 'help' 查看可用命令")
                    
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"命令执行错误: {e}")
    
    def _cmd_start_simulation(self) -> None:
        """启动模拟命令"""
        try:
            if self.brain_simulation and not self.brain_simulation.is_running:
                self.brain_simulation.start(dt=self._brain_dt())
                self.logger.info("模拟已启动")
            else:
                self.logger.warning("模拟已在运行中")
        except Exception as e:
            self.logger.error(f"启动模拟失败: {e}")
    
    def _cmd_stop_simulation(self) -> None:
        """停止模拟命令"""
        try:
            if self.brain_simulation and self.brain_simulation.is_running:
                self.brain_simulation.stop()
                self.logger.info("模拟已停止")
            else:
                self.logger.warning("模拟未在运行")
        except Exception as e:
            self.logger.error(f"停止模拟失败: {e}")
    
    def _cmd_show_status(self) -> None:
        """Display aggregate runtime status"""
        try:
            status: Dict[str, Any] = {
                'system_uptime': time.time() - self.start_time,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage(),
            }
            if self.brain_simulation:
                try:
                    status.update(self.brain_simulation.get_status())
                except Exception as exc:
                    self.logger.warning(f"Failed to extend brain status: {exc}")
            self.logger.info("System status:")
            for key, value in status.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.2f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        except Exception as exc:
            self.logger.error(f"Status retrieval failed: {exc}")

    def _cmd_show_config(self) -> None:
        """显示配置命令"""
        config_info = {
            'mode': self.config.mode,
            'host': self.config.host,
            'port': self.config.port,
            'log_level': self.config.log_level,
            'max_workers': self.config.max_workers,
            'brain_profile': self.brain_config.profile,
            'brain_stage': self.brain_config.stage,
            'brain_config_file': self.brain_config.config_file,
            'brain_timestep_ms': self._brain_dt(),
        }
        
        self.logger.info("系统配置:")
        for key, value in config_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def _cmd_show_metrics(self) -> None:
        """Display collected metrics from the brain simulation"""
        if not self.brain_simulation:
            self.logger.warning("Brain simulation not initialized")
            return

        try:
            metrics = self.brain_simulation.get_metrics()
        except Exception as exc:
            self.logger.error(f"Metric retrieval failed: {exc}")
            return

        self.logger.info("Brain metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def _cmd_show_help(self) -> None:
        """显示帮助命令"""
        help_text = """
可用命令:
  start   - 启动大脑模拟
  stop    - 停止大脑模拟
  status  - 显示系统状态
  config  - 显示系统配置
  metrics - 显示性能指标
  help    - 显示此帮助信息
  quit    - 退出系统
        """
        self.logger.info(help_text.strip())
    
    def _cmd_quit(self) -> str:
        """退出命令"""
        self.logger.info("正在退出系统...")
        return 'quit'
    
    def run_batch_mode(self) -> int:
        """
        运行批处理模式
        
        Returns:
            退出代码
        """
        try:
            self.logger.info("启动批处理模式")
            
            # 加载输入数据
            input_data = self._load_input_data()
            
            # 运行模拟
            results = self._run_batch_simulation(input_data)
            
            # 保存结果
            self._save_results(results)
            
            self.logger.info("批处理完成")
            return 0
            
        except Exception as e:
            self.logger.error(f"批处理模式运行失败: {e}")
            return 1
    
    def _load_input_data(self) -> Dict[str, Any]:
        """加载输入数据"""
        if not self.config.input_file or not self.config.input_file.exists():
            return {}
        
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载输入文件失败: {e}")
            raise
    
    def _run_batch_simulation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a batch simulation using optional overrides"""
        if not self.brain_simulation:
            raise RuntimeError("Brain simulation has not been initialized")

        if input_data:
            self.brain_simulation.update_parameters(input_data)

        return self.brain_simulation.run_simulation(
            duration=self.config.duration,
            dt=self._brain_dt()
        )

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存结果"""
        if not self.config.output_file:
            return
        
        try:
            # 确保输出目录存在
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 添加元数据
            output_data = {
                'timestamp': time.time(),
                'config': {
                    'duration': self.config.duration,
                    'dt': self._brain_dt(),
                    'mode': self.config.mode
                },
                'results': results
            }
            
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"结果已保存到: {self.config.output_file}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            raise
    
    def run_api_mode(self) -> int:
        """
        运行API模式
        
        Returns:
            退出代码
        """
        try:
            self.logger.info(f"启动API服务器 - {self.config.host}:{self.config.port}")
            
            if not self.api_server:
                raise RuntimeError("API服务器未初始化")
            
            # 启动API服务器
            self.api_server.run(
                host=self.config.host,
                port=self.config.port,
                debug=False,
                threaded=True
            )
            
            return 0
            
        except Exception as e:
            self.logger.error(f"API模式运行失败: {e}")
            return 1
    
    def run_visualization_mode(self) -> int:
        """Run visualization workflows in the foreground"""
        try:
            self.logger.info("Starting visualization mode")

            if not self.visualizer:
                raise RuntimeError("Brain visualizer not initialized")

            try:
                self.visualizer.visualize_network_structure()
                if self.config.enable_live_visualization:
                    self.visualizer.start_live_visualization()
                    self.logger.info("Live visualization running - press Ctrl+C to stop")
                    while not self.shutdown_event.is_set():
                        time.sleep(0.5)
                else:
                    self.visualizer.visualize_activity()
            finally:
                if self.config.enable_live_visualization and self.visualizer:
                    self.visualizer.stop_live_visualization()

            return 0
        except KeyboardInterrupt:
            self.logger.info("Visualization interrupted by user")
            return 0
        except Exception as exc:
            self.logger.error(f"Visualization mode failed: {exc}")
            return 1

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_percent()
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def shutdown(self) -> None:
        """优雅关闭系统"""
        try:
            self.logger.info("开始关闭系统")
            
            # 设置关闭标志
            self.shutdown_event.set()
            
            # 关闭大脑模拟
            if self.brain_simulation:
                self.brain_simulation.stop()
            
            # 关闭可视化器
            if self.visualizer:
                try:
                    self.visualizer.stop_live_visualization()
                except AttributeError:
                    pass
            
            # 关闭API服务器
            if self.api_server:
                self.api_server.stop()
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            self.logger.info("系统已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭系统时发生错误: {e}")
    
    @contextmanager
    def managed_execution(self):
        """上下文管理器，确保资源正确清理"""
        try:
            yield self
        finally:
            self.shutdown()


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="大脑模拟系统生产版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  interactive  - 交互式命令行界面
  batch        - 批处理模式，处理输入文件并生成输出
  api          - REST API服务器模式
  visualization - 可视化服务器模式

示例:
  %(prog)s --mode interactive --config config.json
  %(prog)s --mode batch --input data.json --output results.json
  %(prog)s --mode api --host 0.0.0.0 --port 8080
        """
    )
    
    # 基本参数
    parser.add_argument("--config", type=Path, dest="config_file", help="配置文件路径")
    parser.add_argument("--mode", choices=["interactive", "batch", "api", "visualization"],
                        default="interactive", help="运行模式")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="日志级别")
    parser.add_argument("--log-file", type=Path, help="日志文件路径")

    # 大脑配置参数
    parser.add_argument("--profile", help="BrainSimulationSystem 配置文件名称")
    parser.add_argument("--stage", help="BrainSimulationSystem 阶段/课程标识")
    parser.add_argument("--overrides-json", help="JSON 字符串覆盖 BrainSimulation 配置")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="以 key=value 形式提供的额外配置 (可多次指定)",
    )
    parser.add_argument(
        "--timestep-ms",
        type=float,
        help="BrainSimulationSystem 步长 (毫秒)",
    )
    parser.add_argument(
        "--auto-background",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启动 BrainSimulationSystem 后台循环",
    )

    # API模式参数
    parser.add_argument("--host", default="127.0.0.1", help="API服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口")
    parser.add_argument("--enable-cors", action="store_true", help="启用CORS")
    
    # 批处理模式参数
    parser.add_argument("--input", type=Path, help="输入文件路径")
    parser.add_argument("--output", type=Path, help="输出文件路径")
    parser.add_argument("--duration", type=float, default=1000.0, help="模拟持续时间（毫秒）")

    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="启用可视化")
    parser.add_argument("--live", action="store_true", help="启用实时可视化")
    parser.add_argument("--viz-port", type=int, default=8080, help="可视化服务器端口")
    
    # 性能参数
    parser.add_argument("--max-workers", type=int, default=4, help="最大工作线程数")
    parser.add_argument("--memory-limit", type=int, default=1024, help="内存限制（MB）")
    
    return parser.parse_args()


def create_system_config(args: argparse.Namespace) -> SystemConfig:
    """根据命令行参数创建系统配置"""
    return SystemConfig(
        mode=args.mode,
        log_level=args.log_level,
        log_file=args.log_file,
        host=args.host,
        port=args.port,
        enable_cors=args.enable_cors,
        duration=args.duration,
        input_file=args.input,
        output_file=args.output,
        enable_visualization=args.visualize,
        enable_live_visualization=args.live,
        visualization_port=args.viz_port,
        max_workers=args.max_workers,
        memory_limit_mb=args.memory_limit
    )


def _parse_cli_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"覆盖参数 '{item}' 缺少 '=' 分隔符，应使用 key=value 格式"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("覆盖参数的键不能为空")
        value = value.strip()
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def create_brain_simulation_config(args: argparse.Namespace) -> BrainSimulationConfig:
    """Resolve CLI/环境配置为 ``BrainSimulationConfig``."""

    override_dict = _parse_cli_overrides(args.override or [])
    overrides: Mapping[str, Any] | None = override_dict or None

    return resolve_brain_simulation_config(
        profile=args.profile,
        stage=args.stage,
        config_file=args.config_file,
        overrides_json=args.overrides_json,
        overrides=overrides,
        timestep_ms=args.timestep_ms,
        auto_background=args.auto_background,
        use_environment_defaults=True,
    )


def main() -> int:
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建系统配置
        config = create_system_config(args)
        brain_config = create_brain_simulation_config(args)

        # 创建系统实例
        system = ProductionBrainSystem(config, brain_config)

        # 使用上下文管理器确保资源清理
        with system.managed_execution():
            # 初始化组件
            system.initialize_components()

            # 根据模式运行系统
            if config.mode == "interactive":
                return system.run_interactive_mode()
            elif config.mode == "batch":
                return system.run_batch_mode()
            elif config.mode == "api":
                return system.run_api_mode()
            elif config.mode == "visualization":
                return system.run_visualization_mode()
            else:
                system.logger.error(f"不支持的运行模式: {config.mode}")
                return 1
    
    except KeyboardInterrupt:
        logging.info("用户中断程序")
        return 0
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        logging.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
