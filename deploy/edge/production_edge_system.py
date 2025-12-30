"""
生产级边缘计算系统

为AutoGPT组件提供高性能的边缘设备部署和管理功能，
包括资源监控、模型优化、传感器管理和故障恢复。
"""

import json
import logging
import os
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from uuid import uuid4

import numpy as np


class DeviceStatus(Enum):
    """设备状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    TEMPERATURE = "temperature"


@dataclass
class ResourceMetrics:
    """资源指标"""
    resource_type: ResourceType
    current_value: float
    max_value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization_percentage(self) -> float:
        """资源利用率百分比"""
        if self.max_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.max_value) * 100.0)


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    sensor_type: str
    value: Union[float, int, str, bool]
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    quality: float = 1.0  # 数据质量 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeDeviceConfig:
    """边缘设备配置"""
    device_id: str
    device_name: str
    data_directory: str = "./edge_data"
    max_storage_gb: float = 10.0
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    temperature_threshold: float = 70.0
    monitoring_interval: float = 5.0
    enable_gpu: bool = True
    enable_model_optimization: bool = True
    log_level: str = "INFO"


class EdgeResourceManager:
    """
    边缘资源管理器
    
    监控和管理边缘设备的计算资源，包括CPU、内存、GPU等。
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 资源监控
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_history: Dict[ResourceType, List[ResourceMetrics]] = {
            resource_type: [] for resource_type in ResourceType
        }
        
        # 告警系统
        self.alert_callbacks: List[Callable[[ResourceType, ResourceMetrics], None]] = []
        self.alert_thresholds = {
            ResourceType.CPU: config.cpu_threshold,
            ResourceType.MEMORY: config.memory_threshold,
            ResourceType.TEMPERATURE: config.temperature_threshold
        }
        
        # GPU检测
        self.gpu_available = self._detect_gpu()
        
    def _detect_gpu(self) -> bool:
        """检测GPU可用性"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0 and self.config.enable_gpu
        except ImportError:
            self.logger.warning("GPUtil未安装，GPU监控不可用")
            return False
        except Exception as e:
            self.logger.error(f"GPU检测失败: {e}")
            return False
    
    def start_monitoring(self) -> None:
        """启动资源监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("资源监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止资源监控"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("资源监控已停止")
    
    def get_current_metrics(self) -> Dict[ResourceType, ResourceMetrics]:
        """获取当前资源指标"""
        metrics = {}
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics[ResourceType.CPU] = ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_value=cpu_percent,
            max_value=100.0,
            unit="percent"
        )
        
        # 内存使用率
        memory = psutil.virtual_memory()
        metrics[ResourceType.MEMORY] = ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_value=memory.used / (1024**3),  # GB
            max_value=memory.total / (1024**3),  # GB
            unit="GB"
        )
        
        # 存储使用率
        disk = psutil.disk_usage('/')
        metrics[ResourceType.STORAGE] = ResourceMetrics(
            resource_type=ResourceType.STORAGE,
            current_value=disk.used / (1024**3),  # GB
            max_value=disk.total / (1024**3),  # GB
            unit="GB"
        )
        
        # 温度（如果可用）
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # 取第一个可用的温度传感器
                temp_sensor = next(iter(temps.values()))[0]
                metrics[ResourceType.TEMPERATURE] = ResourceMetrics(
                    resource_type=ResourceType.TEMPERATURE,
                    current_value=temp_sensor.current,
                    max_value=temp_sensor.high or 100.0,
                    unit="celsius"
                )
        except (AttributeError, IndexError):
            pass  # 温度传感器不可用
        
        # GPU使用率
        if self.gpu_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    metrics[ResourceType.GPU] = ResourceMetrics(
                        resource_type=ResourceType.GPU,
                        current_value=gpu.memoryUsed,
                        max_value=gpu.memoryTotal,
                        unit="MB"
                    )
            except Exception as e:
                self.logger.error(f"GPU指标获取失败: {e}")
        
        return metrics
    
    def get_resource_history(self, resource_type: ResourceType, 
                           hours: int = 1) -> List[ResourceMetrics]:
        """获取资源历史数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.resource_history.get(resource_type, [])
        
        return [metric for metric in history if metric.timestamp >= cutoff_time]
    
    def add_alert_callback(self, callback: Callable[[ResourceType, ResourceMetrics], None]) -> None:
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                
                for resource_type, metric in metrics.items():
                    # 记录历史数据
                    history = self.resource_history[resource_type]
                    history.append(metric)
                    
                    # 限制历史数据大小
                    if len(history) > 1000:
                        history[:] = history[-500:]
                    
                    # 检查告警阈值
                    threshold = self.alert_thresholds.get(resource_type)
                    if threshold and metric.utilization_percentage > threshold:
                        self._trigger_alert(resource_type, metric)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"资源监控错误: {e}")
                time.sleep(1.0)
    
    def _trigger_alert(self, resource_type: ResourceType, metric: ResourceMetrics) -> None:
        """触发告警"""
        self.logger.warning(
            f"资源告警: {resource_type.value} 使用率 {metric.utilization_percentage:.1f}% "
            f"超过阈值 {self.alert_thresholds[resource_type]:.1f}%"
        )
        
        for callback in self.alert_callbacks:
            try:
                callback(resource_type, metric)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")


class EdgeIOManager:
    """
    边缘IO管理器
    
    管理传感器数据读取、输出数据写入和文件系统操作。
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据目录
        self.data_dir = Path(config.data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 传感器管理
        self.sensors: Dict[str, Dict[str, Any]] = {}
        self.sensor_readings: Dict[str, List[SensorReading]] = {}
        
        # 输出缓存
        self.output_cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()
        
        # 文件监控
        self.file_watchers: Dict[str, Callable] = {}
    
    def register_sensor(self, sensor_id: str, sensor_type: str, 
                       config: Optional[Dict[str, Any]] = None) -> None:
        """注册传感器"""
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'config': config or {},
            'last_reading': None,
            'error_count': 0
        }
        
        self.sensor_readings[sensor_id] = []
        self.logger.info(f"传感器已注册: {sensor_id} ({sensor_type})")
    
    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """读取传感器数据"""
        if sensor_id not in self.sensors:
            self.logger.error(f"未知传感器: {sensor_id}")
            return None
        
        try:
            sensor_info = self.sensors[sensor_id]
            reading = self._read_sensor_value(sensor_id, sensor_info)
            
            if reading:
                # 记录读数
                self.sensor_readings[sensor_id].append(reading)
                sensor_info['last_reading'] = reading
                sensor_info['error_count'] = 0
                
                # 限制历史数据
                readings = self.sensor_readings[sensor_id]
                if len(readings) > 1000:
                    readings[:] = readings[-500:]
            
            return reading
            
        except Exception as e:
            self.sensors[sensor_id]['error_count'] += 1
            self.logger.error(f"传感器读取失败 {sensor_id}: {e}")
            return None
    
    def _read_sensor_value(self, sensor_id: str, sensor_info: Dict[str, Any]) -> Optional[SensorReading]:
        """读取传感器值（可扩展实现）"""
        sensor_type = sensor_info['type']
        sensor_config = sensor_info.get('config', {})
        
        # 实际传感器接口实现
        try:
            if sensor_type == "temperature":
                # 实际温度传感器读取逻辑
                value = self._read_temperature_sensor(sensor_config)
                return SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit="celsius"
                )
            elif sensor_type == "humidity":
                # 实际湿度传感器读取逻辑
                value = self._read_humidity_sensor(sensor_config)
                return SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit="percent"
                )
            elif sensor_type == "pressure":
                # 实际压力传感器读取逻辑
                value = self._read_pressure_sensor(sensor_config)
                return SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit="hPa"
                )
            else:
                # 通用传感器接口
                value = self._read_generic_sensor(sensor_type, sensor_config)
                return SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    value=value,
                    unit=sensor_config.get('unit', 'unit')
                )
        except Exception as e:
            self.logger.error(f"传感器读取错误 {sensor_id}: {e}")
            return None
    
    def _read_temperature_sensor(self, config: Dict[str, Any]) -> float:
        """读取温度传感器"""
        # 实际硬件接口实现
        # 这里应该连接到实际的温度传感器硬件
        device_path = config.get('device_path', '/sys/class/thermal/thermal_zone0/temp')
        
        try:
            if os.path.exists(device_path):
                with open(device_path, 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
        except (OSError, ValueError) as e:
            self.logger.warning(f"无法读取系统温度传感器: {e}")
        
        # 备用方案：使用psutil获取CPU温度
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except AttributeError:
            pass
        
        # 如果无法获取真实温度，抛出异常
        raise RuntimeError("无法获取温度传感器数据")
    
    def _read_humidity_sensor(self, config: Dict[str, Any]) -> float:
        """读取湿度传感器"""
        # 实际硬件接口实现
        # 这里应该连接到实际的湿度传感器硬件
        i2c_address = config.get('i2c_address', 0x40)
        
        try:
            # 尝试使用I2C接口读取湿度传感器
            # 这里需要根据具体的传感器型号实现
            import smbus
            bus = smbus.SMBus(1)  # I2C总线1
            
            # 发送测量命令（具体命令取决于传感器型号）
            bus.write_byte(i2c_address, 0xE5)  # 湿度测量命令
            time.sleep(0.1)  # 等待测量完成
            
            # 读取数据
            data = bus.read_i2c_block_data(i2c_address, 0, 2)
            humidity = ((data[0] << 8) + data[1]) * 125.0 / 65536.0 - 6.0
            
            return max(0.0, min(100.0, humidity))
            
        except ImportError:
            self.logger.warning("smbus模块未安装，无法读取I2C湿度传感器")
        except Exception as e:
            self.logger.warning(f"I2C湿度传感器读取失败: {e}")
        
        # 如果无法获取真实湿度，抛出异常
        raise RuntimeError("无法获取湿度传感器数据")
    
    def _read_pressure_sensor(self, config: Dict[str, Any]) -> float:
        """读取压力传感器"""
        # 实际硬件接口实现
        # 这里应该连接到实际的压力传感器硬件
        i2c_address = config.get('i2c_address', 0x77)
        
        try:
            # 尝试使用I2C接口读取压力传感器
            import smbus
            bus = smbus.SMBus(1)
            
            # BMP280/BME280压力传感器读取逻辑
            # 这里简化实现，实际需要根据传感器规格书实现
            bus.write_byte_data(i2c_address, 0xF4, 0x27)  # 配置寄存器
            time.sleep(0.1)
            
            # 读取压力数据
            data = bus.read_i2c_block_data(i2c_address, 0xF7, 3)
            pressure_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
            
            # 简化的压力计算（实际需要校准参数）
            pressure_hpa = pressure_raw / 256.0
            
            return pressure_hpa
            
        except ImportError:
            self.logger.warning("smbus模块未安装，无法读取I2C压力传感器")
        except Exception as e:
            self.logger.warning(f"I2C压力传感器读取失败: {e}")
        
        # 如果无法获取真实压力，抛出异常
        raise RuntimeError("无法获取压力传感器数据")
    
    def _read_generic_sensor(self, sensor_type: str, config: Dict[str, Any]) -> Union[float, int, str, bool]:
        """读取通用传感器"""
        # 通用传感器接口实现
        device_path = config.get('device_path')
        if device_path and os.path.exists(device_path):
            try:
                with open(device_path, 'r') as f:
                    raw_value = f.read().strip()
                
                # 尝试转换为数值
                try:
                    return float(raw_value)
                except ValueError:
                    return raw_value
                    
            except OSError as e:
                self.logger.error(f"读取设备文件失败 {device_path}: {e}")
        
        # 如果无法读取，抛出异常
        raise RuntimeError(f"无法获取{sensor_type}传感器数据")
    
    def write_output(self, key: str, value: Any, persist: bool = True) -> bool:
        """写入输出数据"""
        try:
            with self.cache_lock:
                self.output_cache[key] = {
                    'value': value,
                    'timestamp': datetime.now().isoformat(),
                    'persist': persist
                }
            
            if persist:
                self._persist_output(key, value)
            
            self.logger.debug(f"输出数据已写入: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"输出数据写入失败 {key}: {e}")
            return False
    
    def read_output(self, key: str) -> Optional[Any]:
        """读取输出数据"""
        with self.cache_lock:
            cached = self.output_cache.get(key)
            if cached:
                return cached['value']
        
        # 尝试从文件读取
        return self._load_persisted_output(key)
    
    def _persist_output(self, key: str, value: Any) -> None:
        """持久化输出数据"""
        output_file = self.data_dir / f"output_{key}.json"
        
        data = {
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_persisted_output(self, key: str) -> Optional[Any]:
        """加载持久化的输出数据"""
        output_file = self.data_dir / f"output_{key}.json"
        
        if not output_file.exists():
            return None
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('value')
        except Exception as e:
            self.logger.error(f"加载持久化数据失败 {key}: {e}")
            return None
    
    def get_sensor_history(self, sensor_id: str, hours: int = 1) -> List[SensorReading]:
        """获取传感器历史数据"""
        if sensor_id not in self.sensor_readings:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        readings = self.sensor_readings[sensor_id]
        
        return [reading for reading in readings if reading.timestamp >= cutoff_time]


class EdgeModelOptimizer:
    """
    边缘模型优化器
    
    为边缘设备优化机器学习模型，提高推理性能和降低资源消耗。
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.optimization_cache: Dict[str, str] = {}
    
    def optimize_model(self, model_path: Path, backend: str = "onnxruntime",
                      target_device: str = "cpu") -> Optional[Path]:
        """
        优化模型
        
        Args:
            model_path: 模型文件路径
            backend: 优化后端
            target_device: 目标设备
            
        Returns:
            优化后的模型路径
        """
        if not self.config.enable_model_optimization:
            self.logger.info("模型优化已禁用")
            return model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 检查缓存
        cache_key = f"{model_path}_{backend}_{target_device}"
        if cache_key in self.optimization_cache:
            cached_path = Path(self.optimization_cache[cache_key])
            if cached_path.exists():
                self.logger.info(f"使用缓存的优化模型: {cached_path}")
                return cached_path
        
        try:
            optimized_path = self._perform_optimization(model_path, backend, target_device)
            
            if optimized_path:
                self.optimization_cache[cache_key] = str(optimized_path)
                self.logger.info(f"模型优化完成: {model_path} -> {optimized_path}")
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            raise RuntimeError(f"模型优化失败: {e}")
    
    def _perform_optimization(self, model_path: Path, backend: str, 
                            target_device: str) -> Optional[Path]:
        """执行模型优化"""
        if backend == "onnxruntime":
            return self._optimize_with_onnxruntime(model_path, target_device)
        elif backend == "tensorrt":
            return self._optimize_with_tensorrt(model_path, target_device)
        elif backend == "openvino":
            return self._optimize_with_openvino(model_path, target_device)
        else:
            raise ValueError(f"不支持的优化后端: {backend}")
    
    def _optimize_with_onnxruntime(self, model_path: Path, target_device: str) -> Optional[Path]:
        """使用ONNX Runtime优化"""
        try:
            import onnxruntime as ort
            
            # 创建优化后的模型路径
            optimized_path = model_path.parent / f"{model_path.stem}_optimized_ort.onnx"
            
            # 实现ONNX Runtime优化
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = str(optimized_path)
            
            # 创建推理会话以触发优化
            providers = ['CPUExecutionProvider']
            if target_device == 'gpu':
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
            
            self.logger.info(f"ONNX Runtime优化完成: {optimized_path}")
            return optimized_path
            
        except ImportError:
            self.logger.warning("ONNX Runtime未安装，跳过优化")
            return model_path
        except Exception as e:
            self.logger.error(f"ONNX Runtime优化失败: {e}")
            return None
    
    def _optimize_with_tensorrt(self, model_path: Path, target_device: str) -> Optional[Path]:
        """使用TensorRT优化"""
        try:
            import tensorrt as trt
            
            # 创建优化后的模型路径
            optimized_path = model_path.parent / f"{model_path.stem}_optimized_trt.engine"
            
            # TensorRT优化实现
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # 解析ONNX模型
            with open(model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(f"TensorRT解析错误: {parser.get_error(error)}")
                    return None
            
            # 配置构建器
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 28  # 256MB
            
            if target_device == 'gpu':
                config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16精度
            
            # 构建引擎
            engine = builder.build_engine(network, config)
            
            if engine:
                # 保存引擎
                with open(optimized_path, 'wb') as f:
                    f.write(engine.serialize())
                
                self.logger.info(f"TensorRT优化完成: {optimized_path}")
                return optimized_path
            else:
                self.logger.error("TensorRT引擎构建失败")
                return None
                
        except ImportError:
            self.logger.warning("TensorRT未安装，跳过优化")
            return model_path
        except Exception as e:
            self.logger.error(f"TensorRT优化失败: {e}")
            return None
    
    def _optimize_with_openvino(self, model_path: Path, target_device: str) -> Optional[Path]:
        """使用OpenVINO优化"""
        try:
            from openvino.runtime import Core
            
            # 创建优化后的模型路径
            optimized_path = model_path.parent / f"{model_path.stem}_optimized_ov.xml"
            
            # OpenVINO优化实现
            core = Core()
            
            # 读取模型
            model = core.read_model(str(model_path))
            
            # 编译模型
            device_name = "GPU" if target_device == 'gpu' else "CPU"
            compiled_model = core.compile_model(model, device_name)
            
            # 保存优化后的模型
            core.save_model(model, str(optimized_path))
            
            self.logger.info(f"OpenVINO优化完成: {optimized_path}")
            return optimized_path
            
        except ImportError:
            self.logger.warning("OpenVINO未安装，跳过优化")
            return model_path
        except Exception as e:
            self.logger.error(f"OpenVINO优化失败: {e}")
            return None


class EdgeInferenceEngine:
    """
    边缘推理引擎
    
    执行优化后的模型推理，支持多种推理后端。
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.loaded_models: Dict[str, Any] = {}
        self.model_lock = threading.RLock()
    
    def load_model(self, model_path: Path, backend: str = "onnxruntime") -> str:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            backend: 推理后端
            
        Returns:
            模型ID
        """
        model_id = f"{model_path.stem}_{backend}"
        
        with self.model_lock:
            if model_id in self.loaded_models:
                self.logger.info(f"模型已加载: {model_id}")
                return model_id
            
            try:
                if backend == "onnxruntime":
                    session = self._load_onnx_model(model_path)
                elif backend == "tensorrt":
                    session = self._load_tensorrt_model(model_path)
                elif backend == "openvino":
                    session = self._load_openvino_model(model_path)
                else:
                    raise ValueError(f"不支持的推理后端: {backend}")
                
                self.loaded_models[model_id] = {
                    'session': session,
                    'backend': backend,
                    'path': model_path,
                    'load_time': datetime.now()
                }
                
                self.logger.info(f"模型加载完成: {model_id}")
                return model_id
                
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise
    
    def run_inference(self, model_id: str, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        运行推理
        
        Args:
            model_id: 模型ID
            input_data: 输入数据
            
        Returns:
            推理结果
        """
        with self.model_lock:
            if model_id not in self.loaded_models:
                raise ValueError(f"模型未加载: {model_id}")
            
            model_info = self.loaded_models[model_id]
            session = model_info['session']
            backend = model_info['backend']
            
            try:
                start_time = time.time()
                
                if backend == "onnxruntime":
                    outputs = self._run_onnx_inference(session, input_data)
                elif backend == "tensorrt":
                    outputs = self._run_tensorrt_inference(session, input_data)
                elif backend == "openvino":
                    outputs = self._run_openvino_inference(session, input_data)
                else:
                    raise ValueError(f"不支持的推理后端: {backend}")
                
                inference_time = time.time() - start_time
                
                self.logger.debug(f"推理完成: {model_id}, 耗时: {inference_time:.3f}s")
                
                return outputs
                
            except Exception as e:
                self.logger.error(f"推理执行失败: {e}")
                raise
    
    def _load_onnx_model(self, model_path: Path):
        """加载ONNX模型"""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if self.config.enable_gpu:
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session
    
    def _load_tensorrt_model(self, model_path: Path):
        """加载TensorRT模型"""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 加载TensorRT引擎
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        return {'engine': engine, 'context': context}
    
    def _load_openvino_model(self, model_path: Path):
        """加载OpenVINO模型"""
        from openvino.runtime import Core
        
        core = Core()
        model = core.read_model(str(model_path))
        
        device_name = "GPU" if self.config.enable_gpu else "CPU"
        compiled_model = core.compile_model(model, device_name)
        
        return compiled_model
    
    def _run_onnx_inference(self, session, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """运行ONNX推理"""
        # 准备输入
        ort_inputs = {}
        for input_name in [inp.name for inp in session.get_inputs()]:
            if input_name in input_data:
                ort_inputs[input_name] = input_data[input_name]
        
        # 运行推理
        ort_outputs = session.run(None, ort_inputs)
        
        # 准备输出
        output_names = [out.name for out in session.get_outputs()]
        outputs = {}
        for i, output_name in enumerate(output_names):
            outputs[output_name] = ort_outputs[i]
        
        return outputs
    
    def _run_tensorrt_inference(self, trt_session, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """运行TensorRT推理"""
        import pycuda.driver as cuda
        
        engine = trt_session['engine']
        context = trt_session['context']
        
        # 分配GPU内存
        inputs = []
        outputs = []
        bindings = []
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        # 创建CUDA流
        stream = cuda.Stream()
        
        # 复制输入数据到GPU
        for i, (binding_name, input_array) in enumerate(input_data.items()):
            np.copyto(inputs[i]['host'], input_array.ravel())
            cuda.memcpy_htod_async(inputs[i]['device'], inputs[i]['host'], stream)
        
        # 运行推理
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # 复制输出数据到CPU
        result_outputs = {}
        for i, output in enumerate(outputs):
            cuda.memcpy_dtoh_async(output['host'], output['device'], stream)
            stream.synchronize()
            
            output_name = f"output_{i}"
            result_outputs[output_name] = output['host'].copy()
        
        return result_outputs
    
    def _run_openvino_inference(self, compiled_model, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """运行OpenVINO推理"""
        # 创建推理请求
        infer_request = compiled_model.create_infer_request()
        
        # 设置输入数据
        for input_name, input_array in input_data.items():
            infer_request.set_tensor(input_name, input_array)
        
        # 运行推理
        infer_request.infer()
        
        # 获取输出
        outputs = {}
        for output in compiled_model.outputs:
            output_tensor = infer_request.get_tensor(output)
            outputs[output.any_name] = output_tensor.data.copy()
        
        return outputs
    
    def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        with self.model_lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                self.logger.info(f"模型已卸载: {model_id}")
                return True
            return False


class EdgeSystem:
    """
    生产级边缘计算系统
    
    集成资源管理、IO操作、模型优化等功能的完整边缘计算解决方案。
    """
    
    def __init__(self, config: EdgeDeviceConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # 系统状态
        self.status = DeviceStatus.INITIALIZING
        self.start_time = datetime.now()
        
        # 核心组件
        self.resource_manager = EdgeResourceManager(config)
        self.io_manager = EdgeIOManager(config)
        self.model_optimizer = EdgeModelOptimizer(config)
        self.inference_engine = EdgeInferenceEngine(config)
        
        # 任务执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 健康检查
        self.health_check_interval = 30.0
        self.health_check_active = False
        self.health_check_thread: Optional[threading.Thread] = None
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("EdgeSystem")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start(self) -> None:
        """启动边缘系统"""
        try:
            self.logger.info(f"启动边缘系统: {self.config.device_name}")
            
            # 启动资源监控
            self.resource_manager.start_monitoring()
            
            # 注册默认传感器
            self._register_default_sensors()
            
            # 启动健康检查
            self._start_health_check()
            
            self.status = DeviceStatus.RUNNING
            self.logger.info("边缘系统启动完成")
            
        except Exception as e:
            self.status = DeviceStatus.ERROR
            self.logger.error(f"边缘系统启动失败: {e}")
            raise
    
    def stop(self) -> None:
        """停止边缘系统"""
        try:
            self.logger.info("停止边缘系统")
            
            self.status = DeviceStatus.SHUTDOWN
            
            # 停止健康检查
            self._stop_health_check()
            
            # 停止资源监控
            self.resource_manager.stop_monitoring()
            
            # 关闭执行器
            self.executor.shutdown(wait=True)
            
            self.logger.info("边缘系统已停止")
            
        except Exception as e:
            self.logger.error(f"边缘系统停止失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = datetime.now() - self.start_time
        
        return {
            'device_id': self.config.device_id,
            'device_name': self.config.device_name,
            'status': self.status.value,
            'uptime_seconds': uptime.total_seconds(),
            'resource_metrics': self.resource_manager.get_current_metrics(),
            'sensor_count': len(self.io_manager.sensors),
            'gpu_available': self.resource_manager.gpu_available,
            'loaded_models': list(self.inference_engine.loaded_models.keys())
        }
    
    def run_inference(self, model_path: str, input_data: Dict[str, np.ndarray], 
                     optimize: bool = True, backend: str = "onnxruntime") -> Future:
        """运行推理任务"""
        def _inference_task():
            try:
                model_file = Path(model_path)
                
                # 模型优化
                if optimize:
                    optimized_model = self.model_optimizer.optimize_model(model_file, backend)
                    if optimized_model:
                        model_file = optimized_model
                
                # 加载模型
                model_id = self.inference_engine.load_model(model_file, backend)
                
                # 执行推理
                start_time = time.time()
                outputs = self.inference_engine.run_inference(model_id, input_data)
                inference_time = time.time() - start_time
                
                result = {
                    'model_id': model_id,
                    'model_path': str(model_file),
                    'backend': backend,
                    'inference_time': inference_time,
                    'outputs': outputs,
                    'status': 'success'
                }
                
                self.logger.info(f"推理完成: {model_path}, 耗时: {inference_time:.3f}s")
                return result
                
            except Exception as e:
                self.logger.error(f"推理失败: {e}")
                return {
                    'model_path': model_path,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return self.executor.submit(_inference_task)
    
    def _register_default_sensors(self) -> None:
        """注册默认传感器"""
        default_sensors = [
            ('cpu_temp', 'temperature', {'device_path': '/sys/class/thermal/thermal_zone0/temp'}),
            ('system_load', 'generic', {'unit': 'percent'}),
        ]
        
        for sensor_id, sensor_type, config in default_sensors:
            try:
                self.io_manager.register_sensor(sensor_id, sensor_type, config)
            except Exception as e:
                self.logger.warning(f"注册传感器失败 {sensor_id}: {e}")
    
    def _start_health_check(self) -> None:
        """启动健康检查"""
        self.health_check_active = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def _stop_health_check(self) -> None:
        """停止健康检查"""
        self.health_check_active = False
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
    
    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self.health_check_active:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"健康检查错误: {e}")
                time.sleep(5.0)
    
    def _perform_health_check(self) -> None:
        """执行健康检查"""
        # 检查资源使用情况
        metrics = self.resource_manager.get_current_metrics()
        
        critical_resources = []
        for resource_type, metric in metrics.items():
            if metric.utilization_percentage > 90.0:
                critical_resources.append(resource_type.value)
        
        if critical_resources:
            if self.status == DeviceStatus.RUNNING:
                self.status = DeviceStatus.DEGRADED
                self.logger.warning(f"系统性能降级，资源紧张: {critical_resources}")
        else:
            if self.status == DeviceStatus.DEGRADED:
                self.status = DeviceStatus.RUNNING
                self.logger.info("系统性能恢复正常")


def create_edge_system(device_id: str, device_name: str, 
                      data_directory: str = "./edge_data") -> EdgeSystem:
    """创建边缘系统实例"""
    config = EdgeDeviceConfig(
        device_id=device_id,
        device_name=device_name,
        data_directory=data_directory
    )
    
    return EdgeSystem(config)