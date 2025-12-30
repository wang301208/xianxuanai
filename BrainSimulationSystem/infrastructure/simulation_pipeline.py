"""
自动化仿真流水线

提供：
- 自动化仿真工作流管理
- 参数扫描与优化
- 批量实验执行
- 结果聚合与分析
- 错误恢复与重试机制
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools
import pickle
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime, timedelta, timezone
import traceback

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

from .experiment_storage import ExperimentStorage

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """流水线阶段"""
    PREPARATION = "preparation"
    PARAMETER_GENERATION = "parameter_generation"
    NETWORK_CREATION = "network_creation"
    SIMULATION_EXECUTION = "simulation_execution"
    RESULT_COLLECTION = "result_collection"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    CLEANUP = "cleanup"

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class OptimizationMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"

@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    param_type: str  # "continuous", "discrete", "categorical"
    bounds: Optional[Tuple[float, float]] = None  # 连续参数范围
    values: Optional[List[Any]] = None  # 离散/分类参数值
    distribution: str = "uniform"  # "uniform", "normal", "log_uniform"
    
@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_id: str
    name: str
    description: str
    
    # 参数空间
    parameter_spaces: List[ParameterSpace]
    
    # 优化配置
    optimization_method: OptimizationMethod
    optimization_config: Dict[str, Any]
    
    # 仿真配置
    base_network_config: Dict[str, Any]
    base_simulation_params: Dict[str, Any]

    # 元数据
    responsible: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    project_name: Optional[str] = None
    project_version: Optional[str] = None
    
    # 执行配置
    max_parallel_jobs: int = 4
    max_retries: int = 3
    timeout_per_job: float = 3600.0  # 秒
    
    # 输出配置
    output_directory: str = "./data/experiments"
    experiment_dirname: Optional[str] = None
    save_intermediate_results: bool = True
    
    # 目标函数
    objective_function: Optional[str] = None  # 函数名或路径

    # 可复现性
    random_seed: Optional[int] = None

    # 初始状态与配置快照
    save_initial_state: bool = True
    save_job_configs: bool = True

    # 仿真输入（任务/刺激/数据集等）
    input_spec: Dict[str, Any] = field(default_factory=dict)
    input_artifacts: List[str] = field(default_factory=list)
    copy_input_artifacts: bool = True

    # 导出格式（可选）
    export_nwb: bool = False
    nwb_filename: str = "spike_trains.nwb"
    export_sonata: bool = False
    sonata_directory: str = "sonata"
    
@dataclass
class SimulationJob:
    """仿真作业"""
    job_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    network_config: Dict[str, Any]
    simulation_params: Dict[str, Any]
    
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    retry_count: int = 0
    
    # 资源使用
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

    # 可复现性
    random_seed: Optional[int] = None
    
@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    
    best_parameters: Dict[str, Any]
    best_objective_value: float
    
    all_results: List[Dict[str, Any]]
    execution_summary: Dict[str, Any]
    
    start_time: datetime
    end_time: datetime
    total_execution_time: float

class ParameterGenerator:
    """参数生成器"""
    
    def __init__(self, parameter_spaces: List[ParameterSpace], rng: Optional[np.random.Generator] = None):
        self.parameter_spaces = parameter_spaces
        self.rng = rng or np.random.default_rng()
        self.logger = logging.getLogger("ParameterGenerator")
    
    def generate_grid_search_parameters(self, samples_per_dimension: int = 10) -> Iterator[Dict[str, Any]]:
        """生成网格搜索参数"""
        param_grids = []
        
        for space in self.parameter_spaces:
            if space.param_type == "continuous":
                if space.bounds:
                    values = np.linspace(space.bounds[0], space.bounds[1], samples_per_dimension)
                else:
                    raise ValueError(f"连续参数 {space.name} 需要指定边界")
            elif space.param_type in ["discrete", "categorical"]:
                if space.values:
                    values = space.values
                else:
                    raise ValueError(f"离散/分类参数 {space.name} 需要指定值列表")
            else:
                raise ValueError(f"未知参数类型: {space.param_type}")
            
            param_grids.append(values)
        
        # 生成笛卡尔积
        for param_combination in itertools.product(*param_grids):
            yield dict(zip([space.name for space in self.parameter_spaces], param_combination))
    
    def generate_random_search_parameters(self, num_samples: int = 100) -> Iterator[Dict[str, Any]]:
        """生成随机搜索参数"""
        for _ in range(num_samples):
            parameters = {}
            
            for space in self.parameter_spaces:
                if space.param_type == "continuous":
                    if space.bounds:
                        if space.distribution == "uniform":
                            value = self.rng.uniform(space.bounds[0], space.bounds[1])
                        elif space.distribution == "normal":
                            mean = (space.bounds[0] + space.bounds[1]) / 2
                            std = (space.bounds[1] - space.bounds[0]) / 6
                            value = np.clip(self.rng.normal(mean, std), space.bounds[0], space.bounds[1])
                        elif space.distribution == "log_uniform":
                            log_low = np.log10(max(space.bounds[0], 1e-10))
                            log_high = np.log10(space.bounds[1])
                            value = 10 ** self.rng.uniform(log_low, log_high)
                        else:
                            value = self.rng.uniform(space.bounds[0], space.bounds[1])
                    else:
                        raise ValueError(f"连续参数 {space.name} 需要指定边界")
                elif space.param_type in ["discrete", "categorical"]:
                    if space.values:
                        value = self.rng.choice(space.values)
                    else:
                        raise ValueError(f"离散/分类参数 {space.name} 需要指定值列表")
                else:
                    raise ValueError(f"未知参数类型: {space.param_type}")
                
                parameters[space.name] = value
            
            yield parameters
    
    def generate_bayesian_optimization_parameters(self, num_samples: int = 100, 
                                                previous_results: List[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """生成贝叶斯优化参数（简化版本）"""
        # 这里是一个简化的贝叶斯优化实现
        # 实际应用中可以使用 scikit-optimize 或 GPyOpt
        
        if not previous_results:
            # 初始随机采样
            yield from self.generate_random_search_parameters(min(10, num_samples))
            return
        
        # 基于历史结果的采样（简化版本）
        for _ in range(num_samples):
            # 在最佳结果附近采样
            best_result = max(previous_results, key=lambda x: x.get('objective_value', -np.inf))
            best_params = best_result.get('parameters', {})
            
            parameters = {}
            for space in self.parameter_spaces:
                if space.name in best_params:
                    base_value = best_params[space.name]
                    
                    if space.param_type == "continuous" and space.bounds:
                        # 在最佳值附近添加噪声
                        noise_scale = (space.bounds[1] - space.bounds[0]) * 0.1
                        value = np.clip(
                            base_value + self.rng.normal(0, noise_scale),
                            space.bounds[0], space.bounds[1]
                        )
                    else:
                        # 对于离散参数，随机选择
                        if space.values:
                            value = self.rng.choice(space.values)
                        else:
                            value = base_value
                else:
                    # 如果没有历史数据，随机采样
                    if space.param_type == "continuous" and space.bounds:
                        value = self.rng.uniform(space.bounds[0], space.bounds[1])
                    elif space.values:
                        value = self.rng.choice(space.values)
                    else:
                        value = 0.0
                
                parameters[space.name] = value
            
            yield parameters

class ObjectiveFunction:
    """目标函数"""
    
    def __init__(self, function_name: str = "default"):
        self.function_name = function_name
        self.logger = logging.getLogger("ObjectiveFunction")
    
    def evaluate(self, simulation_results: Dict[str, Any], 
                parameters: Dict[str, Any]) -> float:
        """评估目标函数值"""
        if self.function_name == "spike_rate_fitness":
            return self._spike_rate_fitness(simulation_results, parameters)
        elif self.function_name == "synchrony_fitness":
            return self._synchrony_fitness(simulation_results, parameters)
        elif self.function_name == "energy_efficiency":
            return self._energy_efficiency(simulation_results, parameters)
        else:
            return self._default_fitness(simulation_results, parameters)
    
    def _spike_rate_fitness(self, results: Dict[str, Any], params: Dict[str, Any]) -> float:
        """基于尖峰频率的适应度函数"""
        target_rate = params.get('target_spike_rate', 10.0)  # Hz
        
        total_spikes = 0
        total_neurons = 0
        simulation_time = results.get('simulation_time', 1000.0)  # ms
        
        for recorder_name, recorder_data in results.items():
            if 'spike_recorder' in recorder_name:
                spikes = recorder_data.get('times', [])
                neurons = len(set(recorder_data.get('senders', [])))
                
                total_spikes += len(spikes)
                total_neurons += neurons
        
        if total_neurons == 0:
            return -1000.0  # 惩罚无神经元活动
        
        actual_rate = (total_spikes / total_neurons) / (simulation_time / 1000.0)
        fitness = -abs(actual_rate - target_rate)  # 越接近目标越好
        
        return fitness
    
    def _synchrony_fitness(self, results: Dict[str, Any], params: Dict[str, Any]) -> float:
        """基于同步性的适应度函数"""
        target_synchrony = params.get('target_synchrony', 0.5)
        
        synchrony_scores = []
        
        for recorder_name, recorder_data in results.items():
            if 'spike_recorder' in recorder_name:
                times = np.array(recorder_data.get('times', []))
                
                if len(times) > 10:
                    # 计算尖峰时间的变异系数作为同步性指标
                    time_bins = np.histogram(times, bins=50)[0]
                    if np.mean(time_bins) > 0:
                        cv = np.std(time_bins) / np.mean(time_bins)
                        synchrony = 1.0 / (1.0 + cv)  # 变异系数越小，同步性越高
                        synchrony_scores.append(synchrony)
        
        if not synchrony_scores:
            return -1000.0
        
        actual_synchrony = np.mean(synchrony_scores)
        fitness = -abs(actual_synchrony - target_synchrony)
        
        return fitness
    
    def _energy_efficiency(self, results: Dict[str, Any], params: Dict[str, Any]) -> float:
        """基于能量效率的适应度函数"""
        # 计算每个尖峰的能量成本
        total_spikes = 0
        for recorder_name, recorder_data in results.items():
            if 'spike_recorder' in recorder_name:
                total_spikes += len(recorder_data.get('times', []))
        
        # 假设的能量模型
        energy_per_spike = 1.0  # pJ
        total_energy = total_spikes * energy_per_spike
        
        # 计算信息传输效率
        information_bits = np.log2(max(total_spikes, 1))
        efficiency = information_bits / max(total_energy, 1e-10)
        
        return efficiency
    
    def _default_fitness(self, results: Dict[str, Any], params: Dict[str, Any]) -> float:
        """默认适应度函数"""
        # 简单地返回总尖峰数
        total_spikes = 0
        for recorder_name, recorder_data in results.items():
            if isinstance(recorder_data, dict) and 'times' in recorder_data:
                total_spikes += len(recorder_data['times'])
        
        return float(total_spikes)

class SimulationPipeline:
    """仿真流水线"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(f"SimulationPipeline.{config.experiment_id}")
        self._file_log_handler: Optional[logging.Handler] = None
        self._file_log_path: Optional[Path] = None
        
        # 组件初始化
        self.rng = np.random.default_rng(config.random_seed)
        self.parameter_generator = ParameterGenerator(config.parameter_spaces, rng=self.rng)
        self.objective_function = ObjectiveFunction(config.objective_function or "default")
        
        # 作业管理
        self.jobs = {}
        self.job_queue = []
        self.completed_jobs = []
        self.failed_jobs = []
        
        # 执行状态
        self.is_running = False
        self.start_time = None
        self.end_time = None
        
        # 结果存储
        self.storage = ExperimentStorage(
            config.output_directory,
            config.experiment_id,
            experiment_dir_name=config.experiment_dirname,
            name=config.name,
            description=config.description,
            meta={
                "project": {"name": config.project_name, "version": config.project_version},
                "responsible": {"name": config.responsible},
                "tags": config.tags,
                "metadata": config.metadata,
            },
        )
        self.storage.initialize()
        self.storage.mark_status("initialized")
        self.results_directory = self.storage.dirs.outputs
        (self.results_directory / "jobs").mkdir(parents=True, exist_ok=True)
        self._setup_file_logging()
        self._persist_experiment_definition()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()

    def _setup_file_logging(self) -> None:
        if self._file_log_handler is not None:
            return

        log_path = self.storage.dirs.logs / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        self.logger.addHandler(handler)
        self._file_log_handler = handler
        self._file_log_path = log_path

    def _finalize_file_logging(self) -> None:
        handler = self._file_log_handler
        path = self._file_log_path
        if handler is None or path is None:
            return

        try:
            self.logger.removeHandler(handler)
        except Exception:
            pass
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass

        try:
            if path.exists():
                self.storage.register_file(path, kind="log", format="log")
        except Exception:
            pass

        self._file_log_handler = None
        self._file_log_path = None

    def _persist_experiment_definition(self) -> None:
        """将实验配置与模型/仿真参数落盘，便于复现与共享"""
        self.storage.write_json(
            "config/sim/experiment_config.json",
            {
                "experiment_id": self.config.experiment_id,
                "name": self.config.name,
                "description": self.config.description,
                "responsible": self.config.responsible,
                "tags": self.config.tags,
                "metadata": self.config.metadata,
                "project_name": self.config.project_name,
                "project_version": self.config.project_version,
                "parameter_spaces": [
                    {
                        "name": space.name,
                        "param_type": space.param_type,
                        "bounds": list(space.bounds) if space.bounds else None,
                        "values": space.values,
                        "distribution": space.distribution,
                    }
                    for space in self.config.parameter_spaces
                ],
                "optimization_method": self.config.optimization_method.value,
                "optimization_config": self.config.optimization_config,
                "base_network_config": self.config.base_network_config,
                "base_simulation_params": self.config.base_simulation_params,
                "max_parallel_jobs": self.config.max_parallel_jobs,
                "max_retries": self.config.max_retries,
                "timeout_per_job": self.config.timeout_per_job,
                "output_directory": self.config.output_directory,
                "experiment_dirname": self.config.experiment_dirname,
                "save_intermediate_results": self.config.save_intermediate_results,
                "objective_function": self.config.objective_function,
                "random_seed": self.config.random_seed,
                "save_initial_state": self.config.save_initial_state,
                "save_job_configs": self.config.save_job_configs,
                "input_spec": self.config.input_spec,
                "input_artifacts": self.config.input_artifacts,
                "copy_input_artifacts": self.config.copy_input_artifacts,
                "export_nwb": self.config.export_nwb,
                "nwb_filename": self.config.nwb_filename,
                "export_sonata": self.config.export_sonata,
                "sonata_directory": self.config.sonata_directory,
            },
            kind="experiment_config",
        )
        self.storage.write_json(
            "config/model/base_network_config.json",
            self.config.base_network_config,
            kind="model_config",
        )
        self.storage.write_json(
            "config/sim/base_simulation_params.json",
            self.config.base_simulation_params,
            kind="simulation_params",
        )
        self._persist_inputs(copy_artifacts=False)
        self._write_flat_config_files(jobs=None)

    def _persist_inputs(self, *, copy_artifacts: bool) -> None:
        self.storage.write_json(
            "config/inputs/input_spec.json",
            self.config.input_spec or {},
            kind="inputs",
        )

        if not copy_artifacts or not self.config.copy_input_artifacts or not self.config.input_artifacts:
            return

        staged: list[Dict[str, Any]] = []
        for source in self.config.input_artifacts:
            try:
                copied = self.storage.copy_into(source, "config/inputs/artifacts", kind="input_artifact")
            except Exception as e:
                self.logger.warning(f"输入数据拷贝失败: {source}: {e}")
                staged.append({"source": source, "status": "failed", "error": str(e)})
                continue

            staged.append(
                {
                    "source": source,
                    "status": "ok",
                    "copied_files": [
                        path.relative_to(self.storage.dirs.root).as_posix() for path in copied
                    ],
                }
            )

        self.storage.write_json(
            "config/inputs/artifacts_manifest.json",
            {
                "schema_version": "inputs-manifest-v1",
                "experiment_id": self.config.experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": staged,
            },
            kind="inputs",
        )

    def _write_flat_config_files(self, jobs: Optional[List[SimulationJob]]) -> None:
        try:
            import importlib.util

            has_pytables = importlib.util.find_spec("tables") is not None
        except Exception:
            has_pytables = False

        spikes_path = "data/spikes.h5" if h5py is not None else "data/spikes.npz"
        states_path = "data/states.h5" if h5py is not None else "data/states.npz"
        detailed_results_path = (
            "results/detailed_results.h5" if h5py is not None else "results/detailed_results.json"
        )

        nwb_path = None
        if self.config.export_nwb:
            candidate = Path(self.config.nwb_filename)
            nwb_path = (
                (Path("data") / candidate).as_posix()
                if candidate.parent == Path(".")
                else candidate.as_posix()
            )

        sonata_dir = None
        if self.config.export_sonata:
            candidate = Path(self.config.sonata_directory)
            if candidate.parts and candidate.parts[0] == "data":
                sonata_dir = candidate.as_posix()
            else:
                sonata_dir = (Path("data") / candidate).as_posix()

        config_payload = {
            "schema_version": "config-v1",
            "experiment_id": self.config.experiment_id,
            "name": self.config.name,
            "description": self.config.description,
            "experiment_dirname": self.storage.experiment_dir_name,
            "project": {"name": self.config.project_name, "version": self.config.project_version},
            "responsible": self.config.responsible,
            "tags": self.config.tags,
            "metadata": self.config.metadata,
            "optimization_method": self.config.optimization_method.value,
            "optimization_config": self.config.optimization_config,
            "random_seed": self.config.random_seed,
            "job_seeds": ({job.job_id: job.random_seed for job in jobs} if jobs else None),
            "base_network_config": self.config.base_network_config,
            "base_simulation_params": self.config.base_simulation_params,
            "input_spec": self.config.input_spec,
            "input_artifacts": self.config.input_artifacts,
            "export": {
                "nwb": self.config.export_nwb,
                "nwb_filename": self.config.nwb_filename,
                "sonata": self.config.export_sonata,
                "sonata_directory": self.config.sonata_directory,
            },
            "paths": {
                "meta": "meta.json",
                "manifest": "manifest.json",
                "provenance": "provenance.json",
                "logs_dir": "results/logs",
                "pipeline_log": "results/logs/pipeline.log",
                "experiment_config": "config/sim/experiment_config.json",
                "seeds": "config/sim/seeds.json",
                "network_structure": "data/network_structure.json",
                "inputs": "config/inputs/input_spec.json",
                "job_configs_dir": "config/sim/jobs",
                "job_results_dir": "results/jobs",
                "initial_states_dir": "data/checkpoints/initial_states",
                "experiment_result": "results/experiment_result.json",
                "detailed_results": detailed_results_path,
                "training_log": "results/training_log.csv",
                "training_log_hdf5": ("results/training_log.h5" if has_pytables else None),
                "results_summary": "results/results_summary.json",
                "spikes": spikes_path,
                "states": states_path,
                "nwb": nwb_path,
                "sonata_dir": sonata_dir,
            },
        }

        self.storage.write_json("config/config.json", config_payload, kind="config")
        try:
            self.storage.write_yaml("config/config.yaml", config_payload, kind="config")
        except ImportError:
            pass

        self.storage.write_json(
            "data/network_structure.json",
            {
                "schema_version": "network-structure-v1",
                "experiment_id": self.config.experiment_id,
                "network_config": self.config.base_network_config,
            },
            kind="network",
        )
        self.storage.write_json(
            "data/network.json",
            {
                "schema_version": "network-v1",
                "experiment_id": self.config.experiment_id,
                "network_config": self.config.base_network_config,
            },
            kind="network",
        )

        if self.config.export_sonata:
            sonata_rel_dir = Path(sonata_dir or (Path("data") / self.config.sonata_directory).as_posix())
            nodes_file = self.storage.dirs.root / sonata_rel_dir / "nodes.csv"
            if not nodes_file.exists():
                try:
                    self.storage.export_network_to_sonata_stub(
                        sonata_rel_dir,
                        network_config=self.config.base_network_config,
                        metadata={"experiment_id": self.config.experiment_id},
                    )
                except Exception as e:
                    self.logger.warning(f"SONATA 导出失败: {e}")

    def _save_experiment_seeds(self, jobs: List[SimulationJob]) -> None:
        self.storage.write_json(
            "config/sim/seeds.json",
            {
                "schema_version": "seeds-v1",
                "experiment_id": self.config.experiment_id,
                "experiment_random_seed": self.config.random_seed,
                "job_seeds": {job.job_id: job.random_seed for job in jobs},
            },
            kind="seeds",
        )
        self._write_flat_config_files(jobs=jobs)

    def _save_job_config(self, job: SimulationJob) -> None:
        self.storage.write_json(
            Path("config/sim/jobs") / f"{job.job_id}.json",
            {
                "schema_version": "job-config-v1",
                "job_id": job.job_id,
                "experiment_id": job.experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": job.random_seed,
                "parameters": job.parameters,
                "network_config": job.network_config,
                "simulation_params": job.simulation_params,
            },
            kind="job_config",
            metadata={"job_id": job.job_id},
        )

    def _save_training_log(self, jobs: List[SimulationJob]) -> None:
        rows: List[Dict[str, Any]] = []
        for idx, job in enumerate(sorted(jobs, key=lambda j: j.job_id)):
            rows.append(
                {
                    "step": idx,
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "objective_value": job.results.get("objective_value"),
                    "execution_time": job.execution_time,
                    "start_time": job.start_time.isoformat() if job.start_time else None,
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                }
            )

        self.storage.write_csv("results/training_log.csv", rows, kind="training_log")
        try:
            self.storage.write_pytables_table(
                "results/training_log.h5",
                rows,
                kind="training_log",
                fieldnames=[
                    "step",
                    "job_id",
                    "status",
                    "objective_value",
                    "execution_time",
                    "start_time",
                    "end_time",
                ],
                table_name="training_log",
            )
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"training_log PyTables 导出失败: {e}")

    def _export_best_run_artifacts(self, job: SimulationJob) -> Dict[str, Optional[str]]:
        exported: Dict[str, Optional[str]] = {"spikes": None, "states": None, "nwb": None}

        simulation_results = job.results.get("simulation_results", {})
        spike_recorder = None
        if isinstance(simulation_results, dict):
            spike_recorder = simulation_results.get("spike_recorder")

        if isinstance(spike_recorder, dict) and "times" in spike_recorder and "senders" in spike_recorder:
            times = np.asarray(spike_recorder.get("times", []), dtype=np.float64)
            senders = np.asarray(spike_recorder.get("senders", []), dtype=np.int32)
            if h5py is not None:
                self.storage.write_hdf5(
                    "data/spikes.h5",
                    {"times": times, "senders": senders},
                    kind="spikes",
                    metadata={"source_job_id": job.job_id},
                )
                exported["spikes"] = "data/spikes.h5"
            else:
                self.storage.write_npz(
                    "data/spikes.npz",
                    {"times": times, "senders": senders},
                    kind="spikes",
                    metadata={"source_job_id": job.job_id},
                )
                exported["spikes"] = "data/spikes.npz"

            if self.config.export_nwb:
                try:
                    nwb_relpath = Path(self.config.nwb_filename)
                    if nwb_relpath.parent == Path("."):
                        nwb_relpath = Path("data") / nwb_relpath
                    self.storage.export_spikes_to_nwb(
                        nwb_relpath,
                        times=times,
                        senders=senders,
                        session_description=self.config.description,
                        metadata={"source_job_id": job.job_id},
                    )
                    exported["nwb"] = nwb_relpath.as_posix()
                except ImportError as e:
                    self.logger.warning(f"NWB 导出跳过: {e}")
                except Exception as e:
                    self.logger.warning(f"NWB 导出失败: {e}")

        init_npz = self.storage.dirs.checkpoints / "initial_states" / f"{job.job_id}.npz"
        if init_npz.exists():
            if h5py is not None:
                with np.load(init_npz) as payload:
                    datasets = {k: payload[k] for k in payload.files}
                self.storage.write_hdf5(
                    "data/states.h5",
                    datasets,
                    kind="states",
                    metadata={"source_job_id": job.job_id, "source_kind": "initial_state"},
                )
                exported["states"] = "data/states.h5"
            else:
                target = self.storage.dirs.root / "data" / "states.npz"
                target.write_bytes(init_npz.read_bytes())
                self.storage.register_file(
                    target,
                    kind="states",
                    format="npz",
                    metadata={"source_job_id": job.job_id, "source_kind": "initial_state"},
                )
                exported["states"] = "data/states.npz"

        return exported

    def _write_results_summary(
        self,
        result: ExperimentResult,
        *,
        best_job: Optional[SimulationJob],
        exported: Dict[str, Optional[str]],
    ) -> None:
        detailed_results_path = (
            "results/detailed_results.h5" if h5py is not None else "results/detailed_results.json"
        )
        meta = json.loads((self.storage.dirs.root / "meta.json").read_text(encoding="utf-8"))
        training_log_hdf5: Optional[str] = "results/training_log.h5"
        if not (self.storage.dirs.root / training_log_hdf5).exists():
            training_log_hdf5 = None
        summary = {
            "schema_version": "results-summary-v1",
            "experiment_id": result.experiment_id,
            "name": self.config.name,
            "description": self.config.description,
            "meta": meta,
            "best": {
                "job_id": best_job.job_id if best_job else None,
                "objective_value": result.best_objective_value,
                "parameters": result.best_parameters,
            },
            "execution_summary": result.execution_summary,
            "paths": {
                "config": "config/config.json",
                "network_structure": "data/network_structure.json",
                "spikes": exported.get("spikes"),
                "states": exported.get("states"),
                "nwb": exported.get("nwb"),
                "pipeline_log": "results/logs/pipeline.log",
                "training_log": "results/training_log.csv",
                "training_log_hdf5": training_log_hdf5,
                "experiment_result": "results/experiment_result.json",
                "detailed_results": detailed_results_path,
                "jobs": "results/jobs",
            },
        }
        self.storage.write_json("results/results_summary.json", summary, kind="results_summary")

    def _save_job_initial_state(self, job: SimulationJob) -> None:
        num_neurons = int(job.network_config.get("num_neurons", 1000))
        rng = np.random.default_rng(job.random_seed)

        v_scalar = job.network_config.get("initial_membrane_potential", job.network_config.get("membrane_potential"))
        if v_scalar is not None:
            v_mean = float(v_scalar)
            v_std = 0.0
            v_source = "network_config.initial_membrane_potential|membrane_potential"
        else:
            v_mean = float(job.network_config.get("membrane_potential_mean", -70.0))
            v_std = float(job.network_config.get("membrane_potential_std", 5.0))
            v_source = "network_config.membrane_potential_mean/membrane_potential_std"

        th_scalar = job.network_config.get("threshold")
        if th_scalar is not None:
            th_mean = float(th_scalar)
            th_std = 0.0
            th_source = "network_config.threshold"
        else:
            th_mean = float(job.network_config.get("threshold_mean", -50.0))
            th_std = float(job.network_config.get("threshold_std", 0.0))
            th_source = "network_config.threshold_mean/threshold_std"

        if v_std > 0:
            membrane_potential = rng.normal(v_mean, v_std, size=num_neurons).astype(np.float32)
        else:
            membrane_potential = np.full((num_neurons,), v_mean, dtype=np.float32)

        if th_std > 0:
            threshold = rng.normal(th_mean, th_std, size=num_neurons).astype(np.float32)
        else:
            threshold = np.full((num_neurons,), th_mean, dtype=np.float32)

        w_mean = job.network_config.get("weight_mean", job.network_config.get("synaptic_weight"))
        w_std = job.network_config.get("weight_std", job.network_config.get("synaptic_weight_std"))
        if w_mean is not None:
            sample_n = int(min(1000, max(1, num_neurons)))
            if w_std is not None and float(w_std) > 0:
                synaptic_weight_samples = rng.normal(float(w_mean), float(w_std), size=sample_n).astype(np.float32)
                w_source = "network_config.weight_mean/weight_std"
            else:
                synaptic_weight_samples = np.full((sample_n,), float(w_mean), dtype=np.float32)
                w_source = "network_config.weight_mean|synaptic_weight"
        else:
            synaptic_weight_samples = np.array([], dtype=np.float32)
            w_source = "unknown"

        npz_rel = Path("data/checkpoints/initial_states") / f"{job.job_id}.npz"
        self.storage.write_npz(
            npz_rel,
            {
                "membrane_potential": membrane_potential,
                "threshold": threshold,
                "synaptic_weight_samples": synaptic_weight_samples,
            },
            kind="initial_state",
            metadata={"job_id": job.job_id},
        )

        def _summary(arr: np.ndarray) -> Dict[str, float]:
            if arr.size == 0:
                return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        summary = {
            "schema_version": "initial-state-v1",
            "job_id": job.job_id,
            "experiment_id": job.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": job.random_seed,
            "num_neurons": num_neurons,
            "membrane_potential": {**_summary(membrane_potential), "source": v_source},
            "threshold": {**_summary(threshold), "source": th_source},
            "synaptic_weight_samples": {**_summary(synaptic_weight_samples), "source": w_source},
            "arrays_npz": str(npz_rel).replace("\\", "/"),
            "array_keys": ["membrane_potential", "threshold", "synaptic_weight_samples"],
            "dtypes": {
                "membrane_potential": str(membrane_potential.dtype),
                "threshold": str(threshold.dtype),
                "synaptic_weight_samples": str(synaptic_weight_samples.dtype),
            },
        }

        self.storage.write_json(
            Path("data/checkpoints/initial_states") / f"{job.job_id}.json",
            summary,
            kind="initial_state_summary",
            metadata={"job_id": job.job_id},
        )
    
    def generate_jobs(self) -> List[SimulationJob]:
        """生成仿真作业"""
        jobs = []
        
        if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            samples_per_dim = self.config.optimization_config.get('samples_per_dimension', 10)
            parameter_iterator = self.parameter_generator.generate_grid_search_parameters(samples_per_dim)
        elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            num_samples = self.config.optimization_config.get('num_samples', 100)
            parameter_iterator = self.parameter_generator.generate_random_search_parameters(num_samples)
        elif self.config.optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
            num_samples = self.config.optimization_config.get('num_samples', 100)
            parameter_iterator = self.parameter_generator.generate_bayesian_optimization_parameters(num_samples)
        else:
            raise ValueError(f"不支持的优化方法: {self.config.optimization_method}")
        
        for i, parameters in enumerate(parameter_iterator):
            job_id = f"{self.config.experiment_id}_job_{i:06d}"
            job_seed = None
            if self.config.random_seed is not None:
                job_seed = int((self.config.random_seed + i) % (2**32))
            
            # 应用参数到网络配置
            network_config = self._apply_parameters_to_config(
                self.config.base_network_config.copy(), parameters
            )
            
            # 应用参数到仿真参数
            simulation_params = self._apply_parameters_to_config(
                self.config.base_simulation_params.copy(), parameters
            )
            
            job = SimulationJob(
                job_id=job_id,
                experiment_id=self.config.experiment_id,
                parameters=parameters,
                network_config=network_config,
                simulation_params=simulation_params,
                random_seed=job_seed,
            )
            
            jobs.append(job)
            self.jobs[job_id] = job
        
        self.job_queue = jobs.copy()
        self.logger.info(f"生成了 {len(jobs)} 个仿真作业")
        
        return jobs
    
    def _apply_parameters_to_config(self, config: Dict[str, Any], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """将参数应用到配置中"""
        # 递归替换配置中的参数占位符
        def replace_placeholders(obj, params):
            if isinstance(obj, dict):
                return {k: replace_placeholders(v, params) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item, params) for item in obj]
            elif isinstance(obj, str):
                # 替换形如 ${param_name} 的占位符
                for param_name, param_value in params.items():
                    placeholder = f"${{{param_name}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, str(param_value))
                return obj
            else:
                return obj
        
        # 直接参数替换
        for param_name, param_value in parameters.items():
            if param_name in config:
                config[param_name] = param_value
        
        # 占位符替换
        config = replace_placeholders(config, parameters)
        
        return config
    
    async def execute_job(self, job: SimulationJob) -> SimulationJob:
        """执行单个仿真作业"""
        job.status = TaskStatus.RUNNING
        job.start_time = datetime.now()
        
        try:
            if self.config.save_initial_state:
                try:
                    self._save_job_initial_state(job)
                except Exception as e:
                    self.logger.warning(f"初始状态快照保存失败 ({job.job_id}): {e}")

            # 这里应该调用实际的仿真后端
            # 调用实际的仿真后端
            await asyncio.sleep(0.1)  # 仿真时间
            
            # 模拟仿真结果
            simulation_results = self._simulate_brain_simulation(job)
            
            # 计算目标函数值
            objective_value = self.objective_function.evaluate(simulation_results, job.parameters)
            
            job.results = {
                'simulation_results': simulation_results,
                'objective_value': objective_value,
                'parameters': job.parameters
            }
            
            job.status = TaskStatus.COMPLETED
            job.end_time = datetime.now()
            job.execution_time = (job.end_time - job.start_time).total_seconds()
            
            # 保存中间结果
            if self.config.save_intermediate_results:
                self._save_job_result(job)
            
            self.completed_jobs.append(job)
            self.logger.info(f"作业 {job.job_id} 完成，目标值: {objective_value:.4f}")
            
        except Exception as e:
            job.status = TaskStatus.FAILED
            job.error_message = str(e)
            job.end_time = datetime.now()
            job.execution_time = (job.end_time - job.start_time).total_seconds()
            
            self.failed_jobs.append(job)
            self.logger.error(f"作业 {job.job_id} 失败: {str(e)}")
            
            # 重试逻辑
            if job.retry_count < self.config.max_retries:
                job.retry_count += 1
                job.status = TaskStatus.RETRYING
                self.job_queue.append(job)  # 重新加入队列
                self.logger.info(f"作业 {job.job_id} 将重试 (第 {job.retry_count} 次)")
        
        return job
    
    def _simulate_brain_simulation(self, job: SimulationJob) -> Dict[str, Any]:
        """模拟大脑仿真（用于测试）"""
        # 这里应该调用实际的大脑仿真系统
        # 生成模拟数据
        
        num_neurons = job.network_config.get('num_neurons', 1000)
        simulation_time = job.simulation_params.get('simulation_time', 1000.0)
        
        # 生成模拟尖峰数据
        spike_rate = job.parameters.get('spike_rate', 10.0)  # Hz
        expected_spikes = int(num_neurons * spike_rate * simulation_time / 1000.0)

        rng = np.random.default_rng(job.random_seed)
        spike_times = np.sort(rng.uniform(0, simulation_time, expected_spikes))
        spike_neurons = rng.integers(0, num_neurons, expected_spikes)
        
        results = {
            'spike_recorder': {
                'times': spike_times.tolist(),
                'senders': spike_neurons.tolist()
            },
            'simulation_time': simulation_time,
            'num_neurons': num_neurons
        }
        
        return results
    
    def _save_job_result(self, job: SimulationJob):
        """保存作业结果"""
        job_data = {
            'job_id': job.job_id,
            'experiment_id': job.experiment_id,
            'parameters': job.parameters,
            'network_config': job.network_config,
            'simulation_params': job.simulation_params,
            'results': job.results,
            'status': job.status.value,
            'execution_time': job.execution_time,
            'start_time': job.start_time.isoformat() if job.start_time else None,
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'error_message': job.error_message,
            'retry_count': job.retry_count,
            'random_seed': job.random_seed,
        }

        self.storage.write_json(
            Path("results/jobs") / f"{job.job_id}.json",
            job_data,
            kind="job_result",
            metadata={"job_id": job.job_id},
        )
    
    async def run_experiment(self) -> ExperimentResult:
        """运行完整实验"""
        self.logger.info(f"开始实验: {self.config.name}")
        self.is_running = True
        self.start_time = datetime.now()
        self.storage.mark_status("running", extra={"started_at": datetime.now(timezone.utc).isoformat()})
        self._persist_inputs(copy_artifacts=True)
        
        try:
            # 生成作业
            jobs = self.generate_jobs()
            self._save_experiment_seeds(jobs)
            if self.config.save_job_configs:
                for job in jobs:
                    self._save_job_config(job)
            
            # 并行执行作业
            semaphore = asyncio.Semaphore(self.config.max_parallel_jobs)
            
            async def execute_with_semaphore(job):
                async with semaphore:
                    return await self.execute_job(job)
            
            # 创建任务
            tasks = [execute_with_semaphore(job) for job in jobs]
            
            # 等待所有任务完成
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            self.end_time = datetime.now()
            total_execution_time = (self.end_time - self.start_time).total_seconds()
            
            # 找到最佳结果
            successful_jobs = [job for job in self.completed_jobs 
                             if job.status == TaskStatus.COMPLETED and 'objective_value' in job.results]
            
            if successful_jobs:
                best_job = max(successful_jobs, key=lambda j: j.results['objective_value'])
                best_parameters = best_job.parameters
                best_objective_value = best_job.results['objective_value']
            else:
                best_job = None
                best_parameters = {}
                best_objective_value = -np.inf
            
            # 创建实验结果
            experiment_result = ExperimentResult(
                experiment_id=self.config.experiment_id,
                total_jobs=len(jobs),
                completed_jobs=len(self.completed_jobs),
                failed_jobs=len(self.failed_jobs),
                best_parameters=best_parameters,
                best_objective_value=best_objective_value,
                all_results=[job.results for job in successful_jobs],
                execution_summary={
                    'total_execution_time': total_execution_time,
                    'average_job_time': np.mean([job.execution_time for job in successful_jobs]) if successful_jobs else 0.0,
                    'success_rate': len(self.completed_jobs) / len(jobs) if jobs else 0.0,
                    'parallel_efficiency': self._calculate_parallel_efficiency()
                },
                start_time=self.start_time,
                end_time=self.end_time,
                total_execution_time=total_execution_time
            )
            
            # 保存实验结果
            self._save_experiment_result(experiment_result)

            self._save_training_log(jobs)
            exported = {"spikes": None, "states": None, "nwb": None}
            if best_job is not None:
                exported = self._export_best_run_artifacts(best_job)

            self.storage.finalize(
                "completed",
                summary={
                    "total_jobs": experiment_result.total_jobs,
                    "completed_jobs": experiment_result.completed_jobs,
                    "failed_jobs": experiment_result.failed_jobs,
                    "best_objective_value": experiment_result.best_objective_value,
                    "total_execution_time": experiment_result.total_execution_time,
                },
            )
            self._write_results_summary(experiment_result, best_job=best_job, exported=exported)
            
            self.logger.info(f"实验完成: {len(self.completed_jobs)}/{len(jobs)} 作业成功")
            self.logger.info(f"最佳目标值: {best_objective_value:.4f}")
            
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"实验执行失败: {str(e)}")
            self.storage.finalize("failed", summary={"error": str(e)})
            try:
                meta = json.loads((self.storage.dirs.root / "meta.json").read_text(encoding="utf-8"))
                self.storage.write_json(
                    "results/results_summary.json",
                    {
                        "schema_version": "results-summary-v1",
                        "experiment_id": self.config.experiment_id,
                        "name": self.config.name,
                        "description": self.config.description,
                        "meta": meta,
                        "error": str(e),
                        "paths": {"jobs": "results/jobs"},
                    },
                    kind="results_summary",
                )
            except Exception:
                pass
            raise
        finally:
            self.is_running = False
            self._finalize_file_logging()
    
    def _calculate_parallel_efficiency(self) -> float:
        """计算并行效率"""
        if not self.completed_jobs:
            return 0.0
        
        total_sequential_time = sum(job.execution_time for job in self.completed_jobs)
        actual_parallel_time = (self.end_time - self.start_time).total_seconds()
        
        theoretical_parallel_time = total_sequential_time / self.config.max_parallel_jobs
        efficiency = theoretical_parallel_time / max(actual_parallel_time, 1e-10)
        
        return min(efficiency, 1.0)
    
    def _save_experiment_result(self, result: ExperimentResult):
        """保存实验结果"""
        result_data = {
            'experiment_id': result.experiment_id,
            'total_jobs': result.total_jobs,
            'completed_jobs': result.completed_jobs,
            'failed_jobs': result.failed_jobs,
            'best_parameters': result.best_parameters,
            'best_objective_value': result.best_objective_value,
            'execution_summary': result.execution_summary,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'total_execution_time': result.total_execution_time
        }

        self.storage.write_json("results/experiment_result.json", result_data, kind="experiment_result")
        
        if h5py is not None:
            # 保存详细结果到HDF5
            h5_file = self.results_directory / "detailed_results.h5"
            with h5py.File(h5_file, 'w') as f:
                # 保存参数和目标值
                if result.all_results:
                    parameters_group = f.create_group('parameters')
                    objectives_group = f.create_group('objectives')

                    for i, job_result in enumerate(result.all_results):
                        job_group = parameters_group.create_group(f'job_{i}')
                        for param_name, param_value in job_result.get('parameters', {}).items():
                            job_group.attrs[param_name] = param_value

                        objectives_group.create_dataset(
                            f'job_{i}', data=job_result.get('objective_value', 0.0)
                        )

            self.storage.register_file(h5_file, kind="experiment_results", format="hdf5")
        else:
            detailed_results = {
                "schema_version": "detailed-results-v1",
                "experiment_id": result.experiment_id,
                "jobs": [
                    {
                        "parameters": job_result.get("parameters", {}),
                        "objective_value": job_result.get("objective_value", 0.0),
                    }
                    for job_result in (result.all_results or [])
                ],
            }
            self.storage.write_json(
                "results/detailed_results.json", detailed_results, kind="experiment_results"
            )

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'gpu_usage': []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        try:
            import psutil
            
            while self.monitoring:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics['cpu_usage'].append({
                    'timestamp': time.time(),
                    'value': cpu_percent
                })
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'timestamp': time.time(),
                    'value': memory.percent
                })
                
                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append({
                        'timestamp': time.time(),
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })
                
                time.sleep(interval)
                
        except ImportError:
            logger.warning("psutil未安装，无法进行性能监控")
        except Exception as e:
            logger.error(f"性能监控错误: {str(e)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for metric_name, metric_data in self.metrics.items():
            if metric_data:
                if metric_name in ['cpu_usage', 'memory_usage']:
                    values = [item['value'] for item in metric_data]
                    summary[metric_name] = {
                        'mean': np.mean(values),
                        'max': np.max(values),
                        'min': np.min(values),
                        'std': np.std(values)
                    }
        
        return summary

def create_experiment_config(name: str, parameter_spaces: List[ParameterSpace],
                           optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
                           num_samples: int = 100) -> ExperimentConfig:
    """创建实验配置的便捷函数"""
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        description=f"自动生成的实验配置: {name}",
        parameter_spaces=parameter_spaces,
        optimization_method=optimization_method,
        optimization_config={'num_samples': num_samples},
        base_network_config={
            'num_neurons': 1000,
            'connection_probability': 0.1
        },
        base_simulation_params={
            'simulation_time': 1000.0,
            'dt': 0.1
        }
    )

