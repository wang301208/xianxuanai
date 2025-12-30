# -*- coding: utf-8 -*-
"""
自动化管线系统
Automation Pipeline System

实现完整的自动化管线：
1. 配置生成
2. 仿真部署
3. 数据收集
4. 统计分析
5. 指标可视化
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import yaml
import os
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd

class PipelineStage(Enum):
    """管线阶段"""
    CONFIG_GENERATION = "config_generation"
    SIMULATION_SETUP = "simulation_setup"
    SIMULATION_EXECUTION = "simulation_execution"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"
    REPORTING = "reporting"

class ExecutionMode(Enum):
    """执行模式"""
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    CLOUD = "cloud"
    HPC = "hpc"

@dataclass
class PipelineConfig:
    """管线配置"""
    name: str
    description: str
    stages: List[PipelineStage]
    execution_mode: ExecutionMode
    parameters: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "pipeline_output"
    max_parallel_jobs: int = 4
    timeout: int = 3600  # 秒

@dataclass
class StageResult:
    """阶段结果"""
    stage: PipelineStage
    success: bool
    execution_time: float
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    artifacts: List[str] = field(default_factory=list)

class AutomationPipeline:
    """自动化管线"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger("AutomationPipeline")
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 阶段处理器
        self.stage_processors = {
            PipelineStage.CONFIG_GENERATION: self._process_config_generation,
            PipelineStage.SIMULATION_SETUP: self._process_simulation_setup,
            PipelineStage.SIMULATION_EXECUTION: self._process_simulation_execution,
            PipelineStage.DATA_COLLECTION: self._process_data_collection,
            PipelineStage.ANALYSIS: self._process_analysis,
            PipelineStage.VISUALIZATION: self._process_visualization,
            PipelineStage.VALIDATION: self._process_validation,
            PipelineStage.REPORTING: self._process_reporting
        }
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        # 资源管理
        self.resource_manager = ResourceManager(config.resources)
        
        # 配置生成器
        self.config_generator = ConfigurationGenerator()
        
        # 数据收集器
        self.data_collector = DataCollector(self.output_dir)
        
        # 分析引擎
        self.analysis_engine = AnalysisEngine()
        
        # 可视化引擎
        self.visualization_engine = VisualizationEngine(self.output_dir)
        
        # 验证器
        from ..validation.experimental_validation import create_experimental_validator
        self.validator = create_experimental_validator()
    
    def execute_pipeline(self) -> Dict[str, Any]:
        """执行管线"""
        
        start_time = time.time()
        
        self.logger.info(f"Starting pipeline execution: {self.config.name}")
        
        # 初始化执行上下文
        execution_context = {
            'pipeline_id': f"pipeline_{int(start_time)}",
            'start_time': start_time,
            'config': self.config,
            'stage_results': {},
            'global_data': {}
        }
        
        try:
            # 按顺序执行各阶段
            for stage in self.config.stages:
                stage_result = self._execute_stage(stage, execution_context)
                execution_context['stage_results'][stage] = stage_result
                
                if not stage_result.success:
                    self.logger.error(f"Stage {stage.value} failed: {stage_result.error_message}")
                    break
                
                self.logger.info(f"Stage {stage.value} completed in {stage_result.execution_time:.2f}s")
            
            # 计算总执行时间
            total_time = time.time() - start_time
            
            # 生成执行报告
            execution_report = self._generate_execution_report(execution_context, total_time)
            
            # 保存执行历史
            self.execution_history.append(execution_report)
            
            self.logger.info(f"Pipeline execution completed in {total_time:.2f}s")
            
            return execution_report
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> StageResult:
        """执行单个阶段"""
        
        stage_start_time = time.time()
        
        try:
            self.logger.info(f"Executing stage: {stage.value}")
            
            # 获取阶段处理器
            processor = self.stage_processors.get(stage)
            if not processor:
                raise ValueError(f"No processor found for stage: {stage.value}")
            
            # 执行阶段
            output_data = processor(context)
            
            execution_time = time.time() - stage_start_time
            
            return StageResult(
                stage=stage,
                success=True,
                execution_time=execution_time,
                output_data=output_data
            )
        
        except Exception as e:
            execution_time = time.time() - stage_start_time
            
            return StageResult(
                stage=stage,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _process_config_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置生成阶段"""
        
        # 生成仿真配置
        base_config = self.config.parameters.get('base_config', {})
        variations = self.config.parameters.get('config_variations', [])
        
        generated_configs = self.config_generator.generate_configurations(
            base_config, variations
        )
        
        # 保存配置文件
        config_dir = self.output_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        config_files = []
        for i, config in enumerate(generated_configs):
            config_file = config_dir / f"config_{i:03d}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            config_files.append(str(config_file))
        
        return {
            'generated_configs': generated_configs,
            'config_files': config_files,
            'num_configs': len(generated_configs)
        }
    
    def _process_simulation_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理仿真设置阶段"""
        
        # 获取配置文件
        config_generation_result = context['stage_results'].get(PipelineStage.CONFIG_GENERATION)
        if not config_generation_result:
            raise ValueError("Config generation stage not completed")
        
        config_files = config_generation_result.output_data['config_files']
        
        # 设置仿真环境
        simulation_jobs = []
        
        for config_file in config_files:
            job_id = f"sim_{Path(config_file).stem}"
            
            job_setup = {
                'job_id': job_id,
                'config_file': config_file,
                'output_dir': str(self.output_dir / "simulations" / job_id),
                'status': 'prepared'
            }
            
            # 创建输出目录
            Path(job_setup['output_dir']).mkdir(parents=True, exist_ok=True)
            
            simulation_jobs.append(job_setup)
        
        return {
            'simulation_jobs': simulation_jobs,
            'num_jobs': len(simulation_jobs)
        }
    
    def _process_simulation_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理仿真执行阶段"""
        
        # 获取仿真作业
        setup_result = context['stage_results'].get(PipelineStage.SIMULATION_SETUP)
        if not setup_result:
            raise ValueError("Simulation setup stage not completed")
        
        simulation_jobs = setup_result.output_data['simulation_jobs']
        
        # 根据执行模式选择执行策略
        if self.config.execution_mode == ExecutionMode.LOCAL:
            results = self._execute_simulations_local(simulation_jobs)
        elif self.config.execution_mode == ExecutionMode.DISTRIBUTED:
            results = self._execute_simulations_distributed(simulation_jobs)
        else:
            raise ValueError(f"Execution mode {self.config.execution_mode} not implemented")
        
        return {
            'simulation_results': results,
            'completed_jobs': len([r for r in results if r['success']]),
            'failed_jobs': len([r for r in results if not r['success']])
        }
    
    def _execute_simulations_local(self, simulation_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """本地执行仿真"""
        
        results = []
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
            futures = []
            
            for job in simulation_jobs:
                future = executor.submit(self._run_single_simulation, job)
                futures.append((job, future))
            
            # 收集结果
            for job, future in futures:
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'job_id': job['job_id'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _execute_simulations_distributed(self, simulation_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分布式执行仿真"""
        
        # 简化的分布式执行（实际应该使用Dask、Ray等）
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_parallel_jobs) as executor:
            futures = []
            
            for job in simulation_jobs:
                future = executor.submit(self._run_single_simulation, job)
                futures.append((job, future))
            
            for job, future in futures:
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'job_id': job['job_id'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _run_single_simulation(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个仿真"""
        
        start_time = time.time()
        
        try:
            # 加载配置
            with open(job['config_file'], 'r') as f:
                config = yaml.safe_load(f)
            
            # 创建仿真实例（简化）
            from ..brain_simulation import BrainSimulation
            simulation = BrainSimulation(config)
            
            # 运行仿真
            duration = config.get('simulation_duration', 1000.0)  # ms
            dt = config.get('time_step', 0.1)  # ms
            
            results = simulation.run(duration, dt)
            
            # 保存结果
            output_file = Path(job['output_dir']) / "simulation_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            execution_time = time.time() - start_time
            
            return {
                'job_id': job['job_id'],
                'success': True,
                'execution_time': execution_time,
                'output_file': str(output_file),
                'results_summary': {
                    'num_neurons': len(results.get('neuron_states', {})),
                    'simulation_duration': duration,
                    'final_time': results.get('current_time', 0)
                }
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'job_id': job['job_id'],
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    def _process_data_collection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据收集阶段"""
        
        # 获取仿真结果
        execution_result = context['stage_results'].get(PipelineStage.SIMULATION_EXECUTION)
        if not execution_result:
            raise ValueError("Simulation execution stage not completed")
        
        simulation_results = execution_result.output_data['simulation_results']
        
        # 收集所有仿真数据
        collected_data = self.data_collector.collect_simulation_data(simulation_results)
        
        # 保存汇总数据
        summary_file = self.output_dir / "collected_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(collected_data['summary'], f, indent=2, default=str)
        
        return collected_data
    
    def _process_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析阶段"""
        
        # 获取收集的数据
        collection_result = context['stage_results'].get(PipelineStage.DATA_COLLECTION)
        if not collection_result:
            raise ValueError("Data collection stage not completed")
        
        collected_data = collection_result.output_data
        
        # 执行统计分析
        analysis_results = self.analysis_engine.analyze_data(collected_data)
        
        # 保存分析结果
        analysis_file = self.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        return analysis_results
    
    def _process_visualization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理可视化阶段"""
        
        # 获取分析结果
        analysis_result = context['stage_results'].get(PipelineStage.ANALYSIS)
        if not analysis_result:
            raise ValueError("Analysis stage not completed")
        
        analysis_data = analysis_result.output_data
        
        # 生成可视化
        visualization_results = self.visualization_engine.create_visualizations(analysis_data)
        
        return visualization_results
    
    def _process_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理验证阶段"""
        
        # 获取分析数据
        analysis_result = context['stage_results'].get(PipelineStage.ANALYSIS)
        if not analysis_result:
            raise ValueError("Analysis stage not completed")
        
        analysis_data = analysis_result.output_data
        
        # 执行验证
        validation_metrics = [
            ValidationMetric.FIRING_RATE,
            ValidationMetric.CONNECTIVITY,
            ValidationMetric.OSCILLATIONS
        ]
        
        validation_results = self.validator.validate_simulation_data(
            analysis_data, validation_metrics
        )
        
        # 生成验证报告
        validation_report = self.validator.generate_validation_report(validation_results)
        
        # 保存验证结果
        validation_file = self.output_dir / "validation_results.json"
        self.validator.save_validation_results(validation_results, str(validation_file))
        
        return {
            'validation_results': validation_results,
            'validation_report': validation_report
        }
    
    def _process_reporting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理报告生成阶段"""
        
        # 收集所有阶段的结果
        all_results = {}
        for stage, result in context['stage_results'].items():
            all_results[stage.value] = result.output_data
        
        # 生成综合报告
        report = self._generate_comprehensive_report(all_results, context)
        
        # 保存报告
        report_file = self.output_dir / "pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成HTML报告
        html_report = self._generate_html_report(report)
        html_file = self.output_dir / "pipeline_report.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        return {
            'report': report,
            'report_file': str(report_file),
            'html_report_file': str(html_file)
        }
    
    def _generate_execution_report(self, context: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """生成执行报告"""
        
        successful_stages = sum(1 for result in context['stage_results'].values() if result.success)
        total_stages = len(context['stage_results'])
        
        return {
            'pipeline_id': context['pipeline_id'],
            'pipeline_name': self.config.name,
            'success': successful_stages == total_stages,
            'execution_time': total_time,
            'stages_completed': successful_stages,
            'total_stages': total_stages,
            'stage_results': {stage.value: {
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            } for stage, result in context['stage_results'].items()},
            'timestamp': context['start_time']
        }
    
    def _generate_comprehensive_report(self, all_results: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合报告"""
        
        report = {
            'pipeline_info': {
                'name': self.config.name,
                'description': self.config.description,
                'execution_mode': self.config.execution_mode.value,
                'pipeline_id': context['pipeline_id']
            },
            'execution_summary': {
                'total_time': time.time() - context['start_time'],
                'stages_executed': list(all_results.keys()),
                'success_rate': len([r for r in context['stage_results'].values() if r.success]) / len(context['stage_results'])
            },
            'results_summary': {}
        }
        
        # 添加各阶段结果摘要
        if 'config_generation' in all_results:
            report['results_summary']['configurations'] = {
                'num_generated': all_results['config_generation']['num_configs']
            }
        
        if 'simulation_execution' in all_results:
            sim_results = all_results['simulation_execution']
            report['results_summary']['simulations'] = {
                'completed_jobs': sim_results['completed_jobs'],
                'failed_jobs': sim_results['failed_jobs'],
                'success_rate': sim_results['completed_jobs'] / (sim_results['completed_jobs'] + sim_results['failed_jobs'])
            }
        
        if 'validation' in all_results:
            val_results = all_results['validation']['validation_report']
            report['results_summary']['validation'] = {
                'overall_score': val_results['summary']['overall_score'],
                'passed_metrics': val_results['summary']['passed_metrics'],
                'total_metrics': val_results['summary']['total_metrics']
            }
        
        return report
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML报告"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Simulation Pipeline Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .success { color: green; }
                .failure { color: red; }
                .metric { margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Brain Simulation Pipeline Report</h1>
                <p><strong>Pipeline:</strong> {pipeline_name}</p>
                <p><strong>Execution Mode:</strong> {execution_mode}</p>
                <p><strong>Total Time:</strong> {total_time:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <p><strong>Success Rate:</strong> {success_rate:.1%}</p>
                <p><strong>Stages Executed:</strong> {stages_executed}</p>
            </div>
            
            <div class="section">
                <h2>Results Summary</h2>
                {results_content}
            </div>
        </body>
        </html>
        """
        
        # 填充模板
        pipeline_info = report['pipeline_info']
        execution_summary = report['execution_summary']
        
        results_content = ""
        for key, value in report['results_summary'].items():
            results_content += f"<h3>{key.title()}</h3>"
            if isinstance(value, dict):
                for k, v in value.items():
                    results_content += f"<p><strong>{k.replace('_', ' ').title()}:</strong> {v}</p>"
            else:
                results_content += f"<p>{value}</p>"
        
        return html_template.format(
            pipeline_name=pipeline_info['name'],
            execution_mode=pipeline_info['execution_mode'],
            total_time=execution_summary['total_time'],
            success_rate=execution_summary['success_rate'],
            stages_executed=', '.join(execution_summary['stages_executed']),
            results_content=results_content
        )

class ResourceManager:
    """资源管理器"""
    
    def __init__(self, resource_config: Dict[str, Any]):
        self.config = resource_config
        self.logger = logging.getLogger("ResourceManager")
        
        # 资源监控
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = 0.0
        
    def allocate_resources(self, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分配资源"""
        
        # 简化的资源分配
        allocated = {
            'cpu_cores': min(job_requirements.get('cpu_cores', 1), 
                           self.config.get('max_cpu_cores', 4)),
            'memory_gb': min(job_requirements.get('memory_gb', 2), 
                           self.config.get('max_memory_gb', 8)),
            'gpu_devices': min(job_requirements.get('gpu_devices', 0), 
                             self.config.get('max_gpu_devices', 0))
        }
        
        return allocated
    
    def monitor_resources(self) -> Dict[str, float]:
        """监控资源使用"""
        
        try:
            import psutil
            
            self.cpu_usage = psutil.cpu_percent()
            self.memory_usage = psutil.virtual_memory().percent
            
            # GPU监控（如果可用）
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage = gpus[0].load * 100
            except ImportError:
                self.gpu_usage = 0.0
        
        except ImportError:
            # 如果psutil不可用，使用模拟数据
            self.cpu_usage = np.random.uniform(20, 80)
            self.memory_usage = np.random.uniform(30, 70)
            self.gpu_usage = np.random.uniform(0, 50)
        
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage
        }

class ConfigurationGenerator:
    """配置生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger("ConfigurationGenerator")
    
    def generate_configurations(self, base_config: Dict[str, Any], 
                              variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成配置变体"""
        
        configurations = []
        
        if not variations:
            # 如果没有变体，返回基础配置
            configurations.append(base_config.copy())
        else:
            # 生成参数组合
            import itertools
            
            # 提取变化的参数
            param_names = []
            param_values = []
            
            for variation in variations:
                param_names.append(variation['parameter'])
                param_values.append(variation['values'])
            
            # 生成所有组合
            for combination in itertools.product(*param_values):
                config = base_config.copy()
                
                for param_name, value in zip(param_names, combination):
                    # 支持嵌套参数
                    self._set_nested_parameter(config, param_name, value)
                
                configurations.append(config)
        
        self.logger.info(f"Generated {len(configurations)} configurations")
        
        return configurations
    
    def _set_nested_parameter(self, config: Dict[str, Any], param_path: str, value: Any):
        """设置嵌套参数"""
        
        keys = param_path.split('.')
        current = config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置值
        current[keys[-1]] = value

class DataCollector:
    """数据收集器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger("DataCollector")
    
    def collect_simulation_data(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """收集仿真数据"""
        
        collected_data = {
            'firing_rates': [],
            'connectivity_matrices': [],
            'spike_times': {},
            'network_statistics': [],
            'summary': {}
        }
        
        successful_results = [r for r in simulation_results if r['success']]
        
        for result in successful_results:
            try:
                # 加载仿真结果
                with open(result['output_file'], 'r') as f:
                    sim_data = json.load(f)
                
                # 提取关键数据
                if 'neuron_states' in sim_data:
                    neuron_states = sim_data['neuron_states']
                    
                    # 发放率
                    firing_rates = [state.get('firing_rate', 0.0) 
                                  for state in neuron_states.values()]
                    collected_data['firing_rates'].extend(firing_rates)
                
                # 连接数据
                if 'connectivity' in sim_data:
                    collected_data['connectivity_matrices'].append(sim_data['connectivity'])
                
                # 尖峰时间
                if 'spike_data' in sim_data:
                    for neuron_id, spikes in sim_data['spike_data'].items():
                        if neuron_id not in collected_data['spike_times']:
                            collected_data['spike_times'][neuron_id] = []
                        collected_data['spike_times'][neuron_id].extend(spikes)
                
                # 网络统计
                if 'network_stats' in sim_data:
                    collected_data['network_statistics'].append(sim_data['network_stats'])
            
            except Exception as e:
                self.logger.warning(f"Failed to collect data from {result['output_file']}: {str(e)}")
        
        # 生成摘要
        collected_data['summary'] = {
            'num_simulations': len(successful_results),
            'total_neurons': len(collected_data['firing_rates']),
            'mean_firing_rate': np.mean(collected_data['firing_rates']) if collected_data['firing_rates'] else 0.0,
            'num_spike_trains': len(collected_data['spike_times'])
        }
        
        return collected_data

class AnalysisEngine:
    """分析引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger("AnalysisEngine")
    
    def analyze_data(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析数据"""
        
        analysis_results = {}
        
        # 发放率分析
        if collected_data['firing_rates']:
            analysis_results['firing_rate_analysis'] = self._analyze_firing_rates(
                collected_data['firing_rates']
            )
        
        # 连接性分析
        if collected_data['connectivity_matrices']:
            analysis_results['connectivity_analysis'] = self._analyze_connectivity(
                collected_data['connectivity_matrices']
            )
        
        # 尖峰时序分析
        if collected_data['spike_times']:
            analysis_results['spike_timing_analysis'] = self._analyze_spike_timing(
                collected_data['spike_times']
            )
        
        # 网络统计分析
        if collected_data['network_statistics']:
            analysis_results['network_analysis'] = self._analyze_network_statistics(
                collected_data['network_statistics']
            )
        
        return analysis_results
    
    def _analyze_firing_rates(self, firing_rates: List[float]) -> Dict[str, Any]:
        """分析发放率"""
        
        rates = np.array(firing_rates)
        
        return {
            'mean': np.mean(rates),
            'std': np.std(rates),
            'median': np.median(rates),
            'min': np.min(rates),
            'max': np.max(rates),
            'percentiles': {
                '25th': np.percentile(rates, 25),
                '75th': np.percentile(rates, 75),
                '95th': np.percentile(rates, 95)
            },
            'distribution_shape': {
                'skewness': float(stats.skew(rates)),
                'kurtosis': float(stats.kurtosis(rates))
            }
        }
    
    def _analyze_connectivity(self, connectivity_matrices: List[Any]) -> Dict[str, Any]:
        """分析连接性"""
        
        # 简化的连接性分析
        if not connectivity_matrices:
            return {}
        
        # 假设连接矩阵是字典格式
        total_connections = 0
        total_possible = 0
        
        for conn_data in connectivity_matrices:
            if isinstance(conn_data, dict):
                total_connections += len(conn_data)
                # 估算可能的连接数
                max_neuron_id = max(max(pre, post) for pre, post in conn_data.keys()) if conn_data else 0
                total_possible += (max_neuron_id + 1) ** 2
        
        connection_density = total_connections / total_possible if total_possible > 0 else 0.0
        
        return {
            'total_connections': total_connections,
            'connection_density': connection_density,
            'average_connections_per_matrix': total_connections / len(connectivity_matrices)
        }
    
    def _analyze_spike_timing(self, spike_times: Dict[str, List[float]]) -> Dict[str, Any]:
        """分析尖峰时序"""
        
        # 计算ISI统计
        all_isis = []
        
        for neuron_spikes in spike_times.values():
            if len(neuron_spikes) > 1:
                isis = np.diff(neuron_spikes)
                all_isis.extend(isis)
        
        if not all_isis:
            return {'error': 'No ISI data available'}
        
        isis = np.array(all_isis)
        
        return {
            'mean_isi': np.mean(isis),
            'std_isi': np.std(isis),
            'cv_isi': np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0.0,
            'median_isi': np.median(isis),
            'isi_distribution': {
                'min': np.min(isis),
                'max': np.max(isis),
                'percentiles': {
                    '25th': np.percentile(isis, 25),
                    '75th': np.percentile(isis, 75)
                }
            }
        }
    
    def _analyze_network_statistics(self, network_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析网络统计"""
        
        # 汇总网络统计
        aggregated_stats = {}
        
        for stats in network_stats:
            for key, value in stats.items():
                if key not in aggregated_stats:
                    aggregated_stats[key] = []
                
                if isinstance(value, (int, float)):
                    aggregated_stats[key].append(value)
        
        # 计算统计摘要
        summary = {}
        for key, values in aggregated_stats.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary

class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("VisualizationEngine")
    
    def create_visualizations(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建可视化"""
        
        visualization_files = []
        
        # 发放率分布图
        if 'firing_rate_analysis' in analysis_data:
            fig_file = self._plot_firing_rate_distribution(analysis_data['firing_rate_analysis'])
            visualization_files.append(fig_file)
        
        # 连接性图
        if 'connectivity_analysis' in analysis_data:
            fig_file = self._plot_connectivity_analysis(analysis_data['connectivity_analysis'])
            visualization_files.append(fig_file)
        
        # ISI分布图
        if 'spike_timing_analysis' in analysis_data:
            fig_file = self._plot_isi_distribution(analysis_data['spike_timing_analysis'])
            visualization_files.append(fig_file)
        
        # 网络统计图
        if 'network_analysis' in analysis_data:
            fig_file = self._plot_network_statistics(analysis_data['network_analysis'])
            visualization_files.append(fig_file)
        
        return {
            'visualization_files': visualization_files,
            'num_visualizations': len(visualization_files)
        }
    
    def _plot_firing_rate_distribution(self, firing_rate_data: Dict[str, Any]) -> str:
        """绘制发放率分布"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 统计摘要
        stats = ['mean', 'std', 'median', 'min', 'max']
        values = [firing_rate_data[stat] for stat in stats]
        
        ax1.bar(stats, values)
        ax1.set_title('Firing Rate Statistics')
        ax1.set_ylabel('Firing Rate (Hz)')
        
        # 百分位数
        percentiles = firing_rate_data['percentiles']
        ax2.bar(percentiles.keys(), percentiles.values())
        ax2.set_title('Firing Rate Percentiles')
        ax2.set_ylabel('Firing Rate (Hz)')
        
        plt.tight_layout()
        
        fig_file = self.viz_dir / "firing_rate_analysis.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_file)
    
    def _plot_connectivity_analysis(self, connectivity_data: Dict[str, Any]) -> str:
        """绘制连接性分析"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ['total_connections', 'connection_density', 'average_connections_per_matrix']
        values = [connectivity_data.get(metric, 0) for metric in metrics]
        
        ax.bar(metrics, values)
        ax.set_title('Connectivity Analysis')
        ax.set_ylabel('Value')
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_file = self.viz_dir / "connectivity_analysis.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_file)
    
    def _plot_isi_distribution(self, isi_data: Dict[str, Any]) -> str:
        """绘制ISI分布"""
        
        if 'error' in isi_data:
            # 创建空图表显示错误
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, isi_data['error'], ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ISI Analysis - Error')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ISI统计
            stats = ['mean_isi', 'std_isi', 'cv_isi', 'median_isi']
            values = [isi_data[stat] for stat in stats]
            
            ax1.bar(stats, values)
            ax1.set_title('ISI Statistics')
            ax1.set_ylabel('Time (ms)')
            
            # ISI分布百分位数
            percentiles = isi_data['isi_distribution']['percentiles']
            ax2.bar(percentiles.keys(), percentiles.values())
            ax2.set_title('ISI Percentiles')
            ax2.set_ylabel('Time (ms)')
        
        plt.tight_layout()
        
        fig_file = self.viz_dir / "isi_analysis.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_file)
    
    def _plot_network_statistics(self, network_data: Dict[str, Any]) -> str:
        """绘制网络统计"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = list(network_data.keys())[:4]  # 最多显示4个指标
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                stats = network_data[metric]
                stat_names = list(stats.keys())
                stat_values = list(stats.values())
                
                axes[i].bar(stat_names, stat_values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # 隐藏未使用的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        fig_file = self.viz_dir / "network_statistics.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(fig_file)

# 工厂函数
def create_automation_pipeline(config: PipelineConfig) -> AutomationPipeline:
    """创建自动化管线"""
    return AutomationPipeline(config)

# 预定义管线配置
def create_standard_pipeline_config() -> PipelineConfig:
    """创建标准管线配置"""
    
    return PipelineConfig(
        name="Standard Brain Simulation Pipeline",
        description="Complete pipeline for brain simulation validation",
        stages=[
            PipelineStage.CONFIG_GENERATION,
            PipelineStage.SIMULATION_SETUP,
            PipelineStage.SIMULATION_EXECUTION,
            PipelineStage.DATA_COLLECTION,
            PipelineStage.ANALYSIS,
            PipelineStage.VISUALIZATION,
            PipelineStage.VALIDATION,
            PipelineStage.REPORTING
        ],
        execution_mode=ExecutionMode.LOCAL,
        parameters={
            'base_config': {
                'num_neurons': 1000,
                'simulation_duration': 1000.0,
                'time_step': 0.1
            },
            'config_variations': [
                {
                    'parameter': 'num_neurons',
                    'values': [500, 1000, 2000]
                },
                {
                    'parameter': 'simulation_duration',
                    'values': [500.0, 1000.0, 2000.0]
                }
            ]
        },
        resources={
            'max_cpu_cores': 4,
            'max_memory_gb': 8,
            'max_gpu_devices': 0
        },
        max_parallel_jobs=2,
        timeout=1800
    )