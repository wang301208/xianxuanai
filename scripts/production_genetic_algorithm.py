"""
生产级遗传算法多指标优化系统

提供高性能、可扩展的遗传算法优化框架，支持多目标优化、
自适应参数调整和实时监控。
"""

from __future__ import annotations

import logging
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import hashlib

try:
    from evolution.generic_ga import GAConfig, GeneticAlgorithm
    from evolution import fitness_plugins
except ImportError:
    # 模拟导入，用于独立运行
    class GAConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class GeneticAlgorithm:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def run(self, generations: int):
            return [0.0, 0.0, 0.0], 1.0
    
    class fitness_plugins:
        _registry = {}
        
        @classmethod
        def register(cls, name: str):
            def decorator(func):
                cls._registry[name] = func
                return func
            return decorator
        
        @classmethod
        def load_from_config(cls, config_path: str):
            return []


@dataclass
class OptimizationMetrics:
    """优化过程指标"""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    convergence_rate: float
    execution_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """优化配置"""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    elitism_ratio: float = 0.1
    diversity_threshold: float = 0.01
    convergence_threshold: float = 1e-6
    max_stagnation: int = 10
    enable_adaptive_params: bool = True
    enable_parallel_evaluation: bool = True
    max_workers: int = 4
    checkpoint_interval: int = 10
    log_level: str = "INFO"


class ProductionFitnessRegistry:
    """生产级适应度函数注册表"""
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, name: str, description: str = "", 
                 input_validation: Optional[Callable] = None,
                 output_range: Optional[Tuple[float, float]] = None):
        """
        注册适应度函数
        
        Args:
            name: 函数名称
            description: 函数描述
            input_validation: 输入验证函数
            output_range: 输出值范围
        """
        def decorator(func: Callable) -> Callable:
            # 创建包装函数，添加验证和错误处理
            def wrapped_func(individual: List[float]) -> float:
                try:
                    # 输入验证
                    if input_validation:
                        if not input_validation(individual):
                            raise ValueError(f"输入验证失败: {individual}")
                    
                    # 执行原函数
                    result = func(individual)
                    
                    # 输出验证
                    if not isinstance(result, (int, float)):
                        raise ValueError(f"适应度函数必须返回数值，得到: {type(result)}")
                    
                    if output_range:
                        min_val, max_val = output_range
                        if not (min_val <= result <= max_val):
                            self.logger.warning(
                                f"适应度值 {result} 超出预期范围 [{min_val}, {max_val}]"
                            )
                    
                    return float(result)
                    
                except Exception as e:
                    self.logger.error(f"适应度函数 {name} 执行失败: {e}")
                    # 返回最差适应度值
                    return float('-inf')
            
            # 注册函数和元数据
            self._functions[name] = wrapped_func
            self._metadata[name] = {
                'description': description,
                'input_validation': input_validation is not None,
                'output_range': output_range,
                'original_function': func
            }
            
            self.logger.info(f"注册适应度函数: {name}")
            return wrapped_func
        
        return decorator
    
    def get_function(self, name: str) -> Callable:
        """获取适应度函数"""
        if name not in self._functions:
            raise KeyError(f"未找到适应度函数: {name}")
        return self._functions[name]
    
    def list_functions(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的函数"""
        return self._metadata.copy()
    
    def load_from_config(self, config_path: Union[str, Path]) -> List[Tuple[Callable, float]]:
        """从配置文件加载适应度函数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            functions = []
            for item in config.get('fitness_functions', []):
                name = item['name']
                weight = item.get('weight', 1.0)
                
                if name in self._functions:
                    functions.append((self._functions[name], weight))
                else:
                    self.logger.warning(f"配置中的函数 {name} 未注册")
            
            return functions
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return []


class ProductionGeneticAlgorithm:
    """生产级遗传算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 fitness_functions: List[Tuple[Callable, float]],
                 config: Optional[OptimizationConfig] = None):
        """
        初始化遗传算法
        
        Args:
            bounds: 变量边界
            fitness_functions: 适应度函数列表，每个元素为(函数, 权重)
            config: 优化配置
        """
        self.bounds = bounds
        self.fitness_functions = fitness_functions
        self.config = config or OptimizationConfig()
        self.logger = self._setup_logging()
        
        # 算法状态
        self.population: Optional[np.ndarray] = None
        self.fitness_values: Optional[np.ndarray] = None
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = float('-inf')
        self.generation = 0
        self.stagnation_count = 0
        
        # 历史记录
        self.metrics_history: List[OptimizationMetrics] = []
        self.convergence_history: List[float] = []
        
        # 并行执行
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # 检查点
        self.checkpoint_data: Dict[str, Any] = {}
        
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
    
    def initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.random.uniform(
            low=[bound[0] for bound in self.bounds],
            high=[bound[1] for bound in self.bounds],
            size=(self.config.population_size, len(self.bounds))
        )
        
        self.logger.info(f"初始化种群，大小: {self.config.population_size}")
        return population
    
    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        if self.config.enable_parallel_evaluation and len(population) > 10:
            return self._evaluate_fitness_parallel(population)
        else:
            return self._evaluate_fitness_sequential(population)
    
    def _evaluate_fitness_sequential(self, population: np.ndarray) -> np.ndarray:
        """顺序评估适应度"""
        fitness_values = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            fitness_values[i] = self._calculate_individual_fitness(individual)
        
        return fitness_values
    
    def _evaluate_fitness_parallel(self, population: np.ndarray) -> np.ndarray:
        """并行评估适应度"""
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 提交任务
        futures = {
            self.executor.submit(self._calculate_individual_fitness, individual): i
            for i, individual in enumerate(population)
        }
        
        # 收集结果
        fitness_values = np.zeros(len(population))
        for future in as_completed(futures):
            index = futures[future]
            try:
                fitness_values[index] = future.result()
            except Exception as e:
                self.logger.error(f"个体 {index} 适应度评估失败: {e}")
                fitness_values[index] = float('-inf')
        
        return fitness_values
    
    def _calculate_individual_fitness(self, individual: np.ndarray) -> float:
        """计算单个个体的适应度"""
        total_fitness = 0.0
        
        for fitness_func, weight in self.fitness_functions:
            try:
                fitness = fitness_func(individual.tolist())
                total_fitness += weight * fitness
            except Exception as e:
                self.logger.error(f"适应度函数执行失败: {e}")
                return float('-inf')
        
        return total_fitness
    
    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """选择操作"""
        # 锦标赛选择
        tournament_size = max(2, int(self.config.selection_pressure))
        selected = np.zeros_like(population)
        
        for i in range(len(population)):
            # 随机选择参赛者
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            tournament_fitness = fitness_values[tournament_indices]
            
            # 选择最优个体
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected[i] = population[winner_index]
        
        return selected
    
    def crossover(self, population: np.ndarray) -> np.ndarray:
        """交叉操作"""
        offspring = population.copy()
        
        for i in range(0, len(population) - 1, 2):
            if np.random.random() < self.config.crossover_rate:
                # 模拟二进制交叉
                parent1, parent2 = population[i], population[i + 1]
                
                # 随机选择交叉点
                crossover_point = np.random.randint(1, len(self.bounds))
                
                # 执行交叉
                offspring[i, crossover_point:] = parent2[crossover_point:]
                offspring[i + 1, crossover_point:] = parent1[crossover_point:]
        
        return offspring
    
    def mutation(self, population: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = population.copy()
        
        for i in range(len(population)):
            for j in range(len(self.bounds)):
                if np.random.random() < self.config.mutation_rate:
                    # 高斯变异
                    mutation_strength = 0.1 * (self.bounds[j][1] - self.bounds[j][0])
                    mutation = np.random.normal(0, mutation_strength)
                    
                    # 应用变异并确保在边界内
                    mutated[i, j] += mutation
                    mutated[i, j] = np.clip(
                        mutated[i, j], self.bounds[j][0], self.bounds[j][1]
                    )
        
        return mutated
    
    def elitism(self, old_population: np.ndarray, old_fitness: np.ndarray,
                new_population: np.ndarray, new_fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """精英保留策略"""
        elite_count = max(1, int(self.config.elitism_ratio * len(old_population)))
        
        # 找到最优个体
        elite_indices = np.argsort(old_fitness)[-elite_count:]
        
        # 替换新种群中最差的个体
        worst_indices = np.argsort(new_fitness)[:elite_count]
        
        new_population[worst_indices] = old_population[elite_indices]
        new_fitness[worst_indices] = old_fitness[elite_indices]
        
        return new_population, new_fitness
    
    def calculate_diversity(self, population: np.ndarray) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        # 计算个体间的平均距离
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def adaptive_parameter_adjustment(self) -> None:
        """自适应参数调整"""
        if not self.config.enable_adaptive_params:
            return
        
        # 根据停滞情况调整变异率
        if self.stagnation_count > 5:
            self.config.mutation_rate = min(0.5, self.config.mutation_rate * 1.1)
        elif self.stagnation_count == 0:
            self.config.mutation_rate = max(0.01, self.config.mutation_rate * 0.9)
        
        # 根据多样性调整选择压力
        if len(self.metrics_history) > 0:
            current_diversity = self.metrics_history[-1].diversity
            if current_diversity < self.config.diversity_threshold:
                self.config.selection_pressure = max(1.5, self.config.selection_pressure * 0.9)
            else:
                self.config.selection_pressure = min(3.0, self.config.selection_pressure * 1.05)
    
    def save_checkpoint(self, filepath: Optional[Path] = None) -> None:
        """保存检查点"""
        if filepath is None:
            filepath = Path(f"ga_checkpoint_{int(time.time())}.json")
        
        checkpoint_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual.tolist() if self.best_individual is not None else None,
            'config': {
                'population_size': self.config.population_size,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'selection_pressure': self.config.selection_pressure
            },
            'metrics_history': [
                {
                    'generation': m.generation,
                    'best_fitness': m.best_fitness,
                    'average_fitness': m.average_fitness,
                    'diversity': m.diversity,
                    'convergence_rate': m.convergence_rate,
                    'execution_time': m.execution_time,
                    'timestamp': m.timestamp
                }
                for m in self.metrics_history
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"检查点已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
    
    def run_optimization(self) -> Tuple[np.ndarray, float]:
        """运行优化过程"""
        start_time = time.time()
        
        try:
            # 初始化种群
            self.population = self.initialize_population()
            self.fitness_values = self.evaluate_fitness(self.population)
            
            # 找到初始最优个体
            best_index = np.argmax(self.fitness_values)
            self.best_individual = self.population[best_index].copy()
            self.best_fitness = self.fitness_values[best_index]
            
            self.logger.info(f"开始优化，初始最优适应度: {self.best_fitness:.6f}")
            
            # 主优化循环
            for generation in range(self.config.generations):
                generation_start = time.time()
                self.generation = generation
                
                # 选择
                selected = self.selection(self.population, self.fitness_values)
                
                # 交叉
                offspring = self.crossover(selected)
                
                # 变异
                offspring = self.mutation(offspring)
                
                # 评估新种群
                offspring_fitness = self.evaluate_fitness(offspring)
                
                # 精英保留
                self.population, self.fitness_values = self.elitism(
                    self.population, self.fitness_values,
                    offspring, offspring_fitness
                )
                
                # 更新最优个体
                current_best_index = np.argmax(self.fitness_values)
                current_best_fitness = self.fitness_values[current_best_index]
                
                if current_best_fitness > self.best_fitness:
                    self.best_individual = self.population[current_best_index].copy()
                    self.best_fitness = current_best_fitness
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1
                
                # 计算指标
                diversity = self.calculate_diversity(self.population)
                average_fitness = np.mean(self.fitness_values)
                convergence_rate = (current_best_fitness - average_fitness) / (abs(average_fitness) + 1e-10)
                generation_time = time.time() - generation_start
                
                # 记录指标
                metrics = OptimizationMetrics(
                    generation=generation,
                    best_fitness=current_best_fitness,
                    average_fitness=average_fitness,
                    diversity=diversity,
                    convergence_rate=convergence_rate,
                    execution_time=generation_time
                )
                self.metrics_history.append(metrics)
                
                # 自适应参数调整
                self.adaptive_parameter_adjustment()
                
                # 日志输出
                if generation % 10 == 0 or generation == self.config.generations - 1:
                    self.logger.info(
                        f"代数 {generation}: 最优={current_best_fitness:.6f}, "
                        f"平均={average_fitness:.6f}, 多样性={diversity:.6f}, "
                        f"停滞={self.stagnation_count}"
                    )
                
                # 保存检查点
                if generation % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()
                
                # 收敛检查
                if self.stagnation_count >= self.config.max_stagnation:
                    self.logger.info(f"算法收敛，停滞代数: {self.stagnation_count}")
                    break
            
            total_time = time.time() - start_time
            self.logger.info(
                f"优化完成，总时间: {total_time:.2f}s, "
                f"最优适应度: {self.best_fitness:.6f}"
            )
            
            return self.best_individual, self.best_fitness
            
        except Exception as e:
            self.logger.error(f"优化过程异常: {e}")
            raise
        
        finally:
            # 清理资源
            if self.executor:
                self.executor.shutdown(wait=True)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        if not self.metrics_history:
            return {}
        
        return {
            'best_individual': self.best_individual.tolist() if self.best_individual is not None else None,
            'best_fitness': self.best_fitness,
            'total_generations': len(self.metrics_history),
            'final_diversity': self.metrics_history[-1].diversity,
            'convergence_history': [m.best_fitness for m in self.metrics_history],
            'average_generation_time': np.mean([m.execution_time for m in self.metrics_history]),
            'total_optimization_time': sum(m.execution_time for m in self.metrics_history),
            'stagnation_count': self.stagnation_count,
            'final_config': {
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'selection_pressure': self.config.selection_pressure
            }
        }


# 生产级适应度函数注册表实例
production_fitness_registry = ProductionFitnessRegistry()


# 注册标准适应度函数
@production_fitness_registry.register(
    "sphere_function",
    description="球面函数，全局最小值在原点",
    output_range=(0.0, float('inf'))
)
def sphere_function(individual: List[float]) -> float:
    """球面函数：f(x) = sum(x_i^2)"""
    return -sum(x**2 for x in individual)  # 负号使其成为最大化问题


@production_fitness_registry.register(
    "rosenbrock_function", 
    description="Rosenbrock函数，经典优化测试函数",
    output_range=(0.0, float('inf'))
)
def rosenbrock_function(individual: List[float]) -> float:
    """Rosenbrock函数"""
    if len(individual) < 2:
        return float('-inf')
    
    total = 0.0
    for i in range(len(individual) - 1):
        total += 100 * (individual[i+1] - individual[i]**2)**2 + (1 - individual[i])**2
    
    return -total  # 负号使其成为最大化问题


@production_fitness_registry.register(
    "rastrigin_function",
    description="Rastrigin函数，多模态测试函数",
    output_range=(0.0, float('inf'))
)
def rastrigin_function(individual: List[float]) -> float:
    """Rastrigin函数"""
    A = 10
    n = len(individual)
    total = A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in individual)
    return -total  # 负号使其成为最大化问题


def main():
    """主函数 - 生产环境使用示例"""
    # 配置优化参数
    config = OptimizationConfig(
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        enable_adaptive_params=True,
        enable_parallel_evaluation=True,
        max_workers=4,
        log_level="INFO"
    )
    
    # 定义优化问题
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # 配置适应度函数
    fitness_functions = [
        (production_fitness_registry.get_function("sphere_function"), 0.5),
        (production_fitness_registry.get_function("rosenbrock_function"), 0.3),
        (production_fitness_registry.get_function("rastrigin_function"), 0.2)
    ]
    
    # 创建优化器
    optimizer = ProductionGeneticAlgorithm(
        bounds=bounds,
        fitness_functions=fitness_functions,
        config=config
    )
    
    try:
        # 运行优化
        best_individual, best_fitness = optimizer.run_optimization()
        
        # 输出结果
        print(f"最优个体: {best_individual}")
        print(f"最优适应度: {best_fitness}")
        
        # 获取优化报告
        report = optimizer.get_optimization_report()
        print(f"总代数: {report['total_generations']}")
        print(f"平均代时间: {report['average_generation_time']:.4f}s")
        print(f"总优化时间: {report['total_optimization_time']:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"优化失败: {e}")
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行主程序
    success = main()
    exit(0 if success else 1)