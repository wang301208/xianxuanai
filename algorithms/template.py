"""
算法基础模板模块

提供算法实现的标准模板和最佳实践指南。
用于指导生产级算法的开发和实现。
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass

from .base import Algorithm


@dataclass
class AlgorithmResult:
    """算法执行结果封装"""
    result: Any
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None


class ProductionAlgorithm(Algorithm):
    """
    生产级算法基础类
    
    提供标准的错误处理、日志记录、性能监控等功能。
    所有生产算法都应继承此类。
    """
    
    def __init__(self, enable_logging: bool = True, enable_metrics: bool = True):
        """
        初始化生产级算法
        
        Args:
            enable_logging: 是否启用日志记录
            enable_metrics: 是否启用性能指标收集
        """
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.logger = logging.getLogger(self.__class__.__name__) if enable_logging else None
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    def execute(self, *args, **kwargs) -> Any:
        """
        执行算法（带监控和错误处理）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            算法执行结果
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 算法执行失败
        """
        start_time = time.perf_counter()
        
        try:
            # 输入验证
            self._validate_inputs(*args, **kwargs)
            
            if self.logger:
                self.logger.info(f"开始执行算法 {self.__class__.__name__}")
            
            # 执行核心算法逻辑
            result = self._execute_core(*args, **kwargs)
            
            # 输出验证
            self._validate_output(result)
            
            execution_time = time.perf_counter() - start_time
            
            # 更新指标
            if self.enable_metrics:
                self._update_metrics(execution_time)
            
            if self.logger:
                self.logger.info(f"算法执行成功，耗时: {execution_time:.4f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            if self.logger:
                self.logger.error(f"算法执行失败: {e}, 耗时: {execution_time:.4f}s")
            
            raise RuntimeError(f"算法 {self.__class__.__name__} 执行失败: {e}") from e
    
    @abstractmethod
    def _execute_core(self, *args, **kwargs) -> Any:
        """
        核心算法逻辑实现
        
        子类必须实现此方法
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            算法执行结果
        """
        pass
    
    def _validate_inputs(self, *args, **kwargs) -> None:
        """
        输入参数验证
        
        子类可以重写此方法实现自定义验证逻辑
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Raises:
            ValueError: 输入参数无效
        """
        pass
    
    def _validate_output(self, result: Any) -> None:
        """
        输出结果验证
        
        子类可以重写此方法实现自定义验证逻辑
        
        Args:
            result: 算法执行结果
            
        Raises:
            ValueError: 输出结果无效
        """
        pass
    
    def _update_metrics(self, execution_time: float) -> None:
        """更新性能指标"""
        self._execution_count += 1
        self._total_execution_time += execution_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if self._execution_count == 0:
            return {"execution_count": 0}
        
        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": self._total_execution_time / self._execution_count,
            "algorithm_name": self.__class__.__name__
        }


class NumericAlgorithm(ProductionAlgorithm):
    """
    数值计算算法基础类
    
    专门用于数值计算的算法，提供数值验证和精度控制。
    """
    
    def __init__(self, precision: float = 1e-10, **kwargs):
        """
        初始化数值算法
        
        Args:
            precision: 数值精度
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.precision = precision
    
    def _validate_inputs(self, *args, **kwargs) -> None:
        """验证数值输入"""
        super()._validate_inputs(*args, **kwargs)
        
        for arg in args:
            if isinstance(arg, (list, tuple)):
                if not all(isinstance(x, (int, float)) for x in arg):
                    raise ValueError("所有元素必须是数值类型")
            elif not isinstance(arg, (int, float)):
                if not hasattr(arg, '__iter__'):  # 允许可迭代对象
                    raise ValueError(f"参数必须是数值类型，得到: {type(arg)}")


class SummationAlgorithm(NumericAlgorithm):
    """
    求和算法实现
    
    生产级的数值求和算法，支持多种数据类型和错误处理。
    """
    
    def _execute_core(self, data: Union[List[Union[int, float]], tuple]) -> Union[int, float]:
        """
        执行求和计算
        
        Args:
            data: 数值列表或元组
            
        Returns:
            求和结果
        """
        if not data:
            return 0
        
        # 使用内置sum函数，性能最优
        return sum(data)
    
    def _validate_inputs(self, data: Union[List[Union[int, float]], tuple], **kwargs) -> None:
        """验证输入参数"""
        super()._validate_inputs(data, **kwargs)
        
        if not isinstance(data, (list, tuple)):
            raise ValueError("数据必须是列表或元组类型")
        
        if not data:
            if self.logger:
                self.logger.warning("输入数据为空，将返回0")
    
    def _validate_output(self, result: Union[int, float]) -> None:
        """验证输出结果"""
        super()._validate_output(result)
        
        if not isinstance(result, (int, float)):
            raise ValueError(f"输出结果必须是数值类型，得到: {type(result)}")
        
        # 检查数值溢出
        if abs(result) == float('inf'):
            raise ValueError("计算结果溢出")
