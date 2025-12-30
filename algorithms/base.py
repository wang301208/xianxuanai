from abc import ABC, abstractmethod
from typing import Any


class Algorithm(ABC):
    """所有算法的基类。

    这是一个抽象基类，定义了算法的通用接口。
    所有具体的算法实现都应该继承这个类并实现 execute 方法。
    
    子类必须实现:
        execute: 执行算法并返回结果的抽象方法
    """

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """执行算法并返回结果。
        
        这是一个抽象方法，必须在子类中实现。
        
        参数:
            *args: 位置参数，具体参数取决于算法实现
            **kwargs: 关键字参数，具体参数取决于算法实现
            
        返回:
            Any: 算法执行的结果，类型取决于具体算法
            
        异常:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError
