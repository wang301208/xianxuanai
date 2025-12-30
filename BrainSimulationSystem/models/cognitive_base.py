"""
认知系统基础模块
Cognitive System Base Module

定义了认知系统的核心抽象和工厂函数。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from .enums import BrainRegion
from .architecture import CognitiveArchitecture
from .regions import PhysiologicalBrainRegion

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from BrainSimulationSystem.core.network import NeuralNetwork


class CognitiveProcess(ABC):
    """认知过程基类"""

    def __init__(
        self,
        network: Optional["NeuralNetwork"] = None,
        params: Optional[Dict[str, Any]] = None,
        name: str = "",
    ):
        """初始化认知过程"""

        # 兼容旧的仅传入名称的调用方式
        if isinstance(network, str) and params is None and not name:
            name = network
            network = None
            params = None

        self.network = network
        self.params: Dict[str, Any] = params or {}
        self.name = name
        self.is_active = False

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入并返回输出"""
        pass

    def activate(self):
        """激活认知过程"""
        self.is_active = True

    def deactivate(self):
        """停用认知过程"""
        self.is_active = False


# 工厂函数

def create_cognitive_architecture(config: Dict[str, Any]) -> CognitiveArchitecture:
    """创建认知架构"""
    return CognitiveArchitecture(config)


def create_brain_region(
    region_type: BrainRegion, config: Dict[str, Any]
) -> PhysiologicalBrainRegion:
    """创建脑区"""
    return PhysiologicalBrainRegion(region_type, config)
