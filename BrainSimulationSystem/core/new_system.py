"""
新一代大脑模拟核心系统
采用神经模拟为主、LLM为辅的架构
"""

from neural_engine import NeuralEngine
from cognitive_layer import CognitiveLayer
from llm_service import LLMService
import random

class BrainSimulationCore:
    def __init__(self):
        # 核心组件初始化
        self.neural_engine = NeuralEngine()
        self.cognitive_layer = CognitiveLayer()
        self.llm_service = LLMService(mode='auxiliary')
        self.migration_ratio = 0.3  # 初始迁移比例

    def process(self, input_data):
        """主处理流程"""
        try:
            # 神经编码阶段
            neural_pattern = self.neural_engine.encode(input_data)
            
            # 认知处理阶段
            cognitive_output = self.cognitive_layer.process(neural_pattern)
            
            # 条件性LLM辅助
            if cognitive_output.get('confidence', 0) < 0.7:
                llm_context = self.llm_service.query(
                    cognitive_output['summary']
                )
                cognitive_output = self._merge_results(
                    cognitive_output, 
                    llm_context
                )
            
            return cognitive_output
            
        except Exception as e:
            self._handle_error(e)
            raise

    def _merge_results(self, cognitive, llm):
        """融合神经认知与LLM结果"""
        return {
            'primary': cognitive,
            'supplement': llm,
            'confidence': min(1.0, cognitive['confidence'] + 0.2)
        }

    def increase_migration_ratio(self, delta=0.1):
        """逐步增加迁移比例"""
        self.migration_ratio = min(1.0, self.migration_ratio + delta)