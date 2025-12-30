"""
注意力过程模块

实现选择性地关注输入的某些部分的注意力机制。
新增功能：
1. 乙酰胆碱调节
2. 工作记忆交互
3. 选择性过滤
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class AttentionSystem(CognitiveProcess):
    """
    注意力系统
    
    特性：
    1. 乙酰胆碱调节的注意力增益
    2. 自上而下和自下而上注意力
    3. 工作记忆交互
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(network, params or {})

        # 注意力参数
        self.ach_sensitivity = self.params.get("ach_sensitivity", 1.0)
        self.bottom_up_weight = self.params.get("bottom_up_weight", 0.5)
        self.top_down_weight = self.params.get("top_down_weight", 0.5)
        self.attention_span = self.params.get("attention_span", 3)
        
        # 注意力状态
        self.focus = None  # 当前注意力焦点
        self.salience_map = {}  # 显著性图
        self.priority_map = {}  # 优先级图
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理注意力分配
        
        Args:
            inputs: 输入数据字典，包含：
                - sensory_input: 感觉输入
                - working_memory: 工作记忆状态
                - neuromod.acetylcholine: ACh水平
                - task_goal: 当前任务目标
                
        Returns:
            注意力处理结果
        """
        # 获取乙酰胆碱水平
        ach_level = inputs.get("neuromod.acetylcholine", 0.5)
        
        # 计算注意力增益
        attention_gain = self._calculate_gain(ach_level)
        
        # 更新显著性图(自下而上)
        self._update_salience_map(inputs.get("sensory_input", {}))
        
        # 更新优先级图(自上而下)
        self._update_priority_map(
            inputs.get("working_memory", {}),
            inputs.get("task_goal", None),
            inputs.get("workspace_attention"),
            inputs.get("workspace_focus"),
        )
        
        # 计算综合注意力图
        attention_map = self._compute_attention_map(attention_gain)
        
        # 选择注意力焦点
        self.focus = self._select_focus(attention_map)
        
        return {
            "attention_focus": self.focus,
            "attention_map": attention_map,
            "attention_gain": attention_gain
        }
    
    def _calculate_gain(self, ach_level: float) -> float:
        """计算乙酰胆碱调节的注意力增益"""
        # 非线性增益函数
        base_gain = 1.0
        ach_effect = np.tanh(self.ach_sensitivity * ach_level)
        return base_gain * (1.0 + ach_effect)
    
    def _update_salience_map(self, sensory_input: Dict[str, Any]) -> None:
        """更新显著性图(自下而上)"""
        self.salience_map = {}
        
        # 计算每个输入项的显著性
        for key, value in sensory_input.items():
            # 使用值的大小作为显著性
            # 实际应用中应使用更复杂的显著性计算
            if isinstance(value, (int, float)):
                salience = abs(value)
            elif isinstance(value, str):
                salience = len(value) / 10.0  # 简单启发式
            else:
                salience = 0.5  # 默认显著性
                
            self.salience_map[key] = salience
    
    def _update_priority_map(
        self,
        working_memory: Dict[str, Any],
        task_goal: Any,
        workspace_attention: Optional[Dict[str, float]] = None,
        workspace_focus: Optional[List[str]] = None,
    ) -> None:
        """更新优先级图(自上而下)"""
        self.priority_map = {}
        
        # 基于工作记忆内容设置优先级
        for key, value in working_memory.items():
            # 从工作记忆中提取优先级信息
            if isinstance(value, dict) and "priority" in value:
                self.priority_map[key] = value["priority"]
            else:
                self.priority_map[key] = 0.5  # 默认优先级
        
        # 基于任务目标调整优先级
        if task_goal:
            goal_str = str(task_goal)
            for key in self.priority_map:
                # 如果项目与目标相关，提高优先级
                if goal_str in str(key):
                    self.priority_map[key] += 0.3
        
        if workspace_attention:
            for key, weight in workspace_attention.items():
                self.priority_map[key] = max(self.priority_map.get(key, 0.0), float(weight))

        if workspace_focus:
            for key in workspace_focus:
                self.priority_map[key] = max(self.priority_map.get(key, 0.0), 0.9)
    
    def _compute_attention_map(self, gain: float) -> Dict[str, float]:
        """计算综合注意力图"""
        attention_map = {}
        
        # 合并显著性和优先级
        all_keys = set(self.salience_map.keys()) | set(self.priority_map.keys())
        
        for key in all_keys:
            salience = self.salience_map.get(key, 0.0)
            priority = self.priority_map.get(key, 0.0)
            
            # 加权组合
            attention = (
                self.bottom_up_weight * salience + 
                self.top_down_weight * priority
            )
            
            # 应用注意力增益
            attention_map[key] = attention * gain
            
        return attention_map
    
    def _select_focus(self, attention_map: Dict[str, float]) -> List[str]:
        """选择注意力焦点"""
        # 按注意力值排序
        sorted_items = sorted(
            attention_map.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 选择前N个项目作为焦点
        focus_items = [item[0] for item in sorted_items[:self.attention_span]]
        return focus_items


class AttentionWorkingMemoryInterface:
    """
    注意力-工作记忆接口
    
    协调注意力系统和工作记忆之间的交互
    """
    
    def __init__(self, 
                 attention_system: AttentionSystem, 
                 working_memory: 'WorkingMemory'):
        self.attention = attention_system
        self.memory = working_memory
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理注意力-记忆交互"""
        # 第一步：注意力处理
        attention_result = self.attention.process(inputs)
        
        # 第二步：增强工作记忆中的注意焦点项
        focus_items = attention_result["attention_focus"]
        memory_input = inputs.copy()
        
        # 为工作记忆添加注意力增强
        memory_input["attention_focus"] = focus_items
        memory_input["attention_gain"] = attention_result["attention_gain"]
        
        # 第三步：工作记忆处理
        memory_result = self.memory.process(memory_input)
        
        # 返回组合结果
        return {
            "attention": attention_result,
            "memory": memory_result,
            "integrated_state": {
                "focus": focus_items,
                "memory_content": memory_result.get("memory_state", {})
            }
        }
    
    def filter_input(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """基于注意力焦点过滤输入"""
        if not self.attention.focus:
            return sensory_input
            
        filtered_input = {}
        for key, value in sensory_input.items():
            # 只保留注意力焦点中的项目
            if key in self.attention.focus:
                filtered_input[key] = value
                
        return filtered_input

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class AttentionProcess(CognitiveProcess):
    """
    注意力过程
    
    选择性地关注输入的某些部分
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化注意力过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - attention_type: 注意力类型
                - focus_size: 注意力焦点大小
                - inhibition_strength: 抑制强度
        """
        super().__init__(network, params or {})
        
        # 注意力状态
        self.focus_position = 0
        self.attention_weights = []
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理注意力
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - perception_output: 感知输出
                - focus_position: 注意力焦点位置（可选）
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        attention_type = self.params.get("attention_type", "spatial")
        focus_size = self.params.get("focus_size", 0.2)  # 相对大小
        inhibition_strength = self.params.get("inhibition_strength", 0.8)
        
        # 获取感知输出
        perception_output = inputs.get("perception_output", [])
        if not perception_output:
            return {"attention_output": [], "attention_weights": []}
        
        # 更新注意力焦点位置
        if "focus_position" in inputs:
            self.focus_position = inputs["focus_position"]
        
        # 应用注意力机制
        if attention_type == "spatial":
            # 空间注意力：关注特定区域
            attention_output, weights = self._apply_spatial_attention(
                perception_output, self.focus_position, focus_size, inhibition_strength
            )
        
        elif attention_type == "feature":
            # 特征注意力：关注特定特征
            target_feature = inputs.get("target_feature", 0.5)
            attention_output, weights = self._apply_feature_attention(
                perception_output, target_feature, focus_size, inhibition_strength
            )
        
        elif attention_type == "global":
            # 全局注意力：基于整体显著性
            attention_output, weights = self._apply_global_attention(
                perception_output, inhibition_strength
            )
        
        else:
            # 默认不应用注意力
            attention_output = perception_output
            weights = [1.0] * len(perception_output)
        
        # 更新注意力权重
        self.attention_weights = weights
        
        return {
            "attention_output": attention_output,
            "attention_weights": weights
        }
    
    def _apply_spatial_attention(
        self, 
        data: List[float], 
        focus_position: float, 
        focus_size: float, 
        inhibition_strength: float
    ) -> Tuple[List[float], List[float]]:
        """
        应用空间注意力
        
        Args:
            data: 输入数据
            focus_position: 注意力焦点位置 (0-1)
            focus_size: 注意力焦点大小 (0-1)
            inhibition_strength: 抑制强度 (0-1)
            
        Returns:
            注意力处理后的数据和注意力权重
        """
        n = len(data)
        weights = []
        
        # 计算每个位置的注意力权重
        for i in range(n):
            # 归一化位置 (0-1)
            pos = i / max(1, n - 1)
            
            # 计算与焦点的距离
            distance = abs(pos - focus_position)
            
            # 如果在焦点范围内，权重为1，否则根据距离衰减
            if distance <= focus_size / 2:
                weight = 1.0
            else:
                # 线性衰减
                weight = max(0.0, 1.0 - (distance - focus_size / 2) / (1.0 - focus_size / 2))
                
                # 应用抑制
                weight *= (1.0 - inhibition_strength)
            
            weights.append(weight)
        
        # 应用权重
        output = [data[i] * weights[i] for i in range(n)]
        
        return output, weights
    
    def _apply_feature_attention(
        self, 
        data: List[float], 
        target_feature: float, 
        focus_size: float, 
        inhibition_strength: float
    ) -> Tuple[List[float], List[float]]:
        """
        应用特征注意力
        
        Args:
            data: 输入数据
            target_feature: 目标特征值
            focus_size: 特征相似度阈值
            inhibition_strength: 抑制强度
            
        Returns:
            注意力处理后的数据和注意力权重
        """
        weights = []
        
        # 计算每个特征的注意力权重
        for value in data:
            # 计算与目标特征的相似度
            similarity = 1.0 - abs(value - target_feature)
            
            # 如果相似度高于阈值，权重为1，否则根据相似度衰减
            if similarity >= 1.0 - focus_size:
                weight = 1.0
            else:
                # 线性衰减
                weight = max(0.0, similarity / (1.0 - focus_size))
                
                # 应用抑制
                weight *= (1.0 - inhibition_strength)
            
            weights.append(weight)
        
        # 应用权重
        output = [data[i] * weights[i] for i in range(len(data))]
        
        return output, weights
    
    def _apply_global_attention(
        self, 
        data: List[float], 
        inhibition_strength: float
    ) -> Tuple[List[float], List[float]]:
        """
        应用全局注意力
        
        Args:
            data: 输入数据
            inhibition_strength: 抑制强度
            
        Returns:
            注意力处理后的数据和注意力权重
        """
        if not data:
            return [], []
        
        # 计算显著性（这里简单地使用值的大小作为显著性）
        saliency = [abs(x) for x in data]
        max_saliency = max(saliency)
        
        if max_saliency == 0:
            return data, [1.0] * len(data)
        
        # 归一化显著性
        normalized_saliency = [s / max_saliency for s in saliency]
        
        # 计算权重：显著性高的保持原值，显著性低的被抑制
        weights = []
        for s in normalized_saliency:
            # 显著性越高，抑制越少
            inhibition = inhibition_strength * (1.0 - s)
            weight = 1.0 - inhibition
            weights.append(weight)
        
        # 应用权重
        output = [data[i] * weights[i] for i in range(len(data))]
        
        return output, weights
