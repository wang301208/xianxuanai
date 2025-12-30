"""
情景记忆系统

实现海马-皮层交互和记忆巩固机制
"""

import numpy as np
from typing import Dict, List
from .events import EventBus

class HippocampalFormation:
    """海马结构模型 (基于Treves-Roll模型)"""
    
    def __init__(self):
        # 新增情境记忆索引
        self.context_index = {}  # {context_hash: [trace_indices]}
        # CA3区参数 (自联想记忆)
        self.ca3 = {
            'pyramidal': 1000,  # 锥体细胞数量
            'recurrent_weights': np.random.rand(1000, 1000) * 0.1,
            'plasticity_rate': 0.01
        }
        
        # CA1区参数 (模式分离)
        self.ca1 = {
            'pyramidal': 800,
            'ff_weights': np.random.rand(800, 1000) * 0.2,
            'context_gain': 1.5
        }
        
        # 记忆痕迹
        self.memory_traces = []
        self.consolidation_counter = 0
        
        # 注册事件处理器
        EventBus.subscribe('encoding_phase', self.on_encoding)
        EventBus.subscribe('consolidation_phase', self.on_consolidation)
    
    def on_encoding(self, sensory_input: np.ndarray, context: np.ndarray):
        """记忆编码阶段"""
        # CA3模式完成
        ca3_input = sensory_input * 0.8 + context * 0.2
        ca3_output = self._ca3_forward(ca3_input)
        
        # CA1模式分离
        ca1_output = self._ca1_forward(ca3_output)
        
        # 创建记忆痕迹
        trace = {
            'sensory': sensory_input,
            'context': context,
            'ca3': ca3_output,
            'ca1': ca1_output,
            'strength': 1.0,
            'replay_count': 0,
            'timestamp': time.time(),
            'last_accessed': time.time()
        }
        self.memory_traces.append(trace)
        
        # 更新情境索引
        ctx_hash = self._hash_context(context)
        if ctx_hash not in self.context_index:
            self.context_index[ctx_hash] = []
        self.context_index[ctx_hash].append(len(self.memory_traces)-1)
    
    def on_consolidation(self, sleep_stage: str):
        """记忆巩固阶段 (慢波睡眠期间)"""
        if sleep_stage == 'SWS':
            # 选择需要巩固的记忆 (基于近期强度)
            recent_traces = sorted(
                [t for t in self.memory_traces if t['strength'] > 0.5],
                key=lambda x: x['strength'],
                reverse=True
            )[:10]
            
            # 海马重放
            for trace in recent_traces:
                self._replay_memory(trace)
                trace['replay_count'] += 1
                trace['strength'] *= 1.2  # 巩固增强
    
    def _ca3_forward(self, input_pattern: np.ndarray) -> np.ndarray:
        """CA3区前向传播 (模式完成)"""
        # 添加随机噪声模拟不完全回忆
        noisy_input = input_pattern * 0.9 + np.random.rand(*input_pattern.shape) * 0.1
        return np.tanh(self.ca3['recurrent_weights'] @ noisy_input)
    
    def _ca1_forward(self, ca3_output: np.ndarray) -> np.ndarray:
        """CA1区前向传播 (模式分离)"""
        return np.tanh(self.ca1['ff_weights'] @ ca3_output * self.ca1['context_gain'])
    
    def retrieve_by_context(self, context: np.ndarray, similarity_threshold=0.7):
        """情境依赖的记忆检索"""
        ctx_hash = self._hash_context(context)
        if ctx_hash not in self.context_index:
            return None
            
        # 计算情境相似度
        candidates = []
        for idx in self.context_index[ctx_hash]:
            trace = self.memory_traces[idx]
            sim = np.dot(context, trace['context']) / (
                np.linalg.norm(context) * np.linalg.norm(trace['context']))
            if sim >= similarity_threshold:
                candidates.append((idx, sim))
                
        # 按强度和最近访问排序
        candidates.sort(key=lambda x: (
            self.memory_traces[x[0]]['strength'],
            self.memory_traces[x[0]]['last_accessed']
        ), reverse=True)
        
        # 更新访问记录
        if candidates:
            self.memory_traces[candidates[0][0]]['last_accessed'] = time.time()
            return self.memory_traces[candidates[0][0]]
        return None
        
    def _hash_context(self, context: np.ndarray) -> str:
        """生成情境特征哈希"""
        return hashlib.md5(context.tobytes()).hexdigest()
        
    def _replay_memory(self, trace: Dict):
        """记忆重放过程 (海马-皮层对话)"""
        # 应用遗忘曲线衰减
        time_elapsed = time.time() - trace['timestamp']
        trace['strength'] *= self._forgetting_curve(time_elapsed)
        # 模拟皮层长时程增强
        cortical_input = trace['ca1'] * 0.7 + trace['context'] * 0.3
        EventBus.publish('cortical_plasticity', 
                        pattern=cortical_input,
                        strength=trace['strength'])

    def _forgetting_curve(self, t: float, a=0.1, b=0.4, c=0.2) -> float:
        """遗忘曲线 (基于Ebbinghaus模型)
        
        Args:
            t: 时间(小时)
            a: 初始衰减率
            b: 长期保留率
            c: 衰减时间常数
        """
        return b + (1 - b) * np.exp(-a * t / c)
        
class CorticalMemory:
    """新皮层记忆存储模型"""
    def __init__(self):
        self.time_decay_factor = 0.999  # 每小时衰减率
    
    def __init__(self):
        self.memory_weights = np.random.rand(5000, 5000) * 0.05
        self.consolidation_rate = 0.001
        
        EventBus.subscribe('cortical_plasticity', self.on_plasticity)
    
    def on_plasticity(self, pattern: np.ndarray, strength: float):
        """突触可塑性更新"""
        # 时程依赖性增强 (基于记忆强度)
        ltp = np.outer(pattern, pattern) * self.consolidation_rate * strength
        self.memory_weights = np.clip(self.memory_weights + ltp, 0, 1)
        
        # 全局抑制保持稀疏性
        self.memory_weights *= 0.995