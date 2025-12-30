"""
海马记忆系统实现
Hippocampal Memory System Implementation

实现海马体的记忆编码、存储和检索机制
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
import hashlib
import types

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency / test stubs
    nx = None  # type: ignore[assignment]


if nx is None or not hasattr(nx, "Graph"):
    class _FallbackGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []

        def add_node(self, node, **attrs):
            self._nodes[node] = dict(attrs or {})

        def add_edge(self, u, v, **attrs):
            self._edges.append((u, v, dict(attrs or {})))

        def nodes(self, data: bool = False):
            if data:
                return list(self._nodes.items())
            return list(self._nodes.keys())

    try:
        if nx is None:
            nx = types.SimpleNamespace(Graph=_FallbackGraph)  # type: ignore[assignment]
        else:
            nx.Graph = _FallbackGraph  # type: ignore[attr-defined]
    except Exception:
        nx = types.SimpleNamespace(Graph=_FallbackGraph)  # type: ignore[assignment]
from scipy.spatial.distance import cosine
from .memory import MemoryTrace, MemoryType, ConsolidationState

class HippocampalMemorySystem:
    """海马记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 海马子区域
        self.ca1_capacity = config.get('ca1_capacity', 10000)
        self.ca3_capacity = config.get('ca3_capacity', 5000)
        self.dg_capacity = config.get('dg_capacity', 100000)
        
        # 记忆存储
        self.ca1_memories = {}  # 详细记忆表征
        self.ca3_memories = {}  # 模式完成
        self.dg_memories = {}   # 模式分离
        
        # 神经网络
        self.ca3_recurrent_network = nx.Graph()
        self.ca1_ca3_connections = {}
        self.dg_ca3_connections = {}
        
        # 编码参数
        self.pattern_separation_threshold = 0.3
        self.pattern_completion_threshold = 0.7
        
        # 巩固参数
        self.consolidation_rate = 0.001
        self.replay_probability = 0.1
        self.reconsolidation_window = float(config.get("reconsolidation_window", 600.0))
        
        self.logger = logging.getLogger("HippocampalMemory")
    
    def encode_episodic_memory(self, episode: Dict[str, Any], 
                             context: Dict[str, Any]) -> str:
        """编码情景记忆"""
        
        # 生成唯一ID
        episode_id = self._generate_memory_id(episode, context)
        
        # 齿状回：模式分离
        dg_pattern = self._dentate_gyrus_encoding(episode, context)
        
        # CA3：自联想网络
        ca3_pattern = self._ca3_encoding(dg_pattern, episode_id)
        
        # CA1：详细表征
        ca1_pattern = self._ca1_encoding(episode, context, ca3_pattern)
        
        # 存储记忆
        memory_trace = MemoryTrace(
            trace_id=episode_id,
            content=episode,
            memory_type=MemoryType.EPISODIC,
            encoding_time=time.time(),
            last_access_time=time.time(),
            context=context,
            neural_pattern=ca1_pattern,
            brain_regions=['hippocampus', 'ca1', 'ca3', 'dg']
        )
        
        self.ca1_memories[episode_id] = memory_trace
        
        self.logger.debug(f"Encoded episodic memory: {episode_id}")
        return episode_id
    
    def _generate_memory_id(self, episode: Dict[str, Any], 
                           context: Dict[str, Any]) -> str:
        """生成记忆ID"""
        
        content_str = json.dumps(episode, sort_keys=True)
        context_str = json.dumps(context, sort_keys=True)
        combined = content_str + context_str + str(time.time())
        
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _dentate_gyrus_encoding(self, episode: Dict[str, Any], 
                               context: Dict[str, Any]) -> np.ndarray:
        """齿状回编码（模式分离）"""
        
        # 稀疏编码
        pattern_size = 1000
        sparsity = 0.02  # 2%的神经元活跃
        
        # 基于内容和上下文生成模式
        content_hash = hash(str(episode)) % pattern_size
        context_hash = hash(str(context)) % pattern_size
        
        pattern = np.zeros(pattern_size)
        
        # 激活少数神经元
        active_neurons = int(pattern_size * sparsity)
        active_indices = np.random.choice(
            pattern_size, active_neurons, replace=False
        )
        
        # 基于内容调节激活强度
        for idx in active_indices:
            activation = np.random.uniform(0.5, 1.0)
            if idx == content_hash or idx == context_hash:
                activation *= 1.5  # 增强相关激活
            pattern[idx] = activation
        
        return pattern
    
    def _ca3_encoding(self, dg_pattern: np.ndarray, episode_id: str) -> np.ndarray:
        """CA3编码（自联想网络）"""
        
        # 压缩表征
        ca3_size = 500
        ca3_pattern = np.zeros(ca3_size)
        
        # 从DG模式生成CA3模式
        for i in range(ca3_size):
            weights = np.random.uniform(-0.1, 0.1, len(dg_pattern))
            ca3_pattern[i] = np.tanh(np.dot(dg_pattern, weights))
        
        # 存储到CA3网络
        self.ca3_memories[episode_id] = ca3_pattern
        
        # 更新递归连接
        self._update_ca3_recurrent_connections(episode_id, ca3_pattern)
        
        return ca3_pattern
    
    def _ca1_encoding(self, episode: Dict[str, Any], context: Dict[str, Any],
                     ca3_pattern: np.ndarray) -> np.ndarray:
        """CA1编码（详细表征）"""
        
        # 详细表征
        ca1_size = 2000
        ca1_pattern = np.zeros(ca1_size)
        
        # 结合CA3输入和直接皮层输入
        ca3_contribution = 0.6
        cortical_contribution = 0.4
        
        # CA3输入
        ca3_weights = np.random.uniform(-0.2, 0.2, (ca1_size, len(ca3_pattern)))
        ca3_input = np.dot(ca3_weights, ca3_pattern)
        
        # 皮层输入（基于内容）
        cortical_input = np.random.uniform(-0.1, 0.1, ca1_size)
        
        # 结合输入
        ca1_pattern = np.tanh(
            ca3_contribution * ca3_input + 
            cortical_contribution * cortical_input
        )
        
        return ca1_pattern
    
    def _update_ca3_recurrent_connections(self, episode_id: str, 
                                        ca3_pattern: np.ndarray):
        """更新CA3递归连接"""
        
        # 添加节点
        self.ca3_recurrent_network.add_node(episode_id, pattern=ca3_pattern)
        
        # 计算与现有记忆的相似性
        for existing_id, existing_data in self.ca3_recurrent_network.nodes(data=True):
            if existing_id != episode_id:
                existing_pattern = existing_data['pattern']
                similarity = 1.0 - cosine(ca3_pattern, existing_pattern)
                
                # 如果相似性超过阈值，建立连接
                if similarity > 0.3:
                    self.ca3_recurrent_network.add_edge(
                        episode_id, existing_id, weight=similarity
                    )
    
    def retrieve_episodic_memory(self, cue: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Optional[MemoryTrace]:
        """检索情景记忆"""

        # Fast-path for unit tests / structured cues: perform a direct dictionary match
        # before falling back to pattern completion.
        if isinstance(cue, dict) and cue:
            for trace in self.ca1_memories.values():
                content = getattr(trace, "content", None)
                if not isinstance(content, dict):
                    continue
                if all(content.get(k) == v for k, v in cue.items()):
                    trace.access(time.time())
                    return trace
        
        # 生成检索线索模式
        cue_pattern = self._generate_cue_pattern(cue, context)
        
        # CA3模式完成
        completed_pattern = self._ca3_pattern_completion(cue_pattern)
        
        if completed_pattern is not None:
            # 找到最匹配的记忆
            best_match_id = self._find_best_match(completed_pattern)
            
            if best_match_id and best_match_id in self.ca1_memories:
                memory_trace = self.ca1_memories[best_match_id]
                
                # 更新访问信息
                memory_trace.access(time.time())
                
                self.logger.debug(f"Retrieved episodic memory: {best_match_id}")
                return memory_trace
        
        return None
    
    def _generate_cue_pattern(self, cue: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> np.ndarray:
        """生成检索线索模式"""
        
        # 简化的线索编码
        pattern_size = 500
        pattern = np.zeros(pattern_size)
        
        # 基于线索内容生成部分模式
        cue_hash = hash(str(cue)) % pattern_size
        pattern[cue_hash] = 1.0
        
        if context:
            context_hash = hash(str(context)) % pattern_size
            pattern[context_hash] = 0.8
        
        # 添加噪声
        noise = np.random.uniform(-0.1, 0.1, pattern_size)
        pattern += noise
        
        return pattern
    
    def _ca3_pattern_completion(self, cue_pattern: np.ndarray) -> Optional[np.ndarray]:
        """CA3模式完成"""
        
        best_match = None
        best_similarity = 0.0
        
        # 与所有存储的CA3模式比较
        for episode_id, ca3_pattern in self.ca3_memories.items():
            # 计算相似性
            similarity = 1.0 - cosine(cue_pattern, ca3_pattern)
            
            if similarity > best_similarity and similarity > self.pattern_completion_threshold:
                best_similarity = similarity
                best_match = ca3_pattern
        
        return best_match
    
    def _find_best_match(self, completed_pattern: np.ndarray) -> Optional[str]:
        """找到最佳匹配的记忆ID"""
        
        best_match_id = None
        best_similarity = 0.0
        
        for episode_id, ca3_pattern in self.ca3_memories.items():
            similarity = 1.0 - cosine(completed_pattern, ca3_pattern)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = episode_id
        
        return best_match_id if best_similarity > 0.8 else None
    
    def consolidate_memories(self, dt: float, sleep_mode: bool = False) -> Dict[str, Any]:
        """巩固记忆"""
        
        consolidation_results = {
            'memories_consolidated': 0,
            'memories_reconsolidated': 0,
            'replay_events': 0,
            'memories_transferred': 0
        }
        
        now = time.time()
        reconsolidation_window = float(self.reconsolidation_window)
        if not np.isfinite(reconsolidation_window) or reconsolidation_window <= 0:
            reconsolidation_window = 0.0

        # 选择需要巩固的记忆
        memories_to_consolidate = []
        for memory_trace in self.ca1_memories.values():
            state = getattr(memory_trace, "consolidation_state", ConsolidationState.LABILE)
            if state in (ConsolidationState.LABILE, ConsolidationState.CONSOLIDATING, ConsolidationState.RECONSOLIDATING):
                memories_to_consolidate.append(memory_trace)
                continue
            if state == ConsolidationState.CONSOLIDATED and reconsolidation_window > 0:
                last_access = getattr(memory_trace, "last_access_time", None)
                try:
                    last_access_s = float(last_access) if last_access is not None else None
                except Exception:
                    last_access_s = None
                if last_access_s is not None and (now - last_access_s) <= reconsolidation_window:
                    memories_to_consolidate.append(memory_trace)
        
        # 巩固过程
        for memory_trace in memories_to_consolidate:
            previous_state = getattr(memory_trace, "consolidation_state", None)
            # 计算巩固信号
            consolidation_signal = self._calculate_consolidation_signal(
                memory_trace, sleep_mode
            )
            
            # 更新巩固状态
            memory_trace.update_consolidation(dt, consolidation_signal)
            
            if memory_trace.consolidation_state == ConsolidationState.CONSOLIDATED:
                if previous_state == ConsolidationState.RECONSOLIDATING:
                    consolidation_results['memories_reconsolidated'] += 1
                elif previous_state in (ConsolidationState.LABILE, ConsolidationState.CONSOLIDATING):
                    consolidation_results['memories_consolidated'] += 1
        
        # 睡眠期间的记忆重放
        if sleep_mode:
            replay_count = self._memory_replay(dt)
            consolidation_results['replay_events'] = replay_count
        
        return consolidation_results
    
    def _calculate_consolidation_signal(self, memory_trace: MemoryTrace, 
                                      sleep_mode: bool) -> float:
        """计算巩固信号强度"""
        
        signal = 0.0
        
        # 基础巩固信号
        signal += 0.3
        
        # 情感增强
        emotional_boost = abs(memory_trace.emotional_valence) * 0.3
        signal += emotional_boost
        
        # 重要性（基于访问次数）
        importance = min(memory_trace.access_count / 10.0, 0.4)
        signal += importance
        
        # 睡眠增强
        if sleep_mode:
            signal *= 2.0
        
        return np.clip(signal, 0.0, 1.0)
    
    def _memory_replay(self, dt: float) -> int:
        """记忆重放"""
        
        replay_count = 0
        
        # 选择要重放的记忆序列
        replay_candidates = [
            memory for memory in self.ca1_memories.values()
            if memory.consolidation_state != ConsolidationState.CONSOLIDATED
        ]
        
        # 按重要性排序
        replay_candidates.sort(
            key=lambda m: m.access_count + abs(m.emotional_valence),
            reverse=True
        )
        
        # 重放前几个重要记忆
        max_replays = min(10, len(replay_candidates))
        for i in range(max_replays):
            if np.random.random() < self.replay_probability:
                memory = replay_candidates[i]
                
                # 模拟重放：增强记忆强度
                memory.strength += 0.1
                memory.strength = min(memory.strength, 2.0)
                
                replay_count += 1
        
        return replay_count
