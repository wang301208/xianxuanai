"""
增强的海马记忆系统

实现完整的海马结构和记忆处理机制，包括：
- 齿状回（DG）的模式分离和神经发生
- CA3的自联想记忆和模式完成
- CA1的时序编码和空间表征
- CA2的社会记忆和时间编码
- 下托（Subiculum）的输出整合
- 内嗅皮层（EC）的网格细胞和边界细胞
- 多种记忆类型的处理机制
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import logging

from .reconstruction import MemoryReconstruction
from ..core.neurons import Neuron, create_neuron
from ..core.synapses import Synapse, create_synapse


def _pad_or_trim_1d(vector: np.ndarray, length: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    length = int(length)
    if length <= 0:
        return np.asarray([], dtype=float)
    if vector.size > length:
        return vector[:length]
    if vector.size < length:
        return np.pad(vector, (0, length - vector.size))
    return vector


class _SparseProjector:
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        fan_in: int,
        rng: np.random.RandomState,
        weight_scale: float = 0.1,
    ):
        self.input_dim = int(max(1, input_dim))
        self.output_dim = int(max(1, output_dim))
        self.fan_in = int(max(1, min(int(fan_in), self.input_dim)))
        self.indices = rng.randint(0, self.input_dim, size=(self.output_dim, self.fan_in))
        self.weights = rng.normal(0.0, float(weight_scale), size=(self.output_dim, self.fan_in))

    def project(self, vector: np.ndarray) -> np.ndarray:
        vector = _pad_or_trim_1d(vector, self.input_dim)
        return np.sum(vector[self.indices] * self.weights, axis=1)


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"           # 情节记忆
    SEMANTIC = "semantic"           # 语义记忆
    SPATIAL = "spatial"             # 空间记忆
    TEMPORAL = "temporal"           # 时间记忆
    SOCIAL = "social"               # 社会记忆
    EMOTIONAL = "emotional"         # 情绪记忆
    PROCEDURAL = "procedural"       # 程序记忆
    WORKING = "working"             # 工作记忆


class MemoryPhase(Enum):
    """记忆处理阶段"""
    ENCODING = "encoding"           # 编码
    CONSOLIDATION = "consolidation" # 巩固
    RETRIEVAL = "retrieval"         # 提取
    RECONSOLIDATION = "reconsolidation" # 再巩固
    FORGETTING = "forgetting"       # 遗忘


@dataclass
class EnhancedMemoryTrace:
    """增强的记忆痕迹"""
    # 基本信息
    trace_id: int
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: float
    
    # 海马各区域编码
    dg_sparse_code: np.ndarray = field(default_factory=lambda: np.array([]))
    ca3_pattern: np.ndarray = field(default_factory=lambda: np.array([]))
    ca1_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    ca2_social_code: np.ndarray = field(default_factory=lambda: np.array([]))
    subiculum_output: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 内嗅皮层编码
    ec_grid_code: np.ndarray = field(default_factory=lambda: np.array([]))
    ec_border_code: np.ndarray = field(default_factory=lambda: np.array([]))
    ec_object_code: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 记忆强度和状态
    encoding_strength: float = 1.0
    consolidation_level: float = 0.0
    retrieval_count: int = 0
    last_accessed: float = 0.0
    
    # 关联信息
    spatial_context: Optional[np.ndarray] = None
    temporal_context: Optional[np.ndarray] = None
    emotional_valence: float = 0.0
    social_context: Optional[Dict[str, Any]] = None
    
    # 连接关系
    associated_traces: List[int] = field(default_factory=list)
    causal_links: List[Tuple[int, str]] = field(default_factory=list)  # (trace_id, relation_type)
    
    # 遗忘参数
    decay_rate: float = 0.01
    interference_level: float = 0.0


class DentateGyrus:
    """齿状回 - 模式分离和神经发生"""
    
    def __init__(self, size: int = 1000000, sparsity: float = 0.005):
        self.size = size
        self.sparsity = sparsity
        self.active_neurons = int(size * sparsity)
        
        # 神经发生参数
        self.neurogenesis_rate = 0.001  # 每天新生神经元比例
        self.young_neuron_ratio = 0.1   # 年轻神经元比例
        self.maturation_time = 30.0     # 神经元成熟时间（天）
        
        # 神经元年龄追踪
        self.neuron_ages = np.random.exponential(100, size)  # 初始年龄分布
        self.young_neurons = np.where(self.neuron_ages < self.maturation_time)[0]
        
        # 活动模式
        self.activity_pattern = np.zeros(size)
        self.pattern_history = deque(maxlen=1000)
        
        # 可塑性参数
        self.learning_rate = 0.01
        self.competition_strength = 2.0
        self.rng = np.random.RandomState(42)

        # 输入 -> DG 的稀疏随机投影（按输入维度惰性初始化）
        self._projection_input_dim: Optional[int] = None
        self._projection_fanout: int = 8
        self._projection_indices: Optional[np.ndarray] = None  # shape: (input_dim, fanout)
        self._projection_weights: Optional[np.ndarray] = None  # shape: (input_dim, fanout)
        
        self.logger = logging.getLogger("DentateGyrus")

    def _ensure_projection(self, input_dim: int) -> None:
        if self._projection_input_dim == int(input_dim) and self._projection_indices is not None:
            return
        input_dim = int(input_dim)
        fanout = int(self._projection_fanout)
        self._projection_input_dim = input_dim
        self._projection_indices = self.rng.randint(0, self.size, size=(input_dim, fanout), dtype=np.int32)
        self._projection_weights = self.rng.normal(0.0, 1.0, size=(input_dim, fanout)).astype(np.float32)
    
    def neurogenesis_update(self, dt: float):
        """更新神经发生过程"""
        # 更新神经元年龄
        self.neuron_ages += dt / (24 * 3600)  # 转换为天
        
        # 生成新神经元
        new_neurons_count = int(self.neurogenesis_rate * self.size * dt / (24 * 3600))
        if new_neurons_count > 0:
            # 随机替换一些老神经元
            old_neurons = np.where(self.neuron_ages > 365)[0]  # 超过1年的神经元
            if len(old_neurons) >= new_neurons_count:
                replace_indices = np.random.choice(old_neurons, new_neurons_count, replace=False)
                self.neuron_ages[replace_indices] = 0.0  # 重置为新生神经元
        
        # 更新年轻神经元列表
        self.young_neurons = np.where(self.neuron_ages < self.maturation_time)[0]
    
    def pattern_separation(self, input_pattern: np.ndarray, 
                          novelty_threshold: float = 0.3) -> np.ndarray:
        """执行模式分离"""
        # 计算与历史模式的相似性
        max_similarity = 0.0
        if self.pattern_history:
            similarities = []
            for hist_pattern in self.pattern_history:
                if len(hist_pattern) == len(input_pattern):
                    sim = np.corrcoef(input_pattern, hist_pattern)[0, 1]
                    if not np.isnan(sim):
                        similarities.append(abs(sim))
            
            if similarities:
                max_similarity = max(similarities)
        
        # 根据新颖性调整分离强度
        if max_similarity > novelty_threshold:
            # 高相似性：增强分离
            separation_strength = 1.5
        else:
            # 低相似性：标准分离
            separation_strength = 1.0
        
        # 使用稀疏随机投影将输入映射到 DG 表示空间，避免相似输入产生高度重叠的 top-k。
        input_pattern = np.asarray(input_pattern, dtype=float).ravel()
        input_dim = int(min(len(input_pattern), 4096))
        if input_dim <= 0:
            return np.zeros(self.size, dtype=float)
        self._ensure_projection(input_dim)

        indices = self._projection_indices[:input_dim]
        weights = self._projection_weights[:input_dim]
        values = (input_pattern[:input_dim, None] * weights).ravel()
        flat_indices = indices.ravel()

        activation = np.zeros(self.size, dtype=float)
        np.add.at(activation, flat_indices, values)

        # 年轻神经元偏置 + 分离强度
        if len(self.young_neurons):
            activation[self.young_neurons] *= 1.5
        activation *= separation_strength

        # 添加噪声：高相似输入时加大随机性以增强分离
        noise_std = 0.15 if max_similarity > novelty_threshold else 0.05
        activation += self.rng.normal(0.0, noise_std, self.size)

        # Winner-take-all：从更大的候选池中随机抽取，降低相似性
        pool_multiplier = 6 if max_similarity > novelty_threshold else 3
        pool_size = min(self.size, max(self.active_neurons, int(self.active_neurons * pool_multiplier)))
        candidate_indices = np.argpartition(activation, -pool_size)[-pool_size:]
        if len(candidate_indices) > self.active_neurons:
            top_indices = self.rng.choice(candidate_indices, self.active_neurons, replace=False)
        else:
            top_indices = candidate_indices
        
        # 创建稀疏表示
        sparse_pattern = np.zeros(self.size)
        sparse_pattern[top_indices] = activation[top_indices]
        
        # 归一化
        if np.sum(sparse_pattern) > 0:
            sparse_pattern /= np.sum(sparse_pattern)
        
        self.activity_pattern = sparse_pattern
        self.pattern_history.append(sparse_pattern.copy())
        
        return sparse_pattern
    
    def get_neurogenesis_state(self) -> Dict[str, Any]:
        """获取神经发生状态"""
        return {
            'total_neurons': self.size,
            'young_neurons': len(self.young_neurons),
            'young_ratio': len(self.young_neurons) / self.size,
            'mean_age': np.mean(self.neuron_ages),
            'neurogenesis_rate': self.neurogenesis_rate
        }


class CA3AutoAssociative:
    """CA3自联想网络 - 模式完成和联想记忆"""
    
    def __init__(self, size: int = 300000):
        self.size = size
        
        # 从DG输入生成CA3表示：使用稀疏随机投影，避免巨型稠密矩阵导致内存错误
        self.rng = np.random.RandomState(123)
        self._dg_projector: Optional[_SparseProjector] = None
        self._dg_fanin = 32
        
        # 存储的模式
        self.stored_patterns = []
        self.pattern_associations = defaultdict(list)
        
        # 动力学参数
        self.retrieval_threshold = 0.1
        self.completion_iterations = 10
        self.learning_rate = 0.005
        
        # 状态变量
        self.activity = np.zeros(size)
        self.retrieval_success_rate = 0.0
        
        self.logger = logging.getLogger("CA3AutoAssociative")
    
    def _initialize_recurrent_weights(self) -> np.ndarray:
        """初始化稀疏的自联想权重"""
        # 创建稀疏连接矩阵
        connection_prob = 0.02  # 2%的连接概率
        weights = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            # 随机选择连接目标
            n_connections = int(self.size * connection_prob)
            targets = np.random.choice(self.size, n_connections, replace=False)
            
            for j in targets:
                if i != j:  # 无自连接
                    weights[i, j] = np.random.normal(0, 0.1)
        
        return weights

    def _ensure_dg_projector(self, input_dim: int) -> None:
        input_dim = int(max(1, input_dim))
        if self._dg_projector is not None and self._dg_projector.input_dim == input_dim:
            return
        fan_in = min(int(self._dg_fanin), input_dim)
        self._dg_projector = _SparseProjector(
            input_dim=input_dim,
            output_dim=int(self.size),
            fan_in=fan_in,
            rng=self.rng,
            weight_scale=0.1,
        )

    def encode_input(self, dg_input: np.ndarray) -> np.ndarray:
        """将DG输入编码为CA3表示（不修改内部存储）"""
        dg_vec = np.asarray(dg_input, dtype=float).reshape(-1)
        self._ensure_dg_projector(int(dg_vec.size) or 1)
        if self._dg_projector is None:
            return np.zeros(int(self.size))
        projected = self._dg_projector.project(dg_vec)
        return np.tanh(projected)
    
    def store_pattern(self, dg_input: np.ndarray, pattern_id: int) -> np.ndarray:
        """存储新模式"""
        ca3_pattern = self.encode_input(dg_input)
        
        # 存储模式
        self.stored_patterns.append((pattern_id, ca3_pattern.copy()))
        
        self.activity = ca3_pattern
        return ca3_pattern
    
    def _hebbian_update(self, pattern: np.ndarray):
        """Hebbian学习更新"""
        # 外积更新
        outer_product = np.outer(pattern, pattern)
        
        # 移除对角线（无自连接）
        np.fill_diagonal(outer_product, 0)
        
        # 更新权重
        self.recurrent_weights += self.learning_rate * outer_product
        
        # 权重衰减和归一化
        self.recurrent_weights *= 0.999
        self.recurrent_weights = np.clip(self.recurrent_weights, -1.0, 1.0)
    
    def pattern_completion(self, partial_cue: np.ndarray, 
                          noise_level: float = 0.1) -> Tuple[np.ndarray, float]:
        """模式完成检索"""
        cue = _pad_or_trim_1d(partial_cue, int(self.size))
        if not self.stored_patterns:
            self.activity = cue
            return cue, 0.0

        best_pattern: Optional[np.ndarray] = None
        best_score = -1.0
        cue_norm = float(np.linalg.norm(cue) or 1.0)
        for _, stored_pattern in self.stored_patterns:
            denom = cue_norm * float(np.linalg.norm(stored_pattern) or 1.0)
            score = abs(float(np.dot(cue, stored_pattern) / denom))
            if score > best_score:
                best_score = score
                best_pattern = stored_pattern

        activity = best_pattern.copy() if best_pattern is not None else cue
        if noise_level and noise_level > 0:
            activity = activity + self.rng.normal(0.0, float(noise_level), int(self.size))
            activity = np.tanh(activity)

        retrieval_quality = self._compute_retrieval_quality(activity)
        self.activity = activity
        return activity, retrieval_quality
    
    def _compute_retrieval_quality(self, retrieved_pattern: np.ndarray) -> float:
        """计算检索质量"""
        if not self.stored_patterns:
            return 0.0
        
        max_similarity = 0.0
        for _, stored_pattern in self.stored_patterns:
            similarity = np.corrcoef(retrieved_pattern, stored_pattern)[0, 1]
            if not np.isnan(similarity):
                max_similarity = max(max_similarity, abs(similarity))
        
        return max_similarity
    
    def create_association(self, pattern_id1: int, pattern_id2: int, 
                          association_strength: float = 1.0):
        """创建模式间关联"""
        self.pattern_associations[pattern_id1].append((pattern_id2, association_strength))
        self.pattern_associations[pattern_id2].append((pattern_id1, association_strength))
    
    def associative_retrieval(self, cue_pattern_id: int) -> List[Tuple[int, float]]:
        """基于关联的检索"""
        if cue_pattern_id in self.pattern_associations:
            return self.pattern_associations[cue_pattern_id]
        return []


class CA1TemporalSequence:
    """CA1时序编码网络"""
    
    def __init__(self, size: int = 400000):
        self.size = size
        
        # 时间细胞参数
        self.time_cells_ratio = 0.3
        self.time_cells = np.random.choice(size, int(size * self.time_cells_ratio), replace=False)
        
        # 位置细胞参数
        self.place_cells_ratio = 0.4
        self.place_cells = np.random.choice(size, int(size * self.place_cells_ratio), replace=False)
        
        # 序列缓冲区
        self.sequence_buffer = deque(maxlen=50)
        self.temporal_context = np.zeros(size)
        
        # 权重投影：避免内存上的 O(N^2) 巨型稠密矩阵
        self.rng = np.random.RandomState(123)
        self._ca3_projector: Optional[_SparseProjector] = None
        self._temporal_projector = _SparseProjector(
            input_dim=int(size),
            output_dim=int(size),
            fan_in=min(16, int(size)),
            rng=self.rng,
            weight_scale=0.05,
        )
        
        # 时间编码参数
        self.time_scales = np.logspace(0, 3, len(self.time_cells))  # 1ms到1s的时间尺度
        self.temporal_decay = 0.95
        
        # 状态变量
        self.activity = np.zeros(size)
        self.current_time = 0.0
        
        self.logger = logging.getLogger("CA1TemporalSequence")
    
    def encode_temporal_sequence(self, ca3_sequence: List[np.ndarray], 
                               time_intervals: List[float]) -> np.ndarray:
        """编码时间序列"""
        sequence_representation = np.zeros(self.size)
        
        # 重置时间上下文
        self.temporal_context = np.zeros(self.size)
        
        for i, (ca3_pattern, dt) in enumerate(zip(ca3_sequence, time_intervals)):
            # 更新时间
            self.current_time += dt
            
            # 处理CA3输入
            ca1_activity = self._process_ca3_input(ca3_pattern)
            
            # 时间细胞编码
            time_encoding = self._encode_time_interval(dt)
            ca1_activity[self.time_cells] *= time_encoding
            
            # 更新序列表示
            weight = 1.0 / (i + 1)  # 递减权重
            sequence_representation += weight * ca1_activity
            
            # 更新时间上下文
            self.temporal_context = (self.temporal_decay * self.temporal_context + 
                                   (1 - self.temporal_decay) * ca1_activity)
            
            # 存储到序列缓冲区
            self.sequence_buffer.append({
                'pattern': ca1_activity.copy(),
                'time': self.current_time,
                'interval': dt
            })
        
        self.activity = sequence_representation
        return sequence_representation
    
    def _process_ca3_input(self, ca3_pattern: np.ndarray) -> np.ndarray:
        """处理CA3输入"""
        ca3_vec = np.asarray(ca3_pattern, dtype=float).reshape(-1)
        if self._ca3_projector is None or self._ca3_projector.input_dim != int(ca3_vec.size or 1):
            fan_in = min(32, int(ca3_vec.size) or 1)
            self._ca3_projector = _SparseProjector(
                input_dim=int(ca3_vec.size) or 1,
                output_dim=int(self.size),
                fan_in=fan_in,
                rng=self.rng,
                weight_scale=0.1,
            )

        feedforward = np.tanh(
            self._ca3_projector.project(ca3_vec) if self._ca3_projector else np.zeros(int(self.size))
        )
        temporal_input = self._temporal_projector.project(self.temporal_context)
        combined = feedforward + 0.3 * temporal_input
        return np.tanh(combined)
    
    def _encode_time_interval(self, dt: float) -> np.ndarray:
        """编码时间间隔"""
        time_encoding = np.zeros(len(self.time_cells))
        
        for i, time_scale in enumerate(self.time_scales):
            # 不同时间尺度的响应
            response = np.exp(-dt / time_scale) if dt > 0 else 1.0
            time_encoding[i] = response
        
        return time_encoding
    
    def predict_next_in_sequence(self, current_pattern: np.ndarray) -> np.ndarray:
        """预测序列中的下一个模式"""
        prediction = self._temporal_projector.project(current_pattern)
        return np.tanh(prediction)
    
    def encode_spatial_location(self, position: np.ndarray, environment_size: float = 100.0) -> np.ndarray:
        """编码空间位置（位置细胞）"""
        spatial_encoding = np.zeros(len(self.place_cells))
        
        # 为每个位置细胞分配随机的感受野中心
        for i, cell_idx in enumerate(self.place_cells):
            # 使用细胞索引作为种子生成一致的感受野
            np.random.seed(cell_idx)
            field_center = np.random.uniform(0, environment_size, len(position))
            field_size = np.random.uniform(5, 20)  # 感受野大小
            
            # 计算距离和激活
            distance = np.linalg.norm(position - field_center)
            activation = np.exp(-(distance ** 2) / (2 * field_size ** 2))
            spatial_encoding[i] = activation
        
        # 重置随机种子
        np.random.seed()
        
        return spatial_encoding


class CA2SocialMemory:
    """CA2社会记忆网络"""
    
    def __init__(self, size: int = 200000):
        self.size = size
        
        # 社会记忆特异性参数
        self.social_cells_ratio = 0.6
        self.social_cells = np.random.choice(size, int(size * self.social_cells_ratio), replace=False)
        
        # 社会信息存储
        self.social_memories = {}
        self.social_relationships = defaultdict(dict)
        
        # 社会编码参数
        self.familiarity_threshold = 0.7
        self.social_learning_rate = 0.01
        
        # 状态变量
        self.activity = np.zeros(size)
        
        self.logger = logging.getLogger("CA2SocialMemory")
    
    def encode_social_interaction(self, agent_id: str, interaction_type: str, 
                                context: Dict[str, Any]) -> np.ndarray:
        """编码社会互动"""
        # 创建社会特征向量
        social_features = self._extract_social_features(agent_id, interaction_type, context)
        
        # 社会细胞编码
        social_encoding = np.zeros(self.size)
        social_encoding[self.social_cells] = social_features[:len(self.social_cells)]
        
        # 更新社会记忆
        if agent_id not in self.social_memories:
            self.social_memories[agent_id] = []
        
        self.social_memories[agent_id].append({
            'interaction_type': interaction_type,
            'context': context,
            'encoding': social_encoding.copy(),
            'timestamp': time.time()
        })
        
        # 更新社会关系
        self._update_social_relationship(agent_id, interaction_type, context)
        
        self.activity = social_encoding
        return social_encoding
    
    def _extract_social_features(self, agent_id: str, interaction_type: str, 
                               context: Dict[str, Any]) -> np.ndarray:
        """提取社会特征"""
        features = []
        
        # 代理ID编码（简化为哈希）
        agent_hash = hash(agent_id) % 1000
        features.extend([agent_hash / 1000.0])
        
        # 互动类型编码
        interaction_types = ['cooperation', 'competition', 'communication', 'conflict', 'neutral']
        interaction_encoding = [1.0 if interaction_type == itype else 0.0 for itype in interaction_types]
        features.extend(interaction_encoding)
        
        # 上下文特征
        if 'valence' in context:
            features.append(context['valence'])
        else:
            features.append(0.0)
        
        if 'intensity' in context:
            features.append(context['intensity'])
        else:
            features.append(0.5)
        
        # 填充到所需长度
        target_length = len(self.social_cells)
        while len(features) < target_length:
            features.extend(features[:min(len(features), target_length - len(features))])
        
        return np.array(features[:target_length])
    
    def _update_social_relationship(self, agent_id: str, interaction_type: str, 
                                  context: Dict[str, Any]):
        """更新社会关系"""
        if agent_id not in self.social_relationships:
            self.social_relationships[agent_id] = {
                'familiarity': 0.0,
                'trust': 0.5,
                'cooperation_history': [],
                'interaction_count': 0
            }
        
        relationship = self.social_relationships[agent_id]
        relationship['interaction_count'] += 1
        
        # 更新熟悉度
        relationship['familiarity'] = min(1.0, relationship['familiarity'] + 0.1)
        
        # 根据互动类型更新信任度
        if interaction_type == 'cooperation':
            relationship['trust'] = min(1.0, relationship['trust'] + 0.1)
        elif interaction_type == 'conflict':
            relationship['trust'] = max(0.0, relationship['trust'] - 0.1)
        
        # 记录合作历史
        if interaction_type in ['cooperation', 'conflict']:
            relationship['cooperation_history'].append({
                'type': interaction_type,
                'timestamp': time.time(),
                'context': context
            })
    
    def retrieve_social_memory(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """检索社会记忆"""
        if agent_id in self.social_memories:
            return {
                'memories': self.social_memories[agent_id],
                'relationship': self.social_relationships.get(agent_id, {}),
                'familiarity': self.social_relationships.get(agent_id, {}).get('familiarity', 0.0)
            }
        return None
    
    def get_social_network_state(self) -> Dict[str, Any]:
        """获取社会网络状态"""
        return {
            'known_agents': list(self.social_memories.keys()),
            'total_interactions': sum(len(memories) for memories in self.social_memories.values()),
            'relationship_summary': {
                agent_id: {
                    'familiarity': rel.get('familiarity', 0.0),
                    'trust': rel.get('trust', 0.5),
                    'interactions': rel.get('interaction_count', 0)
                }
                for agent_id, rel in self.social_relationships.items()
            }
        }


class Subiculum:
    """下托 - 海马输出整合"""
    
    def __init__(self, size: int = 100000):
        self.size = size
        
        # 输入/输出投影（稀疏随机，避免巨型稠密矩阵）
        self.rng = np.random.RandomState(123)
        self._ca1_projector: Optional[_SparseProjector] = None
        self._ca3_projector: Optional[_SparseProjector] = None
        self._cortical_projector = _SparseProjector(
            input_dim=int(size),
            output_dim=int(size),
            fan_in=min(16, int(size)),
            rng=self.rng,
            weight_scale=0.1,
        )
        
        # 状态变量
        self.activity = np.zeros(size)
        
        self.logger = logging.getLogger("Subiculum")
    
    def integrate_hippocampal_output(self, ca1_activity: np.ndarray, 
                                   ca3_activity: Optional[np.ndarray] = None) -> np.ndarray:
        """整合海马输出"""
        ca1_vec = np.asarray(ca1_activity, dtype=float).reshape(-1)
        if self._ca1_projector is None or self._ca1_projector.input_dim != int(ca1_vec.size or 1):
            self._ca1_projector = _SparseProjector(
                input_dim=int(ca1_vec.size) or 1,
                output_dim=int(self.size),
                fan_in=min(32, int(ca1_vec.size) or 1),
                rng=self.rng,
                weight_scale=0.1,
            )
        ca1_input = self._ca1_projector.project(ca1_vec) if self._ca1_projector else np.zeros(int(self.size))
        
        # 处理CA3输入（如果提供）
        ca3_input = np.zeros(int(self.size))
        if ca3_activity is not None:
            ca3_vec = np.asarray(ca3_activity, dtype=float).reshape(-1)
            if self._ca3_projector is None or self._ca3_projector.input_dim != int(ca3_vec.size or 1):
                self._ca3_projector = _SparseProjector(
                    input_dim=int(ca3_vec.size) or 1,
                    output_dim=int(self.size),
                    fan_in=min(32, int(ca3_vec.size) or 1),
                    rng=self.rng,
                    weight_scale=0.05,
                )
            ca3_input = self._ca3_projector.project(ca3_vec) if self._ca3_projector else ca3_input
        
        # 整合输入
        integrated = np.tanh(ca1_input + 0.3 * ca3_input)
        
        self.activity = integrated
        return integrated
    
    def generate_cortical_output(self, integrated_activity: Optional[np.ndarray] = None) -> np.ndarray:
        """生成皮层输出"""
        if integrated_activity is None:
            integrated_activity = self.activity
        
        cortical_output = self._cortical_projector.project(integrated_activity)
        return np.tanh(cortical_output)


class EnhancedHippocampalSystem:
    """增强的海马记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各个海马区域
        self.dentate_gyrus = DentateGyrus(
            size=config.get('dg_size', 1000000),
            sparsity=config.get('dg_sparsity', 0.005)
        )
        
        self.ca3 = CA3AutoAssociative(
            size=config.get('ca3_size', 300000)
        )
        
        self.ca1 = CA1TemporalSequence(
            size=config.get('ca1_size', 400000)
        )
        
        self.ca2 = CA2SocialMemory(
            size=config.get('ca2_size', 200000)
        )
        
        self.subiculum = Subiculum(
            size=config.get('subiculum_size', 100000)
        )
        
        # 记忆存储
        self.memory_traces: List[EnhancedMemoryTrace] = []
        self.trace_counter = 0
        
        # 记忆索引
        self.memory_index = {memory_type: [] for memory_type in MemoryType}
        self.spatial_index = {}
        self.temporal_index = defaultdict(list)
        self.social_index = defaultdict(list)
        
        # 系统状态
        self.current_phase = MemoryPhase.ENCODING
        self.consolidation_scheduler = []
        
        # 性能监控
        self.encoding_success_rate = 0.0
        self.retrieval_success_rate = 0.0
        self.consolidation_progress = 0.0
        
        self.logger = logging.getLogger("EnhancedHippocampalSystem")
        self.logger.info("增强海马记忆系统初始化完成")
    
    def encode_memory(self, content: Dict[str, Any], memory_type: MemoryType,
                     context: Optional[Dict[str, Any]] = None) -> int:
        """编码记忆"""
        self.current_phase = MemoryPhase.ENCODING
        
        # 创建输入模式
        input_pattern = self._create_input_pattern(content, memory_type)
        
        # DG模式分离
        dg_code = self.dentate_gyrus.pattern_separation(input_pattern)
        
        # CA3存储和联想
        ca3_pattern = self.ca3.store_pattern(dg_code, self.trace_counter)
        
        # CA1时序编码
        ca1_sequence = self.ca1.encode_temporal_sequence([ca3_pattern], [1.0])
        
        # CA2社会记忆（如果适用）
        ca2_social = np.array([])
        if memory_type == MemoryType.SOCIAL and context:
            agent_id = context.get('agent_id', 'unknown')
            interaction_type = context.get('interaction_type', 'neutral')
            ca2_social = self.ca2.encode_social_interaction(agent_id, interaction_type, context)
        
        # Subiculum整合
        subiculum_output = self.subiculum.integrate_hippocampal_output(ca1_sequence, ca3_pattern)
        
        # 创建记忆痕迹
        trace = EnhancedMemoryTrace(
            trace_id=self.trace_counter,
            memory_type=memory_type,
            content=content,
            timestamp=time.time(),
            dg_sparse_code=dg_code,
            ca3_pattern=ca3_pattern,
            ca1_sequence=ca1_sequence,
            ca2_social_code=ca2_social,
            subiculum_output=subiculum_output
        )
        
        # 添加上下文信息
        if context:
            if 'spatial_location' in context:
                trace.spatial_context = np.array(context['spatial_location'])
            if 'emotional_valence' in context:
                trace.emotional_valence = context['emotional_valence']
            if 'social_context' in context:
                trace.social_context = context['social_context']
        
        # 存储记忆痕迹
        self.memory_traces.append(trace)
        
        # 更新索引
        self._update_memory_indices(trace)
        
        # 调度巩固
        self._schedule_consolidation(trace)
        
        self.trace_counter += 1
        self.logger.info(f"编码记忆 {self.trace_counter-1}: {memory_type.value}")
        
        return self.trace_counter - 1
    
    def _create_input_pattern(self, content: Dict[str, Any], memory_type: MemoryType) -> np.ndarray:
        """创建输入模式"""
        pattern_components = []
        
        # 基于内容类型创建特征
        for key, value in content.items():
            if isinstance(value, (int, float)):
                pattern_components.append(float(value))
            elif isinstance(value, str):
                # 简单字符串哈希编码
                hash_val = hash(value) % 1000
                pattern_components.append(hash_val / 1000.0)
            elif isinstance(value, (list, np.ndarray)):
                if isinstance(value, list):
                    value = np.array(value)
                pattern_components.extend(value.flatten()[:100])  # 限制长度
        
        # 记忆类型编码
        type_encoding = [0.0] * len(MemoryType)
        type_encoding[list(MemoryType).index(memory_type)] = 1.0
        pattern_components.extend(type_encoding)
        
        # 时间戳编码
        time_encoding = [time.time() % 86400 / 86400]  # 一天内的时间比例
        pattern_components.extend(time_encoding)
        
        # 转换为numpy数组并归一化
        pattern = np.array(pattern_components)
        if len(pattern) > 1000:
            pattern = pattern[:1000]
        elif len(pattern) < 1000:
            pattern = np.pad(pattern, (0, 1000 - len(pattern)))
        
        # 归一化
        if np.std(pattern) > 0:
            pattern = (pattern - np.mean(pattern)) / np.std(pattern)
        
        return pattern
    
    def retrieve_memory(self, cue: Dict[str, Any], memory_type: Optional[MemoryType] = None,
                       context: Optional[Dict[str, Any]] = None) -> List[EnhancedMemoryTrace]:
        """检索记忆"""
        self.current_phase = MemoryPhase.RETRIEVAL
        
        # 创建检索线索
        cue_pattern = self._create_input_pattern(cue, memory_type or MemoryType.EPISODIC)
        
        # 对检索线索使用与编码相同的 DG->CA3 路径，保证特征空间一致
        dg_code = self.dentate_gyrus.pattern_separation(cue_pattern)
        ca3_cue = self.ca3.encode_input(dg_code)
        completed_pattern, retrieval_quality = self.ca3.pattern_completion(ca3_cue)
        
        # 查找匹配的记忆痕迹
        candidates = []
        
        # 基于记忆类型筛选
        if memory_type:
            search_traces = self.memory_index[memory_type]
        else:
            search_traces = list(range(len(self.memory_traces)))
        
        # 计算相似性
        for trace_id in search_traces:
            if trace_id < len(self.memory_traces):
                trace = self.memory_traces[trace_id]
                
                # CA3模式相似性
                ca3_similarity = np.corrcoef(completed_pattern, trace.ca3_pattern)[0, 1]
                if np.isnan(ca3_similarity):
                    ca3_similarity = 0.0
                
                # 上下文匹配
                context_match = self._compute_context_match(trace, context)
                
                # 综合相似性
                total_similarity = 0.7 * abs(ca3_similarity) + 0.3 * context_match
                
                candidates.append((trace, total_similarity))
                if total_similarity > 0.3:  # 检索阈值
                    trace.retrieval_count += 1
                    trace.last_accessed = time.time()
        
        # 按相似性排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回最佳匹配
        retrieved_traces = [trace for trace, _ in candidates[:5]]  # 返回前5个
        
        self.logger.info(f"检索到 {len(retrieved_traces)} 个记忆")
        return retrieved_traces
    
    def _compute_context_match(self, trace: EnhancedMemoryTrace, 
                             context: Optional[Dict[str, Any]]) -> float:
        """计算上下文匹配度"""
        if not context:
            return 0.5  # 中性匹配
        
        match_score = 0.0
        match_count = 0
        
        # 空间上下文匹配
        if 'spatial_location' in context and trace.spatial_context is not None:
            spatial_distance = np.linalg.norm(
                np.array(context['spatial_location']) - trace.spatial_context
            )
            spatial_match = np.exp(-spatial_distance / 10.0)  # 空间衰减
            match_score += spatial_match
            match_count += 1
        
        # 情绪上下文匹配
        if 'emotional_valence' in context:
            emotion_diff = abs(context['emotional_valence'] - trace.emotional_valence)
            emotion_match = 1.0 - emotion_diff
            match_score += emotion_match
            match_count += 1
        
        # 社会上下文匹配
        if 'social_context' in context and trace.social_context:
            # 简化的社会匹配
            social_match = 0.5  # 默认匹配
            if context['social_context'].get('agent_id') == trace.social_context.get('agent_id'):
                social_match = 1.0
            match_score += social_match
            match_count += 1
        
        return match_score / max(match_count, 1)
    
    def consolidate_memories(self, consolidation_type: str = 'systems'):
        """巩固记忆"""
        self.current_phase = MemoryPhase.CONSOLIDATION
        
        if consolidation_type == 'systems':
            # 系统巩固：将海马记忆转移到皮层
            self._systems_consolidation()
        elif consolidation_type == 'synaptic':
            # 突触巩固：强化突触连接
            self._synaptic_consolidation()
        
        self.logger.info(f"执行 {consolidation_type} 巩固")
    
    def _systems_consolidation(self):
        """系统巩固"""
        # 选择需要巩固的记忆
        consolidation_candidates = [
            trace for trace in self.memory_traces
            if trace.consolidation_level < 1.0 and 
            (time.time() - trace.timestamp) > 3600  # 1小时后开始巩固
        ]
        
        # 按重要性排序
        consolidation_candidates.sort(
            key=lambda t: t.encoding_strength + t.retrieval_count * 0.1,
            reverse=True
        )
        
        # 巩固前10个记忆
        for trace in consolidation_candidates[:10]:
            # 重放记忆
            self._replay_memory(trace)
            
            # 更新巩固水平
            trace.consolidation_level = min(1.0, trace.consolidation_level + 0.1)
    
    def _synaptic_consolidation(self):
        """突触巩固"""
        # 强化最近编码的记忆的突触连接
        recent_traces = [
            trace for trace in self.memory_traces
            if (time.time() - trace.timestamp) < 3600  # 1小时内
        ]
        
        for trace in recent_traces:
            # 增强编码强度
            trace.encoding_strength = min(2.0, trace.encoding_strength * 1.1)
    
    def _replay_memory(self, trace: EnhancedMemoryTrace):
        """重放记忆"""
        # 在海马回路中重新激活记忆模式
        self.ca3.activity = trace.ca3_pattern
        self.ca1.activity = trace.ca1_sequence
        
        if len(trace.ca2_social_code) > 0:
            self.ca2.activity = trace.ca2_social_code
        
        self.subiculum.activity = trace.subiculum_output
    
    def _update_memory_indices(self, trace: EnhancedMemoryTrace):
        """更新记忆索引"""
        # 类型索引
        self.memory_index[trace.memory_type].append(trace.trace_id)
        
        # 空间索引
        if trace.spatial_context is not None:
            spatial_key = tuple(np.round(trace.spatial_context, 1))
            if spatial_key not in self.spatial_index:
                self.spatial_index[spatial_key] = []
            self.spatial_index[spatial_key].append(trace.trace_id)
        
        # 时间索引
        time_key = int(trace.timestamp // 3600)  # 小时级别
        self.temporal_index[time_key].append(trace.trace_id)
        
        # 社会索引
        if trace.social_context:
            agent_id = trace.social_context.get('agent_id')
            if agent_id:
                self.social_index[agent_id].append(trace.trace_id)
    
    def _schedule_consolidation(self, trace: EnhancedMemoryTrace):
        """调度巩固"""
        # 根据记忆重要性调度巩固时间
        importance = trace.encoding_strength + abs(trace.emotional_valence)
        
        if importance > 1.5:
            # 高重要性：快速巩固
            consolidation_delay = 1800  # 30分钟
        else:
            # 标准巩固
            consolidation_delay = 7200  # 2小时
        
        consolidation_time = trace.timestamp + consolidation_delay
        self.consolidation_scheduler.append((consolidation_time, trace.trace_id))
    
    def step(self, dt: float):
        """系统步进更新"""
        # 更新神经发生
        self.dentate_gyrus.neurogenesis_update(dt)
        
        # 检查巩固调度
        current_time = time.time()
        due_consolidations = [
            (t, trace_id) for t, trace_id in self.consolidation_scheduler
            if t <= current_time
        ]
        
        for _, trace_id in due_consolidations:
            if trace_id < len(self.memory_traces):
                trace = self.memory_traces[trace_id]
                self._replay_memory(trace)
                trace.consolidation_level = min(1.0, trace.consolidation_level + 0.05)
        
        # 移除已处理的巩固任务
        self.consolidation_scheduler = [
            (t, trace_id) for t, trace_id in self.consolidation_scheduler
            if t > current_time
        ]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.memory_traces:
            return {'total_memories': 0}
        
        # 基本统计
        total_memories = len(self.memory_traces)
        memory_type_counts = {
            memory_type.value: len(traces)
            for memory_type, traces in self.memory_index.items()
        }
        
        # 巩固统计
        consolidation_levels = [trace.consolidation_level for trace in self.memory_traces]
        mean_consolidation = np.mean(consolidation_levels)
        
        # 检索统计
        retrieval_counts = [trace.retrieval_count for trace in self.memory_traces]
        mean_retrievals = np.mean(retrieval_counts)
        
        # 神经发生统计
        neurogenesis_stats = self.dentate_gyrus.get_neurogenesis_state()
        
        # 社会网络统计
        social_stats = self.ca2.get_social_network_state()
        
        return {
            'total_memories': total_memories,
            'memory_type_distribution': memory_type_counts,
            'consolidation': {
                'mean_level': mean_consolidation,
                'pending_tasks': len(self.consolidation_scheduler)
            },
            'retrieval': {
                'mean_count': mean_retrievals,
                'success_rate': self.retrieval_success_rate
            },
            'neurogenesis': neurogenesis_stats,
            'social_network': social_stats,
            'system_phase': self.current_phase.value
        }
    
    def reset(self):
        """重置系统"""
        self.memory_traces.clear()
        self.trace_counter = 0
        self.memory_index = {memory_type: [] for memory_type in MemoryType}
        self.spatial_index.clear()
        self.temporal_index.clear()
        self.social_index.clear()
        self.consolidation_scheduler.clear()
        
        # 重置各区域
        self.dentate_gyrus.activity_pattern = np.zeros(self.dentate_gyrus.size)
        self.ca3.activity = np.zeros(self.ca3.size)
        self.ca1.activity = np.zeros(self.ca1.size)
        self.ca2.activity = np.zeros(self.ca2.size)
        self.subiculum.activity = np.zeros(self.subiculum.size)
        
        self.logger.info("海马记忆系统已重置")
