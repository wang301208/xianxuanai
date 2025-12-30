"""
情节记忆系统

实现详细的情节记忆编码、存储和检索机制：
- 时空上下文绑定
- 情节边界检测
- 自传体记忆组织
- 情节重构和想象
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import logging

from .enhanced_hippocampal_system import EnhancedMemoryTrace, MemoryType


class EpisodicComponent(Enum):
    """情节记忆组成要素"""
    WHAT = "what"           # 事件内容
    WHERE = "where"         # 空间位置
    WHEN = "when"           # 时间信息
    WHO = "who"             # 参与者
    WHY = "why"             # 动机/原因
    HOW = "how"             # 方式/过程


@dataclass
class EpisodicEvent:
    """情节事件"""
    event_id: int
    event_type: str
    content: Dict[str, Any]
    
    # 时空信息
    spatial_location: np.ndarray
    temporal_timestamp: float
    duration: float
    
    # 参与者信息
    participants: List[str] = field(default_factory=list)
    self_involvement: float = 1.0  # 自我参与度
    
    # 上下文信息
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    emotional_context: Dict[str, float] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)
    
    # 事件结构
    sub_events: List[int] = field(default_factory=list)
    causal_relations: List[Tuple[int, str]] = field(default_factory=list)  # (event_id, relation_type)
    
    # 记忆质量
    vividness: float = 1.0
    confidence: float = 1.0
    completeness: float = 1.0


@dataclass
class EpisodicEpisode:
    """情节片段（多个相关事件的集合）"""
    episode_id: int
    title: str
    events: List[int]  # event_ids
    
    # 时空边界
    start_time: float
    end_time: float
    spatial_boundary: Optional[np.ndarray] = None
    
    # 主题和意义
    theme: str = ""
    significance: float = 0.5
    emotional_tone: Dict[str, float] = field(default_factory=dict)
    
    # 叙事结构
    narrative_structure: Dict[str, Any] = field(default_factory=dict)
    key_moments: List[int] = field(default_factory=list)


class EpisodicBoundaryDetector:
    """情节边界检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 边界检测参数
        self.temporal_threshold = config.get('temporal_threshold', 300)  # 5分钟
        self.spatial_threshold = config.get('spatial_threshold', 10.0)   # 10米
        self.context_threshold = config.get('context_threshold', 0.3)    # 上下文变化阈值
        
        # 特征权重
        self.feature_weights = {
            'temporal': 0.3,
            'spatial': 0.3,
            'contextual': 0.2,
            'semantic': 0.2
        }
        
        # 状态追踪
        self.current_context = {}
        self.context_history = deque(maxlen=100)
        
        self.logger = logging.getLogger("EpisodicBoundaryDetector")
    
    def detect_boundary(self, current_event: EpisodicEvent, 
                       previous_events: List[EpisodicEvent]) -> Tuple[bool, float]:
        """检测情节边界"""
        if not previous_events:
            return True, 1.0  # 第一个事件总是边界
        
        last_event = previous_events[-1]
        
        # 计算各维度的变化
        temporal_change = self._compute_temporal_change(current_event, last_event)
        spatial_change = self._compute_spatial_change(current_event, last_event)
        contextual_change = self._compute_contextual_change(current_event, last_event)
        semantic_change = self._compute_semantic_change(current_event, last_event)
        
        # 加权组合
        boundary_score = (
            self.feature_weights['temporal'] * temporal_change +
            self.feature_weights['spatial'] * spatial_change +
            self.feature_weights['contextual'] * contextual_change +
            self.feature_weights['semantic'] * semantic_change
        )
        
        # 判断是否为边界
        is_boundary = boundary_score > 0.5
        
        return is_boundary, boundary_score
    
    def _compute_temporal_change(self, current: EpisodicEvent, previous: EpisodicEvent) -> float:
        """计算时间变化"""
        time_gap = current.temporal_timestamp - previous.temporal_timestamp
        return min(1.0, time_gap / self.temporal_threshold)
    
    def _compute_spatial_change(self, current: EpisodicEvent, previous: EpisodicEvent) -> float:
        """计算空间变化"""
        if current.spatial_location is None or previous.spatial_location is None:
            return 0.0
        
        distance = np.linalg.norm(current.spatial_location - previous.spatial_location)
        return min(1.0, distance / self.spatial_threshold)
    
    def _compute_contextual_change(self, current: EpisodicEvent, previous: EpisodicEvent) -> float:
        """计算上下文变化"""
        # 环境上下文变化
        env_change = self._compute_context_similarity(
            current.environmental_context, 
            previous.environmental_context
        )
        
        # 社会上下文变化
        social_change = self._compute_context_similarity(
            current.social_context,
            previous.social_context
        )
        
        return (env_change + social_change) / 2.0
    
    def _compute_semantic_change(self, current: EpisodicEvent, previous: EpisodicEvent) -> float:
        """计算语义变化"""
        # 事件类型变化
        type_change = 1.0 if current.event_type != previous.event_type else 0.0
        
        # 参与者变化
        current_participants = set(current.participants)
        previous_participants = set(previous.participants)
        
        if current_participants or previous_participants:
            participant_overlap = len(current_participants & previous_participants)
            participant_union = len(current_participants | previous_participants)
            participant_change = 1.0 - (participant_overlap / participant_union)
        else:
            participant_change = 0.0
        
        return (type_change + participant_change) / 2.0
    
    def _compute_context_similarity(self, context1: Dict[str, Any], 
                                  context2: Dict[str, Any]) -> float:
        """计算上下文相似性"""
        if not context1 and not context2:
            return 0.0
        
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 0.0
        
        differences = 0
        for key in all_keys:
            val1 = context1.get(key)
            val2 = context2.get(key)
            
            if val1 != val2:
                differences += 1
        
        return differences / len(all_keys)


class EpisodicMemorySystem:
    """情节记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 组件初始化
        self.boundary_detector = EpisodicBoundaryDetector(config.get('boundary_detection', {}))
        
        # 存储结构
        self.events: Dict[int, EpisodicEvent] = {}
        self.episodes: Dict[int, EpisodicEpisode] = {}
        self.event_counter = 0
        self.episode_counter = 0
        
        # 索引结构
        self.temporal_index = defaultdict(list)  # timestamp -> event_ids
        self.spatial_index = defaultdict(list)   # spatial_region -> event_ids
        self.participant_index = defaultdict(list)  # participant -> event_ids
        self.type_index = defaultdict(list)      # event_type -> event_ids
        
        # 当前情节状态
        self.current_episode: Optional[EpisodicEpisode] = None
        self.current_episode_events: List[int] = []
        
        # 自传体记忆组织
        self.life_periods = {}  # period_name -> (start_time, end_time, event_ids)
        self.significant_events = []  # 重要事件列表
        
        # 重构和想象
        self.reconstruction_cache = {}
        self.imagination_templates = {}
        
        self.logger = logging.getLogger("EpisodicMemorySystem")
    
    def encode_event(self, event_content: Dict[str, Any], 
                    spatial_location: Optional[np.ndarray] = None,
                    participants: Optional[List[str]] = None,
                    context: Optional[Dict[str, Any]] = None) -> int:
        """编码情节事件"""
        # 创建事件对象
        event = EpisodicEvent(
            event_id=self.event_counter,
            event_type=event_content.get('type', 'general'),
            content=event_content,
            spatial_location=spatial_location if spatial_location is not None else np.zeros(3),
            temporal_timestamp=time.time(),
            duration=event_content.get('duration', 1.0),
            participants=participants or []
        )
        
        # 添加上下文信息
        if context:
            event.environmental_context = context.get('environment', {})
            event.emotional_context = context.get('emotion', {})
            event.social_context = context.get('social', {})
        
        # 检测情节边界
        previous_events = list(self.events.values())[-10:]  # 最近10个事件
        is_boundary, boundary_score = self.boundary_detector.detect_boundary(event, previous_events)
        
        # 处理情节边界
        if is_boundary and self.current_episode is not None:
            # 结束当前情节
            self._finalize_current_episode()
        
        if is_boundary or self.current_episode is None:
            # 开始新情节
            self._start_new_episode(event)
        
        # 添加事件到当前情节
        self.current_episode_events.append(event.event_id)

        # 同步当前情节到 episodes（允许“进行中”的情节可被观察到）
        if self.current_episode is not None:
            self.current_episode.events = self.current_episode_events.copy()
            self.current_episode.end_time = event.temporal_timestamp
            self.episodes[self.current_episode.episode_id] = self.current_episode
        
        # 存储事件
        self.events[event.event_id] = event
        
        # 更新索引
        self._update_event_indices(event)
        
        # 检测因果关系
        self._detect_causal_relations(event, previous_events)
        
        self.event_counter += 1
        self.logger.info(f"编码事件 {event.event_id}: {event.event_type}")
        
        return event.event_id
    
    def _start_new_episode(self, initial_event: EpisodicEvent):
        """开始新情节"""
        episode = EpisodicEpisode(
            episode_id=self.episode_counter,
            title=f"Episode_{self.episode_counter}",
            events=[],
            start_time=initial_event.temporal_timestamp,
            end_time=initial_event.temporal_timestamp,
            spatial_boundary=initial_event.spatial_location.copy() if initial_event.spatial_location is not None else None
        )
        
        self.current_episode = episode
        self.current_episode_events = []
        self.episode_counter += 1
        
        self.logger.info(f"开始新情节 {episode.episode_id}")
    
    def _finalize_current_episode(self):
        """完成当前情节"""
        if self.current_episode is None:
            return
        
        # 设置情节事件
        self.current_episode.events = self.current_episode_events.copy()
        
        # 计算情节结束时间
        if self.current_episode_events:
            last_event_id = self.current_episode_events[-1]
            if last_event_id in self.events:
                self.current_episode.end_time = self.events[last_event_id].temporal_timestamp
        
        # 生成情节标题和主题
        self._generate_episode_metadata(self.current_episode)
        
        # 存储情节
        self.episodes[self.current_episode.episode_id] = self.current_episode
        
        self.logger.info(f"完成情节 {self.current_episode.episode_id}: {self.current_episode.title}")
        
        self.current_episode = None
        self.current_episode_events = []
    
    def _generate_episode_metadata(self, episode: EpisodicEpisode):
        """生成情节元数据"""
        if not episode.events:
            return
        
        # 收集事件信息
        event_types = []
        participants = set()
        
        for event_id in episode.events:
            if event_id in self.events:
                event = self.events[event_id]
                event_types.append(event.event_type)
                participants.update(event.participants)
        
        # 生成标题
        if event_types:
            dominant_type = max(set(event_types), key=event_types.count)
            episode.title = f"{dominant_type.title()} Episode"
        
        # 生成主题
        if len(participants) > 1:
            episode.theme = "social_interaction"
        elif len(set(event_types)) > 3:
            episode.theme = "complex_activity"
        else:
            episode.theme = "routine_activity"
        
        # 计算重要性
        episode.significance = self._compute_episode_significance(episode)
    
    def _compute_episode_significance(self, episode: EpisodicEpisode) -> float:
        """计算情节重要性"""
        significance = 0.0
        
        # 基于事件数量
        significance += min(1.0, len(episode.events) / 10.0) * 0.3
        
        # 基于持续时间
        duration = episode.end_time - episode.start_time
        significance += min(1.0, duration / 3600.0) * 0.2  # 1小时为满分
        
        return min(1.0, significance)
    
    def _update_event_indices(self, event: EpisodicEvent):
        """更新事件索引"""
        # 时间索引
        time_key = int(event.temporal_timestamp // 3600)  # 小时级别
        self.temporal_index[time_key].append(event.event_id)
        
        # 空间索引
        if event.spatial_location is not None:
            spatial_key = tuple(np.round(event.spatial_location, 1))
            self.spatial_index[spatial_key].append(event.event_id)
        
        # 参与者索引
        for participant in event.participants:
            self.participant_index[participant].append(event.event_id)
        
        # 类型索引
        self.type_index[event.event_type].append(event.event_id)
    
    def _detect_causal_relations(self, current_event: EpisodicEvent, 
                               previous_events: List[EpisodicEvent]):
        """检测因果关系"""
        # 简化的因果关系检测
        for prev_event in previous_events[-5:]:  # 检查最近5个事件
            time_diff = current_event.temporal_timestamp - prev_event.temporal_timestamp
            
            # 时间窗口内的事件可能有因果关系
            if 0 < time_diff < 300:  # 5分钟内
                # 检查空间接近性
                if (current_event.spatial_location is not None and 
                    prev_event.spatial_location is not None):
                    distance = np.linalg.norm(
                        current_event.spatial_location - prev_event.spatial_location
                    )
                    
                    if distance < 5.0:  # 5米内
                        # 添加因果关系
                        current_event.causal_relations.append(
                            (prev_event.event_id, "temporal_spatial")
                        )
    
    def retrieve_events(self, query: Dict[str, Any]) -> List[EpisodicEvent]:
        """检索事件"""
        candidate_events = set()
        
        # 基于不同维度检索
        if 'time_range' in query:
            start_time, end_time = query['time_range']
            for time_key in self.temporal_index:
                if start_time <= time_key * 3600 <= end_time:
                    candidate_events.update(self.temporal_index[time_key])
        
        if 'participants' in query:
            for participant in query['participants']:
                if participant in self.participant_index:
                    candidate_events.update(self.participant_index[participant])
        
        if 'event_type' in query:
            event_type = query['event_type']
            if event_type in self.type_index:
                candidate_events.update(self.type_index[event_type])
        
        # 返回事件对象
        return [self.events[event_id] for event_id in candidate_events if event_id in self.events]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            'total_events': len(self.events),
            'total_episodes': len(self.episodes),
            'current_episode_events': len(self.current_episode_events),
            'significant_events': len(self.significant_events),
            'life_periods': len(self.life_periods)
        }
