"""
记忆巩固机制

实现不同类型的记忆巩固过程：
- 突触巩固（数小时内）
- 系统巩固（数天到数年）
- 重新巩固（检索后的再巩固）
- 睡眠依赖的巩固
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
import logging

from .enhanced_hippocampal_system import EnhancedMemoryTrace, MemoryType, MemoryPhase


class ConsolidationType(Enum):
    """巩固类型"""
    SYNAPTIC = "synaptic"           # 突触巩固
    SYSTEMS = "systems"             # 系统巩固
    RECONSOLIDATION = "reconsolidation"  # 重新巩固
    SLEEP_DEPENDENT = "sleep_dependent"  # 睡眠依赖巩固


class SleepStage(Enum):
    """睡眠阶段"""
    WAKE = "wake"
    N1 = "n1"                       # 浅睡眠1期
    N2 = "n2"                       # 浅睡眠2期
    N3 = "n3"                       # 深睡眠（慢波睡眠）
    REM = "rem"                     # 快速眼动睡眠


@dataclass
class ConsolidationEvent:
    """巩固事件"""
    trace_id: int
    consolidation_type: ConsolidationType
    start_time: float
    duration: float
    strength: float
    sleep_stage: Optional[SleepStage] = None
    completed: bool = False


class SynapticConsolidation:
    """突触巩固机制"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 巩固参数
        self.consolidation_window = config.get('synaptic_window', 3600)  # 1小时
        self.protein_synthesis_delay = config.get('protein_delay', 1800)  # 30分钟
        self.consolidation_rate = config.get('synaptic_rate', 0.1)
        
        # 分子机制参数
        self.creb_activation_threshold = 0.5
        self.protein_synthesis_rate = 0.05
        self.ltp_maintenance_factor = 1.2
        
        # 状态追踪
        self.active_consolidations: List[ConsolidationEvent] = []
        self.protein_levels = {}  # trace_id -> protein_level
        
        self.logger = logging.getLogger("SynapticConsolidation")
    
    def initiate_consolidation(self, trace: EnhancedMemoryTrace) -> ConsolidationEvent:
        """启动突触巩固"""
        # 检查是否满足巩固条件
        if not self._check_consolidation_conditions(trace):
            return None
        
        # 创建巩固事件
        event = ConsolidationEvent(
            trace_id=trace.trace_id,
            consolidation_type=ConsolidationType.SYNAPTIC,
            start_time=time.time(),
            duration=self.consolidation_window,
            strength=trace.encoding_strength
        )
        
        self.active_consolidations.append(event)
        self.protein_levels[trace.trace_id] = 0.0
        
        self.logger.info(f"启动突触巩固: 记忆 {trace.trace_id}")
        return event
    
    def _check_consolidation_conditions(self, trace: EnhancedMemoryTrace) -> bool:
        """检查巩固条件"""
        # 检查编码强度
        if trace.encoding_strength < self.creb_activation_threshold:
            return False
        
        # 检查时间窗口
        time_since_encoding = time.time() - trace.timestamp
        if time_since_encoding > self.consolidation_window:
            return False
        
        # 检查是否已在巩固中
        for event in self.active_consolidations:
            if event.trace_id == trace.trace_id and not event.completed:
                return False
        
        return True
    
    def update_consolidation(self, dt: float):
        """更新巩固过程"""
        current_time = time.time()
        
        for event in self.active_consolidations:
            if event.completed:
                continue
            
            elapsed_time = current_time - event.start_time
            
            # 检查是否完成
            if elapsed_time >= event.duration:
                event.completed = True
                self.logger.info(f"突触巩固完成: 记忆 {event.trace_id}")
                continue
            
            # 更新蛋白质合成
            if elapsed_time > self.protein_synthesis_delay:
                if event.trace_id in self.protein_levels:
                    self.protein_levels[event.trace_id] += self.protein_synthesis_rate * dt
                    self.protein_levels[event.trace_id] = min(1.0, self.protein_levels[event.trace_id])
    
    def apply_consolidation_effects(self, trace: EnhancedMemoryTrace) -> EnhancedMemoryTrace:
        """应用巩固效果"""
        if trace.trace_id not in self.protein_levels:
            return trace
        
        protein_level = self.protein_levels[trace.trace_id]
        
        # 增强记忆强度
        consolidation_boost = protein_level * self.ltp_maintenance_factor
        trace.encoding_strength *= (1.0 + consolidation_boost)
        trace.encoding_strength = min(2.0, trace.encoding_strength)
        
        # 更新巩固水平
        trace.consolidation_level = max(trace.consolidation_level, protein_level)
        
        return trace
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """获取巩固状态"""
        active_count = sum(1 for event in self.active_consolidations if not event.completed)
        completed_count = sum(1 for event in self.active_consolidations if event.completed)
        
        return {
            'active_consolidations': active_count,
            'completed_consolidations': completed_count,
            'protein_synthesis_active': len(self.protein_levels),
            'average_protein_level': np.mean(list(self.protein_levels.values())) if self.protein_levels else 0.0
        }


class SystemsConsolidation:
    """系统巩固机制"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 巩固时间尺度
        self.consolidation_phases = {
            'early': 24 * 3600,      # 1天
            'intermediate': 7 * 24 * 3600,  # 1周
            'late': 30 * 24 * 3600   # 1个月
        }
        
        # 海马-皮层转移参数
        self.transfer_rate = config.get('transfer_rate', 0.01)
        self.hippocampal_decay_rate = config.get('hippocampal_decay', 0.005)
        self.cortical_strengthening_rate = config.get('cortical_strengthening', 0.02)
        
        # 记忆类型特异性参数
        self.type_consolidation_rates = {
            MemoryType.EPISODIC: 0.8,      # 情节记忆慢巩固
            MemoryType.SEMANTIC: 1.2,      # 语义记忆快巩固
            MemoryType.SPATIAL: 1.0,       # 空间记忆标准巩固
            MemoryType.EMOTIONAL: 1.5,     # 情绪记忆快速巩固
            MemoryType.SOCIAL: 0.9         # 社会记忆中等巩固
        }
        
        # 状态追踪
        self.consolidation_progress = {}  # trace_id -> progress
        self.cortical_representations = {}  # trace_id -> cortical_pattern
        
        self.logger = logging.getLogger("SystemsConsolidation")
    
    def initiate_systems_consolidation(self, trace: EnhancedMemoryTrace) -> bool:
        """启动系统巩固"""
        # 检查是否已开始系统巩固
        if trace.trace_id in self.consolidation_progress:
            return False
        
        # 检查突触巩固是否完成
        if trace.consolidation_level < 0.5:
            return False
        
        # 初始化系统巩固
        self.consolidation_progress[trace.trace_id] = {
            'phase': 'early',
            'progress': 0.0,
            'start_time': time.time(),
            'hippocampal_strength': 1.0,
            'cortical_strength': 0.0
        }
        
        # 创建初始皮层表示
        self.cortical_representations[trace.trace_id] = self._create_cortical_representation(trace)
        
        self.logger.info(f"启动系统巩固: 记忆 {trace.trace_id}")
        return True
    
    def _create_cortical_representation(self, trace: EnhancedMemoryTrace) -> np.ndarray:
        """创建皮层表示"""
        # 基于海马表示创建皮层模式
        hippocampal_pattern = np.concatenate([
            trace.ca3_pattern[:1000] if len(trace.ca3_pattern) > 0 else np.zeros(1000),
            trace.ca1_sequence[:1000] if len(trace.ca1_sequence) > 0 else np.zeros(1000)
        ])
        
        # 转换为皮层模式（更分布式、更稳定）
        cortical_pattern = np.tanh(hippocampal_pattern * 0.5)
        
        # 添加皮层特异性噪声
        cortical_pattern += np.random.normal(0, 0.1, len(cortical_pattern))
        
        return cortical_pattern
    
    def update_systems_consolidation(self, dt: float):
        """更新系统巩固"""
        current_time = time.time()
        
        for trace_id, progress_info in self.consolidation_progress.items():
            elapsed_time = current_time - progress_info['start_time']
            
            # 确定当前阶段
            current_phase = self._determine_consolidation_phase(elapsed_time)
            progress_info['phase'] = current_phase
            
            # 计算巩固进度
            phase_duration = self.consolidation_phases[current_phase]
            phase_progress = min(1.0, elapsed_time / phase_duration)
            progress_info['progress'] = phase_progress
            
            # 更新海马和皮层强度
            self._update_representation_strengths(progress_info, dt)
    
    def _determine_consolidation_phase(self, elapsed_time: float) -> str:
        """确定巩固阶段"""
        if elapsed_time < self.consolidation_phases['early']:
            return 'early'
        elif elapsed_time < self.consolidation_phases['intermediate']:
            return 'intermediate'
        else:
            return 'late'
    
    def _update_representation_strengths(self, progress_info: Dict[str, Any], dt: float):
        """更新表示强度"""
        # 海马强度衰减
        decay_rate = self.hippocampal_decay_rate * dt
        progress_info['hippocampal_strength'] *= (1.0 - decay_rate)
        progress_info['hippocampal_strength'] = max(0.1, progress_info['hippocampal_strength'])
        
        # 皮层强度增强
        strengthening_rate = self.cortical_strengthening_rate * dt
        progress_info['cortical_strength'] += strengthening_rate
        progress_info['cortical_strength'] = min(1.0, progress_info['cortical_strength'])
    
    def apply_systems_consolidation_effects(self, trace: EnhancedMemoryTrace) -> EnhancedMemoryTrace:
        """应用系统巩固效果"""
        if trace.trace_id not in self.consolidation_progress:
            return trace
        
        progress_info = self.consolidation_progress[trace.trace_id]
        
        # 根据巩固进度调整记忆表示
        hippocampal_weight = progress_info['hippocampal_strength']
        cortical_weight = progress_info['cortical_strength']
        
        # 更新记忆强度（皮层表示更稳定）
        stability_boost = cortical_weight * 0.5
        trace.encoding_strength *= (1.0 + stability_boost)
        
        # 更新巩固水平
        systems_consolidation_level = cortical_weight
        trace.consolidation_level = max(trace.consolidation_level, systems_consolidation_level)
        
        return trace
    
    def get_systems_consolidation_status(self) -> Dict[str, Any]:
        """获取系统巩固状态"""
        if not self.consolidation_progress:
            return {'active_consolidations': 0}
        
        phase_counts = {'early': 0, 'intermediate': 0, 'late': 0}
        total_hippocampal_strength = 0.0
        total_cortical_strength = 0.0
        
        for progress_info in self.consolidation_progress.values():
            phase_counts[progress_info['phase']] += 1
            total_hippocampal_strength += progress_info['hippocampal_strength']
            total_cortical_strength += progress_info['cortical_strength']
        
        n_consolidations = len(self.consolidation_progress)
        
        return {
            'active_consolidations': n_consolidations,
            'phase_distribution': phase_counts,
            'average_hippocampal_strength': total_hippocampal_strength / n_consolidations,
            'average_cortical_strength': total_cortical_strength / n_consolidations
        }


class SleepDependentConsolidation:
    """睡眠依赖巩固"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 睡眠阶段特异性参数
        self.stage_consolidation_rates = {
            SleepStage.N2: 0.1,     # N2期：纺锤波增强
            SleepStage.N3: 0.3,     # N3期：慢波睡眠，主要巩固期
            SleepStage.REM: 0.2     # REM期：程序记忆和情绪记忆
        }
        
        # 记忆类型与睡眠阶段的关联
        self.memory_sleep_preferences = {
            MemoryType.EPISODIC: [SleepStage.N3, SleepStage.REM],
            MemoryType.SEMANTIC: [SleepStage.N3],
            MemoryType.PROCEDURAL: [SleepStage.REM],
            MemoryType.EMOTIONAL: [SleepStage.REM],
            MemoryType.SPATIAL: [SleepStage.N3]
        }
        
        # 睡眠状态
        self.current_sleep_stage = SleepStage.WAKE
        self.sleep_stage_duration = 0.0
        self.total_sleep_time = 0.0
        
        # 重放机制
        self.replay_probability = 0.3
        self.replay_sequences = []
        
        self.logger = logging.getLogger("SleepDependentConsolidation")
    
    def set_sleep_stage(self, sleep_stage: SleepStage):
        """设置睡眠阶段"""
        if sleep_stage != self.current_sleep_stage:
            self.logger.info(f"睡眠阶段转换: {self.current_sleep_stage.value} -> {sleep_stage.value}")
            self.current_sleep_stage = sleep_stage
            self.sleep_stage_duration = 0.0
    
    def update_sleep_consolidation(self, memory_traces: List[EnhancedMemoryTrace], dt: float):
        """更新睡眠巩固"""
        if self.current_sleep_stage == SleepStage.WAKE:
            return
        
        self.sleep_stage_duration += dt
        self.total_sleep_time += dt
        
        # 获取当前阶段的巩固率
        consolidation_rate = self.stage_consolidation_rates.get(self.current_sleep_stage, 0.0)
        
        if consolidation_rate > 0:
            # 选择适合当前睡眠阶段的记忆
            suitable_memories = self._select_memories_for_stage(memory_traces)
            
            # 执行巩固
            for trace in suitable_memories:
                self._consolidate_during_sleep(trace, consolidation_rate, dt)
            
            # 执行记忆重放
            if np.random.random() < self.replay_probability * dt:
                self._execute_memory_replay(suitable_memories)
    
    def _select_memories_for_stage(self, memory_traces: List[EnhancedMemoryTrace]) -> List[EnhancedMemoryTrace]:
        """选择适合当前睡眠阶段的记忆"""
        suitable_memories = []
        
        for trace in memory_traces:
            # 检查记忆类型是否适合当前阶段
            preferred_stages = self.memory_sleep_preferences.get(trace.memory_type, [])
            
            if self.current_sleep_stage in preferred_stages:
                # 检查记忆是否需要巩固
                if trace.consolidation_level < 1.0:
                    # 优先选择最近的记忆
                    time_since_encoding = time.time() - trace.timestamp
                    if time_since_encoding < 24 * 3600:  # 24小时内
                        suitable_memories.append(trace)
        
        # 按重要性排序
        suitable_memories.sort(
            key=lambda t: t.encoding_strength + t.retrieval_count * 0.1 + abs(t.emotional_valence),
            reverse=True
        )
        
        return suitable_memories[:20]  # 限制数量
    
    def _consolidate_during_sleep(self, trace: EnhancedMemoryTrace, 
                                consolidation_rate: float, dt: float):
        """睡眠期间巩固"""
        # 计算巩固增量
        consolidation_increment = consolidation_rate * dt / 3600  # 每小时的巩固量
        
        # 应用睡眠阶段特异性增强
        if self.current_sleep_stage == SleepStage.N3:
            # 慢波睡眠：增强陈述性记忆
            if trace.memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
                consolidation_increment *= 1.5
        
        elif self.current_sleep_stage == SleepStage.REM:
            # REM睡眠：增强程序记忆和情绪记忆
            if trace.memory_type in [MemoryType.PROCEDURAL, MemoryType.EMOTIONAL]:
                consolidation_increment *= 1.3
        
        # 更新巩固水平
        trace.consolidation_level = min(1.0, trace.consolidation_level + consolidation_increment)
        
        # 增强记忆强度
        strength_increment = consolidation_increment * 0.1
        trace.encoding_strength = min(2.0, trace.encoding_strength + strength_increment)
    
    def _execute_memory_replay(self, memory_traces: List[EnhancedMemoryTrace]):
        """执行记忆重放"""
        if len(memory_traces) < 2:
            return
        
        # 选择重放序列
        replay_length = min(5, len(memory_traces))
        replay_sequence = np.random.choice(memory_traces, replay_length, replace=False)
        
        # 记录重放序列
        self.replay_sequences.append({
            'sequence': [trace.trace_id for trace in replay_sequence],
            'sleep_stage': self.current_sleep_stage,
            'timestamp': time.time()
        })
        
        # 增强序列中记忆的关联
        for i in range(len(replay_sequence) - 1):
            current_trace = replay_sequence[i]
            next_trace = replay_sequence[i + 1]
            
            # 添加关联
            if next_trace.trace_id not in current_trace.associated_traces:
                current_trace.associated_traces.append(next_trace.trace_id)
        
        self.logger.info(f"执行记忆重放: {len(replay_sequence)} 个记忆")
    
    def get_sleep_consolidation_status(self) -> Dict[str, Any]:
        """获取睡眠巩固状态"""
        return {
            'current_sleep_stage': self.current_sleep_stage.value,
            'stage_duration': self.sleep_stage_duration,
            'total_sleep_time': self.total_sleep_time,
            'replay_sequences_count': len(self.replay_sequences),
            'consolidation_rates': {
                stage.value: rate for stage, rate in self.stage_consolidation_rates.items()
            }
        }


class MemoryConsolidationManager:
    """记忆巩固管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各种巩固机制
        self.synaptic_consolidation = SynapticConsolidation(config.get('synaptic', {}))
        self.systems_consolidation = SystemsConsolidation(config.get('systems', {}))
        self.sleep_consolidation = SleepDependentConsolidation(config.get('sleep', {}))
        
        # 巩固调度
        self.consolidation_queue = []
        self.active_consolidations = {}
        
        self.logger = logging.getLogger("MemoryConsolidationManager")
    
    def schedule_consolidation(self, trace: EnhancedMemoryTrace):
        """调度记忆巩固"""
        # 立即启动突触巩固
        synaptic_event = self.synaptic_consolidation.initiate_consolidation(trace)
        if synaptic_event:
            self.active_consolidations[trace.trace_id] = [synaptic_event]
        
        # 调度系统巩固（延迟启动）
        self.consolidation_queue.append({
            'trace_id': trace.trace_id,
            'consolidation_type': ConsolidationType.SYSTEMS,
            'scheduled_time': time.time() + 3600  # 1小时后
        })
    
    def update_all_consolidations(self, memory_traces: List[EnhancedMemoryTrace], dt: float):
        """更新所有巩固过程"""
        # 更新突触巩固
        self.synaptic_consolidation.update_consolidation(dt)
        
        # 更新系统巩固
        self.systems_consolidation.update_systems_consolidation(dt)
        
        # 更新睡眠巩固
        self.sleep_consolidation.update_sleep_consolidation(memory_traces, dt)
        
        # 处理调度队列
        self._process_consolidation_queue(memory_traces)
        
        # 应用巩固效果
        self._apply_consolidation_effects(memory_traces)
    
    def _process_consolidation_queue(self, memory_traces: List[EnhancedMemoryTrace]):
        """处理巩固调度队列"""
        current_time = time.time()
        
        # 处理到期的巩固任务
        due_tasks = [task for task in self.consolidation_queue if task['scheduled_time'] <= current_time]
        
        for task in due_tasks:
            trace_id = task['trace_id']
            consolidation_type = task['consolidation_type']
            
            # 找到对应的记忆痕迹
            trace = next((t for t in memory_traces if t.trace_id == trace_id), None)
            if not trace:
                continue
            
            # 启动相应的巩固过程
            if consolidation_type == ConsolidationType.SYSTEMS:
                success = self.systems_consolidation.initiate_systems_consolidation(trace)
                if success:
                    self.logger.info(f"启动系统巩固: 记忆 {trace_id}")
        
        # 移除已处理的任务
        self.consolidation_queue = [
            task for task in self.consolidation_queue 
            if task['scheduled_time'] > current_time
        ]
    
    def _apply_consolidation_effects(self, memory_traces: List[EnhancedMemoryTrace]):
        """应用巩固效果"""
        for trace in memory_traces:
            # 应用突触巩固效果
            trace = self.synaptic_consolidation.apply_consolidation_effects(trace)
            
            # 应用系统巩固效果
            trace = self.systems_consolidation.apply_systems_consolidation_effects(trace)
    
    def set_sleep_stage(self, sleep_stage: SleepStage):
        """设置睡眠阶段"""
        self.sleep_consolidation.set_sleep_stage(sleep_stage)
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """获取巩固总结"""
        return {
            'synaptic': self.synaptic_consolidation.get_consolidation_status(),
            'systems': self.systems_consolidation.get_systems_consolidation_status(),
            'sleep': self.sleep_consolidation.get_sleep_consolidation_status(),
            'queue_length': len(self.consolidation_queue),
            'active_consolidations': len(self.active_consolidations)
        }