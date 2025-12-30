"""
睡眠周期管理器

实现优先记忆巩固和创伤记忆处理
"""

from .memory_system import MemoryTrace
from .events import EventBus

class SleepCycleManager:
    def __init__(self):
        self.rem_duration = 20  # REM期占比(分钟)
        self.sws_duration = 70   # SWS期占比(分钟)
        self.cycle_length = 90   # 完整周期(分钟)
        
        # 优先巩固参数
        self.priority_boost = 1.3  # 优先记忆增强系数
        self.max_priority_replays = 5  # 每周期最大优先重放次数
        
    def process_sleep_cycle(self, memories: List[MemoryTrace]):
        """处理一个完整睡眠周期"""
        # SWS期 - 优先巩固高优先级记忆
        priority_memories = sorted(
            [m for m in memories if m.priority > 0.7],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for mem in priority_memories[:self.max_priority_replays]:
            self._reinforce_memory(mem)
            
        # REM期 - 创伤记忆处理
        trauma_memories = [m for m in memories if m.is_trauma]
        for mem in trauma_memories:
            self._process_trauma_memory(mem)
            
    def _reinforce_memory(self, memory: MemoryTrace):
        """增强优先记忆"""
        memory.strength *= self.priority_boost
        memory.replay_count += 1
        EventBus.publish('memory_replay', 
                        memory=memory,
                        stage='SWS')
        
    def _process_trauma_memory(self, memory: MemoryTrace):
        """处理创伤记忆"""
        if not memory.suppressed:
            # 自然状态下减弱创伤记忆
            memory.retrievability *= 0.8
            memory.detail *= 0.7
        EventBus.publish('trauma_processing',
                        memory=memory,
                        stage='REM')