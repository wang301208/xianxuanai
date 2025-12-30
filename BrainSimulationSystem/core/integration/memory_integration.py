"""
记忆系统集成层

连接海马与皮层记忆系统
"""

from ..memory_system import HippocampalFormation, CorticalMemory
from ..events import EventBus

class MemoryIntegration:
    def __init__(self):
        # 情绪状态参数
        self.emotion_state = {
            'valence': 0.5,    # 效价 [-1,1] 
            'arousal': 0.5,    # 唤醒度 [0,1]
            'dominance': 0.5   # 控制度 [0,1]
        }
        
        # 记忆层级结构
        self.memory_hierarchy = {
            'episodic': [],    # 情景记忆
            'semantic': [],    # 语义记忆
            'procedural': []   # 程序记忆
        }
        # 扩展睡眠周期参数
        self.sleep_stages = {
            'awake': {
                'duration': 16 * 60,  # 分钟
                'memory_ops': 'encoding'
            },
            'sleep': {
                'cycles': 5,
                'sws_ratio': 0.8,
                'rem_ratio': 0.2,
                'memory_ops': 'consolidation' 
            }
        }
        self.biological_clock = 0  # 模拟生物钟时间(分钟)
        self.hippocampus = HippocampalFormation()
        self.cortex = CorticalMemory()
        
        # 睡眠周期参数
        self.sleep_cycle = {
            'REM': 0,
            'SWS': 0,
            'cycle_duration': 90  # 分钟
        }
    
    def update_biological_clock(self, dt: float):
        """更新生物钟状态"""
        self.biological_clock += dt
        
        # 昼夜节律判断
        if self.biological_clock % (24 * 60) < self.sleep_stages['awake']['duration']:
            # 觉醒期 - 记忆编码
            EventBus.publish('encoding_phase')
        else:
            # 睡眠期 - 记忆巩固
            sleep_phase = self.biological_clock % 90  # 睡眠周期90分钟
            if sleep_phase < 90 * self.sleep_stages['sleep']['sws_ratio']:
                EventBus.publish('consolidation_phase', sleep_stage='SWS')
            else:
                EventBus.publish('consolidation_phase', sleep_stage='REM')
                
        # 每天重置生物钟
        if self.biological_clock >= 24 * 60:
            self.biological_clock = 0
            # 应用每日记忆衰减
            self.hippocampus.apply_daily_forgetting()
            
    def update_emotion(self, valence: float, arousal: float, dominance: float):
        """更新情绪状态 (PAD模型)"""
        self.emotion_state = {
            'valence': max(-1, min(1, valence)),
            'arousal': max(0, min(1, arousal)),
            'dominance': max(0, min(1, dominance))
        }
        
    def retrieve_memory(self, context: np.ndarray):
        """情境记忆检索接口"""
        memory = self.hippocampus.retrieve_by_context(context)
        if memory:
            # 情绪增强效应 (高唤醒增强检索)
            strength_boost = 1 + self.emotion_state['arousal'] * 0.5
            memory['strength'] *= strength_boost
            
            # 触发皮层模式完成
            EventBus.publish('cortical_retrieval', 
                           pattern=memory['ca1'],
                           context=context,
                           emotion=self.emotion_state)
            
            # 根据情绪效价分类记忆
            if self.emotion_state['valence'] > 0.7:
                self.memory_hierarchy['episodic'].append(memory)
            elif abs(self.emotion_state['valence']) < 0.3:
                self.memory_hierarchy['semantic'].append(memory)
            else:
                self.memory_hierarchy['procedural'].append(memory)
        return memory
        
    def consolidate_to_semantic(self, memory: Dict):
        """情景记忆→语义记忆转化"""
        if memory in self.memory_hierarchy['episodic']:
            self.memory_hierarchy['episodic'].remove(memory)
            
            # 提取语义特征
            semantic_memory = {
                'concepts': self._extract_concepts(memory['sensory']),
                'relations': [],
                'strength': memory['strength'] * 0.8  # 转化损耗
            }
            self.memory_hierarchy['semantic'].append(semantic_memory)
            
    def _extract_concepts(self, sensory_input: np.ndarray) -> List[str]:
        """从感觉输入提取语义概念 (简化版)"""
        # 实际实现应接入NLP模块
        return ['object', 'action', 'location']  # 返回值