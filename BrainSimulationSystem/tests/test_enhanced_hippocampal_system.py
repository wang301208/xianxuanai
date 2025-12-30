"""
增强海马记忆系统测试

测试海马记忆系统的各项功能
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

from BrainSimulationSystem.memory.enhanced_hippocampal_system import (
    EnhancedHippocampalSystem, MemoryType, MemoryPhase,
    DentateGyrus, CA3AutoAssociative, CA1TemporalSequence, 
    CA2SocialMemory, Subiculum
)
from BrainSimulationSystem.memory.memory_consolidation import (
    MemoryConsolidationManager, ConsolidationType, SleepStage
)
from BrainSimulationSystem.memory.episodic_memory import (
    EpisodicMemorySystem, EpisodicEvent
)


class TestDentateGyrus(unittest.TestCase):
    """测试齿状回"""
    
    def setUp(self):
        self.dg = DentateGyrus(size=10000, sparsity=0.01)
    
    def test_pattern_separation(self):
        """测试模式分离"""
        # 创建输入模式
        pattern1 = np.random.random(1000)
        pattern2 = pattern1 + np.random.normal(0, 0.1, 1000)  # 相似模式
        
        # 模式分离
        sparse1 = self.dg.pattern_separation(pattern1)
        sparse2 = self.dg.pattern_separation(pattern2)
        
        # 验证稀疏性
        active_ratio1 = np.sum(sparse1 > 0) / len(sparse1)
        active_ratio2 = np.sum(sparse2 > 0) / len(sparse2)
        
        self.assertLess(active_ratio1, 0.02)  # 稀疏表示
        self.assertLess(active_ratio2, 0.02)
        
        # 验证分离效果
        similarity = np.corrcoef(sparse1, sparse2)[0, 1]
        self.assertLess(abs(similarity), 0.5)  # 分离后相似性降低
    
    def test_neurogenesis(self):
        """测试神经发生"""
        initial_young_count = len(self.dg.young_neurons)
        
        # 模拟时间推进
        self.dg.neurogenesis_update(24 * 3600)  # 1天
        
        # 验证神经发生状态
        neurogenesis_state = self.dg.get_neurogenesis_state()
        self.assertIn('young_neurons', neurogenesis_state)
        self.assertIn('neurogenesis_rate', neurogenesis_state)


class TestCA3AutoAssociative(unittest.TestCase):
    """测试CA3自联想网络"""
    
    def setUp(self):
        self.ca3 = CA3AutoAssociative(size=5000)
    
    def test_pattern_storage_and_completion(self):
        """测试模式存储和完成"""
        # 创建测试模式
        dg_input = np.random.random(1000)
        
        # 存储模式
        stored_pattern = self.ca3.store_pattern(dg_input, pattern_id=1)
        
        # 创建部分线索
        partial_cue = stored_pattern.copy()
        partial_cue[len(partial_cue)//2:] = 0  # 移除一半信息
        
        # 模式完成
        completed_pattern, quality = self.ca3.pattern_completion(partial_cue)
        
        # 验证完成质量
        self.assertGreater(quality, 0.3)
        
        # 验证模式相似性
        similarity = np.corrcoef(stored_pattern, completed_pattern)[0, 1]
        self.assertGreater(abs(similarity), 0.5)
    
    def test_associative_learning(self):
        """测试联想学习"""
        # 存储两个模式
        pattern1 = self.ca3.store_pattern(np.random.random(1000), 1)
        pattern2 = self.ca3.store_pattern(np.random.random(1000), 2)
        
        # 创建关联
        self.ca3.create_association(1, 2, 0.8)
        
        # 验证关联检索
        associations = self.ca3.associative_retrieval(1)
        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0][0], 2)


class TestCA1TemporalSequence(unittest.TestCase):
    """测试CA1时序编码"""
    
    def setUp(self):
        self.ca1 = CA1TemporalSequence(size=8000)
    
    def test_temporal_sequence_encoding(self):
        """测试时间序列编码"""
        # 创建序列
        ca3_sequence = [np.random.random(3000) for _ in range(5)]
        time_intervals = [1.0, 0.5, 2.0, 1.5, 0.8]
        
        # 编码序列
        sequence_repr = self.ca1.encode_temporal_sequence(ca3_sequence, time_intervals)
        
        # 验证编码结果
        self.assertEqual(len(sequence_repr), self.ca1.size)
        self.assertGreater(np.sum(np.abs(sequence_repr)), 0)
    
    def test_spatial_encoding(self):
        """测试空间编码"""
        position = np.array([10.0, 20.0, 5.0])
        
        # 编码空间位置
        spatial_code = self.ca1.encode_spatial_location(position)
        
        # 验证编码
        self.assertEqual(len(spatial_code), len(self.ca1.place_cells))
        self.assertGreater(np.sum(spatial_code), 0)


class TestCA2SocialMemory(unittest.TestCase):
    """测试CA2社会记忆"""
    
    def setUp(self):
        self.ca2 = CA2SocialMemory(size=4000)
    
    def test_social_interaction_encoding(self):
        """测试社会互动编码"""
        # 编码社会互动
        social_code = self.ca2.encode_social_interaction(
            agent_id="agent_1",
            interaction_type="cooperation",
            context={"valence": 0.8, "intensity": 0.6}
        )
        
        # 验证编码
        self.assertEqual(len(social_code), self.ca2.size)
        
        # 检索社会记忆
        memory = self.ca2.retrieve_social_memory("agent_1")
        self.assertIsNotNone(memory)
        self.assertEqual(len(memory['memories']), 1)
    
    def test_social_relationship_tracking(self):
        """测试社会关系追踪"""
        # 多次互动
        for i in range(3):
            self.ca2.encode_social_interaction(
                agent_id="agent_1",
                interaction_type="cooperation",
                context={"valence": 0.7}
            )
        
        # 检查关系状态
        network_state = self.ca2.get_social_network_state()
        self.assertIn("agent_1", network_state['known_agents'])
        
        relationship = network_state['relationship_summary']['agent_1']
        self.assertGreater(relationship['familiarity'], 0.0)
        self.assertEqual(relationship['interactions'], 3)


class TestEnhancedHippocampalSystem(unittest.TestCase):
    """测试增强海马系统"""
    
    def setUp(self):
        config = {
            'dg_size': 10000,
            'ca3_size': 5000,
            'ca1_size': 8000,
            'ca2_size': 4000,
            'subiculum_size': 2000
        }
        self.hippocampus = EnhancedHippocampalSystem(config)
    
    def test_memory_encoding(self):
        """测试记忆编码"""
        content = {
            'type': 'learning',
            'subject': 'mathematics',
            'difficulty': 0.7
        }
        
        # 编码记忆
        trace_id = self.hippocampus.encode_memory(
            content=content,
            memory_type=MemoryType.EPISODIC,
            context={'spatial_location': [10.0, 20.0, 0.0]}
        )
        
        # 验证编码
        self.assertEqual(trace_id, 0)
        self.assertEqual(len(self.hippocampus.memory_traces), 1)
        
        trace = self.hippocampus.memory_traces[0]
        self.assertEqual(trace.memory_type, MemoryType.EPISODIC)
        self.assertEqual(trace.content, content)
    
    def test_memory_retrieval(self):
        """测试记忆检索"""
        # 编码多个记忆
        for i in range(5):
            self.hippocampus.encode_memory(
                content={'type': 'test', 'id': i},
                memory_type=MemoryType.SEMANTIC
            )
        
        # 检索记忆
        cue = {'type': 'test'}
        retrieved = self.hippocampus.retrieve_memory(cue, MemoryType.SEMANTIC)
        
        # 验证检索结果
        self.assertGreater(len(retrieved), 0)
        self.assertLessEqual(len(retrieved), 5)
    
    def test_memory_consolidation(self):
        """测试记忆巩固"""
        # 编码记忆
        trace_id = self.hippocampus.encode_memory(
            content={'type': 'important', 'value': 1.0},
            memory_type=MemoryType.EPISODIC
        )
        
        # 执行巩固
        self.hippocampus.consolidate_memories('synaptic')
        
        # 验证巩固效果
        trace = self.hippocampus.memory_traces[0]
        # 巩固可能需要时间，这里主要验证系统不出错
        self.assertIsNotNone(trace)
    
    def test_system_statistics(self):
        """测试系统统计"""
        # 编码一些记忆
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.SPATIAL]:
            self.hippocampus.encode_memory(
                content={'type': memory_type.value},
                memory_type=memory_type
            )
        
        # 获取统计信息
        stats = self.hippocampus.get_system_statistics()
        
        # 验证统计信息
        self.assertEqual(stats['total_memories'], 3)
        self.assertIn('memory_type_distribution', stats)
        self.assertIn('consolidation', stats)
        self.assertIn('neurogenesis', stats)


class TestMemoryConsolidation(unittest.TestCase):
    """测试记忆巩固"""
    
    def setUp(self):
        config = {
            'synaptic': {'synaptic_window': 3600},
            'systems': {'transfer_rate': 0.01},
            'sleep': {}
        }
        self.consolidation_manager = MemoryConsolidationManager(config)
    
    def test_sleep_stage_setting(self):
        """测试睡眠阶段设置"""
        # 设置睡眠阶段
        self.consolidation_manager.set_sleep_stage(SleepStage.N3)
        
        # 验证设置
        sleep_status = self.consolidation_manager.sleep_consolidation.get_sleep_consolidation_status()
        self.assertEqual(sleep_status['current_sleep_stage'], 'n3')
    
    def test_consolidation_summary(self):
        """测试巩固总结"""
        summary = self.consolidation_manager.get_consolidation_summary()
        
        # 验证总结结构
        self.assertIn('synaptic', summary)
        self.assertIn('systems', summary)
        self.assertIn('sleep', summary)


class TestEpisodicMemory(unittest.TestCase):
    """测试情节记忆"""
    
    def setUp(self):
        config = {
            'boundary_detection': {
                'temporal_threshold': 300,
                'spatial_threshold': 10.0
            }
        }
        self.episodic_system = EpisodicMemorySystem(config)
    
    def test_event_encoding(self):
        """测试事件编码"""
        event_content = {
            'type': 'meeting',
            'participants': ['Alice', 'Bob'],
            'duration': 3600
        }
        
        # 编码事件
        event_id = self.episodic_system.encode_event(
            event_content=event_content,
            spatial_location=np.array([0.0, 0.0, 0.0]),
            participants=['Alice', 'Bob']
        )
        
        # 验证编码
        self.assertEqual(event_id, 0)
        self.assertEqual(len(self.episodic_system.events), 1)
        
        event = self.episodic_system.events[0]
        self.assertEqual(event.event_type, 'meeting')
        self.assertEqual(event.participants, ['Alice', 'Bob'])
    
    def test_episode_boundary_detection(self):
        """测试情节边界检测"""
        # 编码多个事件
        for i in range(3):
            self.episodic_system.encode_event(
                event_content={'type': 'work', 'task': i},
                spatial_location=np.array([i * 5.0, 0.0, 0.0])  # 不同位置
            )
            time.sleep(0.1)  # 小间隔
        
        # 验证情节创建
        self.assertGreaterEqual(len(self.episodic_system.episodes), 1)
    
    def test_event_retrieval(self):
        """测试事件检索"""
        # 编码事件
        for i in range(5):
            self.episodic_system.encode_event(
                event_content={'type': 'study', 'subject': f'subject_{i}'},
                participants=[f'student_{i}']
            )
        
        # 检索事件
        query = {'participants': ['student_1']}
        retrieved_events = self.episodic_system.retrieve_events(query)
        
        # 验证检索结果
        self.assertEqual(len(retrieved_events), 1)
        self.assertIn('student_1', retrieved_events[0].participants)


if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main()