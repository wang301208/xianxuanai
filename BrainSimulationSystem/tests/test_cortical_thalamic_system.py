"""
皮层柱+丘脑环路系统测试

这个模块提供了完整的皮层-丘脑系统测试，包括：
- 基本功能测试
- 集成测试
- 性能测试
- 同步化测试
- 可塑性测试
"""

import unittest
import numpy as np
import time
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)

# 导入测试模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_cortical_thalamic_system import (
    CorticalThalamicSystem, 
    CorticalThalamicConfig,
    SensoryModalityType,
    AttentionType,
    create_cortical_thalamic_system,
    create_minimal_config
)

from core.cortical_column_manager import (
    CorticalColumnManager,
    CorticalAreaType,
    ConnectionType,
    create_cortical_column_manager
)

from core.enhanced_thalamic_nuclei import (
    EnhancedThalamicNucleus,
    ThalamicOscillationType,
    create_enhanced_thalamic_nucleus,
    create_standard_thalamic_nuclei
)

from core.cortical_thalamic_integration import (
    CorticalThalamicIntegration,
    IntegrationConfig,
    IntegrationMode,
    create_cortical_thalamic_integration,
    create_minimal_integration
)


class TestCorticalThalamicSystem(unittest.TestCase):
    """皮层-丘脑系统基础测试"""
    
    def setUp(self):
        """测试设置"""
        self.config = create_minimal_config()
        self.system = CorticalThalamicSystem(self.config)
    
    def tearDown(self):
        """测试清理"""
        if hasattr(self.system, 'reset_system'):
            self.system.reset_system()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.system)
        self.assertEqual(len(self.system.cortical_columns), self.config.num_columns)
        self.assertIsNotNone(self.system.thalamocortical_loop)
    
    def test_sensory_input_processing(self):
        """测试感觉输入处理"""
        # 创建测试输入
        visual_input = np.random.randn(100) * 0.5
        
        # 处理视觉输入
        success = self.system.process_sensory_input(
            SensoryModalityType.VISUAL, 
            visual_input
        )
        
        self.assertTrue(success)
    
    def test_attention_control(self):
        """测试注意力控制"""
        # 测试空间注意力
        self.system.update_attention(AttentionType.SPATIAL, 0.8)
        self.assertEqual(self.system.attention_state[AttentionType.SPATIAL], 0.8)
        
        # 测试注意力目标设置
        self.system.set_attention_target("visual", 0.9)
        self.assertEqual(self.system.attention_targets["visual"], 0.9)
    
    def test_arousal_modulation(self):
        """测试觉醒调节"""
        # 测试高觉醒
        self.system.update_arousal(0.9)
        self.assertEqual(self.system.global_arousal, 0.9)
        
        # 测试低觉醒
        self.system.update_arousal(0.3)
        self.assertEqual(self.system.global_arousal, 0.3)
    
    def test_sleep_transitions(self):
        """测试睡眠转换"""
        # 测试进入睡眠
        self.system.simulate_sleep_transition(2)  # N2睡眠
        self.assertEqual(self.system.sleep_stage, 2)
        
        # 测试觉醒
        self.system.simulate_sleep_transition(0)  # 觉醒
        self.assertEqual(self.system.sleep_stage, 0)
    
    def test_system_step(self):
        """测试系统步进"""
        dt = 1.0  # 1ms
        
        # 执行步进
        results = self.system.step(dt)
        
        # 验证结果结构
        self.assertIn('timestamp', results)
        self.assertIn('thalamic_results', results)
        self.assertIn('cortical_results', results)
        self.assertIn('global_state', results)
    
    def test_system_state_management(self):
        """测试系统状态管理"""
        # 获取系统状态
        state = self.system.get_system_state()
        
        self.assertIn('config', state)
        self.assertIn('global_state', state)
        self.assertIn('cortical_columns', state)
        
        # 测试重置
        self.system.reset_system()
        self.assertEqual(self.system.sleep_stage, 0)
        self.assertEqual(self.system.global_arousal, 0.8)


class TestCorticalColumnManager(unittest.TestCase):
    """皮层柱管理器测试"""
    
    def setUp(self):
        """测试设置"""
        self.manager = create_cortical_column_manager()
        
        # 创建测试皮层柱
        from core.enhanced_cortical_column import EnhancedCorticalColumnWithLoop
        
        for i in range(3):
            column_config = {
                'total_neurons': 500,
                'position': (i * 1.0, 0.0, 0.0)
            }
            column = EnhancedCorticalColumnWithLoop(column_config, None)
            self.manager.add_column(i, column, (i * 1.0, 0.0, 0.0))
    
    def test_column_management(self):
        """测试皮层柱管理"""
        # 验证皮层柱数量
        self.assertEqual(len(self.manager.columns), 3)
        
        # 测试移除皮层柱
        self.manager.remove_column(2)
        self.assertEqual(len(self.manager.columns), 2)
        self.assertNotIn(2, self.manager.columns)
    
    def test_cortical_area_creation(self):
        """测试皮层区域创建"""
        from core.cortical_column_manager import CorticalAreaConfig
        
        # 创建视觉区域
        area_config = CorticalAreaConfig(
            area_type=CorticalAreaType.PRIMARY_VISUAL,
            column_ids=[0, 1],
            position=(0.0, 0.0, 0.0)
        )
        
        self.manager.create_cortical_area(area_config)
        
        # 验证区域创建
        self.assertIn(CorticalAreaType.PRIMARY_VISUAL, self.manager.cortical_areas)
        self.assertEqual(
            len(self.manager.area_column_mapping[CorticalAreaType.PRIMARY_VISUAL]), 
            2
        )
    
    def test_inter_area_connections(self):
        """测试区域间连接"""
        from core.cortical_column_manager import CorticalAreaConfig
        
        # 创建两个区域
        v1_config = CorticalAreaConfig(
            area_type=CorticalAreaType.PRIMARY_VISUAL,
            column_ids=[0],
            position=(0.0, 0.0, 0.0)
        )
        
        pfc_config = CorticalAreaConfig(
            area_type=CorticalAreaType.PREFRONTAL,
            column_ids=[1],
            position=(2.0, 0.0, 0.0)
        )
        
        self.manager.create_cortical_area(v1_config)
        self.manager.create_cortical_area(pfc_config)
        
        # 创建前馈连接
        self.manager.create_inter_area_connection(
            CorticalAreaType.PRIMARY_VISUAL,
            CorticalAreaType.PREFRONTAL,
            ConnectionType.FEEDFORWARD
        )
        
        # 验证连接创建
        feedforward_connections = [
            conn for conn in self.manager.inter_column_connections
            if conn.connection_type == ConnectionType.FEEDFORWARD
        ]
        
        self.assertGreater(len(feedforward_connections), 0)
    
    def test_synchronization_groups(self):
        """测试同步组"""
        # 创建同步组
        self.manager.create_synchronization_group("test_group", [0, 1])
        
        # 验证同步组创建
        self.assertIn("test_group", self.manager.synchronization_groups)
        self.assertEqual(len(self.manager.synchronization_groups["test_group"]), 2)
    
    def test_manager_step(self):
        """测试管理器步进"""
        dt = 1.0
        
        # 执行步进
        results = self.manager.step(dt)
        
        # 验证结果
        self.assertIn('column_results', results)
        self.assertIn('synchrony_metrics', results)
        self.assertIn('connection_stats', results)


class TestThalamicNuclei(unittest.TestCase):
    """丘脑核团测试"""
    
    def setUp(self):
        """测试设置"""
        self.nucleus = create_enhanced_thalamic_nucleus('LGN', {
            'size': 500,
            'position': (0.0, 0.0, 0.0)
        })
    
    def test_nucleus_initialization(self):
        """测试核团初始化"""
        self.assertEqual(self.nucleus.nucleus_type, 'LGN')
        self.assertEqual(self.nucleus.size, 500)
        self.assertGreater(len(self.nucleus.all_neurons), 0)
    
    def test_sensory_input_processing(self):
        """测试感觉输入处理"""
        # 创建视觉输入
        visual_input = np.random.randn(50) * 0.3
        
        # 设置输入
        self.nucleus.set_sensory_input(visual_input)
        
        # 验证输入设置
        self.assertIsNotNone(self.nucleus.sensory_input)
        self.assertEqual(len(self.nucleus.sensory_input), len(visual_input))
    
    def test_cortical_feedback(self):
        """测试皮层反馈"""
        # 创建反馈信号
        feedback = np.random.randn(20) * 0.2
        
        # 设置反馈
        self.nucleus.set_cortical_feedback(feedback)
        
        # 验证反馈设置
        self.assertIsNotNone(self.nucleus.cortical_feedback)
    
    def test_arousal_modulation(self):
        """测试觉醒调节"""
        # 测试高觉醒
        self.nucleus.update_arousal(0.9)
        self.assertEqual(self.nucleus.arousal_level, 0.9)
        
        # 验证振荡变化
        self.assertEqual(
            self.nucleus.oscillation_state.current_oscillation,
            ThalamicOscillationType.GAMMA
        )
    
    def test_attention_focus(self):
        """测试注意力聚焦"""
        # 设置高注意力
        self.nucleus.update_attention_focus(0.8)
        self.assertEqual(self.nucleus.attention_focus, 0.8)
        
        # 验证门控状态变化
        self.assertGreater(self.nucleus.gating_state, 0.5)
    
    def test_nucleus_step(self):
        """测试核团步进"""
        dt = 1.0
        
        # 执行步进
        results = self.nucleus.step(dt)
        
        # 验证结果结构
        self.assertIn('cell_activities', results)
        self.assertIn('oscillation_state', results)
        self.assertIn('functional_state', results)
        self.assertIn('spike_counts', results)
    
    def test_output_activity(self):
        """测试输出活动"""
        # 获取输出
        output = self.nucleus.get_output_activity()
        
        # 验证输出
        self.assertIsInstance(output, np.ndarray)
    
    def test_nucleus_reset(self):
        """测试核团重置"""
        # 修改状态
        self.nucleus.update_arousal(0.3)
        self.nucleus.update_attention_focus(0.2)
        
        # 重置
        self.nucleus.reset()
        
        # 验证重置
        self.assertEqual(self.nucleus.arousal_level, 0.8)
        self.assertEqual(self.nucleus.attention_focus, 0.5)


class TestCorticalThalamicIntegration(unittest.TestCase):
    """皮层-丘脑集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.integration = create_minimal_integration()
    
    def tearDown(self):
        """测试清理"""
        if hasattr(self.integration, 'shutdown_system'):
            self.integration.shutdown_system()
    
    def test_integration_initialization(self):
        """测试集成初始化"""
        self.assertIsNotNone(self.integration.cortical_thalamic_system)
        self.assertIsNotNone(self.integration.column_manager)
        self.assertGreater(len(self.integration.thalamic_nuclei), 0)
    
    def test_multimodal_processing(self):
        """测试多模态处理"""
        # 创建多模态输入
        visual_input = np.random.randn(50) * 0.4
        auditory_input = np.random.randn(30) * 0.3
        
        # 处理输入
        visual_success = self.integration.process_sensory_input(
            SensoryModalityType.VISUAL, visual_input
        )
        auditory_success = self.integration.process_sensory_input(
            SensoryModalityType.AUDITORY, auditory_input
        )
        
        self.assertTrue(visual_success)
        self.assertTrue(auditory_success)
    
    def test_attention_coordination(self):
        """测试注意力协调"""
        # 设置空间注意力
        self.integration.update_attention(AttentionType.SPATIAL, 0.7)
        
        # 设置特定目标注意力
        self.integration.update_attention(
            AttentionType.FEATURE, 0.8, target="visual"
        )
        
        # 验证注意力设置成功（通过系统状态）
        state = self.integration.get_system_state()
        self.assertIn('cortical_thalamic_state', state)
    
    def test_sleep_wake_coordination(self):
        """测试睡眠-觉醒协调"""
        # 模拟睡眠转换
        self.integration.simulate_sleep_transition(2)  # N2睡眠
        
        # 验证所有组件都响应了睡眠转换
        for nucleus in self.integration.thalamic_nuclei.values():
            # 检查振荡模式是否适合睡眠
            self.assertIn(
                nucleus.oscillation_state.current_oscillation,
                [ThalamicOscillationType.SPINDLE, ThalamicOscillationType.DELTA]
            )
    
    def test_integration_step(self):
        """测试集成步进"""
        dt = 1.0
        
        # 执行步进
        results = self.integration.step(dt)
        
        # 验证结果结构
        self.assertIn('cortical_results', results)
        self.assertIn('thalamic_results', results)
        self.assertIn('integration_metrics', results)
        self.assertIn('synchronization', results)
        self.assertIn('performance', results)
    
    def test_system_state_management(self):
        """测试系统状态管理"""
        # 获取完整状态
        state = self.integration.get_system_state()
        
        # 验证状态结构
        self.assertIn('system_status', state)
        self.assertIn('components', state)
        self.assertIn('connections', state)
        self.assertIn('integration_metrics', state)
    
    def test_pause_resume_system(self):
        """测试暂停恢复系统"""
        # 暂停系统
        self.integration.pause_system()
        
        # 尝试步进（应该返回错误）
        results = self.integration.step(1.0)
        self.assertIn('error', results)
        
        # 恢复系统
        self.integration.resume_system()
        
        # 再次步进（应该成功）
        results = self.integration.step(1.0)
        self.assertNotIn('error', results)
    
    def test_system_reset(self):
        """测试系统重置"""
        # 运行几步
        for _ in range(5):
            self.integration.step(1.0)
        
        # 记录步数
        step_count_before = self.integration.step_count
        
        # 重置系统
        self.integration.reset_system()
        
        # 验证重置
        self.assertEqual(self.integration.current_time, 0.0)
        self.assertEqual(self.integration.step_count, 0)


class TestPerformanceAndScaling(unittest.TestCase):
    """性能和扩展性测试"""
    
    def test_system_performance(self):
        """测试系统性能"""
        # 创建较大的系统
        config = IntegrationConfig(
            num_cortical_columns=4,
            neurons_per_column=1000,
            parallel_processing=True
        )
        
        integration = CorticalThalamicIntegration(config)
        
        try:
            # 测试处理时间
            start_time = time.time()
            
            for i in range(10):
                results = integration.step(1.0)
                
                # 验证实时性能
                if 'performance' in results and 'processing_time' in results['performance']:
                    processing_time = results['performance']['processing_time']
                    self.assertLess(processing_time, 0.1)  # 应该小于100ms
            
            total_time = time.time() - start_time
            self.assertLess(total_time, 2.0)  # 10步应该在2秒内完成
            
        finally:
            integration.shutdown_system()
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        # 获取初始内存
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建系统
        integration = create_minimal_integration()
        
        try:
            # 运行一段时间
            for _ in range(50):
                integration.step(1.0)
            
            # 检查内存增长
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # 内存增长应该合理（小于100MB）
            self.assertLess(memory_growth, 100)
            
        finally:
            integration.shutdown_system()
    
    def test_scalability(self):
        """测试可扩展性"""
        # 测试不同规模的系统
        scales = [
            (2, 500),   # 小规模
            (4, 1000),  # 中等规模
            (6, 1500)   # 较大规模
        ]
        
        for num_columns, neurons_per_column in scales:
            config = IntegrationConfig(
                num_cortical_columns=num_columns,
                neurons_per_column=neurons_per_column,
                parallel_processing=True
            )
            
            integration = CorticalThalamicIntegration(config)
            
            try:
                # 测试初始化时间
                start_time = time.time()
                
                # 运行几步
                for _ in range(3):
                    results = integration.step(1.0)
                    self.assertNotIn('error', results)
                
                execution_time = time.time() - start_time
                
                # 执行时间应该随规模合理增长
                expected_max_time = (num_columns * neurons_per_column) / 1000.0
                self.assertLess(execution_time, expected_max_time)
                
            finally:
                integration.shutdown_system()


def run_comprehensive_test():
    """运行综合测试"""
    print("开始皮层柱+丘脑环路系统综合测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCorticalThalamicSystem,
        TestCorticalColumnManager,
        TestThalamicNuclei,
        TestCorticalThalamicIntegration,
        TestPerformanceAndScaling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n测试完成:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # 运行综合测试
    success = run_comprehensive_test()
    
    if success:
        print("\n✅ 所有测试通过！皮层柱+丘脑环路系统功能正常。")
    else:
        print("\n❌ 部分测试失败，请检查系统实现。")
    
    exit(0 if success else 1)