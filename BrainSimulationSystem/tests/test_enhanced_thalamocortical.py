"""
增强丘脑-皮层环路测试

测试完善的皮层柱-丘脑环路系统的各项功能
"""

import pytest
import numpy as np
from typing import Dict, Any

from BrainSimulationSystem.core.enhanced_thalamocortical_loop import (
    ThalamocorticalLoop, ThalamicNucleusType, ThalamicNucleus
)
from BrainSimulationSystem.core.enhanced_cortical_column import (
    EnhancedCorticalColumnWithLoop
)
from BrainSimulationSystem.integration.thalamocortical_integration import (
    ThalamocorticalIntegration, ThalamocorticalConfig
)


class TestThalamicNucleus:
    """测试丘脑核团功能"""
    
    def test_nucleus_initialization(self):
        """测试核团初始化"""
        nucleus = ThalamicNucleus(
            ThalamicNucleusType.VPL, 
            100, 
            (0, 0, 0), 
            {}
        )
        
        assert nucleus.size == 100
        assert len(nucleus.neurons) == 100
        assert len(nucleus.interneurons) > 0
        assert nucleus.nucleus_type == ThalamicNucleusType.VPL
    
    def test_sensory_input_processing(self):
        """测试感觉输入处理"""
        nucleus = ThalamicNucleus(
            ThalamicNucleusType.LGN, 
            50, 
            (0, 0, 0), 
            {}
        )
        
        # 设置感觉输入
        input_data = np.random.randn(50)
        nucleus.set_sensory_input(input_data)
        
        assert nucleus.sensory_input is not None
        assert len(nucleus.sensory_input) == 50
    
    def test_oscillation_states(self):
        """测试振荡状态"""
        nucleus = ThalamicNucleus(
            ThalamicNucleusType.MD, 
            80, 
            (0, 0, 0), 
            {}
        )
        
        # 测试觉醒水平调节
        nucleus.update_arousal(0.3)  # 低觉醒（睡眠状态）
        assert nucleus.oscillation_state.arousal_level == 0.3
        
        nucleus.update_arousal(0.9)  # 高觉醒（觉醒状态）
        assert nucleus.oscillation_state.arousal_level == 0.9
        
        # 测试注意力调节
        nucleus.update_attention(0.8)
        assert nucleus.oscillation_state.attention_focus == 0.8
    
    def test_nucleus_step_function(self):
        """测试核团步进函数"""
        nucleus = ThalamicNucleus(
            ThalamicNucleusType.VPL, 
            30, 
            (0, 0, 0), 
            {}
        )
        
        # 设置输入并运行步进
        input_data = np.ones(30) * 0.5
        nucleus.set_sensory_input(input_data)
        
        result = nucleus.step(1.0)
        
        assert 'relay_spikes' in result
        assert 'interneuron_spikes' in result
        assert 'oscillation_phase' in result
        assert 'oscillation_amplitude' in result


class TestThalamocorticalLoop:
    """测试丘脑-皮层环路"""
    
    def test_loop_initialization(self):
        """测试环路初始化"""
        config = {
            'enabled_nuclei': {
                'ventral_posterior_lateral': True,
                'lateral_geniculate': True,
                'mediodorsal': True,
                'reticular': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        assert len(loop.thalamic_nuclei) >= 4
        assert ThalamicNucleusType.VPL in loop.thalamic_nuclei
        assert ThalamicNucleusType.LGN in loop.thalamic_nuclei
        assert ThalamicNucleusType.RETICULAR in loop.thalamic_nuclei
    
    def test_cortical_column_integration(self):
        """测试皮层柱集成"""
        config = {
            'enabled_nuclei': {
                'ventral_posterior_lateral': True,
                'lateral_geniculate': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        # 创建皮层柱
        cortical_config = {
            'total_neurons': 500,
            'oscillation_enabled': True,
            'plasticity_enabled': True
        }
        
        cortical_column = EnhancedCorticalColumnWithLoop(cortical_config, loop)
        loop.add_cortical_column(0, cortical_column)
        
        assert 0 in loop.cortical_columns
        assert len(loop.thalamo_cortical_connections) > 0
        assert len(loop.cortico_thalamic_connections) > 0
    
    def test_sensory_input_distribution(self):
        """测试感觉输入分发"""
        config = {
            'enabled_nuclei': {
                'lateral_geniculate': True,
                'ventral_posterior_lateral': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        # 设置视觉输入
        visual_input = np.random.randn(200)
        loop.set_sensory_input(ThalamicNucleusType.LGN, visual_input)
        
        # 设置体感输入
        somatosensory_input = np.random.randn(150)
        loop.set_sensory_input(ThalamicNucleusType.VPL, somatosensory_input)
        
        # 验证输入已设置
        lgn_nucleus = loop.thalamic_nuclei[ThalamicNucleusType.LGN]
        vpl_nucleus = loop.thalamic_nuclei[ThalamicNucleusType.VPL]
        
        assert lgn_nucleus.sensory_input is not None
        assert vpl_nucleus.sensory_input is not None
    
    def test_attention_modulation(self):
        """测试注意力调节"""
        config = {
            'enabled_nuclei': {
                'lateral_geniculate': True,
                'pulvinar': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        # 更新注意力聚焦
        loop.update_attention_focus('visual', 0.8)
        
        # 验证相应核团的注意力水平
        if ThalamicNucleusType.LGN in loop.thalamic_nuclei:
            lgn_nucleus = loop.thalamic_nuclei[ThalamicNucleusType.LGN]
            assert lgn_nucleus.oscillation_state.attention_focus == 0.8
    
    def test_sleep_wake_transitions(self):
        """测试睡眠-觉醒转换"""
        config = {
            'enabled_nuclei': {
                'ventral_posterior_lateral': True,
                'reticular': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        # 测试不同睡眠阶段
        for sleep_stage in [0, 1, 2, 3, 4]:  # 觉醒到REM
            loop.simulate_sleep_transition(sleep_stage)
            
            # 验证觉醒水平变化
            expected_arousal = {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.2, 4: 0.5}
            assert abs(loop.global_arousal - expected_arousal[sleep_stage]) < 0.1
            
            # 验证所有核团的睡眠阶段
            for nucleus in loop.thalamic_nuclei.values():
                assert nucleus.oscillation_state.sleep_stage == sleep_stage
    
    def test_synchronization_calculation(self):
        """测试同步化计算"""
        config = {
            'enabled_nuclei': {
                'ventral_posterior_lateral': True,
                'lateral_geniculate': True,
                'mediodorsal': True
            }
        }
        
        loop = ThalamocorticalLoop(config)
        
        # 运行几步以建立振荡
        for _ in range(10):
            loop.step(1.0)
        
        # 计算同步化指数
        sync_indices = loop.get_synchronization_index()
        
        assert isinstance(sync_indices, dict)
        assert len(sync_indices) > 0
        
        # 验证同步化指数在合理范围内
        for sync_value in sync_indices.values():
            assert 0.0 <= sync_value <= 1.0


class TestEnhancedCorticalColumn:
    """测试增强皮层柱"""
    
    def test_enhanced_column_initialization(self):
        """测试增强皮层柱初始化"""
        config = {
            'total_neurons': 800,
            'oscillation_enabled': True,
            'plasticity_enabled': True,
            'learning_rate': 0.002
        }
        
        column = EnhancedCorticalColumnWithLoop(config)
        
        assert len(column.neurons) == 800
        assert column.oscillation_state is not None
        assert len(column.plasticity_traces) >= 0
        assert column.learning_rate == 0.002
    
    def test_thalamic_input_processing(self):
        """测试丘脑输入处理"""
        # 创建丘脑环路
        loop_config = {
            'enabled_nuclei': {
                'lateral_geniculate': True,
                'ventral_posterior_lateral': True
            }
        }
        loop = ThalamocorticalLoop(loop_config)
        
        # 创建皮层柱
        cortical_config = {
            'total_neurons': 600,
            'oscillation_enabled': True
        }
        column = EnhancedCorticalColumnWithLoop(cortical_config, loop)
        
        # 处理丘脑输入
        thalamic_inputs = {
            ThalamicNucleusType.LGN: np.random.randn(100),
            ThalamicNucleusType.VPL: np.random.randn(80)
        }
        
        column.process_thalamic_input(thalamic_inputs)
        
        # 验证输入已分发到相应层
        l4_layer = column.layers.get('L4_exc')
        assert l4_layer is not None
    
    def test_oscillation_modulation(self):
        """测试振荡调节"""
        config = {
            'total_neurons': 400,
            'oscillation_enabled': True
        }
        
        column = EnhancedCorticalColumnWithLoop(config)
        
        # 运行几步以建立振荡
        for _ in range(5):
            result = column.step(1.0)
        
        # 验证振荡信息
        assert 'layer_oscillations' in result
        assert len(result['layer_oscillations']) > 0
        
        # 验证振荡相干性
        coherence = column.get_oscillation_coherence()
        assert 'gamma' in coherence
        assert 'beta' in coherence
        assert 'alpha' in coherence
    
    def test_inter_layer_synchrony(self):
        """测试层间同步"""
        config = {
            'total_neurons': 500,
            'oscillation_enabled': True
        }
        
        column = EnhancedCorticalColumnWithLoop(config)
        
        # 运行足够的步数以建立同步历史
        for _ in range(15):
            column.step(1.0)
        
        # 验证层间同步计算
        assert len(column.inter_layer_synchrony) > 0
        
        # 验证同步值在合理范围内
        for sync_value in column.inter_layer_synchrony.values():
            assert -1.0 <= sync_value <= 1.0
    
    def test_cortical_feedback(self):
        """测试皮层反馈"""
        config = {
            'total_neurons': 300,
            'oscillation_enabled': True
        }
        
        column = EnhancedCorticalColumnWithLoop(config)
        
        # 运行几步
        for _ in range(3):
            column.step(1.0)
        
        # 获取皮层反馈
        feedback = column.get_cortical_feedback_for_thalamus()
        
        assert isinstance(feedback, dict)
        assert 'primary' in feedback or 'secondary' in feedback


class TestThalamocorticalIntegration:
    """测试丘脑-皮层集成"""
    
    def test_integration_initialization(self):
        """测试集成系统初始化"""
        config = ThalamocorticalConfig(
            enabled=True,
            num_cortical_columns=2,
            oscillation_enabled=True,
            plasticity_enabled=True
        )
        
        integration = ThalamocorticalIntegration(config)
        
        assert integration.is_initialized
        assert len(integration.cortical_columns) == 2
        assert integration.thalamocortical_loop is not None
    
    def test_sensory_input_interface(self):
        """测试感觉输入接口"""
        config = ThalamocorticalConfig(
            enabled=True,
            num_cortical_columns=1
        )
        
        integration = ThalamocorticalIntegration(config)
        
        # 设置不同模态的感觉输入
        integration.set_sensory_input('visual', np.random.randn(100))
        integration.set_sensory_input('somatosensory', np.random.randn(80))
        integration.set_sensory_input('auditory', np.random.randn(60))
        
        # 验证输入已正确设置
        assert integration.is_initialized
    
    def test_system_step_function(self):
        """测试系统步进函数"""
        config = ThalamocorticalConfig(
            enabled=True,
            num_cortical_columns=1
        )
        
        integration = ThalamocorticalIntegration(config)
        
        # 运行系统步进
        result = integration.step(1.0)
        
        assert 'thalamic_result' in result
        assert 'cortical_results' in result
        assert 'synchronization_indices' in result
        assert 'total_spikes' in result
        assert 'update_time' in result
    
    def test_system_state_monitoring(self):
        """测试系统状态监控"""
        config = ThalamocorticalConfig(
            enabled=True,
            num_cortical_columns=2
        )
        
        integration = ThalamocorticalIntegration(config)
        
        # 运行几步
        for _ in range(5):
            integration.step(1.0)
        
        # 获取系统状态
        state = integration.get_system_state()
        
        assert state['initialized']
        assert state['num_cortical_columns'] == 2
        assert 'performance' in state
        assert 'global_arousal' in state
    
    def test_system_reset(self):
        """测试系统重置"""
        config = ThalamocorticalConfig(
            enabled=True,
            num_cortical_columns=1
        )
        
        integration = ThalamocorticalIntegration(config)
        
        # 运行几步
        for _ in range(3):
            integration.step(1.0)
        
        initial_time = integration.current_time
        
        # 重置系统
        integration.reset()
        
        assert integration.current_time == 0.0
        assert len(integration.performance_metrics['update_times']) == 0


def test_comprehensive_integration():
    """综合集成测试"""
    # 创建完整的丘脑-皮层系统
    config = ThalamocorticalConfig(
        enabled=True,
        num_cortical_columns=2,
        oscillation_enabled=True,
        plasticity_enabled=True
    )
    
    integration = ThalamocorticalIntegration(config)
    
    # 设置多模态输入
    integration.set_sensory_input('visual', np.sin(np.linspace(0, 2*np.pi, 200)))
    integration.set_sensory_input('somatosensory', np.cos(np.linspace(0, 2*np.pi, 150)))
    
    # 调节注意力和觉醒
    integration.update_attention_focus('visual', 0.8)
    integration.update_arousal_level(0.9)
    
    # 运行仿真
    results = []
    for step in range(10):
        result = integration.step(1.0)
        results.append(result)
    
    # 验证结果
    assert len(results) == 10
    assert all('total_spikes' in result for result in results)
    
    # 测试睡眠转换
    integration.simulate_sleep_transition(2)  # N2睡眠
    
    sleep_result = integration.step(1.0)
    assert 'thalamic_result' in sleep_result
    
    # 获取最终状态
    final_state = integration.get_system_state()
    assert final_state['initialized']
    assert final_state['current_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__])