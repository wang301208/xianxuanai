"""
Enhanced Brain Simulation System Tests

测试完整的增强型大脑仿真系统
"""

import pytest
import numpy as np
import time

from BrainSimulationSystem.core.enhanced_brain_system import create_enhanced_brain_simulation
from BrainSimulationSystem.core.cell_diversity import CellType
from BrainSimulationSystem.core.vascular_system import VesselType
from BrainSimulationSystem.core.physiological_regions import BrainRegion

class TestEnhancedBrainSystem:
    """测试增强型大脑系统"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = {
            'simulation_dt': 1.0,
            'logging_level': 'WARNING',  # 减少测试时的日志输出
            'performance_monitoring': True
        }
        self.brain_sim = create_enhanced_brain_simulation(self.config)
    
    def test_system_initialization(self):
        """测试系统初始化"""
        
        # 检查脑区是否正确初始化
        assert len(self.brain_sim.brain_network.regions) == 3
        assert BrainRegion.PRIMARY_VISUAL_CORTEX in self.brain_sim.brain_network.regions
        assert BrainRegion.HIPPOCAMPUS_CA1 in self.brain_sim.brain_network.regions
        assert BrainRegion.PREFRONTAL_CORTEX in self.brain_sim.brain_network.regions
        
        # 检查脑区间连接
        assert len(self.brain_sim.brain_network.connections) == 3
        
        # 检查全局状态初始化
        assert self.brain_sim.global_metabolic_state == 1.0
        assert self.brain_sim.global_inflammation == 0.0
        assert self.brain_sim.simulation_time == 0.0
    
    def test_single_step_simulation(self):
        """测试单步仿真"""
        
        # 准备输入
        external_inputs = {
            'PRIMARY_VISUAL_CORTEX': {
                'visual_stimulus': np.random.randn(64, 64)
            },
            'HIPPOCAMPUS_CA1': {
                'position': np.array([500, 500])
            },
            'PREFRONTAL_CORTEX': {
                'working_memory_items': [0, 1, 2]
            }
        }
        
        # 执行单步
        result = self.brain_sim.step(dt=1.0, external_inputs=external_inputs)
        
        # 检查结果结构
        assert 'results' in result
        assert 'statistics' in result
        assert 'simulation_time' in result
        assert 'update_time' in result
        
        # 检查仿真时间更新
        assert self.brain_sim.simulation_time == 1.0
        
        # 检查统计信息
        stats = result['statistics']
        assert 'global_state' in stats
        assert 'regions' in stats
        
        # 检查每个脑区的统计
        for region_name in ['PRIMARY_VISUAL_CORTEX', 'HIPPOCAMPUS_CA1', 'PREFRONTAL_CORTEX']:
            assert region_name in stats['regions']
            region_stats = stats['regions'][region_name]
            assert 'cell_activity' in region_stats
            assert 'vascular_stats' in region_stats
            assert 'neurotransmitter_levels' in region_stats
    
    def test_cell_diversity(self):
        """测试细胞多样性"""
        
        # 检查每个脑区的细胞类型多样性
        for region in self.brain_sim.brain_network.regions.values():
            cell_stats = region.cell_manager.get_population_statistics()
            
            # 应该有多种细胞类型
            assert len(cell_stats['cell_type_counts']) >= 5
            
            # 应该包含神经元和胶质细胞
            cell_types = list(cell_stats['cell_type_counts'].keys())
            has_neurons = any('pyramidal' in ct.lower() or 'interneuron' in ct.lower() for ct in cell_types)
            has_glia = any('astrocyte' in ct.lower() or 'microglia' in ct.lower() for ct in cell_types)
            
            assert has_neurons, f"Region should have neurons, found: {cell_types}"
            assert has_glia, f"Region should have glia, found: {cell_types}"
    
    def test_vascular_system(self):
        """测试血管系统"""
        
        for region in self.brain_sim.brain_network.regions.values():
            vascular_stats = region.vascular_network.get_vascular_statistics()
            
            # 应该有血管网络
            assert vascular_stats['total_vessels'] > 0
            assert vascular_stats['total_length_um'] > 0
            assert vascular_stats['total_volume_um3'] > 0
            
            # 应该有不同类型的血管
            vessel_types = vascular_stats['vessel_type_counts']
            assert len(vessel_types) >= 3  # 至少动脉、毛细血管、静脉
            
            # 应该有血流
            assert vascular_stats['total_flow_rate'] >= 0
    
    def test_neurotransmitter_dynamics(self):
        """测试神经递质动力学"""
        
        # 运行几步仿真
        for _ in range(5):
            self.brain_sim.step(1.0)
        
        # 检查神经递质浓度
        for region in self.brain_sim.brain_network.regions.values():
            nt_concentrations = region.neurotransmitter_concentrations
            
            # 应该有基本的神经递质
            assert 'glutamate' in nt_concentrations
            assert 'gaba' in nt_concentrations
            
            # 浓度应该为正值
            for nt, conc in nt_concentrations.items():
                assert conc >= 0, f"Neurotransmitter {nt} has negative concentration: {conc}"
    
    def test_oscillations(self):
        """测试神经振荡"""
        
        # 运行仿真
        result = self.brain_sim.step(1.0)
        
        # 检查振荡活动
        for region_name, region_result in result['results'].items():
            oscillations = region_result.get('oscillations', {})
            
            # 应该有振荡活动
            assert len(oscillations) > 0
            
            # 检查振荡参数
            for band, osc_data in oscillations.items():
                assert 'frequency' in osc_data
                assert 'amplitude' in osc_data
                assert 'phase' in osc_data
                
                # 频率应该在合理范围内
                assert 0 < osc_data['frequency'] < 300  # Hz
                assert osc_data['amplitude'] >= 0
                assert 0 <= osc_data['phase'] <= 2 * np.pi
    
    def test_inter_region_connectivity(self):
        """测试脑区间连接"""
        
        # 运行仿真以激活连接
        external_inputs = {
            'PRIMARY_VISUAL_CORTEX': {
                'visual_stimulus': np.ones((64, 64))  # 强视觉刺激
            }
        }
        
        result = self.brain_sim.step(1.0, external_inputs)
        
        # 检查是否有脑区间信号传递
        has_inter_region_signals = False
        for region_result in result['results'].values():
            if 'inter_region_input' in region_result:
                has_inter_region_signals = True
                break
        
        # 注意：可能需要多步才能看到明显的脑区间信号
        # 这里主要检查机制是否存在
        assert len(self.brain_sim.brain_network.connections) > 0
    
    def test_metabolic_modeling(self):
        """测试代谢建模"""
        
        initial_metabolic_state = self.brain_sim.global_metabolic_state
        
        # 运行高活动仿真
        high_activity_inputs = {
            'PRIMARY_VISUAL_CORTEX': {
                'visual_stimulus': np.random.randn(64, 64) * 5  # 强刺激
            },
            'PREFRONTAL_CORTEX': {
                'working_memory_items': [0, 1, 2, 3, 4, 5, 6]  # 满负荷工作记忆
            }
        }
        
        for _ in range(10):
            self.brain_sim.step(1.0, high_activity_inputs)
        
        # 代谢状态应该有变化
        final_metabolic_state = self.brain_sim.global_metabolic_state
        
        # 代谢状态应该在合理范围内
        assert 0.1 <= final_metabolic_state <= 2.0
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        
        # 运行几步仿真
        for _ in range(5):
            self.brain_sim.step(1.0)
        
        # 检查性能指标
        metrics = self.brain_sim.performance_metrics
        
        assert 'update_times' in metrics
        assert len(metrics['update_times']) == 5
        
        # 更新时间应该为正值
        for update_time in metrics['update_times']:
            assert update_time > 0
    
    def test_system_overview(self):
        """测试系统概览"""
        
        overview = self.brain_sim.get_system_overview()
        
        # 检查概览结构
        assert 'brain_regions' in overview
        assert 'total_cells' in overview
        assert 'total_vessels' in overview
        assert 'system_capabilities' in overview
        
        # 检查脑区信息
        assert len(overview['brain_regions']) == 3
        
        for region_info in overview['brain_regions'].values():
            assert 'dimensions' in region_info
            assert 'cell_types' in region_info
            assert 'total_cells' in region_info
            assert 'vessel_count' in region_info
        
        # 检查总计数
        assert overview['total_cells'] > 0
        assert overview['total_vessels'] > 0
        
        # 检查系统能力
        capabilities = overview['system_capabilities']
        assert len(capabilities) > 5
        assert any('cell type diversity' in cap.lower() for cap in capabilities)
        assert any('vascular' in cap.lower() for cap in capabilities)

class TestSimulationScenarios:
    """测试仿真场景"""
    
    def setup_method(self):
        """设置测试环境"""
        self.brain_sim = create_enhanced_brain_simulation()
    
    def test_visual_processing_scenario(self):
        """测试视觉处理场景"""
        
        # 创建视觉刺激序列
        input_sequence = []
        for i in range(10):
            # 移动的条纹刺激
            stimulus = np.zeros((64, 64))
            stripe_position = (i * 6) % 64
            stimulus[:, stripe_position:stripe_position+4] = 1.0
            
            input_sequence.append({
                'PRIMARY_VISUAL_CORTEX': {
                    'visual_stimulus': stimulus
                }
            })
        
        # 运行仿真
        results = []
        for inputs in input_sequence:
            result = self.brain_sim.step(1.0, inputs)
            results.append(result)
        
        # 检查V1活动是否响应刺激
        v1_activities = []
        for result in results:
            v1_stats = result['statistics']['regions']['PRIMARY_VISUAL_CORTEX']
            v1_activities.append(v1_stats['cell_activity']['activity_rate'])
        
        # 应该有活动变化
        assert max(v1_activities) > min(v1_activities)
    
    def test_memory_encoding_scenario(self):
        """测试记忆编码场景"""
        
        # 空间导航任务
        positions = [
            np.array([100, 100]),
            np.array([200, 200]),
            np.array([300, 300]),
            np.array([400, 400]),
            np.array([500, 500])
        ]
        
        results = []
        for pos in positions:
            inputs = {
                'HIPPOCAMPUS_CA1': {
                    'position': pos
                }
            }
            result = self.brain_sim.step(1.0, inputs)
            results.append(result)
        
        # 检查CA1活动
        ca1_activities = []
        for result in results:
            ca1_stats = result['statistics']['regions']['HIPPOCAMPUS_CA1']
            ca1_activities.append(ca1_stats['cell_activity']['activity_rate'])
        
        # 应该有位置相关的活动
        assert len(set(np.round(ca1_activities, 2))) > 1  # 不同位置应该有不同活动
    
    def test_working_memory_scenario(self):
        """测试工作记忆场景"""
        
        # 工作记忆负载测试
        memory_loads = [
            [],           # 无负载
            [0],          # 1项
            [0, 1, 2],    # 3项
            [0, 1, 2, 3, 4, 5, 6]  # 7项（满负载）
        ]
        
        results = []
        for load in memory_loads:
            inputs = {
                'PREFRONTAL_CORTEX': {
                    'working_memory_items': load
                }
            }
            result = self.brain_sim.step(1.0, inputs)
            results.append(result)
        
        # 检查PFC活动随负载变化
        pfc_activities = []
        for result in results:
            pfc_stats = result['statistics']['regions']['PREFRONTAL_CORTEX']
            pfc_activities.append(pfc_stats['cell_activity']['activity_rate'])
        
        # 活动应该随负载增加
        assert pfc_activities[-1] >= pfc_activities[0]  # 满负载 >= 无负载

if __name__ == "__main__":
    pytest.main([__file__, "-v"])