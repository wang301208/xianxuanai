"""
多类型神经元模型测试
Multi-Type Neuron Models Tests
"""

import unittest
import numpy as np
from typing import Dict, Any

from BrainSimulationSystem.core.multi_neuron_models import (
    NeuronType, BaseNeuronModel, LIFNeuron, HodgkinHuxleyNeuron,
    AdExNeuron, IzhikevichNeuron, MultiCompartmentNeuron,
    Astrocyte, Microglia, NeuromodulatorSystem,
    create_neuron, get_default_parameters,
    CompartmentParameters, IonChannelState
)

class TestNeuronModels(unittest.TestCase):
    """神经元模型测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.dt = 0.1  # ms
        self.simulation_time = 100.0  # ms
        self.test_current = 10.0  # pA
    
    def test_lif_neuron(self):
        """测试LIF神经元"""
        
        params = get_default_parameters(NeuronType.LIF)
        neuron = LIFNeuron(neuron_id=1, params=params)
        
        # 检查初始状态
        self.assertEqual(neuron.V, params['V_rest'])
        self.assertEqual(len(neuron.spike_times), 0)
        
        # 添加输入电流
        neuron.add_input_current(self.test_current)
        
        # 仿真
        spike_count = 0
        for t in np.arange(0, self.simulation_time, self.dt):
            if neuron.update(self.dt, t):
                spike_count += 1
        
        # 检查是否发放
        self.assertGreater(spike_count, 0)
        self.assertEqual(len(neuron.spike_times), spike_count)
        
        # 测试重置
        neuron.reset()
        self.assertEqual(neuron.V, params['V_rest'])
        self.assertEqual(neuron.I_ext, 0.0)
    
    def test_hodgkin_huxley_neuron(self):
        """测试Hodgkin-Huxley神经元"""
        
        params = get_default_parameters(NeuronType.HODGKIN_HUXLEY)
        neuron = HodgkinHuxleyNeuron(neuron_id=2, params=params)
        
        # 检查初始状态
        self.assertAlmostEqual(neuron.V, -65.0, places=1)
        self.assertGreater(neuron.m, 0)
        self.assertGreater(neuron.h, 0)
        self.assertGreater(neuron.n, 0)
        
        # 添加强电流刺激
        neuron.add_input_current(50.0)  # 较大电流
        
        # 仿真
        spike_count = 0
        for t in np.arange(0, self.simulation_time, self.dt):
            if neuron.update(self.dt, t):
                spike_count += 1
        
        # HH神经元应该能发放
        self.assertGreater(spike_count, 0)
        
        # 测试重置
        neuron.reset()
        self.assertAlmostEqual(neuron.V, -65.0, places=1)
    
    def test_adex_neuron(self):
        """测试AdEx神经元"""
        
        params = get_default_parameters(NeuronType.ADAPTIVE_EXPONENTIAL)
        neuron = AdExNeuron(neuron_id=3, params=params)
        
        # 检查初始状态
        self.assertEqual(neuron.V, params['E_L'])
        self.assertEqual(neuron.w, 0.0)
        
        # 添加输入电流
        neuron.add_input_current(200.0)  # pA
        
        # 仿真
        spike_count = 0
        w_values = []
        for t in np.arange(0, self.simulation_time, self.dt):
            w_values.append(neuron.w)
            if neuron.update(self.dt, t):
                spike_count += 1
        
        # 检查适应性
        self.assertGreater(spike_count, 0)
        self.assertGreater(max(w_values), min(w_values))  # 适应变量应该变化
    
    def test_izhikevich_neuron(self):
        """测试Izhikevich神经元"""
        
        params = get_default_parameters(NeuronType.IZHIKEVICH)
        neuron = IzhikevichNeuron(neuron_id=4, params=params)
        
        # 检查初始状态
        self.assertEqual(neuron.V, params['c'])
        self.assertEqual(neuron.u, params['b'] * params['c'])
        
        # 添加输入电流
        neuron.add_input_current(20.0)
        
        # 仿真
        spike_count = 0
        for t in np.arange(0, self.simulation_time, self.dt):
            if neuron.update(self.dt, t):
                spike_count += 1
        
        # 检查发放
        self.assertGreater(spike_count, 0)
    
    def test_multi_compartment_neuron(self):
        """测试多室神经元"""
        
        morphology = {
            'soma': {'length': 20.0, 'diameter': 20.0},
            'dendrite': {'length': 200.0, 'diameter': 2.0},
            'axon': {'length': 500.0, 'diameter': 1.0, 'Na_density': 300.0}
        }
        
        params = {'morphology': morphology}
        neuron = MultiCompartmentNeuron(neuron_id=5, params=params)
        
        # 检查室结构
        self.assertIn('soma', neuron.compartments)
        self.assertIn('dendrite', neuron.compartments)
        self.assertIn('axon', neuron.compartments)
        
        # 检查连接
        self.assertGreater(len(neuron.connections), 0)
        
        # 检查离子通道状态
        self.assertIn('soma', neuron.ion_channels)
        
        # 添加输入电流到胞体
        neuron.add_input_current(100.0)
        
        # 仿真
        spike_count = 0
        for t in np.arange(0, 50.0, self.dt):  # 较短时间
            if neuron.update(self.dt, t):
                spike_count += 1
        
        # 检查各室电位
        soma_v = neuron.get_compartment_voltage('soma')
        dendrite_v = neuron.get_compartment_voltage('dendrite')
        
        self.assertNotEqual(soma_v, dendrite_v)  # 各室电位应该不同
    
    def test_astrocyte(self):
        """测试星形胶质细胞"""
        
        params = get_default_parameters(NeuronType.ASTROCYTE)
        astrocyte = Astrocyte(neuron_id=6, params=params)
        
        # 检查初始状态
        self.assertGreater(astrocyte.Ca_cyt, 0)
        self.assertGreater(astrocyte.Ca_ER, 0)
        self.assertGreater(astrocyte.glucose, 0)
        
        # 创建测试神经元
        test_neuron = LIFNeuron(neuron_id=7, params=get_default_parameters(NeuronType.LIF))
        astrocyte.connect_neuron(test_neuron)
        
        # 仿真
        initial_ca = astrocyte.Ca_cyt
        for t in np.arange(0, 50.0, self.dt):
            astrocyte.update(self.dt, t)
        
        # 检查钙动力学
        self.assertNotEqual(astrocyte.Ca_cyt, initial_ca)
        
        # 检查代谢状态
        self.assertGreater(astrocyte.glucose, 0)
        self.assertGreater(astrocyte.lactate, 0)
    
    def test_microglia(self):
        """测试小胶质细胞"""
        
        params = get_default_parameters(NeuronType.MICROGLIA)
        microglia = Microglia(neuron_id=8, params=params)
        
        # 检查初始状态
        self.assertEqual(microglia.activation_level, 0.0)
        self.assertTrue(microglia.surveillance_mode)
        self.assertEqual(microglia.cytokines['TNF_alpha'], 0.0)
        
        # 仿真正常状态
        for t in np.arange(0, 20.0, self.dt):
            microglia.update(self.dt, t)
        
        # 应该保持监视状态
        self.assertTrue(microglia.surveillance_mode)
        self.assertLess(microglia.activation_level, 0.1)
    
    def test_neuromodulator_system(self):
        """测试神经调质系统"""
        
        config = {'release_rates': {'dopamine': 0.02}}
        neuromod_system = NeuromodulatorSystem(config)
        
        # 检查初始状态
        self.assertEqual(neuromod_system.concentrations['dopamine'], 0.0)
        
        # 模拟神经活动
        neural_activity = {'dopamine': 1.0}
        
        # 更新系统
        for t in np.arange(0, 50.0, self.dt):
            neuromod_system.update(self.dt, neural_activity)
        
        # 检查多巴胺浓度增加
        self.assertGreater(neuromod_system.concentrations['dopamine'], 0.0)
        
        # 测试应用到神经元
        test_neurons = [
            LIFNeuron(neuron_id=9, params=get_default_parameters(NeuronType.LIF)),
            LIFNeuron(neuron_id=10, params=get_default_parameters(NeuronType.LIF))
        ]
        
        neuromod_system.apply_to_neurons(test_neurons)
        
        # 检查神经调质是否应用
        for neuron in test_neurons:
            self.assertGreater(neuron.neuromodulation['dopamine'], 0.0)
    
    def test_neuron_factory(self):
        """测试神经元工厂函数"""
        
        # 测试各种神经元类型
        neuron_types = [
            NeuronType.LIF,
            NeuronType.HODGKIN_HUXLEY,
            NeuronType.ADAPTIVE_EXPONENTIAL,
            NeuronType.IZHIKEVICH,
            NeuronType.ASTROCYTE,
            NeuronType.MICROGLIA
        ]
        
        for neuron_type in neuron_types:
            params = get_default_parameters(neuron_type)
            neuron = create_neuron(neuron_type, neuron_id=1, params=params)
            
            self.assertIsInstance(neuron, BaseNeuronModel)
            self.assertEqual(neuron.neuron_type, neuron_type)
            self.assertEqual(neuron.neuron_id, 1)

    def test_cell_type_aliases_and_auto_factory(self):
        """测试 CellType/群体标签到具体模型的映射与 auto 推断"""

        neuron = create_neuron("pyramidal_l2/3", neuron_id=100, params={})
        self.assertIsInstance(neuron, AdExNeuron)
        self.assertEqual(neuron.neuron_type, NeuronType.ADAPTIVE_EXPONENTIAL)

        neuron = create_neuron("pv_interneuron", neuron_id=101, params={})
        self.assertIsInstance(neuron, IzhikevichNeuron)
        self.assertEqual(neuron.neuron_type, NeuronType.IZHIKEVICH)

        neuron = create_neuron("auto", neuron_id=102, params={"population_type": "inhibitory"})
        self.assertIsInstance(neuron, IzhikevichNeuron)

        neuron = create_neuron("auto", neuron_id=103, params={"population_type": "excitatory"})
        self.assertIsInstance(neuron, AdExNeuron)
    
    def test_neuromodulation_effects(self):
        """测试神经调质效应"""
        
        # 创建LIF神经元
        params = get_default_parameters(NeuronType.LIF)
        neuron = LIFNeuron(neuron_id=11, params=params)
        
        # 添加基础电流
        base_current = 8.0  # 接近阈值但不足以发放
        neuron.add_input_current(base_current)
        
        # 不加神经调质的仿真
        spike_count_baseline = 0
        neuron_copy = LIFNeuron(neuron_id=12, params=params)
        neuron_copy.add_input_current(base_current)
        
        for t in np.arange(0, self.simulation_time, self.dt):
            if neuron_copy.update(self.dt, t):
                spike_count_baseline += 1
        
        # 添加多巴胺调节
        neuron.apply_neuromodulation('dopamine', 0.5)
        
        # 加神经调质的仿真
        spike_count_modulated = 0
        for t in np.arange(0, self.simulation_time, self.dt):
            if neuron.update(self.dt, t):
                spike_count_modulated += 1
        
        # 神经调质应该影响发放
        # 注意：由于随机性，这个测试可能需要调整
        self.assertNotEqual(spike_count_baseline, spike_count_modulated)

class TestCompartmentParameters(unittest.TestCase):
    """室参数测试"""
    
    def test_default_compartment_parameters(self):
        """测试默认室参数"""
        
        params = CompartmentParameters()
        
        # 检查几何参数
        self.assertEqual(params.length, 100.0)
        self.assertEqual(params.diameter, 2.0)
        
        # 检查电学参数
        self.assertEqual(params.Ra, 150.0)
        self.assertEqual(params.Cm, 1.0)
        self.assertEqual(params.Rm, 30000.0)
        
        # 检查离子通道密度
        self.assertEqual(params.Na_density, 120.0)
        self.assertEqual(params.K_density, 36.0)
    
    def test_custom_compartment_parameters(self):
        """测试自定义室参数"""
        
        params = CompartmentParameters(
            length=200.0,
            diameter=5.0,
            Na_density=200.0
        )
        
        self.assertEqual(params.length, 200.0)
        self.assertEqual(params.diameter, 5.0)
        self.assertEqual(params.Na_density, 200.0)
        # 其他参数应该保持默认值
        self.assertEqual(params.K_density, 36.0)

class TestIonChannelState(unittest.TestCase):
    """离子通道状态测试"""
    
    def test_default_ion_channel_state(self):
        """测试默认离子通道状态"""
        
        state = IonChannelState()
        
        # 检查HH门控变量
        self.assertEqual(state.m, 0.0)
        self.assertEqual(state.h, 1.0)
        self.assertEqual(state.n, 0.0)
        
        # 检查钙浓度
        self.assertEqual(state.ca_i, 50e-6)

if __name__ == '__main__':
    unittest.main()
