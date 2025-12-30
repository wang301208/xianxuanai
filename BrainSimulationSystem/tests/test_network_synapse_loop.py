# -*- coding: utf-8 -*-
"""
网络-突触闭环最小化集成测试
"""

import unittest

from BrainSimulationSystem.core.network import NeuralNetwork, create_full_brain_network
from BrainSimulationSystem.core.network_models import FeedForwardNetwork
from BrainSimulationSystem.core.neuron_base import NeuronType

class TestNetworkSynapseLoop(unittest.TestCase):
    """测试网络更新中的突触闭环是否工作"""

    def setUp(self):
        # 创建默认配置下的完整网络对象（小规模可运行）
        self.net = create_full_brain_network()

    def test_single_update_step_synapse_loop(self):
        """单步更新后应产生突触统计输出"""
        dt = 0.1  # 毫秒
        res = self.net.update(dt, external_inputs={})
        # 断言存在突触更新结果
        self.assertIn("synapse_update", res)
        syn = res["synapse_update"]
        # 活跃突触计数应为整数
        self.assertIn("active_synapses", syn)
        self.assertIsInstance(syn["active_synapses"], int)
        # 统计信息应包含总突触数与分布
        self.assertIn("stats", syn)
        self.assertIn("total_synapses", syn["stats"])
        self.assertIn("neurotransmitter_distribution", syn["stats"])

    def test_unified_neural_network_interface(self):
        """结构化网络模型应复用统一的 NeuralNetwork 接口"""

        structured = FeedForwardNetwork(network_id=42, params={"dt": 0.1})

        # 添加最小结构：两层两个神经元以及突触连接
        structured.add_neuron(0, NeuronType.LIF, {"V_rest": -65.0, "threshold": -50.0})
        structured.add_neuron(1, NeuronType.LIF, {"V_rest": -65.0, "threshold": -50.0})
        structured.add_layer([0])
        structured.add_layer([1])
        structured.connect_layers(0, 1, connection_prob=1.0, synapse_params={"weight": 0.5})

        # 两种网络都应是统一基类的实例
        self.assertIsInstance(self.net, NeuralNetwork)
        self.assertIsInstance(structured, NeuralNetwork)

        # 统一接口方法在不同实现中应保持一致
        structured.set_input([0.3])
        self.net.set_input([0.7])
        self.assertTrue(structured._input_buffer)
        self.assertTrue(self.net._input_buffer)

        structured.reset()
        self.net.reset()
        self.assertEqual(structured._input_buffer, [])
        self.assertEqual(self.net._input_buffer, [])

        # step 支持统一的 dt 参数
        result = structured.step(0.1)
        self.assertIn("time", result)
        self.assertIn("network_state", result)
        self.assertIn(0, result["network_state"])

if __name__ == "__main__":
    unittest.main()
