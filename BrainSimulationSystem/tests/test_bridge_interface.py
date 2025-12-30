# -*- coding: utf-8 -*-
"""
桥接器事件接口最小化测试
"""

import unittest

from BrainSimulationSystem.core.network import create_full_brain_network

class DummyBridge(object):
    """占位桥接器，仅用于验证事件收集接口"""
    def __init__(self):
        self.received = []

    async def process_brain_spikes(self, spikes):
        # 记录输入尖峰事件
        self.received.extend(spikes)
        # 回传空结果（占位）
        return []

class TestBridgeInterface(unittest.TestCase):
    """测试网络与桥接器的事件接口"""

    def setUp(self):
        self.net = create_full_brain_network()
        # 设置占位桥接器
        self.bridge = DummyBridge()
        self.net.set_neuromorphic_bridge(self.bridge)

    def test_collect_bridge_spikes(self):
        """单步更新后应可获取桥接尖峰事件"""
        dt = 0.1
        res = self.net.update(dt, external_inputs={})
        spikes = self.net.get_last_bridge_spikes()
        # 尖峰事件列表应存在
        self.assertIsInstance(spikes, list)
        # 每个事件应为(神经元ID, 时间戳ms)格式
        for item in spikes[:10]:
            self.assertIsInstance(item, tuple)
            self.assertIsInstance(item[0], int)
            self.assertIsInstance(item[1], float)

if __name__ == "__main__":
    unittest.main()