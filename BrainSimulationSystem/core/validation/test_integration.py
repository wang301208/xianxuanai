"""
单细胞记录集成测试模块

测试与神经元模型的完整集成
"""

import numpy as np
from neurons import HodgkinHuxleyNeuron, PositionalNeuron
from single_cell_recording import IntracellularRecording, MultiElectrodeArray

def test_positional_recording():
    """测试带位置信息的神经元记录"""
    print("=== 带位置神经元测试 ===")
    
    # 创建基础神经元
    base_neuron = HodgkinHuxleyNeuron()
    
    # 添加位置信息
    neuron = PositionalNeuron(base_neuron, position=[10, 20, 30])
    
    # 记录测试
    recorder = IntracellularRecording()
    recording = recorder.record_membrane_potential(neuron, duration=200)
    
    print(f"记录成功，膜电位范围: {np.min(recording['membrane_potential']):.1f}mV 到 {np.max(recording['membrane_potential']):.1f}mV")
    print(f"神经元位置: {neuron.position}")

def test_multi_electrode_array():
    """测试多电极阵列功能"""
    print("\n=== 多电极阵列测试 ===")
    
    # 创建3个带位置的神经元
    neurons = [
        PositionalNeuron(HodgkinHuxleyNeuron(), position=[i*100, 0, 0]) 
        for i in range(3)
    ]
    
    # 创建4电极阵列
    mea = MultiElectrodeArray(n_electrodes=4, spacing=150.0)
    
    # 记录网络活动
    recordings = mea.record_network(neurons, duration=500)
    
    print(f"成功记录 {len(recordings)} 个电极数据")
    for i, rec in enumerate(recordings):
        spikes = np.sum(rec['spikes'])
        print(f"电极 {i+1} (位置: {rec['electrode_pos']}): 检测到 {spikes} 个尖峰")

if __name__ == "__main__":
    test_positional_recording()
    test_multi_electrode_array()