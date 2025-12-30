"""
单细胞记录模块测试脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from single_cell_recording import IntracellularRecording, ExtracellularRecording
from neurons import HodgkinHuxleyNeuron

def test_intracellular_recording():
    """测试细胞内记录功能"""
    print("=== 细胞内记录测试 ===")
    neuron = HodgkinHuxleyNeuron()
    recorder = IntracellularRecording(sampling_rate=10000)
    
    # 测试不同输入电流下的响应
    for current in [0.2, 0.5, 1.0]:
        recording = recorder.record_membrane_potential(
            neuron, duration=500, input_current=current)
        analysis = recorder.analyze_membrane_potential(recording)
        
        print(f"\n输入电流: {current}nA")
        print(f"尖峰数量: {analysis['n_spikes']}")
        print(f"平均频率: {analysis['firing_rate']:.1f}Hz")
        print(f"输入电阻: {analysis['r_input']:.1f}MΩ")
        
        # 可视化
        fig = recorder.plot_recording(recording, 
            title=f"Intracellular Recording ({current}nA)")
        plt.show()

def test_extracellular_recording():
    """测试细胞外记录功能"""
    print("\n=== 细胞外记录测试 ===")
    neuron = HodgkinHuxleyNeuron()
    recorder = ExtracellularRecording(sampling_rate=30000)
    
    recording = recorder.record_spikes(neuron, duration=1000, input_current=0.7)
    spikes = recorder.detect_spikes(recording)
    
    print(f"检测到尖峰数量: {len(spikes['spike_times'])}")
    
    # 可视化
    fig = recorder.plot_recording(recording)
    plt.scatter(spikes['spike_times'], 
               [recording['extracellular_potential'][i] for i in spikes['spike_indices']],
               color='red', label='Detected Spikes')
    plt.legend()
    plt.show()

def test_data_persistence():
    """测试数据保存/加载功能"""
    print("\n=== 数据持久化测试 ===")
    import pickle
    import os
    
    neuron = HodgkinHuxleyNeuron()
    recorder = IntracellularRecording()
    recording = recorder.record_membrane_potential(neuron, duration=200)
    
    # 保存数据
    with open('recording_data.pkl', 'wb') as f:
        pickle.dump(recording, f)
    print("数据已保存到 recording_data.pkl")
    
    # 加载数据
    with open('recording_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    print("数据加载成功，尖峰数量:", np.sum(loaded_data['spikes']))
    
    # 清理测试文件
    os.remove('recording_data.pkl')

if __name__ == "__main__":
    test_intracellular_recording()
    test_extracellular_recording()
    test_data_persistence()