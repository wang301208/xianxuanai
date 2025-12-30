"""
FPGA硬件接口实现
FPGA Hardware Interface Implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from queue import Queue, Empty

from .neuromorphic_hardware import (
    NeuromorphicHardwareInterface, 
    HardwareConfiguration, 
    SpikeEvent
)

# FPGA特定导入
try:
    import pynq
    from pynq import Overlay
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False

class FPGAInterface(NeuromorphicHardwareInterface):
    """自定义FPGA硬件接口"""
    
    def __init__(self, config: HardwareConfiguration):
        super().__init__(config)
        self.overlay = None
        self.dma_buffers = {}
        self.interrupt_handlers = {}
        
        if not PYNQ_AVAILABLE:
            self.logger.warning("PYNQ不可用，使用仿真模式")
    
    async def connect(self) -> bool:
        """连接到FPGA硬件"""
        try:
            if PYNQ_AVAILABLE:
                # 加载FPGA覆盖层
                bitstream_path = self.config.custom_config.get('bitstream_path', 'neural_network.bit')
                self.overlay = Overlay(bitstream_path)
                
                # 初始化DMA引擎
                self._initialize_dma()
                
                # 设置中断处理
                self._setup_interrupts()
            
            self.is_connected = True
            self.logger.info("成功连接到FPGA硬件")
            return True
            
        except Exception as e:
            self.logger.error(f"连接FPGA硬件失败: {e}")
            return False
    
    def _initialize_dma(self):
        """初始化DMA引擎"""
        if self.overlay and hasattr(self.overlay, 'axi_dma_0'):
            self.dma = self.overlay.axi_dma_0
            
            # 分配DMA缓冲区
            buffer_size = self.config.custom_config.get('buffer_size', 1024 * 1024)  # 1MB
            self.dma_buffers['input'] = pynq.allocate(shape=(buffer_size,), dtype=np.uint32)
            self.dma_buffers['output'] = pynq.allocate(shape=(buffer_size,), dtype=np.uint32)
    
    def _setup_interrupts(self):
        """设置中断处理"""
        if self.overlay and hasattr(self.overlay, 'interrupt_controller'):
            # 注册中断处理函数
            self.overlay.interrupt_controller.register_interrupt_handler(
                0, self._spike_interrupt_handler
            )
    
    def _spike_interrupt_handler(self, interrupt_id):
        """尖峰中断处理函数"""
        # 从FPGA读取尖峰数据
        spike_data = self._read_spike_buffer()
        
        # 转换为事件对象
        for data in spike_data:
            event = SpikeEvent.from_bytes(data)
            self.output_queue.put(event)
    
    def _read_spike_buffer(self) -> List[bytes]:
        """从FPGA读取尖峰缓冲区"""
        spike_data = []
        
        if self.dma_buffers.get('output'):
            # 从DMA缓冲区读取数据
            buffer = self.dma_buffers['output']
            
            # 解析缓冲区中的尖峰事件
            for i in range(0, len(buffer), 8):  # 每个事件8字节
                if buffer[i] != 0:  # 检查是否有有效数据
                    event_bytes = buffer[i:i+8].tobytes()
                    spike_data.append(event_bytes)
        
        return spike_data
    
    async def configure_network(self, network_config: Dict[str, Any]) -> bool:
        """配置网络到FPGA"""
        try:
            if not self.overlay:
                self.logger.warning("FPGA覆盖层未加载，跳过配置")
                return True
            
            # 配置神经元参数
            neuron_config = self._prepare_neuron_config(network_config)
            self._write_neuron_config(neuron_config)
            
            # 配置连接权重
            weight_config = self._prepare_weight_config(network_config)
            self._write_weight_config(weight_config)
            
            # 启动FPGA神经网络核心
            self._start_neural_core()
            
            self.logger.info("FPGA网络配置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"FPGA网络配置失败: {e}")
            return False
    
    def _prepare_neuron_config(self, network_config: Dict[str, Any]) -> np.ndarray:
        """准备神经元配置数据"""
        neuron_groups = network_config.get('neuron_groups', [])
        
        # 创建配置数组
        config_array = np.zeros((self.config.max_neurons, 8), dtype=np.uint32)
        
        neuron_idx = 0
        for group in neuron_groups:
            for i in range(group['size']):
                if neuron_idx < self.config.max_neurons:
                    # 配置神经元参数
                    config_array[neuron_idx, 0] = int(group.get('threshold', -50) * 1000)  # mV -> μV
                    config_array[neuron_idx, 1] = int(group.get('reset', -65) * 1000)
                    config_array[neuron_idx, 2] = int(group.get('tau_m', 20) * 1000)  # ms -> μs
                    config_array[neuron_idx, 3] = int(group.get('refractory', 2) * 1000)
                    
                    neuron_idx += 1
        
        return config_array
    
    def _write_neuron_config(self, config_array: np.ndarray):
        """写入神经元配置到FPGA"""
        if self.overlay and hasattr(self.overlay, 'neuron_config_memory'):
            # 通过AXI接口写入配置
            memory = self.overlay.neuron_config_memory
            
            for i, neuron_config in enumerate(config_array):
                for j, param in enumerate(neuron_config):
                    memory.write(i * 32 + j * 4, param)  # 每个参数4字节
    
    def _prepare_weight_config(self, network_config: Dict[str, Any]) -> np.ndarray:
        """准备权重配置数据"""
        connections = network_config.get('connections', [])
        
        # 创建权重矩阵
        weight_matrix = np.zeros((self.config.max_neurons, self.config.max_neurons), dtype=np.int16)
        
        for conn in connections:
            source_id = conn['source_group']
            target_id = conn['target_group']
            weights = conn.get('weight_matrix', 1.0)
            
            if isinstance(weights, (int, float)):
                weight_matrix[source_id, target_id] = int(weights * 1000)  # 转换为定点数
            else:
                # 处理权重矩阵
                for i, row in enumerate(weights):
                    for j, weight in enumerate(row):
                        if source_id + i < self.config.max_neurons and target_id + j < self.config.max_neurons:
                            weight_matrix[source_id + i, target_id + j] = int(weight * 1000)
        
        return weight_matrix
    
    def _write_weight_config(self, weight_matrix: np.ndarray):
        """写入权重配置到FPGA"""
        if self.overlay and hasattr(self.overlay, 'weight_memory'):
            # 通过DMA写入权重矩阵
            flattened_weights = weight_matrix.flatten()
            
            if 'input' in self.dma_buffers:
                buffer = self.dma_buffers['input']
                buffer[:len(flattened_weights)] = flattened_weights
                
                # 启动DMA传输
                self.dma.sendchannel.transfer(buffer)
                self.dma.sendchannel.wait()
    
    def _start_neural_core(self):
        """启动FPGA神经网络核心"""
        if self.overlay and hasattr(self.overlay, 'neural_core_control'):
            control_reg = self.overlay.neural_core_control
            control_reg.write(0x00, 0x01)  # 启动信号
    
    async def send_spike_events(self, events: List[SpikeEvent]):
        """发送尖峰事件到FPGA"""
        if not events:
            return
        
        # 将事件转换为字节流
        event_data = b''.join(event.to_bytes() for event in events)
        
        if 'input' in self.dma_buffers:
            buffer = self.dma_buffers['input']
            
            # 将事件数据复制到DMA缓冲区
            event_array = np.frombuffer(event_data, dtype=np.uint32)
            buffer[:len(event_array)] = event_array
            
            # 启动DMA传输
            if self.dma:
                self.dma.sendchannel.transfer(buffer[:len(event_array)])
                self.dma.sendchannel.wait()
    
    async def receive_spike_events(self) -> List[SpikeEvent]:
        """从FPGA接收尖峰事件"""
        events = []
        
        try:
            while True:
                event = self.output_queue.get_nowait()
                events.append(event)
        except Empty:
            pass
        
        return events
    
    async def start_execution(self):
        """开始FPGA执行"""
        try:
            # 启动神经网络处理
            self._start_neural_core()
            
            # 启用中断
            if self.overlay and hasattr(self.overlay, 'interrupt_controller'):
                self.overlay.interrupt_controller.enable_interrupt(0)
            
            self.is_running = True
            self.logger.info("FPGA执行已开始")
            
        except Exception as e:
            self.logger.error(f"启动FPGA执行失败: {e}")
    
    async def stop_execution(self):
        """停止FPGA执行"""
        try:
            # 停止神经网络处理
            if self.overlay and hasattr(self.overlay, 'neural_core_control'):
                control_reg = self.overlay.neural_core_control
                control_reg.write(0x00, 0x00)  # 停止信号
            
            # 禁用中断
            if self.overlay and hasattr(self.overlay, 'interrupt_controller'):
                self.overlay.interrupt_controller.disable_interrupt(0)
            
            self.is_running = False
            self.logger.info("FPGA执行已停止")
            
        except Exception as e:
            self.logger.error(f"停止FPGA执行失败: {e}")
    
    async def disconnect(self):
        """断开FPGA连接"""
        await self.stop_execution()
        
        # 释放DMA缓冲区
        for buffer in self.dma_buffers.values():
            if hasattr(buffer, 'freebuffer'):
                buffer.freebuffer()
        
        self.is_connected = False
        self.logger.info("已断开FPGA硬件连接")
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """获取FPGA硬件状态"""
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'overlay_loaded': self.overlay is not None,
            'dma_buffers': len(self.dma_buffers),
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'performance': self.performance_metrics.copy()
        }