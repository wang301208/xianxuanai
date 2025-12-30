"""
可视化数据处理器

将神经网络数据转换为可视化数据。
"""

import numpy as np
import json


class VisualizationDataProcessor:
    """可视化数据处理器基类"""
    
    def __init__(self, params=None):
        """
        初始化可视化数据处理器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
    
    def process_data(self, data):
        """
        处理数据
        
        参数:
            data: 输入数据
            
        返回:
            处理后的数据
        """
        return data
    
    def to_json(self, data):
        """
        将数据转换为JSON格式
        
        参数:
            data: 输入数据
            
        返回:
            JSON格式的数据
        """
        return json.dumps(data)


class NeuronActivityProcessor(VisualizationDataProcessor):
    """神经元活动数据处理器"""
    
    def process_data(self, data):
        """
        处理神经元活动数据
        
        参数:
            data: 神经元活动数据，格式为 {neuron_id: activity_value}
            
        返回:
            处理后的数据，格式为 {nodes: [{id, value, x, y, z}]}
        """
        nodes = []
        
        # 获取布局参数
        layout_type = self.params.get("layout", "grid")
        
        # 处理每个神经元
        for i, (neuron_id, activity) in enumerate(data.items()):
            # 根据布局类型计算位置
            if layout_type == "grid":
                # 网格布局
                grid_size = int(np.ceil(np.sqrt(len(data))))
                x = (i % grid_size) / grid_size
                y = (i // grid_size) / grid_size
                z = 0
            elif layout_type == "sphere":
                # 球形布局
                phi = np.arccos(1 - 2 * (i / len(data)))
                theta = np.pi * (1 + 5**0.5) * i
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
            else:
                # 默认随机布局
                x = np.random.random()
                y = np.random.random()
                z = np.random.random()
            
            # 添加节点
            nodes.append({
                "id": neuron_id,
                "value": float(activity),
                "x": float(x),
                "y": float(y),
                "z": float(z)
            })
        
        return {"nodes": nodes}


class SynapseActivityProcessor(VisualizationDataProcessor):
    """突触活动数据处理器"""
    
    def process_data(self, data):
        """
        处理突触活动数据
        
        参数:
            data: 突触活动数据，格式为 {(pre_id, post_id): weight}
            
        返回:
            处理后的数据，格式为 {links: [{source, target, value}]}
        """
        links = []
        
        # 处理每个突触
        for (pre_id, post_id), weight in data.items():
            # 添加连接
            links.append({
                "source": pre_id,
                "target": post_id,
                "value": float(weight)
            })
        
        return {"links": links}


class NetworkActivityProcessor(VisualizationDataProcessor):
    """网络活动数据处理器"""
    
    def __init__(self, params=None):
        """
        初始化网络活动数据处理器
        
        参数:
            params (dict): 配置参数
        """
        super().__init__(params)
        self.neuron_processor = NeuronActivityProcessor(params)
        self.synapse_processor = SynapseActivityProcessor(params)
    
    def process_data(self, data):
        """
        处理网络活动数据
        
        参数:
            data: 网络活动数据，格式为 {
                neurons: {neuron_id: activity_value},
                synapses: {(pre_id, post_id): weight}
            }
            
        返回:
            处理后的数据，格式为 {
                nodes: [{id, value, x, y, z}],
                links: [{source, target, value}]
            }
        """
        # 处理神经元数据
        neuron_data = self.neuron_processor.process_data(data.get("neurons", {}))
        
        # 处理突触数据
        synapse_data = self.synapse_processor.process_data(data.get("synapses", {}))
        
        # 合并数据
        return {
            "nodes": neuron_data.get("nodes", []),
            "links": synapse_data.get("links", [])
        }


class CognitiveActivityProcessor(VisualizationDataProcessor):
    """认知活动数据处理器"""
    
    def process_data(self, data):
        """
        处理认知活动数据
        
        参数:
            data: 认知活动数据，格式为 {
                attention: {region: value},
                working_memory: {item_id: {content, activation}},
                neuromodulators: {name: level}
            }
            
        返回:
            处理后的数据
        """
        result = {
            "attention": {},
            "working_memory": {},
            "neuromodulators": {}
        }
        
        # 处理注意力数据
        if "attention" in data:
            result["attention"] = {
                "regions": list(data["attention"].keys()),
                "values": list(data["attention"].values())
            }
        
        # 处理工作记忆数据
        if "working_memory" in data:
            items = []
            for item_id, item_data in data["working_memory"].items():
                items.append({
                    "id": item_id,
                    "content": item_data.get("content", ""),
                    "activation": float(item_data.get("activation", 0))
                })
            result["working_memory"] = {
                "items": items,
                "capacity": len(items)
            }
        
        # 处理神经调质数据
        if "neuromodulators" in data:
            result["neuromodulators"] = {
                "names": list(data["neuromodulators"].keys()),
                "levels": list(data["neuromodulators"].values())
            }
        
        return result


def create_data_processor(processor_type, params=None):
    """
    创建数据处理器
    
    参数:
        processor_type (str): 处理器类型
        params (dict): 配置参数
        
    返回:
        数据处理器实例
    """
    processors = {
        "neuron": NeuronActivityProcessor,
        "synapse": SynapseActivityProcessor,
        "network": NetworkActivityProcessor,
        "cognitive": CognitiveActivityProcessor
    }
    
    if processor_type not in processors:
        raise ValueError(f"未知的处理器类型: {processor_type}")
    
    return processors[processor_type](params)