"""
大脑模拟系统可视化模块

提供神经活动和网络状态的可视化功能。
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from IPython.display import display, clear_output
import time
import threading

class BrainVisualizer:
    """
    大脑模拟系统可视化器
    
    提供神经活动和网络状态的可视化功能。
    """
    
    def __init__(self, brain_simulation):
        """
        初始化可视化器
        
        Args:
            brain_simulation: 大脑模拟系统实例
        """
        self.brain_simulation = brain_simulation
        self.network = brain_simulation.network
        
        # 可视化配置
        self.config = {
            "update_interval": 100,  # 更新间隔（毫秒）
            "neuron_colors": {
                "excitatory": "red",
                "inhibitory": "blue",
                "sensory": "green",
                "motor": "purple"
            },
            "spike_color": "yellow",
            "connection_color": "gray",
            "connection_width_scale": 2.0,
            "neuron_size_scale": 100,
            "layout": "spring"  # 可选：spring, circular, kamada_kawai, spectral
        }
        
        # 可视化状态
        self.is_visualizing = False
        self.visualization_thread = None
        self.figures = {}
        self.axes = {}
        self.plots = {}
        self._latest_step_event = None
        self._latest_end_event = None
        self._stop_requested = False
        
        # 注册事件回调
        self.brain_simulation.register_event_callback("step", self._on_simulation_step)
        self.brain_simulation.register_event_callback("simulation_end", self._on_simulation_end)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        配置可视化器
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
    
    def create_network_graph(self) -> nx.DiGraph:
        """
        创建网络图
        
        Returns:
            NetworkX有向图
        """
        G = nx.DiGraph()
        
        # 添加节点
        for neuron_id, neuron in self.network.neurons.items():
            G.add_node(neuron_id, 
                       type=neuron.type,
                       position=neuron.position if hasattr(neuron, "position") else None)
        
        # 添加边
        for (pre_id, post_id), synapse in self.network.synapses.items():
            G.add_edge(pre_id, post_id, 
                       weight=synapse.weight,
                       type=synapse.type if hasattr(synapse, "type") else "excitatory")
        
        return G
    
    def visualize_network_structure(self, figsize=(10, 8)) -> None:
        """
        可视化网络结构
        
        Args:
            figsize: 图形大小
        """
        # 创建网络图
        G = self.create_network_graph()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        self.figures["network_structure"] = fig
        self.axes["network_structure"] = ax
        
        # 设置布局
        if self.config["layout"] == "spring":
            pos = nx.spring_layout(G)
        elif self.config["layout"] == "circular":
            pos = nx.circular_layout(G)
        elif self.config["layout"] == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif self.config["layout"] == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "excitatory")
            node_colors.append(self.config["neuron_colors"].get(node_type, "gray"))
            node_sizes.append(self.config["neuron_size_scale"])
        
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        # 绘制边
        edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        edge_colors = []
        for u, v in G.edges():
            synapse_type = G[u][v].get("type", "excitatory")
            if synapse_type == "excitatory":
                edge_colors.append("green")
            else:
                edge_colors.append("red")
        
        nx.draw_networkx_edges(G, pos, 
                              width=[abs(w) * self.config["connection_width_scale"] for w in edge_weights],
                              edge_color=edge_colors,
                              alpha=0.6,
                              ax=ax)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)
        
        # 设置标题和坐标轴
        ax.set_title("神经网络结构")
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_activity(self, figsize=(12, 10)) -> None:
        """
        可视化神经活动
        
        Args:
            figsize: 图形大小
        """
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        self.figures["activity"] = fig
        self.axes["activity"] = axes
        
        # 初始化数据
        times = self.brain_simulation.simulation_results["times"]
        if not times:
            times = [0]
            spikes = [[]]
            voltages = [[]]
            weights = [[]]
        else:
            spikes = self.brain_simulation.simulation_results["spikes"]
            voltages = self.brain_simulation.simulation_results["voltages"]
            weights = self.brain_simulation.simulation_results["weights"]
        
        # 绘制脉冲图
        ax_spikes = axes[0]
        ax_spikes.set_title("神经元脉冲活动")
        ax_spikes.set_ylabel("神经元ID")
        self.plots["spikes"] = ax_spikes.spy(np.array(spikes).T, markersize=2)
        
        # 绘制电压图
        ax_voltages = axes[1]
        ax_voltages.set_title("神经元膜电位")
        ax_voltages.set_ylabel("膜电位 (mV)")
        
        # 选择一些神经元进行可视化
        neuron_ids = list(self.network.neurons.keys())
        selected_neurons = neuron_ids[:min(5, len(neuron_ids))]
        
        for neuron_id in selected_neurons:
            neuron_voltages = [v[neuron_id] if neuron_id < len(v) else 0 for v in voltages]
            line, = ax_voltages.plot(times, neuron_voltages, label=f"神经元 {neuron_id}")
            self.plots[f"voltage_{neuron_id}"] = line
        
        ax_voltages.legend(loc="upper right")
        
        # 绘制权重图
        ax_weights = axes[2]
        ax_weights.set_title("突触权重变化")
        ax_weights.set_xlabel("时间 (ms)")
        ax_weights.set_ylabel("权重")
        
        # 选择一些突触进行可视化
        synapse_keys = list(self.network.synapses.keys())
        selected_synapses = synapse_keys[:min(5, len(synapse_keys))]
        
        for i, (pre_id, post_id) in enumerate(selected_synapses):
            synapse_weights = []
            for w in weights:
                if i < len(w):
                    synapse_weights.append(w[i])
                else:
                    synapse_weights.append(0)
            
            line, = ax_weights.plot(times, synapse_weights, label=f"{pre_id}->{post_id}")
            self.plots[f"weight_{pre_id}_{post_id}"] = line
        
        ax_weights.legend(loc="upper right")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cognitive_state(self, figsize=(12, 10)) -> None:
        """
        可视化认知状态
        
        Args:
            figsize: 图形大小
        """
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.figures["cognitive"] = fig
        self.axes["cognitive"] = axes
        
        # 初始化数据
        cognitive_states = self.brain_simulation.simulation_results.get("cognitive_states", [])
        if not cognitive_states:
            cognitive_states = [{"perception": {}, "attention": {}, "memory": {}, "decision": {}}]
        
        # 绘制感知状态
        ax_perception = axes[0, 0]
        ax_perception.set_title("感知状态")
        ax_perception.set_xlabel("特征")
        ax_perception.set_ylabel("激活强度")
        
        # 获取最新的感知状态
        perception_data = cognitive_states[-1]["perception"].get("perception_output", {})
        if perception_data:
            features = list(perception_data.keys())
            values = list(perception_data.values())
            self.plots["perception"] = ax_perception.bar(features, values)
        
        # 绘制注意力状态
        ax_attention = axes[0, 1]
        ax_attention.set_title("注意力状态")
        ax_attention.set_xlabel("位置")
        ax_attention.set_ylabel("注意力强度")
        
        # 获取最新的注意力状态
        attention_data = cognitive_states[-1]["attention"].get("attention_map", {})
        if attention_data:
            positions = list(attention_data.keys())
            values = list(attention_data.values())
            self.plots["attention"] = ax_attention.plot(positions, values)[0]
        
        # 绘制记忆状态
        ax_memory = axes[1, 0]
        ax_memory.set_title("记忆状态")
        ax_memory.set_xlabel("记忆项")
        ax_memory.set_ylabel("记忆强度")
        
        # 获取最新的记忆状态
        memory_data = cognitive_states[-1]["memory"].get("memory_strengths", {})
        if memory_data:
            memory_items = list(memory_data.keys())
            values = list(memory_data.values())
            self.plots["memory"] = ax_memory.bar(memory_items, values)
        
        # 绘制决策状态
        ax_decision = axes[1, 1]
        ax_decision.set_title("决策状态")
        ax_decision.set_xlabel("选项")
        ax_decision.set_ylabel("选择概率")
        
        # 获取最新的决策状态
        decision_data = cognitive_states[-1]["decision"].get("action_values", {})
        if decision_data:
            options = list(decision_data.keys())
            values = list(decision_data.values())
            self.plots["decision"] = ax_decision.bar(options, values)
        
        plt.tight_layout()
        plt.show()
    
    def start_live_visualization(self) -> None:
        """开始实时可视化"""
        if self.is_visualizing:
            return
        
        self.is_visualizing = True
        
        # 创建可视化线程
        self.visualization_thread = threading.Thread(
            target=self._live_visualization_loop
        )
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
    
    def stop_live_visualization(self) -> None:
        """停止实时可视化"""
        if not self.is_visualizing:
            return
        
        self.is_visualizing = False
        
        # 等待线程结束
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
            self.visualization_thread = None
    
    def _live_visualization_loop(self) -> None:
        """实时可视化循环"""
        # 创建图形
        plt.ion()  # 开启交互模式
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        self.figures["live"] = fig
        self.axes["live"] = axes
        
        # 初始化绘图
        ax_spikes = axes[0, 0]
        ax_spikes.set_title("神经元脉冲活动")
        ax_spikes.set_ylabel("神经元ID")
        ax_spikes.set_xlabel("时间")
        
        ax_voltages = axes[0, 1]
        ax_voltages.set_title("神经元膜电位")
        ax_voltages.set_ylabel("膜电位 (mV)")
        ax_voltages.set_xlabel("时间")
        
        ax_weights = axes[1, 0]
        ax_weights.set_title("突触权重变化")
        ax_weights.set_ylabel("权重")
        ax_weights.set_xlabel("时间")
        
        ax_cognitive = axes[1, 1]
        ax_cognitive.set_title("认知状态")
        ax_cognitive.set_ylabel("激活强度")
        ax_cognitive.set_xlabel("认知组件")
        
        plt.tight_layout()
        
        # 选择一些神经元和突触进行可视化
        neuron_ids = list(self.network.neurons.keys())
        selected_neurons = neuron_ids[:min(5, len(neuron_ids))]
        
        synapse_keys = list(self.network.synapses.keys())
        selected_synapses = synapse_keys[:min(5, len(synapse_keys))]
        
        # 初始化绘图对象
        voltage_lines = []
        for neuron_id in selected_neurons:
            line, = ax_voltages.plot([], [], label=f"神经元 {neuron_id}")
            voltage_lines.append(line)
        
        weight_lines = []
        for pre_id, post_id in selected_synapses:
            line, = ax_weights.plot([], [], label=f"{pre_id}->{post_id}")
            weight_lines.append(line)
        
        # 添加图例
        ax_voltages.legend(loc="upper right")
        ax_weights.legend(loc="upper right")
        
        # 可视化循环
        while self.is_visualizing:
            try:
                # 获取最新数据
                times = self.brain_simulation.simulation_results["times"]
                spikes = self.brain_simulation.simulation_results["spikes"]
                voltages = self.brain_simulation.simulation_results["voltages"]
                weights = self.brain_simulation.simulation_results["weights"]
                cognitive_states = self.brain_simulation.simulation_results.get("cognitive_states", [])
                
                if not times:
                    time.sleep(0.1)
                    continue
                
                # 更新脉冲图
                ax_spikes.clear()
                ax_spikes.set_title("神经元脉冲活动")
                ax_spikes.set_ylabel("神经元ID")
                ax_spikes.set_xlabel("时间")
                
                # 创建脉冲矩阵
                spike_matrix = np.zeros((len(neuron_ids), len(times)))
                for t, spike_list in enumerate(spikes):
                    for neuron_id in spike_list:
                        if neuron_id < len(neuron_ids):
                            spike_matrix[neuron_id, t] = 1
                
                ax_spikes.spy(spike_matrix, markersize=2)
                
                # 更新电压图
                for i, neuron_id in enumerate(selected_neurons):
                    neuron_voltages = [v[neuron_id] if neuron_id < len(v) else 0 for v in voltages]
                    voltage_lines[i].set_data(times, neuron_voltages)
                
                ax_voltages.relim()
                ax_voltages.autoscale_view()
                
                # 更新权重图
                for i, ((pre_id, post_id), line) in enumerate(zip(selected_synapses, weight_lines)):
                    synapse_weights = []
                    for w in weights:
                        if i < len(w):
                            synapse_weights.append(w[i])
                        else:
                            synapse_weights.append(0)
                    
                    line.set_data(times, synapse_weights)
                
                ax_weights.relim()
                ax_weights.autoscale_view()
                
                # 更新认知状态图
                if cognitive_states:
                    ax_cognitive.clear()
                    ax_cognitive.set_title("认知状态")
                    ax_cognitive.set_ylabel("激活强度")
                    ax_cognitive.set_xlabel("认知组件")
                    
                    # 获取最新的认知状态
                    latest_state = cognitive_states[-1]
                    
                    # 提取各个认知组件的激活强度
                    components = []
                    values = []
                    
                    # 感知
                    perception_output = latest_state["perception"].get("perception_output", {})
                    if perception_output:
                        avg_perception = sum(perception_output.values()) / len(perception_output)
                        components.append("感知")
                        values.append(avg_perception)
                    
                    # 注意力
                    attention_map = latest_state["attention"].get("attention_map", {})
                    if attention_map:
                        max_attention = max(attention_map.values())
                        components.append("注意力")
                        values.append(max_attention)
                    
                    # 记忆
                    memory_strengths = latest_state["memory"].get("memory_strengths", {})
                    if memory_strengths:
                        avg_memory = sum(memory_strengths.values()) / len(memory_strengths)
                        components.append("记忆")
                        values.append(avg_memory)
                    
                    # 决策
                    action_values = latest_state["decision"].get("action_values", {})
                    if action_values:
                        max_decision = max(action_values.values())
                        components.append("决策")
                        values.append(max_decision)
                    
                    # 绘制条形图
                    if components:
                        ax_cognitive.bar(components, values)
                
                # 更新图形
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                # 控制更新频率
                time.sleep(self.config["update_interval"] / 1000.0)
                
            except Exception as e:
                print(f"可视化错误: {e}")
                time.sleep(1.0)
        
        plt.ioff()  # 关闭交互模式
    
    def _on_simulation_step(self, event_data: Dict[str, Any]) -> None:
        """
        模拟步骤事件回调
        
        Args:
            event_data: 事件数据
        """
        self._latest_step_event = dict(event_data or {})
        if not self.is_visualizing:
            return
        if self._stop_requested:
            self.is_visualizing = False
    
    def _on_simulation_end(self, event_data: Dict[str, Any]) -> None:
        """
        模拟结束事件回调
        
        Args:
            event_data: 事件数据
        """
        self._latest_end_event = dict(event_data or {})
        self._stop_requested = True
        self.is_visualizing = False
    
    def save_visualization(self, filepath: str, figure_name: str = None) -> None:
        """
        保存可视化图形
        
        Args:
            filepath: 文件路径
            figure_name: 图形名称，如果为None则保存所有图形
        """
        if figure_name is not None and figure_name in self.figures:
            self.figures[figure_name].savefig(filepath)
        else:
            # 保存所有图形
            for name, fig in self.figures.items():
                filename = f"{filepath}_{name}.png"
                fig.savefig(filename)
