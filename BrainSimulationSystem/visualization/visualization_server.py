"""
大脑模拟系统可视化服务器

提供Web界面，用于可视化大脑模拟系统的神经活动和认知过程。
"""

import sys
import os
import json
import threading
import time
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, render_template, Response
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from BrainSimulationSystem.models.cognitive_controller import CognitiveControllerBuilder
from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.config.cognitive_defaults import (
    DEFAULT_ATTENTION_PARAMS,
    DEFAULT_WORKING_MEMORY_PARAMS,
    load_cognitive_defaults,
)


class VisualizationServer:
    """大脑模拟系统可视化服务器类"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        """初始化可视化服务器"""
        self.app = Flask(__name__, 
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        self.host = host
        self.port = port
        
        # 创建认知控制器
        self.controller = self._create_controller()
        
        # 模拟状态
        self.simulation_running = False
        self.simulation_thread = None
        self.simulation_results = []
        self.current_step = 0
        self.max_steps = 100
        self.step_interval = 0.5  # 秒
        
        # 可视化数据
        self.visualization_data = {
            "neural_activity": [],
            "attention_focus": [],
            "memory_content": [],
            "neuromodulators": []
        }
        
        # 注册路由
        self._register_routes()
    
    def _create_controller(self):
        """Create cognitive controller."""
        defaults = load_cognitive_defaults()
        attention_params = dict(defaults.get("attention", DEFAULT_ATTENTION_PARAMS))
        memory_params = dict(defaults.get("working_memory", DEFAULT_WORKING_MEMORY_PARAMS))

        builder = CognitiveControllerBuilder()
        builder.with_attention_params(attention_params)
        builder.with_working_memory_params(memory_params)

        return builder.build()

    def _register_routes(self):
        """注册Web路由"""
        # 主页
        self.app.route('/')(self.index)
        
        # 可视化页面
        self.app.route('/visualization')(self.visualization_page)
        self.app.route('/network')(self.network_page)
        self.app.route('/cognitive')(self.cognitive_page)
        
        # API端点
        self.app.route('/api/visualization/data')(self.get_visualization_data)
        self.app.route('/api/simulation/start', methods=['POST'])(self.start_simulation)
        self.app.route('/api/simulation/stop', methods=['POST'])(self.stop_simulation)
        self.app.route('/api/simulation/status')(self.simulation_status)
        self.app.route('/api/cognitive/state')(self.get_cognitive_state)
        self.app.route('/api/neuromodulators')(self.get_neuromodulators)
        
        # 实时数据流
        self.app.route('/api/stream/neural_activity')(self.stream_neural_activity)
        self.app.route('/api/stream/cognitive_state')(self.stream_cognitive_state)
    
    def run(self):
        """运行可视化服务器"""
        # 确保模板和静态文件目录存在
        self._ensure_directories()
        
        # 创建默认模板和静态文件
        from BrainSimulationSystem.visualization.template_generator import create_default_templates
        from BrainSimulationSystem.visualization.static_generator import create_default_static_files
        
        create_default_templates()
        create_default_static_files()
        
        # 运行Flask应用
        self.app.run(host=self.host, port=self.port, debug=False)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        static_js_dir = os.path.join(static_dir, 'js')
        static_css_dir = os.path.join(static_dir, 'css')
        
        # 创建目录
        for directory in [template_dir, static_dir, static_js_dir, static_css_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    # 页面路由处理函数
    
    def index(self):
        """首页"""
        return render_template('index.html')
    
    def visualization_page(self):
        """可视化页面"""
        return render_template('visualization.html')
    
    def network_page(self):
        """神经网络页面"""
        return render_template('network.html')
    
    def cognitive_page(self):
        """认知过程页面"""
        return render_template('cognitive.html')
    
    # API端点处理函数
    
    def get_visualization_data(self):
        """获取可视化数据"""
        return jsonify(self.visualization_data)
    
    def start_simulation(self):
        """启动模拟"""
        if self.simulation_running:
            return jsonify({"error": "模拟已在运行中"}), 400
            
        try:
            data = request.json or {}
            self.max_steps = int(data.get("steps", 100))
            self.step_interval = float(data.get("interval", 0.5))
            
            # 重置模拟状态
            self.current_step = 0
            self.simulation_results = []
            self.simulation_running = True
            
            # 启动模拟线程
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            return jsonify({
                "status": "started",
                "max_steps": self.max_steps,
                "step_interval": self.step_interval
            })
        except Exception as e:
            self.simulation_running = False
            return jsonify({"error": str(e)}), 500
    
    def stop_simulation(self):
        """停止模拟"""
        if not self.simulation_running:
            return jsonify({"error": "模拟未在运行"}), 400
            
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
            
        return jsonify({
            "status": "stopped",
            "completed_steps": self.current_step,
            "results_count": len(self.simulation_results)
        })
    
    def simulation_status(self):
        """获取模拟状态"""
        status = {
            "running": self.simulation_running,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress": (self.current_step / self.max_steps) * 100 if self.max_steps > 0 else 0,
            "results_count": len(self.simulation_results)
        }
        return jsonify(status)
    
    def get_cognitive_state(self):
        """获取认知状态"""
        state = {
            "cognitive_state": self.controller.state.name,
            "state_history": [s.name for s in self.controller.state_history[-10:]],
            "neuromodulators": self.controller.neuromodulators
        }
        return jsonify(state)
    
    def get_neuromodulators(self):
        """获取神经调质水平"""
        return jsonify(self.controller.neuromodulators)
    
    def stream_neural_activity(self):
        """流式传输神经活动数据"""
        def generate():
            while True:
                if self.simulation_running and self.visualization_data["neural_activity"]:
                    data = json.dumps({
                        "timestamp": time.time(),
                        "data": self.visualization_data["neural_activity"][-1]
                    })
                    yield f"data: {data}\n\n"
                time.sleep(0.5)
                
        return Response(generate(), mimetype="text/event-stream")
    
    def stream_cognitive_state(self):
        """流式传输认知状态数据"""
        def generate():
            while True:
                if self.simulation_running:
                    data = json.dumps({
                        "timestamp": time.time(),
                        "cognitive_state": self.controller.state.name,
                        "neuromodulators": self.controller.neuromodulators
                    })
                    yield f"data: {data}\n\n"
                time.sleep(0.5)
                
        return Response(generate(), mimetype="text/event-stream")
    
    def _run_simulation(self):
        """运行模拟线程"""
        try:
            for step in range(self.max_steps):
                if not self.simulation_running:
                    break
                    
                self.current_step = step
                
                # 生成模拟输入
                sensory_input = self._generate_sensory_input(step)
                neuromodulators = self._modulate_neuromodulators(step)
                
                # 准备控制器输入
                controller_input = {
                    "sensory_input": sensory_input,
                    "neuromodulators": neuromodulators,
                    "task_goal": "simulation_task",
                    "decision_required": step % 10 == 0,
                    "response_required": step % 10 == 5,
                    "response_complete": step % 10 == 6
                }
                
                # 处理认知周期
                result = self.controller.process(controller_input)
                self.simulation_results.append(result)
                
                # 更新可视化数据
                self._update_visualization_data(result)
                
                # 等待下一步
                time.sleep(self.step_interval)
                
        except Exception as e:
            print(f"模拟线程错误: {e}")
        finally:
            self.simulation_running = False
    
    def _generate_sensory_input(self, step: int) -> Dict[str, Any]:
        """生成模拟的感觉输入"""
        # 模拟不同的感觉输入项
        sensory_input = {
            "visual_object_1": {"shape": "circle", "color": "red", "size": 0.8},
            "visual_object_2": {"shape": "square", "color": "blue", "size": 0.5},
            "auditory_input": {"frequency": 440, "volume": 0.7}
        }
        
        # 每3个时间步改变一个对象的属性
        if step % 3 == 0:
            colors = ["red", "blue", "green", "yellow", "purple"]
            sensory_input["visual_object_1"]["color"] = colors[step % len(colors)]
        
        # 每5个时间步添加一个新对象
        if step % 5 == 0:
            sensory_input[f"new_object_{step}"] = {
                "shape": "star", 
                "color": "yellow", 
                "size": 0.9,
                "novelty": 1.0
            }
            
        return sensory_input
    
    def _modulate_neuromodulators(self, step: int) -> Dict[str, float]:
        """模拟神经调质水平变化"""
        # 基础水平
        base_level = 0.5
        
        # 模拟乙酰胆碱水平变化 (注意力相关)
        ach_level = base_level + 0.3 * np.sin(step / 10)
        
        # 模拟多巴胺水平变化 (奖励相关)
        dopa_level = base_level
        if step % 15 == 0:
            dopa_level += 0.4  # 奖励峰值
        
        # 模拟去甲肾上腺素水平变化 (警觉相关)
        ne_level = base_level + 0.2 * np.random.random()
        
        # 模拟5-羟色胺水平变化 (情绪相关)
        serotonin_level = base_level + 0.1 * np.sin(step / 20)
        
        return {
            "acetylcholine": max(0.0, min(1.0, ach_level)),
            "dopamine": max(0.0, min(1.0, dopa_level)),
            "norepinephrine": max(0.0, min(1.0, ne_level)),
            "serotonin": max(0.0, min(1.0, serotonin_level))
        }
    
    def _update_visualization_data(self, result: Dict[str, Any]) -> None:
        """更新可视化数据"""
        # 更新神经活动数据
        neural_activity = {
            "timestamp": time.time(),
            "state": result["cognitive_state"],
            "activity_level": np.random.random()  # 模拟活动水平
        }
        self.visualization_data["neural_activity"].append(neural_activity)
        
        # 更新注意力焦点
        if "integrated_output" in result and "focus" in result["integrated_output"]:
            focus = result["integrated_output"]["focus"]
            self.visualization_data["attention_focus"].append({
                "timestamp": time.time(),
                "focus": focus
            })
        
        # 更新工作记忆内容
        if "integrated_output" in result and "memory_content" in result["integrated_output"]:
            memory = result["integrated_output"]["memory_content"]
            self.visualization_data["memory_content"].append({
                "timestamp": time.time(),
                "content": memory
            })
        
        # 更新神经调质水平
        if "neuromodulators" in result:
            self.visualization_data["neuromodulators"].append({
                "timestamp": time.time(),
                "levels": result["neuromodulators"]
            })
        
        # 限制数据长度
        max_data_points = 100
        for key in self.visualization_data:
            if len(self.visualization_data[key]) > max_data_points:
                self.visualization_data[key] = self.visualization_data[key][-max_data_points:]


