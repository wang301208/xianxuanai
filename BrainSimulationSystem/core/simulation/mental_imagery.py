"""
心理想象模拟器

实现约束性心理模拟和反事实推理
"""

from typing import Dict, List
import numpy as np

class MentalCanvas:
    """想象画布"""
    def __init__(self):
        self.objects = {}
        self.physical_laws = {
            'gravity': 9.8,
            'friction': 0.3
        }
        
    def add_object(self, name: str, properties: Dict):
        """添加想象对象"""
        self.objects[name] = properties
        
    def apply_physics(self):
        """应用物理约束"""
        for obj in self.objects.values():
            if 'position' in obj and 'velocity' in obj:
                obj['position'] += obj['velocity']
                obj['velocity'] *= (1 - self.physical_laws['friction'])
                
    def reset(self):
        """重置画布"""
        self.objects = {}

class MentalImagerySimulator:
    def __init__(self):
        self.canvas = MentalCanvas()
        self.reality_constraint = 0.8  # 现实约束强度(0-1)
        self.max_steps = 100          # 最大模拟步数
        
    def simulate_scenario(self, scenario: Dict) -> Dict:
        """运行想象模拟
        
        Args:
            scenario: 包含initial_conditions和rules的场景描述
            
        Returns:
            模拟结果和关键事件序列
        """
        self.canvas.reset()
        
        # 初始化场景
        for name, props in scenario.get('initial_conditions', {}).items():
            self.canvas.add_object(name, props)
            
        # 运行模拟
        events = []
        for step in range(self.max_steps):
            # 应用物理规则
            self.canvas.apply_physics()
            
            # 应用领域特定规则
            for rule in scenario.get('rules', []):
                self._apply_rule(rule)
                
            # 记录关键事件
            if self._check_event_conditions(scenario):
                events.append(self._capture_event(step))
                
            # 检查终止条件
            if self._check_termination(scenario):
                break
                
        return {
            'final_state': self.canvas.objects,
            'events': events,
            'steps': step + 1
        }
        
    def _apply_rule(self, rule: Dict):
        """应用特定领域规则"""
        # 简化实现 - 实际应使用规则引擎
        if rule['type'] == 'collision':
            obj1 = self.canvas.objects.get(rule['object1'])
            obj2 = self.canvas.objects.get(rule['object2'])
            if obj1 and obj2:
                if self._check_collision(obj1, obj2):
                    self._handle_collision(obj1, obj2, rule.get('effect'))
                    
    def _check_collision(self, obj1: Dict, obj2: Dict) -> bool:
        """检测对象碰撞"""
        return (abs(obj1.get('position', 0) - obj2.get('position', 0)) 
                < (obj1.get('size', 1) + obj2.get('size', 1))/2)
                
    def _handle_collision(self, obj1: Dict, obj2: Dict, effect: str):
        """处理碰撞事件"""
        if effect == 'bounce':
            obj1['velocity'], obj2['velocity'] = -obj2['velocity'], -obj1['velocity']
            
    def _check_event_conditions(self, scenario: Dict) -> bool:
        """检查事件触发条件"""
        # 简化实现
        return np.random.rand() < 0.1
        
    def _capture_event(self, step: int) -> Dict:
        """捕获关键事件"""
        return {
            'step': step,
            'description': f"事件发生在步骤{step}",
            'objects': list(self.canvas.objects.keys())
        }
        
    def _check_termination(self, scenario: Dict) -> bool:
        """检查模拟终止条件"""
        # 简化实现
        return np.random.rand() > 0.95 * self.reality_constraint
        
    def counterfactual_simulation(self, scenario: Dict, changes: Dict) -> Dict:
        """反事实情景模拟"""
        # 克隆原始场景
        alt_scenario = scenario.copy()
        alt_scenario['initial_conditions'] = {
            **scenario['initial_conditions'],
            **changes
        }
        return self.simulate_scenario(alt_scenario)