"""
决策制定模块

实现大脑的决策制定功能，包括风险评估、奖励预测和行动选择。
"""

import numpy as np
from collections import deque


class DecisionMaker:
    """决策制定器"""
    
    def __init__(self, params=None):
        """
        初始化决策制定器
        
        参数:
            params (dict): 配置参数，可包含：
                - risk_aversion: 风险规避系数，0-1之间
                - discount_factor: 未来奖励折扣因子
                - exploration_rate: 探索率，0-1之间
                - memory_size: 决策记忆容量
        """
        self.params = params or {}
        self.risk_aversion = self.params.get("risk_aversion", 0.5)
        self.discount_factor = self.params.get("discount_factor", 0.9)
        self.exploration_rate = self.params.get("exploration_rate", 0.1)
        self.memory_size = self.params.get("memory_size", 100)
        
        # 决策记忆
        self.decision_memory = deque(maxlen=self.memory_size)
        
        # 价值函数
        self.value_function = {}
    
    def evaluate_options(self, options):
        """
        评估选项
        
        参数:
            options (list): 选项列表，每个选项应包含：
                - reward: 预期奖励
                - risk: 风险值
                - cost: 执行成本
                
        返回:
            评估后的选项列表，包含评估分数
        """
        evaluated_options = []
        for option in options:
            # 计算净价值 (奖励 - 风险 * 风险规避系数 - 成本)
            net_value = option.get("reward", 0) - \
                       (option.get("risk", 0) * self.risk_aversion) - \
                       option.get("cost", 0)
            
            # 应用折扣因子 (如果是未来奖励)
            if "delay" in option:
                net_value *= (self.discount_factor ** option["delay"])
            
            evaluated_options.append({
                **option,
                "net_value": net_value
            })
        
        return evaluated_options
    
    def make_decision(self, options):
        """
        做出决策
        
        参数:
            options (list): 选项列表
            
        返回:
            选择的选项索引
        """
        if not options:
            return None
        
        # 评估选项
        evaluated_options = self.evaluate_options(options)
        
        # ε-贪婪策略选择
        if np.random.random() < self.exploration_rate:
            # 探索：随机选择
            chosen_idx = np.random.randint(0, len(evaluated_options))
        else:
            # 利用：选择最高净价值的选项
            net_values = [opt["net_value"] for opt in evaluated_options]
            chosen_idx = np.argmax(net_values)
        
        # 记录决策
        self.record_decision(evaluated_options[chosen_idx])
        
        return chosen_idx
    
    def record_decision(self, decision):
        """
        记录决策结果
        
        参数:
            decision (dict): 决策信息
        """
        self.decision_memory.append(decision)
        
        # 更新价值函数
        option_key = str(decision.get("action", ""))
        if option_key in self.value_function:
            # 指数移动平均更新
            self.value_function[option_key] = \
                0.9 * self.value_function[option_key] + \
                0.1 * decision.get("net_value", 0)
        else:
            self.value_function[option_key] = decision.get("net_value", 0)
    
    def update_parameters(self, feedback):
        """
        根据反馈更新决策参数
        
        参数:
            feedback (dict): 反馈信息，可包含：
                - outcome: 结果 (success/failure)
                - actual_reward: 实际获得的奖励
                - actual_risk: 实际风险
        """
        if not self.decision_memory:
            return
        
        # 获取最近的决策
        last_decision = self.decision_memory[-1]
        
        # 根据结果调整风险规避系数
        if feedback.get("outcome") == "failure":
            self.risk_aversion = min(1.0, self.risk_aversion + 0.05)
        elif feedback.get("outcome") == "success":
            self.risk_aversion = max(0.0, self.risk_aversion - 0.02)
        
        # 根据实际奖励调整折扣因子
        predicted_reward = last_decision.get("reward", 0)
        actual_reward = feedback.get("actual_reward", 0)
        if actual_reward > predicted_reward:
            self.discount_factor = min(0.99, self.discount_factor + 0.01)
        elif actual_reward < predicted_reward:
            self.discount_factor = max(0.5, self.discount_factor - 0.01)
        
        # 根据经验调整探索率
        self.exploration_rate = max(0.01, min(0.3, 1.0 / (1 + len(self.decision_memory))))


class PrefrontalCortex:
    """前额叶皮层决策系统"""
    
    def __init__(self, params=None):
        """
        初始化前额叶皮层决策系统
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        self.decision_maker = DecisionMaker(self.params.get("decision", {}))
        
        # 工作记忆接口
        self.working_memory = None
        
        # 情绪系统接口
        self.emotion_system = None
        
        # 神经调质影响
        self.dopamine_sensitivity = self.params.get("dopamine_sensitivity", 1.0)
        self.serotonin_sensitivity = self.params.get("serotonin_sensitivity", 1.0)
    
    def connect_working_memory(self, working_memory):
        """连接工作记忆系统"""
        self.working_memory = working_memory
    
    def connect_emotion_system(self, emotion_system):
        """连接情绪系统"""
        self.emotion_system = emotion_system
    
    def make_complex_decision(self, options):
        """
        做出复杂决策
        
        参数:
            options (list): 选项列表
            
        返回:
            选择的选项索引
        """
        if not options:
            return None
        
        # 从工作记忆中获取相关信息
        contextual_info = []
        if self.working_memory:
            contextual_info = [item["content"] for item in self.working_memory.items]
        
        # 从情绪系统中获取情绪状态
        emotional_state = {}
        if self.emotion_system:
            emotional_state = self.emotion_system.current_state()
        
        # 增强高风险选项的感知风险 (受血清素影响)
        for option in options:
            if "risk" in option:
                option["perceived_risk"] = option["risk"] * \
                    (1 + (1 - self.serotonin_sensitivity) * 0.5)
        
        # 增强奖励敏感性 (受多巴胺影响)
        for option in options:
            if "reward" in option:
                option["perceived_reward"] = option["reward"] * \
                    (1 + (self.dopamine_sensitivity - 1) * 0.3)
        
        # 做出决策
        return self.decision_maker.make_decision(options)
    
    def receive_feedback(self, feedback):
        """
        接收反馈并学习
        
        参数:
            feedback (dict): 反馈信息
        """
        self.decision_maker.update_parameters(feedback)
        
        # 根据反馈调整神经调质敏感性
        if feedback.get("outcome") == "success":
            self.dopamine_sensitivity = min(2.0, self.dopamine_sensitivity + 0.05)
        elif feedback.get("outcome") == "failure":
            self.serotonin_sensitivity = max(0.5, self.serotonin_sensitivity - 0.03)


class BasalGanglia:
    """基底神经节决策系统"""
    
    def __init__(self, params=None):
        """
        初始化基底神经节决策系统
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 直接通路 (促进动作执行)
        self.direct_pathway_gain = self.params.get("direct_pathway_gain", 1.0)
        
        # 间接通路 (抑制动作执行)
        self.indirect_pathway_gain = self.params.get("indirect_pathway_gain", 1.0)
        
        # 超直接通路 (快速抑制)
        self.hyperdirect_pathway_gain = self.params.get("hyperdirect_pathway_gain", 1.0)
        
        # 多巴胺调节
        self.dopamine_level = 1.0  # 正常水平为1.0
    
    def update_dopamine(self, level):
        """
        更新多巴胺水平
        
        参数:
            level (float): 多巴胺水平 (0-2之间)
        """
        self.dopamine_level = max(0, min(2.0, level))
        
        # 多巴胺调节通路增益
        # D1受体 (直接通路): 正相关
        self.direct_pathway_gain = 0.5 + self.dopamine_level * 0.5
        
        # D2受体 (间接通路): 负相关
        self.indirect_pathway_gain = 1.5 - self.dopamine_level * 0.5
    
    def select_action(self, action_values):
        """
        选择动作
        
        参数:
            action_values (dict): 动作价值映射
            
        返回:
            选择的动作
        """
        if not action_values:
            return None
        
        # 计算直接通路激活 (促进动作)
        direct_activation = {}
        for action, value in action_values.items():
            direct_activation[action] = value * self.direct_pathway_gain
        
        # 计算间接通路激活 (抑制动作)
        indirect_activation = {}
        for action, value in action_values.items():
            indirect_activation[action] = -value * self.indirect_pathway_gain
        
        # 计算净激活
        net_activation = {}
        for action in action_values:
            net_activation[action] = \
                direct_activation.get(action, 0) + \
                indirect_activation.get(action, 0)
        
        # 选择最高净激活的动作
        return max(net_activation.items(), key=lambda x: x[1])[0]


class DecisionSystem:
    """整合决策系统"""
    
    def __init__(self, params=None):
        """
        初始化决策系统
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 前额叶皮层系统 (慢速、审慎决策)
        self.pfc = PrefrontalCortex(self.params.get("pfc", {}))
        
        # 基底神经节系统 (快速、习惯性决策)
        self.bg = BasalGanglia(self.params.get("bg", {}))
        
        # 当前决策模式
        self.decision_mode = "pfc"  # 默认为前额叶皮层主导
        
        # 计时器 (用于模式切换)
        self.timer = 0
        self.time_threshold = self.params.get("time_threshold", 5)
    
    def connect_working_memory(self, working_memory):
        """连接工作记忆系统"""
        self.pfc.connect_working_memory(working_memory)
    
    def connect_emotion_system(self, emotion_system):
        """连接情绪系统"""
        self.pfc.connect_emotion_system(emotion_system)
    
    def update_dopamine(self, level):
        """
        更新多巴胺水平
        
        参数:
            level (float): 多巴胺水平
        """
        self.bg.update_dopamine(level)
        
        # 高多巴胺促进习惯性决策
        if level > 1.2:
            self.decision_mode = "bg"
        elif level < 0.8:
            self.decision_mode = "pfc"
    
    def make_decision(self, options):
        """
        做出决策
        
        参数:
            options (list): 选项列表
            
        返回:
            选择的选项索引
        """
        # 根据决策模式选择决策系统
        if self.decision_mode == "pfc":
            # 前额叶皮层决策
            decision = self.pfc.make_complex_decision(options)
            self.timer += 1
            
            # 如果决策时间过长，切换到基底神经节模式
            if self.timer > self.time_threshold:
                self.decision_mode = "bg"
                self.timer = 0
        else:
            # 基底神经节决策 (基于动作价值)
            action_values = {i: opt.get("reward", 0) for i, opt in enumerate(options)}
            decision = self.bg.select_action(action_values)
            self.timer = 0  # 重置计时器
        
        return decision
    
    def receive_feedback(self, feedback):
        """
        接收反馈
        
        参数:
            feedback (dict): 反馈信息
        """
        self.pfc.receive_feedback(feedback)
        
        # 根据反馈调整决策模式阈值
        if feedback.get("outcome") == "success":
            self.time_threshold = max(3, self.time_threshold - 1)
        else:
            self.time_threshold = min(10, self.time_threshold + 1)