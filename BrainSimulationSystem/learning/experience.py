"""
经验学习系统
实现基于强化学习的决策经验积累与优化
"""

import numpy as np
from typing import Dict, List
import pickle
import os

class ExperienceLearningSystem:
    def __init__(self, memory_capacity: int = 10000):
        # 经验记忆库
        self.experience_memory = []
        self.memory_capacity = memory_capacity
        
        # 学习参数
        self.learning_rate = 0.05
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        
        # 价值网络 (简化表示为字典)
        self.value_network = {}
        
        # 经验类别
        self.experience_categories = [
            'success', 'failure', 'unexpected', 'novel'
        ]
        
        # 经验统计
        self.statistics = {cat: 0 for cat in self.experience_categories}
        
    def store_experience(self, state: Dict, action: int,
                        reward: float, next_state: Dict,
                        available_actions: List[int] = None,
                        next_available_actions: List[int] = None,
                        category: str = 'general') -> None:
        """存储决策经验"""
        # 创建经验元组
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': np.datetime64('now'),
            'category': category,
            'available_actions': list(available_actions) if available_actions is not None else None,
            'next_available_actions': (list(next_available_actions)
                                       if next_available_actions is not None
                                       else (list(available_actions)
                                             if available_actions is not None else None))
        }
        
        # 添加到记忆库
        self.experience_memory.append(experience)
        
        # 更新统计
        if category in self.statistics:
            self.statistics[category] += 1
        
        # 如果超出容量，移除最旧的经验
        if len(self.experience_memory) > self.memory_capacity:
            self.experience_memory.pop(0)
            
    def learn_from_experience(self, batch_size: int = 32) -> Dict:
        """从经验中学习"""
        if len(self.experience_memory) < batch_size:
            return {'status': 'insufficient_data'}
            
        # 随机采样经验批次
        batch_indices = np.random.choice(
            len(self.experience_memory), 
            batch_size, 
            replace=False
        )
        batch = [self.experience_memory[i] for i in batch_indices]
        
        # 计算价值更新
        total_loss = 0
        for exp in batch:
            # 获取当前状态-行动的估计价值
            state_key = self._state_to_key(exp['state'])
            state_action_key = f"{state_key}_{exp['action']}"
            current_value = self.value_network.get(state_action_key, 0)

            # 计算目标价值 (TD目标)
            next_state_key = self._state_to_key(exp['next_state'])
            candidate_actions = exp.get('next_available_actions') or exp.get('available_actions') or []
            if candidate_actions:
                next_value = max(
                    self.value_network.get(f"{next_state_key}_{action}", 0)
                    for action in candidate_actions
                )
            else:
                next_value = 0
            target_value = exp['reward'] + self.discount_factor * next_value

            # 更新价值网络
            loss = target_value - current_value
            self.value_network[state_action_key] = (
                current_value + self.learning_rate * loss
            )
            total_loss += abs(loss)
            
        return {
            'status': 'success',
            'average_loss': total_loss / batch_size,
            'batch_size': batch_size
        }
        
    def get_action_recommendation(self, state: Dict, 
                                 available_actions: List[int]) -> int:
        """基于学习经验推荐行动"""
        # 标准化行动列表
        actions = list(available_actions)

        # 探索-利用权衡
        if np.random.random() < self.exploration_rate:
            # 探索: 随机选择
            return np.random.choice(actions)
        else:
            # 利用: 选择价值最高的行动
            state_key = self._state_to_key(state)
            action_values = {}

            for action in actions:
                # 构建状态-行动键
                action_key = f"{state_key}_{action}"
                action_values[action] = self.value_network.get(action_key, 0)

            # 如果所有行动都没有记录，默认选择第一个
            if not any(value != 0 for value in action_values.values()):
                return actions[0]

            # 选择价值最高的行动
            return max(action_values, key=action_values.get)
    
    def extract_decision_patterns(self) -> Dict:
        """提取决策模式"""
        if len(self.experience_memory) < 100:
            return {'status': 'insufficient_data'}
            
        # 按类别分组
        categorized = {cat: [] for cat in self.experience_categories}
        for exp in self.experience_memory:
            if exp['category'] in categorized:
                categorized[exp['category']].append(exp)
                
        # 提取每类的特征模式
        patterns = {}
        for cat, experiences in categorized.items():
            if len(experiences) > 10:
                # 提取状态特征
                state_features = self._extract_common_features(
                    [e['state'] for e in experiences]
                )
                
                # 提取行动模式
                action_distribution = {}
                for e in experiences:
                    action = e['action']
                    action_distribution[action] = action_distribution.get(action, 0) + 1
                    
                # 计算平均奖励
                avg_reward = sum(e['reward'] for e in experiences) / len(experiences)
                
                patterns[cat] = {
                    'state_features': state_features,
                    'action_distribution': action_distribution,
                    'average_reward': avg_reward,
                    'sample_count': len(experiences)
                }
                
        return patterns
    
    def save_experience(self, filepath: str) -> bool:
        """保存经验到文件"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'experience_memory': self.experience_memory,
                    'value_network': self.value_network,
                    'statistics': self.statistics
                }, f)
            return True
        except Exception as e:
            print(f"保存经验失败: {e}")
            return False
            
    def load_experience(self, filepath: str) -> bool:
        """从文件加载经验"""
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.experience_memory = data['experience_memory']
                self.value_network = data['value_network']
                self.statistics = data['statistics']
            return True
        except Exception as e:
            print(f"加载经验失败: {e}")
            return False
    
    def _state_to_key(self, state: Dict) -> str:
        """将状态转换为键"""
        # 简化实现，实际应用中需要更复杂的状态表示
        key_parts = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, (int, float)):
                # 离散化连续值
                key_parts.append(f"{k}:{int(v*10)/10}")
            else:
                key_parts.append(f"{k}:{v}")
        return "_".join(key_parts)
        
    def _extract_common_features(self, states: List[Dict]) -> Dict:
        """提取状态列表中的共同特征"""
        if not states:
            return {}

        # 获取所有可能的特征
        all_features = set()
        for state in states:
            all_features.update(state.keys())

        # 计算每个特征的平均值和方差
        feature_stats = {}
        for feature in all_features:
            values = [s.get(feature, 0) for s in states if feature in s]
            if values:
                feature_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'presence': len(values) / len(states)
                }

        return feature_stats
