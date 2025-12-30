"""
认知重评策略学习模块

实现能够从经验中学习的情绪调节策略优化系统
"""

from typing import Dict

class ReappraisalLearner:
    def __init__(self):
        # 初始化策略库
        self.strategies = {
            'reinterpretation': {
                'description': '重新解释情绪刺激的意义',
                'efficacy': 0.6,
                'applicability': {
                    'anger': 0.7,
                    'fear': 0.5,
                    'disgust': 0.6
                },
                'activation_threshold': 0.4
            },
            'perspective_taking': {
                'description': '采取第三方视角看待情境',
                'efficacy': 0.4,
                'applicability': {
                    'sadness': 0.8,
                    'guilt': 0.6,
                    'shame': 0.7
                },
                'activation_threshold': 0.5
            },
            'positive_reframing': {
                'description': '寻找情境中的积极面',
                'efficacy': 0.5,
                'applicability': {
                    'fear': 0.4,
                    'sadness': 0.6
                },
                'activation_threshold': 0.3
            }
        }
        
        # 学习参数
        self.learning_rate = 0.1
        self.decay_rate = 0.98
        self.transfer_factor = 0.3  # 策略间知识迁移系数

    def select_strategy(self, emotion_type: str) -> str:
        """选择最适合当前情绪类型的策略"""
        viable = []
        for name, params in self.strategies.items():
            applicability = params['applicability'].get(emotion_type, 0)
            if applicability >= params['activation_threshold']:
                score = applicability * params['efficacy']
                viable.append((name, score))
        
        if not viable:
            return 'reinterpretation'  # 默认策略
        
        return max(viable, key=lambda x: x[1])[0]

    def update_from_experience(self, strategy: str, 
                             emotion_type: str, 
                             success: bool):
        """根据调节结果更新策略参数"""
        if strategy not in self.strategies:
            return
            
        # 更新适用性
        current = self.strategies[strategy]['applicability'].get(
            emotion_type, 0.5)
        delta = self.learning_rate * (1 if success else -1)
        self.strategies[strategy]['applicability'][emotion_type] = max(
            0.1, min(1.0, current + delta))
        
        # 更新效能
        if success:
            self.strategies[strategy]['efficacy'] = min(
                1.0, self.strategies[strategy]['efficacy'] + self.learning_rate/2)
        else:
            self.strategies[strategy]['efficacy'] *= self.decay_rate
            
        # 跨情绪类型知识迁移
        if success:
            for other_emotion in self.strategies[strategy]['applicability']:
                if other_emotion != emotion_type:
                    current = self.strategies[strategy]['applicability'][other_emotion]
                    self.strategies[strategy]['applicability'][other_emotion] = min(
                        1.0, current + self.learning_rate * self.transfer_factor)

    def add_custom_strategy(self, name: str, description: str,
                          base_efficacy: float = 0.5):
        """添加自定义重评策略"""
        if name not in self.strategies:
            self.strategies[name] = {
                'description': description,
                'efficacy': base_efficacy,
                'applicability': {},
                'activation_threshold': 0.4
            }

    def get_strategy_stats(self) -> Dict:
        """获取当前策略统计数据"""
        return {
            'strategy_count': len(self.strategies),
            'average_efficacy': sum(
                s['efficacy'] for s in self.strategies.values()) / len(self.strategies),
            'most_effective': max(
                self.strategies.items(), 
                key=lambda x: x[1]['efficacy'])[0]
        }