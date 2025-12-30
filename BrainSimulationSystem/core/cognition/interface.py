"""
认知-情绪交互接口层

实现认知过程和情绪系统的双向调节机制
"""

from typing import Dict

class CognitiveEmotionInterface:
    def __init__(self):
        # 认知资源分配参数
        self.cognitive_resources = {
        
    def get_reappraisal_strategies(self) -> Dict:
        """获取当前重评策略配置"""
        return {
            'available_strategies': list(self.strategies.keys()),
            'learning_rate': self.learning_rate
        }
        
    def reset_learning(self):
        """重置策略学习状态"""
        self.learning_rate = 0.1
        for strategy in self.strategies.values():
            strategy['efficacy'] = 0.5
            strategy['applicability'].update(
                {k: 0.5 for k in strategy['applicability']})

class EnhancedCognitiveInterface(CognitiveEmotionInterface):
    def __init__(self):
        super().__init__()
        self.reappraisal_learner = ReappraisalLearner()
        
    def regulate_with_learning(self, emotion_state: Dict) -> Dict:
        """带学习能力的情绪调节"""
        emotion_type = emotion_state['type']
        strategy = self.reappraisal_learner.select_strategy(emotion_type)
        
        # 应用策略
        if strategy == 'reinterpretation':
            return self._apply_reinterpretation(emotion_state)
        elif strategy == 'perspective_taking':
            return self._apply_perspective_taking(emotion_state)
        return {}
        
    def feedback_learning(self, strategy: str, emotion_type: str, 
                        success: bool):
        """提供学习反馈"""
        self.reappraisal_learner.update_strategy(
            strategy, emotion_type, success)
            'attention': {
                'baseline': 0.5,
                'emotion_gain': {
                    'positive': 1.3,
                    'negative': 0.8,
                    'neutral': 1.0
                }
            },
            'working_memory': {
                'capacity': 7,  # 经典7±2项
                'emotion_impact': {
                    'high_arousal': 0.7,
                    'low_arousal': 1.1
                }
            }
        }
        
        # 情绪调节参数
        self.emotion_regulation = {
            'cognitive_reappraisal': {
                'efficacy': 0.6,
                'time_window': 200  # ms
            },
            'attentional_deployment': {
                'redirect_strength': 0.8
            }
        }

    def apply_emotion_effects(self, cognitive_process: str, 
                            emotion_state: Dict) -> float:
        """应用情绪对认知过程的影响"""
        process_params = self.cognitive_resources.get(cognitive_process, {})
        if not process_params:
            return 1.0
            
        # 计算情绪增益
        emotion_type = emotion_state.get('type', 'neutral')
        emotion_gain = process_params['emotion_gain'].get(
            emotion_type, 1.0)
            
        # 计算唤醒度影响
        arousal = emotion_state.get('arousal', 0.5)
        arousal_impact = (process_params['emotion_impact']['high_arousal'] 
                         if arousal > 0.7 else 
                         process_params['emotion_impact']['low_arousal'])
        
        return emotion_gain * arousal_impact

    def regulate_emotion(self, cognitive_act: Dict) -> Dict:
        """应用认知策略调节情绪反应"""
        regulation = {}
        
        # 认知重评策略
        if cognitive_act.get('reappraisal', False):
            efficacy = self.emotion_regulation['cognitive_reappraisal']['efficacy']
            regulation['intensity_reduction'] = efficacy * 0.8
            
        # 注意部署策略
        if cognitive_act.get('redirect_attention', False):
            strength = self.emotion_regulation['attentional_deployment']['redirect_strength']
            regulation['attention_shift'] = strength
            
        return regulation

    def update_parameters(self, new_params: Dict):
        """动态更新系统参数"""
        for key, value in new_params.items():
            if key in self.cognitive_resources:
                self.cognitive_resources[key].update(value)
            elif key in self.emotion_regulation:
                self.emotion_regulation[key].update(value)