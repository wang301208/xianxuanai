"""
增强型决策引擎

整合认知评估、情绪过滤、风险偏好、经验学习和元决策
"""

from typing import List, Dict
import time
from ..learning.experience import ExperienceLearningSystem
from ..meta_decision.core import MetaDecisionSystem

class DecisionEngine:
    def __init__(self):
        # 决策权重配置
        self.weights = {
            'cognitive': 0.6,  # 认知理性权重
            'emotional': 0.3,  # 情绪影响权重
            'social': 0.1,     # 社会规范权重
            'gender': 0.15,    # 性别影响权重
            'personality': 0.2,  # 个性特质权重
            'experience': 0.25  # 经验学习权重
        }
        
        # 初始化经验学习系统
        self.experience_learner = ExperienceLearningSystem()
        
        # 初始化元决策系统
        self.meta_decider = MetaDecisionSystem()
        
        # 性别模型 (兼容新旧版本)
        try:
            from ..personality.gender_custom import GenderCustomizer
            self.gender_model = GenderCustomizer()
        except ImportError:
            from ..personality.gender import GenderRoleInternalizer
            self.gender_model = GenderRoleInternalizer()
            
        # 个性定制系统
        from ..personality.custom_traits import PersonalityCustomizer
        self.personality_customizer = PersonalityCustomizer()
        
        # 决策阈值
        self.thresholds = {
            'confidence': 0.7,  # 最小置信阈值
            'risk_tolerance': 0.4  # 风险承受基线
        }
    
    def make_decision(self, options: List[Dict], context: Dict = None) -> int:
        """执行增强型决策流程"""
        # 应用上一轮的元决策调整
        self._apply_meta_adjustments()

        # 监控决策开始
        process_data = {
            'options': options,
            'context': context or {},
            'start_time': time.time()
        }
        self.meta_decider.monitor_decision_process(process_data)
        
        # 多维度评估
        cognitive = [self._evaluate_cognitive(o) for o in options]
        emotional = [self._evaluate_emotional(o) for o in options]
        social = [self._evaluate_social(o) for o in options]
        
        available_indices = list(range(len(options)))

        # 经验学习推荐
        if context:
            exp_recommendations = [
                self.experience_learner.get_action_recommendation(
                    {'option': o, 'context': context},
                    available_indices
                )
                for o in options
            ]
        else:
            exp_recommendations = [0.5] * len(options)
        
        # 加权综合评分
        scores = [
            c * self.weights['cognitive'] + 
            e * self.weights['emotional'] + 
            s * self.weights['social'] +
            r * self.weights['experience']
            for c, e, s, r in zip(cognitive, emotional, social, exp_recommendations)
        ]
        
        # 应用决策阈值
        valid_scores = [
            s if s > self.thresholds['confidence'] else 0 
            for s in scores
        ]
        
        selected_index = valid_scores.index(max(valid_scores))
        
        # 记录决策结果
        if context:
            # 准备经验学习数据
            state = {
                'context': context,
                'options': options,
                'selected': selected_index
            }
            # 在实际应用中，reward应该来自环境反馈
            reward = 0.5  # 默认值，实际应用中应从环境中获取
            
            # 存储经验
            self.experience_learner.store_experience(
                state,
                selected_index,
                reward,
                state,
                available_actions=available_indices
            )
            
            # 从经验中学习
            self.experience_learner.learn_from_experience()
            
            # 更新元决策系统
            self.meta_decider.evaluate_decision_quality({
                'selected_option': options[selected_index],
                'expected_outcome': context.get('expected', {}),
                'actual_outcome': {}  # 实际应用中应从环境中获取
            })

            # 获取元决策调整建议
            self.meta_decider.suggest_adjustments()

        return selected_index
    
    def _evaluate_cognitive(self, option: Dict) -> float:
        """认知维度评估"""
        value = option.get('expected_value', 0)
        risk = option.get('risk', 0)
        cost = option.get('cost', 0)
        
        # 预期效用计算
        return (value * (1 - risk)) - cost
    
    def _evaluate_emotional(self, option: Dict) -> float:
        """情绪维度评估"""
        stress = option.get('stress', 0)
        arousal = option.get('arousal', 0.5)
        
        # 情绪效价计算
        return (1 - stress) * (0.5 + 0.5 * arousal)
    
    def _evaluate_social(self, option: Dict) -> float:
        """社会维度评估"""
        norms = option.get('social_norms', {})
        return sum(norms.values()) / len(norms) if norms else 0.5
    
    def update_weights(self, new_weights: Dict):
        """动态调整决策权重"""
        for k, v in new_weights.items():
            if k in self.weights:
                self.weights[k] = max(0.0, min(1.0, self.weights[k] + v))

    def _apply_meta_adjustments(self) -> None:
        """应用来自元决策系统的待定调整"""
        if not hasattr(self.meta_decider, 'consume_pending_adjustments'):
            return

        adjustments = self.meta_decider.consume_pending_adjustments('engine')
        if not adjustments:
            return

        heuristic_weights = adjustments.get('heuristic_weights', {})
        if heuristic_weights:
            self.update_weights(heuristic_weights)

        risk_delta = adjustments.get('risk_tolerance_delta')
        if risk_delta:
            new_risk = self.thresholds.get('risk_tolerance', 0.4) + risk_delta
            self.thresholds['risk_tolerance'] = max(0.0, min(1.0, new_risk))