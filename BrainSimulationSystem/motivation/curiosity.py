"""
社交增强型好奇心驱动系统

整合社会认知维度的探索动机模型
"""

from typing import Dict
import numpy as np

class SocialCuriosityEngine:
    def __init__(self):
        # 基础好奇心参数
        self.base_curiosity = {
            'novelty_weight': 0.6,
            'complexity_weight': 0.4,
            'uncertainty_threshold': 0.3
        }
        
        # 社交好奇心参数
        self.social_components = {
            'face_processing': {
                'eye_gaze_sensitivity': 0.7,
                'expression_recognition': 0.8
            },
            'voice_processing': {
                'intonation_sensitivity': 0.6,
                'speech_rate_analysis': 0.5
            },
            'gesture_analysis': {
                'hand_movement': 0.7,
                'body_language': 0.6
            }
        }
        
        # 神经调节参数
        self.neuro_modulation = {
            'oxytocin_effect': 0.4,  # 催产素对社会好奇心的增强
            'dopamine_baseline': 0.5
        }
    
    def compute_integrated_curiosity(self, stimulus: Dict) -> float:
        """计算整合社交线索的综合好奇心值"""
        # 基础好奇心
        base_drive = self._compute_base_curiosity(stimulus)
        
        # 社交好奇心
        social_drive = self._compute_social_curiosity(stimulus.get('social_cues', {}))
        
        # 神经调节
        neuro_factor = self.neuro_modulation['dopamine_baseline']
        if stimulus.get('social_context'):
            neuro_factor += self.neuro_modulation['oxytocin_effect']
        
        return base_drive * 0.6 + social_drive * 0.4 * neuro_factor
    
    def _compute_base_curiosity(self, stimulus: Dict) -> float:
        """计算非社交好奇心基础值"""
        novelty = stimulus.get('novelty', 0) * self.base_curiosity['novelty_weight']
        complexity = stimulus.get('complexity', 0) * self.base_curiosity['complexity_weight']
        return (novelty + complexity) / 2
    
    def _compute_social_curiosity(self, social_cues: Dict) -> float:
        """计算社交好奇心专项值"""
        # 面部处理
        face_score = (
            social_cues.get('eye_contact', 0) * self.social_components['face_processing']['eye_gaze_sensitivity'] +
            social_cues.get('expression', 0) * self.social_components['face_processing']['expression_recognition']
        ) / 2
        
        # 声音处理
        voice_score = (
            social_cues.get('intonation', 0) * self.social_components['voice_processing']['intonation_sensitivity'] +
            social_cues.get('speech_rate', 0) * self.social_components['voice_processing']['speech_rate_analysis']
        ) / 2
        
        # 手势分析
        gesture_score = (
            social_cues.get('hand_movement', 0) * self.social_components['gesture_analysis']['hand_movement'] +
            social_cues.get('body_posture', 0) * self.social_components['gesture_analysis']['body_language']
        ) / 2
        
        return (face_score + voice_score + gesture_score) / 3
    
    def update_social_parameters(self, reward: float, social_type: str):
        """根据社交反馈更新参数"""
        if reward > 0.5:  # 正反馈
            if social_type == 'face':
                self.social_components['face_processing']['expression_recognition'] = min(
                    1.0, self.social_components['face_processing']['expression_recognition'] + 0.05)
            elif social_type == 'voice':
                self.social_components['voice_processing']['intonation_sensitivity'] = min(
                    1.0, self.social_components['voice_processing']['intonation_sensitivity'] + 0.03)
        else:  # 负反馈
            self.neuro_modulation['oxytocin_effect'] = max(
                0.1, self.neuro_modulation['oxytocin_effect'] - 0.02)