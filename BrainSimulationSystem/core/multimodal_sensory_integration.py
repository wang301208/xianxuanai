# -*- coding: utf-8 -*-
"""
多模态感觉输入整合系统
Multimodal Sensory Integration System

实现多种感觉模态的整合处理：
1. 视觉、听觉、触觉、本体感觉输入
2. 感觉皮层层级处理
3. 跨模态绑定与整合
4. 注意力调制
5. 预测编码机制
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

class SensoryModality(Enum):
    """感觉模态枚举"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    VESTIBULAR = "vestibular"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"

class ProcessingLevel(Enum):
    """处理层级枚举"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ASSOCIATION = "association"
    MULTIMODAL = "multimodal"

@dataclass
class SensoryInput:
    """感觉输入数据结构"""
    modality: SensoryModality
    data: np.ndarray
    timestamp: float
    spatial_location: Optional[Tuple[float, float, float]] = None
    intensity: float = 1.0
    frequency: Optional[float] = None
    duration: Optional[float] = None

@dataclass
class SensoryFeature:
    """感觉特征"""
    feature_type: str
    value: float
    confidence: float
    spatial_location: Optional[Tuple[float, float, float]] = None
    temporal_dynamics: Optional[Dict[str, float]] = None

class MultimodalIntegrator:
    """多模态整合器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("MultimodalIntegrator")
        
        # 导入各个感觉皮层
        from .visual_cortex import VisualCortex
        from .auditory_cortex import AuditoryCortex
        from .somatosensory_cortex import SomatosensoryCortex
        
        # 初始化各感觉皮层
        self.visual_cortex = VisualCortex(config)
        self.auditory_cortex = AuditoryCortex(config)
        self.somatosensory_cortex = SomatosensoryCortex(config)
        
        # 多模态整合区域
        self.integration_areas = {
            'superior_temporal_sulcus': {  # 上颞沟（视听整合）
                'neurons': {},
                'audiovisual_binding': {},
                'temporal_synchrony': 0.0
            },
            'posterior_parietal_cortex': {  # 后顶叶皮层（空间整合）
                'neurons': {},
                'spatial_maps': {},
                'attention_control': 0.0
            },
            'insular_cortex': {  # 岛叶皮层（内感受整合）
                'neurons': {},
                'interoceptive_signals': {},
                'emotional_integration': 0.0
            }
        }
        
        # 跨模态绑定机制
        self.binding_mechanisms = {
            'temporal_binding': {},  # 时间绑定
            'spatial_binding': {},   # 空间绑定
            'feature_binding': {},   # 特征绑定
            'semantic_binding': {}   # 语义绑定
        }
        
        # 预测编码
        self.predictive_models = {
            'visual_predictions': {},
            'auditory_predictions': {},
            'tactile_predictions': {},
            'cross_modal_predictions': {}
        }
        
        # 注意力系统
        self.attention_system = {
            'spatial_attention': np.zeros((64, 64, 64)),  # 3D注意力地图
            'feature_attention': {},
            'temporal_attention': {},
            'cross_modal_attention': {}
        }
        
        self._initialize_integration_system()
    
    def _initialize_integration_system(self):
        """初始化整合系统"""
        
        # 初始化整合区域神经元
        for area_name, area_info in self.integration_areas.items():
            neuron_count = {
                'superior_temporal_sulcus': 500,
                'posterior_parietal_cortex': 600,
                'insular_cortex': 400
            }[area_name]
            
            for i in range(neuron_count):
                area_info['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'modality_preferences': np.random.random(len(SensoryModality)),
                    'receptive_field': {
                        'spatial': (np.random.uniform(0, 64), np.random.uniform(0, 64), np.random.uniform(0, 64)),
                        'temporal': np.random.uniform(10, 500),  # ms
                        'frequency': np.random.uniform(1, 100)   # Hz
                    },
                    'integration_weights': np.random.uniform(0.1, 1.0, len(SensoryModality))
                }
    
    def process_multimodal_input(self, sensory_inputs: List[SensoryInput]) -> Dict[str, Any]:
        """处理多模态感觉输入"""
        
        # 分别处理各模态输入
        modality_responses = {}
        
        for sensory_input in sensory_inputs:
            if sensory_input.modality == SensoryModality.VISUAL:
                response = self.visual_cortex.process_visual_input(sensory_input)
                modality_responses['visual'] = response
                
            elif sensory_input.modality == SensoryModality.AUDITORY:
                response = self.auditory_cortex.process_auditory_input(sensory_input)
                modality_responses['auditory'] = response
                
            elif sensory_input.modality in [SensoryModality.TACTILE, SensoryModality.PROPRIOCEPTIVE]:
                response = self.somatosensory_cortex.process_somatosensory_input(sensory_input)
                modality_responses['somatosensory'] = response
        
        # 跨模态整合
        integration_results = self._cross_modal_integration(modality_responses, sensory_inputs)
        
        # 预测编码
        prediction_results = self._predictive_coding(modality_responses, sensory_inputs)
        
        # 注意力调制
        attention_results = self._attention_modulation(modality_responses, integration_results)
        
        return {
            'modality_responses': modality_responses,
            'integration_results': integration_results,
            'prediction_results': prediction_results,
            'attention_results': attention_results,
            'unified_percept': self._create_unified_percept(integration_results, attention_results)
        }
    
    def _cross_modal_integration(self, modality_responses: Dict[str, Any], 
                               sensory_inputs: List[SensoryInput]) -> Dict[str, Any]:
        """跨模态整合"""
        
        integration_results = {}
        
        # 视听整合（上颞沟）
        if 'visual' in modality_responses and 'auditory' in modality_responses:
            audiovisual_integration = self._audiovisual_integration(
                modality_responses['visual'], 
                modality_responses['auditory'],
                sensory_inputs
            )
            integration_results['audiovisual'] = audiovisual_integration
        
        # 视触整合（后顶叶皮层）
        if 'visual' in modality_responses and 'somatosensory' in modality_responses:
            visuotactile_integration = self._visuotactile_integration(
                modality_responses['visual'],
                modality_responses['somatosensory'],
                sensory_inputs
            )
            integration_results['visuotactile'] = visuotactile_integration
        
        # 空间整合
        spatial_integration = self._spatial_integration(modality_responses, sensory_inputs)
        integration_results['spatial'] = spatial_integration
        
        # 时间整合
        temporal_integration = self._temporal_integration(modality_responses, sensory_inputs)
        integration_results['temporal'] = temporal_integration
        
        return integration_results
    
    def _audiovisual_integration(self, visual_response: Dict[str, Any], 
                               auditory_response: Dict[str, Any],
                               sensory_inputs: List[SensoryInput]) -> Dict[str, Any]:
        """视听整合"""
        
        # 时间同步检测
        temporal_synchrony = self._detect_temporal_synchrony(visual_response, auditory_response, sensory_inputs)
        
        # 空间一致性
        spatial_consistency = self._compute_spatial_consistency(visual_response, auditory_response, sensory_inputs)
        
        # McGurk效应模拟
        mcgurk_effect = self._simulate_mcgurk_effect(visual_response, auditory_response)
        
        # 整合强度
        integration_strength = temporal_synchrony * spatial_consistency
        
        # 更新上颞沟活动
        sts_activity = self._update_sts_activity(integration_strength, visual_response, auditory_response)
        
        return {
            'temporal_synchrony': temporal_synchrony,
            'spatial_consistency': spatial_consistency,
            'mcgurk_effect': mcgurk_effect,
            'integration_strength': integration_strength,
            'sts_activity': sts_activity,
            'binding_quality': self._assess_binding_quality(temporal_synchrony, spatial_consistency)
        }
    
    def _detect_temporal_synchrony(self, visual_response: Dict[str, Any], 
                                 auditory_response: Dict[str, Any],
                                 sensory_inputs: List[SensoryInput]) -> float:
        """检测时间同步性"""
        
        # 获取视觉和听觉输入的时间戳
        visual_timestamps = [inp.timestamp for inp in sensory_inputs if inp.modality == SensoryModality.VISUAL]
        auditory_timestamps = [inp.timestamp for inp in sensory_inputs if inp.modality == SensoryModality.AUDITORY]
        
        if not visual_timestamps or not auditory_timestamps:
            return 0.0
        
        # 计算时间差
        time_diff = abs(visual_timestamps[0] - auditory_timestamps[0])
        
        # 时间窗口（McGurk效应的时间窗口约为±200ms）
        temporal_window = 200.0  # ms
        
        # 同步性强度（高斯函数）
        synchrony = np.exp(-(time_diff**2) / (2 * (temporal_window/3)**2))
        
        return synchrony
    
    def _compute_spatial_consistency(self, visual_response: Dict[str, Any], 
                                   auditory_response: Dict[str, Any],
                                   sensory_inputs: List[SensoryInput]) -> float:
        """计算空间一致性"""
        
        # 获取空间位置信息
        visual_locations = [inp.spatial_location for inp in sensory_inputs 
                          if inp.modality == SensoryModality.VISUAL and inp.spatial_location]
        auditory_locations = [inp.spatial_location for inp in sensory_inputs 
                            if inp.modality == SensoryModality.AUDITORY and inp.spatial_location]
        
        if not visual_locations or not auditory_locations:
            return 0.5  # 默认中等一致性
        
        # 计算空间距离
        visual_pos = np.array(visual_locations[0])
        auditory_pos = np.array(auditory_locations[0])
        spatial_distance = np.linalg.norm(visual_pos - auditory_pos)
        
        # 空间窗口
        spatial_window = 10.0  # 空间单位
        
        # 一致性强度
        consistency = np.exp(-(spatial_distance**2) / (2 * (spatial_window/3)**2))
        
        return consistency
    
    def _simulate_mcgurk_effect(self, visual_response: Dict[str, Any], 
                              auditory_response: Dict[str, Any]) -> Dict[str, Any]:
        """模拟McGurk效应"""
        
        # 简化的McGurk效应模拟
        # 视觉语音信息
        visual_speech_strength = visual_response.get('speech_features', {}).get('lip_movement', 0.0)
        
        # 听觉语音信息
        auditory_speech_strength = auditory_response.get('speech_features', {}).get('phoneme_clarity', 0.0)
        
        # McGurk融合
        if visual_speech_strength > 0.5 and auditory_speech_strength > 0.5:
            # 发生McGurk效应
            mcgurk_strength = min(visual_speech_strength, auditory_speech_strength)
            perceived_phoneme = self._fuse_phonemes(visual_response, auditory_response)
        else:
            mcgurk_strength = 0.0
            perceived_phoneme = None
        
        return {
            'mcgurk_strength': mcgurk_strength,
            'perceived_phoneme': perceived_phoneme,
            'visual_dominance': visual_speech_strength / (visual_speech_strength + auditory_speech_strength + 1e-6),
            'auditory_dominance': auditory_speech_strength / (visual_speech_strength + auditory_speech_strength + 1e-6)
        }
    
    def _fuse_phonemes(self, visual_response: Dict[str, Any], 
                      auditory_response: Dict[str, Any]) -> str:
        """融合音素"""
        # 简化的音素融合规则
        visual_phoneme = visual_response.get('speech_features', {}).get('detected_phoneme', 'ba')
        auditory_phoneme = auditory_response.get('speech_features', {}).get('detected_phoneme', 'ga')
        
        # McGurk融合规则（简化）
        fusion_rules = {
            ('ba', 'ga'): 'da',  # 经典McGurk效应
            ('pa', 'ka'): 'ta',
            ('ma', 'na'): 'na'
        }
        
        fused_phoneme = fusion_rules.get((visual_phoneme, auditory_phoneme), visual_phoneme)
        return fused_phoneme
    
    def _update_sts_activity(self, integration_strength: float, 
                           visual_response: Dict[str, Any], 
                           auditory_response: Dict[str, Any]) -> Dict[str, Any]:
        """更新上颞沟活动"""
        
        sts_neurons = self.integration_areas['superior_temporal_sulcus']['neurons']
        
        # 更新神经元活动
        for neuron_id, neuron in sts_neurons.items():
            # 多模态输入整合
            visual_input = np.random.random() * integration_strength  # 简化
            auditory_input = np.random.random() * integration_strength
            
            # 加权整合
            visual_weight = neuron['integration_weights'][SensoryModality.VISUAL.value[0]]
            auditory_weight = neuron['integration_weights'][SensoryModality.AUDITORY.value[0]]
            
            total_input = visual_input * visual_weight + auditory_input * auditory_weight
            
            # 更新发放率
            neuron['firing_rate'] = np.tanh(total_input)
        
        # 计算整体STS活动
        mean_activity = np.mean([n['firing_rate'] for n in sts_neurons.values()])
        
        return {
            'mean_activity': mean_activity,
            'integration_neurons_active': sum(1 for n in sts_neurons.values() if n['firing_rate'] > 0.5),
            'temporal_synchrony': integration_strength
        }
    
    def _assess_binding_quality(self, temporal_synchrony: float, spatial_consistency: float) -> Dict[str, float]:
        """评估绑定质量"""
        
        # 绑定强度
        binding_strength = (temporal_synchrony + spatial_consistency) / 2
        
        # 绑定稳定性
        binding_stability = min(temporal_synchrony, spatial_consistency)
        
        # 绑定置信度
        binding_confidence = temporal_synchrony * spatial_consistency
        
        return {
            'binding_strength': binding_strength,
            'binding_stability': binding_stability,
            'binding_confidence': binding_confidence,
            'binding_quality': (binding_strength + binding_stability + binding_confidence) / 3
        }
    
    def _visuotactile_integration(self, visual_response: Dict[str, Any],
                                somatosensory_response: Dict[str, Any],
                                sensory_inputs: List[SensoryInput]) -> Dict[str, Any]:
        """视触整合"""
        
        # 橡胶手错觉模拟
        rubber_hand_illusion = self._simulate_rubber_hand_illusion(visual_response, somatosensory_response)
        
        # 空间重映射
        spatial_remapping = self._compute_spatial_remapping(visual_response, somatosensory_response)
        
        # 身体图式更新
        body_schema_update = self._update_body_schema(visual_response, somatosensory_response)
        
        return {
            'rubber_hand_illusion': rubber_hand_illusion,
            'spatial_remapping': spatial_remapping,
            'body_schema_update': body_schema_update,
            'visuotactile_coherence': self._compute_visuotactile_coherence(visual_response, somatosensory_response)
        }
    
    def _simulate_rubber_hand_illusion(self, visual_response: Dict[str, Any],
                                     somatosensory_response: Dict[str, Any]) -> Dict[str, float]:
        """模拟橡胶手错觉"""
        
        # 视觉手部信息
        visual_hand_presence = visual_response.get('hand_detection', {}).get('confidence', 0.0)
        
        # 触觉刺激
        tactile_stimulation = somatosensory_response.get('tactile_intensity', 0.0)
        
        # 时空一致性
        spatiotemporal_congruence = visual_response.get('spatiotemporal_match', 0.5)
        
        # 错觉强度
        illusion_strength = visual_hand_presence * tactile_stimulation * spatiotemporal_congruence
        
        # 所有权转移
        ownership_transfer = min(1.0, illusion_strength * 1.2)
        
        return {
            'illusion_strength': illusion_strength,
            'ownership_transfer': ownership_transfer,
            'proprioceptive_drift': illusion_strength * 0.8,
            'embodiment_level': ownership_transfer
        }
    
    def _compute_spatial_remapping(self, visual_response: Dict[str, Any],
                                 somatosensory_response: Dict[str, Any]) -> Dict[str, Any]:
        """计算空间重映射"""
        
        # 视觉空间坐标
        visual_coordinates = visual_response.get('spatial_coordinates', np.array([0, 0, 0]))
        
        # 体感空间坐标
        somatosensory_coordinates = somatosensory_response.get('spatial_coordinates', np.array([0, 0, 0]))
        
        # 坐标变换矩阵
        transformation_matrix = self._compute_coordinate_transformation(visual_coordinates, somatosensory_coordinates)
        
        # 重映射精度
        remapping_accuracy = self._assess_remapping_accuracy(transformation_matrix)
        
        return {
            'transformation_matrix': transformation_matrix,
            'remapping_accuracy': remapping_accuracy,
            'coordinate_alignment': self._compute_coordinate_alignment(visual_coordinates, somatosensory_coordinates)
        }
    
    def _compute_coordinate_transformation(self, visual_coords: np.ndarray, 
                                        somatosensory_coords: np.ndarray) -> np.ndarray:
        """计算坐标变换矩阵"""
        # 简化的3D变换矩阵
        if len(visual_coords) != 3 or len(somatosensory_coords) != 3:
            return np.eye(3)
        
        # 计算旋转和平移
        translation = somatosensory_coords - visual_coords
        
        # 简化的变换矩阵（仅平移）
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        
        return transformation
    
    def _assess_remapping_accuracy(self, transformation_matrix: np.ndarray) -> float:
        """评估重映射精度"""
        # 基于变换矩阵的条件数评估精度
        if transformation_matrix.shape[0] < 3:
            return 0.5
        
        # 计算平移向量的模长
        translation_magnitude = np.linalg.norm(transformation_matrix[:3, 3])
        
        # 精度与平移距离成反比
        accuracy = 1.0 / (1.0 + translation_magnitude / 10.0)
        
        return accuracy
    
    def _compute_coordinate_alignment(self, visual_coords: np.ndarray, 
                                    somatosensory_coords: np.ndarray) -> float:
        """计算坐标对齐度"""
        if len(visual_coords) != len(somatosensory_coords):
            return 0.0
        
        # 计算欧氏距离
        distance = np.linalg.norm(visual_coords - somatosensory_coords)
        
        # 对齐度（距离越小对齐度越高）
        alignment = np.exp(-distance / 5.0)
        
        return alignment
    
    def _update_body_schema(self, visual_response: Dict[str, Any],
                          somatosensory_response: Dict[str, Any]) -> Dict[str, Any]:
        """更新身体图式"""
        
        # 身体部位检测
        body_parts_visual = visual_response.get('body_parts', {})
        body_parts_tactile = somatosensory_response.get('body_parts', {})
        
        # 身体图式一致性
        schema_consistency = self._compute_schema_consistency(body_parts_visual, body_parts_tactile)
        
        # 可塑性更新
        plasticity_update = self._compute_plasticity_update(schema_consistency)
        
        return {
            'schema_consistency': schema_consistency,
            'plasticity_update': plasticity_update,
            'body_representation': self._integrate_body_representation(body_parts_visual, body_parts_tactile)
        }
    
    def _compute_schema_consistency(self, visual_parts: Dict[str, Any], 
                                  tactile_parts: Dict[str, Any]) -> float:
        """计算身体图式一致性"""
        
        # 找到共同的身体部位
        common_parts = set(visual_parts.keys()) & set(tactile_parts.keys())
        
        if not common_parts:
            return 0.0
        
        # 计算每个部位的一致性
        consistencies = []
        for part in common_parts:
            visual_conf = visual_parts[part].get('confidence', 0.0)
            tactile_conf = tactile_parts[part].get('confidence', 0.0)
            
            # 一致性 = 两个模态置信度的几何平均
            consistency = np.sqrt(visual_conf * tactile_conf)
            consistencies.append(consistency)
        
        return np.mean(consistencies)
    
    def _compute_plasticity_update(self, consistency: float) -> Dict[str, float]:
        """计算可塑性更新"""
        
        # 学习率与一致性相关
        learning_rate = consistency * 0.1
        
        # 不同类型的可塑性
        return {
            'synaptic_plasticity': learning_rate,
            'structural_plasticity': learning_rate * 0.5,
            'homeostatic_plasticity': (1.0 - consistency) * 0.05
        }
    
    def _integrate_body_representation(self, visual_parts: Dict[str, Any], 
                                     tactile_parts: Dict[str, Any]) -> Dict[str, Any]:
        """整合身体表征"""
        
        integrated_representation = {}
        
        # 合并所有身体部位
        all_parts = set(visual_parts.keys()) | set(tactile_parts.keys())
        
        for part in all_parts:
            visual_info = visual_parts.get(part, {})
            tactile_info = tactile_parts.get(part, {})
            
            # 整合置信度
            visual_conf = visual_info.get('confidence', 0.0)
            tactile_conf = tactile_info.get('confidence', 0.0)
            integrated_conf = (visual_conf + tactile_conf) / 2
            
            # 整合位置信息
            visual_pos = visual_info.get('position', np.array([0, 0, 0]))
            tactile_pos = tactile_info.get('position', np.array([0, 0, 0]))
            integrated_pos = (visual_pos + tactile_pos) / 2
            
            integrated_representation[part] = {
                'confidence': integrated_conf,
                'position': integrated_pos,
                'modality_weights': {
                    'visual': visual_conf / (visual_conf + tactile_conf + 1e-6),
                    'tactile': tactile_conf / (visual_conf + tactile_conf + 1e-6)
                }
            }
        
        return integrated_representation
    
    def _compute_visuotactile_coherence(self, visual_response: Dict[str, Any],
                                      somatosensory_response: Dict[str, Any]) -> float:
        """计算视触一致性"""
        
        # 空间一致性
        spatial_coherence = self._compute_coordinate_alignment(
            visual_response.get('spatial_coordinates', np.array([0, 0, 0])),
            somatosensory_response.get('spatial_coordinates', np.array([0, 0, 0]))
        )
        
        # 时间一致性
        temporal_coherence = 0.8  # 简化假设
        
        # 特征一致性
        feature_coherence = self._compute_schema_consistency(
            visual_response.get('body_parts', {}),
            somatosensory_response.get('body_parts', {})
        )
        
        # 综合一致性
        overall_coherence = (spatial_coherence + temporal_coherence + feature_coherence) / 3
        
        return overall_coherence
    
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新多模态整合系统"""
        
        results = {}
        
        # 处理多模态输入
        if 'sensory_inputs' in inputs:
            multimodal_result = self.process_multimodal_input(inputs['sensory_inputs'])
            results['multimodal_processing'] = multimodal_result
        
        # 更新预测模型
        if 'prediction_targets' in inputs:
            prediction_update = self._update_predictive_models(inputs['prediction_targets'])
            results['prediction_update'] = prediction_update
        
        # 更新注意力系统
        if 'attention_cues' in inputs:
            attention_update = self._update_attention_system(inputs['attention_cues'])
            results['attention_update'] = attention_update
        
        return results

# 工厂函数
def create_multimodal_integrator(config: Optional[Dict[str, Any]] = None) -> MultimodalIntegrator:
    """创建多模态整合器"""
    if config is None:
        config = {}
    
    return MultimodalIntegrator(config)