# -*- coding: utf-8 -*-
"""
体感皮层处理系统
Somatosensory Cortex Processing System

实现体感处理：
1. S1: 触觉、本体感觉、痛觉
2. S2: 复杂触觉特征
3. 身体图式表征
4. 运动感觉整合
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

class SomatosensoryCortex:
    """体感皮层"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("SomatosensoryCortex")
        
        # 体感皮层区域
        self.areas = {
            'S1': {  # 初级体感皮层
                'areas': {
                    '3a': {'neurons': {}, 'modality': 'proprioception'},  # 本体感觉
                    '3b': {'neurons': {}, 'modality': 'tactile'},        # 触觉
                    '1': {'neurons': {}, 'modality': 'texture'},         # 纹理
                    '2': {'neurons': {}, 'modality': 'shape_size'}       # 形状大小
                },
                'somatotopic_map': {},  # 体感拓扑图
                'receptive_fields': {}
            },
            'S2': {  # 次级体感皮层
                'neurons': {},
                'bilateral_integration': {},
                'complex_tactile_features': {},
                'tactile_memory': {}
            },
            'posterior_parietal': {  # 后顶叶
                'neurons': {},
                'body_schema': {},
                'spatial_attention': {},
                'visuotactile_integration': {}
            }
        }
        
        # 身体部位映射
        self.body_parts = {
            'hand': {'left': {}, 'right': {}},
            'arm': {'left': {}, 'right': {}},
            'face': {},
            'leg': {'left': {}, 'right': {}},
            'trunk': {}
        }
        
        # 感受器类型
        self.receptor_types = {
            'mechanoreceptors': {
                'SA1': {'adaptation': 'slow', 'sensitivity': 'texture'},
                'SA2': {'adaptation': 'slow', 'sensitivity': 'stretch'},
                'RA': {'adaptation': 'rapid', 'sensitivity': 'vibration'},
                'PC': {'adaptation': 'rapid', 'sensitivity': 'high_freq_vibration'}
            },
            'thermoreceptors': {
                'cold': {'threshold': 15, 'optimal': 25},
                'warm': {'threshold': 35, 'optimal': 40}
            },
            'nociceptors': {
                'mechanical': {'threshold': 0.8},
                'thermal': {'threshold': 45},
                'chemical': {'threshold': 0.5}
            },
            'proprioceptors': {
                'muscle_spindles': {'sensitivity': 'length_velocity'},
                'golgi_tendon': {'sensitivity': 'tension'},
                'joint_receptors': {'sensitivity': 'position_movement'}
            }
        }
        
        self._initialize_somatosensory_cortex()
    
    def _initialize_somatosensory_cortex(self):
        """初始化体感皮层"""
        
        # 初始化S1各区域
        for area_name, area_info in self.areas['S1']['areas'].items():
            neuron_count = {'3a': 400, '3b': 600, '1': 500, '2': 400}[area_name]
            
            for i in range(neuron_count):
                area_info['neurons'][i] = {
                    'body_part': np.random.choice(list(self.body_parts.keys())),
                    'receptive_field_size': np.random.uniform(1, 10),  # mm²
                    'receptive_field_location': (
                        np.random.uniform(0, 100),  # x坐标
                        np.random.uniform(0, 100)   # y坐标
                    ),
                    'preferred_stimulus': self._assign_preferred_stimulus(area_name),
                    'firing_rate': 0.0,
                    'membrane_potential': -70.0,
                    'adaptation_state': 1.0
                }
        
        # 初始化S2
        for i in range(300):
            self.areas['S2']['neurons'][i] = {
                'bilateral_receptive_field': True,
                'complex_feature_selectivity': np.random.random(10),
                'tactile_memory_trace': 0.0,
                'firing_rate': 0.0,
                'membrane_potential': -70.0
            }
        
        # 初始化后顶叶
        for i in range(400):
            self.areas['posterior_parietal']['neurons'][i] = {
                'body_schema_contribution': np.random.random(5),
                'spatial_attention_field': (
                    np.random.uniform(0, 100),
                    np.random.uniform(0, 100),
                    np.random.uniform(10, 50)  # 注意力范围
                ),
                'multimodal_integration_weights': {
                    'tactile': np.random.uniform(0.3, 0.8),
                    'visual': np.random.uniform(0.2, 0.7),
                    'proprioceptive': np.random.uniform(0.4, 0.9)
                },
                'firing_rate': 0.0,
                'membrane_potential': -70.0
            }
        
        # 初始化体感拓扑图
        self._initialize_somatotopic_map()
    
    def _assign_preferred_stimulus(self, area_name: str) -> Dict[str, Any]:
        """分配偏好刺激"""
        
        preferences = {
            '3a': {
                'modality': 'proprioception',
                'feature': np.random.choice(['joint_angle', 'muscle_length', 'movement_velocity']),
                'sensitivity_range': (0.1, 10.0)  # 度或mm/s
            },
            '3b': {
                'modality': 'tactile',
                'feature': np.random.choice(['pressure', 'indentation', 'contact']),
                'sensitivity_range': (0.01, 1.0)  # N或mm
            },
            '1': {
                'modality': 'texture',
                'feature': np.random.choice(['roughness', 'friction', 'spatial_pattern']),
                'sensitivity_range': (0.1, 5.0)  # 纹理单位
            },
            '2': {
                'modality': 'shape_size',
                'feature': np.random.choice(['curvature', 'edge_orientation', 'object_size']),
                'sensitivity_range': (1.0, 50.0)  # mm或度
            }
        }
        
        return preferences[area_name]
    
    def _initialize_somatotopic_map(self):
        """初始化体感拓扑图"""
        
        # 简化的体感拓扑图（Penfield小人）
        somatotopic_areas = {
            'face': {'size': 30, 'position': (10, 50)},
            'hand': {'size': 25, 'position': (40, 30)},
            'arm': {'size': 15, 'position': (60, 35)},
            'trunk': {'size': 10, 'position': (70, 50)},
            'leg': {'size': 20, 'position': (80, 70)}
        }
        
        for body_part, info in somatotopic_areas.items():
            self.areas['S1']['somatotopic_map'][body_part] = {
                'cortical_area': info['size'],  # 皮层面积（相对单位）
                'cortical_position': info['position'],  # 皮层位置
                'magnification_factor': info['size'] / 10.0,  # 放大因子
                'resolution': 1.0 / info['size']  # 空间分辨率
            }
    
    def process_somatosensory_input(self, somatosensory_input) -> Dict[str, Any]:
        """处理体感输入"""
        
        # 预处理体感数据
        tactile_data = somatosensory_input.data
        modality = somatosensory_input.modality
        
        # 感受器级处理
        receptor_responses = self._receptor_processing(tactile_data, modality)
        
        # S1处理：初级体感特征
        s1_response = self._s1_processing(receptor_responses, somatosensory_input)
        
        # S2处理：复杂触觉特征
        s2_response = self._s2_processing(s1_response)
        
        # 后顶叶处理：身体图式和空间整合
        parietal_response = self._posterior_parietal_processing(s1_response, s2_response)
        
        # 身体图式更新
        body_schema_update = self._update_body_schema(s1_response, parietal_response)
        
        return {
            'receptor_responses': receptor_responses,
            'S1_response': s1_response,
            'S2_response': s2_response,
            'parietal_response': parietal_response,
            'body_schema_update': body_schema_update,
            'tactile_intensity': np.mean(tactile_data) if len(tactile_data) > 0 else 0.0,
            'spatial_coordinates': somatosensory_input.spatial_location or np.array([0, 0, 0]),
            'body_parts': self._extract_body_parts_info(s1_response)
        }
    
    def _receptor_processing(self, tactile_data: np.ndarray, 
                           modality: Any) -> Dict[str, Any]:
        """感受器级处理"""
        
        receptor_responses = {}
        
        # 机械感受器
        if hasattr(modality, 'value') and modality.value in ['tactile', 'proprioceptive']:
            mechanoreceptor_response = self._process_mechanoreceptors(tactile_data)
            receptor_responses['mechanoreceptors'] = mechanoreceptor_response
        
        # 温度感受器
        if len(tactile_data) > 0:
            thermoreceptor_response = self._process_thermoreceptors(tactile_data)
            receptor_responses['thermoreceptors'] = thermoreceptor_response
        
        # 伤害感受器
        nociceptor_response = self._process_nociceptors(tactile_data)
        receptor_responses['nociceptors'] = nociceptor_response
        
        # 本体感受器
        if hasattr(modality, 'value') and modality.value == 'proprioceptive':
            proprioceptor_response = self._process_proprioceptors(tactile_data)
            receptor_responses['proprioceptors'] = proprioceptor_response
        
        return receptor_responses
    
    def _process_mechanoreceptors(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """处理机械感受器"""
        
        if len(tactile_data) == 0:
            return {receptor_type: 0.0 for receptor_type in self.receptor_types['mechanoreceptors']}
        
        responses = {}
        
        # SA1: 慢适应，纹理敏感
        sa1_response = np.mean(np.abs(np.gradient(tactile_data)))  # 纹理变化
        responses['SA1'] = min(1.0, sa1_response)
        
        # SA2: 慢适应，拉伸敏感
        sa2_response = np.std(tactile_data)  # 变形程度
        responses['SA2'] = min(1.0, sa2_response)
        
        # RA: 快适应，振动敏感
        if len(tactile_data) > 1:
            ra_response = np.mean(np.abs(np.diff(tactile_data)))  # 变化率
        else:
            ra_response = 0.0
        responses['RA'] = min(1.0, ra_response)
        
        # PC: 快适应，高频振动敏感
        if len(tactile_data) > 2:
            pc_response = np.mean(np.abs(np.diff(tactile_data, n=2)))  # 二阶变化
        else:
            pc_response = 0.0
        responses['PC'] = min(1.0, pc_response)
        
        return responses
    
    def _process_thermoreceptors(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """处理温度感受器"""
        
        if len(tactile_data) == 0:
            return {'cold': 0.0, 'warm': 0.0}
        
        # 假设tactile_data包含温度信息
        temperature = np.mean(tactile_data) * 40 + 20  # 映射到温度范围
        
        # 冷感受器
        cold_response = 0.0
        if temperature < 30:
            cold_response = (30 - temperature) / 15.0
        
        # 温感受器
        warm_response = 0.0
        if temperature > 32:
            warm_response = (temperature - 32) / 15.0
        
        return {
            'cold': min(1.0, max(0.0, cold_response)),
            'warm': min(1.0, max(0.0, warm_response)),
            'temperature': temperature
        }
    
    def _process_nociceptors(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """处理伤害感受器"""
        
        if len(tactile_data) == 0:
            return {'mechanical': 0.0, 'thermal': 0.0, 'chemical': 0.0}
        
        # 机械性疼痛
        mechanical_pain = 0.0
        if np.max(np.abs(tactile_data)) > 0.8:
            mechanical_pain = (np.max(np.abs(tactile_data)) - 0.8) / 0.2
        
        # 热性疼痛（基于温度）
        temperature = np.mean(tactile_data) * 40 + 20
        thermal_pain = 0.0
        if temperature > 45 or temperature < 10:
            if temperature > 45:
                thermal_pain = (temperature - 45) / 20.0
            else:
                thermal_pain = (10 - temperature) / 10.0
        
        # 化学性疼痛（简化）
        chemical_pain = 0.0
        if np.std(tactile_data) > 0.5:
            chemical_pain = (np.std(tactile_data) - 0.5) / 0.5
        
        return {
            'mechanical': min(1.0, max(0.0, mechanical_pain)),
            'thermal': min(1.0, max(0.0, thermal_pain)),
            'chemical': min(1.0, max(0.0, chemical_pain))
        }
    
    def _process_proprioceptors(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """处理本体感受器"""
        
        if len(tactile_data) == 0:
            return {'muscle_spindles': 0.0, 'golgi_tendon': 0.0, 'joint_receptors': 0.0}
        
        # 肌梭：长度和速度敏感
        muscle_length = np.mean(tactile_data)
        if len(tactile_data) > 1:
            muscle_velocity = np.mean(np.abs(np.diff(tactile_data)))
        else:
            muscle_velocity = 0.0
        
        muscle_spindle_response = (muscle_length + muscle_velocity) / 2
        
        # 高尔基腱器官：张力敏感
        muscle_tension = np.max(np.abs(tactile_data))
        golgi_response = muscle_tension
        
        # 关节感受器：位置和运动敏感
        joint_position = muscle_length  # 简化
        joint_movement = muscle_velocity
        joint_response = (joint_position + joint_movement) / 2
        
        return {
            'muscle_spindles': min(1.0, muscle_spindle_response),
            'golgi_tendon': min(1.0, golgi_response),
            'joint_receptors': min(1.0, joint_response)
        }
    
    def _s1_processing(self, receptor_responses: Dict[str, Any], 
                      somatosensory_input) -> Dict[str, Any]:
        """S1处理：初级体感特征"""
        
        area_responses = {}
        
        # 处理各S1区域
        for area_name, area_info in self.areas['S1']['areas'].items():
            area_response = self._process_s1_area(area_name, area_info, receptor_responses, somatosensory_input)
            area_responses[area_name] = area_response
        
        # 体感拓扑激活
        somatotopic_activation = self._compute_somatotopic_activation(area_responses, somatosensory_input)
        
        # 侧抑制
        lateral_inhibition = self._apply_lateral_inhibition(area_responses)
        
        return {
            'area_responses': area_responses,
            'somatotopic_activation': somatotopic_activation,
            'lateral_inhibition': lateral_inhibition,
            'feature_maps': self._create_feature_maps(area_responses)
        }
    
    def _process_s1_area(self, area_name: str, area_info: Dict[str, Any], 
                        receptor_responses: Dict[str, Any], 
                        somatosensory_input) -> Dict[str, Any]:
        """处理单个S1区域"""
        
        neuron_responses = {}
        modality = area_info['modality']
        
        for neuron_id, neuron in area_info['neurons'].items():
            # 获取相关的感受器输入
            relevant_input = self._get_relevant_receptor_input(modality, receptor_responses)
            
            # 感受野匹配
            rf_match = self._compute_receptive_field_match(neuron, somatosensory_input)
            
            # 特征选择性
            feature_selectivity = self._compute_feature_selectivity(neuron, relevant_input)
            
            # 综合响应
            total_response = relevant_input * rf_match * feature_selectivity * neuron['adaptation_state']
            neuron['firing_rate'] = max(0, total_response)
            neuron_responses[neuron_id] = neuron['firing_rate']
            
            # 适应性更新
            if neuron['firing_rate'] > 0.7:
                neuron['adaptation_state'] *= 0.9
            else:
                neuron['adaptation_state'] = min(1.0, neuron['adaptation_state'] * 1.05)
        
        return {
            'neuron_responses': neuron_responses,
            'mean_activity': np.mean(list(neuron_responses.values())),
            'peak_activity': np.max(list(neuron_responses.values())) if neuron_responses else 0.0,
            'active_neurons': sum(1 for r in neuron_responses.values() if r > 0.1)
        }
    
    def _get_relevant_receptor_input(self, modality: str, 
                                   receptor_responses: Dict[str, Any]) -> float:
        """获取相关的感受器输入"""
        
        if modality == 'proprioception' and 'proprioceptors' in receptor_responses:
            return np.mean(list(receptor_responses['proprioceptors'].values()))
        elif modality == 'tactile' and 'mechanoreceptors' in receptor_responses:
            return np.mean(list(receptor_responses['mechanoreceptors'].values()))
        elif modality in ['texture', 'shape_size'] and 'mechanoreceptors' in receptor_responses:
            # 纹理和形状主要依赖特定的机械感受器
            if modality == 'texture':
                return receptor_responses['mechanoreceptors'].get('SA1', 0.0)
            else:  # shape_size
                return (receptor_responses['mechanoreceptors'].get('SA1', 0.0) + 
                       receptor_responses['mechanoreceptors'].get('SA2', 0.0)) / 2
        else:
            return 0.0
    
    def _compute_receptive_field_match(self, neuron: Dict[str, Any], 
                                     somatosensory_input) -> float:
        """计算感受野匹配度"""
        
        if not hasattr(somatosensory_input, 'spatial_location') or somatosensory_input.spatial_location is None:
            return 0.5  # 默认匹配度
        
        # 获取刺激位置
        stimulus_location = somatosensory_input.spatial_location[:2]  # 取x, y坐标
        
        # 神经元感受野中心
        rf_center = neuron['receptive_field_location']
        rf_size = neuron['receptive_field_size']
        
        # 计算距离
        distance = np.linalg.norm(np.array(stimulus_location) - np.array(rf_center))
        
        # 高斯感受野
        rf_match = np.exp(-(distance**2) / (2 * (rf_size/3)**2))
        
        return rf_match
    
    def _compute_feature_selectivity(self, neuron: Dict[str, Any], 
                                   receptor_input: float) -> float:
        """计算特征选择性"""
        
        preferred_stimulus = neuron['preferred_stimulus']
        feature = preferred_stimulus['feature']
        sensitivity_range = preferred_stimulus['sensitivity_range']
        
        # 简化的特征匹配
        # 假设receptor_input在0-1范围内，映射到敏感范围
        mapped_input = receptor_input * (sensitivity_range[1] - sensitivity_range[0]) + sensitivity_range[0]
        
        # 最优响应在敏感范围中间
        optimal_value = (sensitivity_range[0] + sensitivity_range[1]) / 2
        
        # 高斯调谐
        tuning_width = (sensitivity_range[1] - sensitivity_range[0]) / 4
        selectivity = np.exp(-((mapped_input - optimal_value)**2) / (2 * tuning_width**2))
        
        return selectivity
    
    def _compute_somatotopic_activation(self, area_responses: Dict[str, Any], 
                                      somatosensory_input) -> Dict[str, Any]:
        """计算体感拓扑激活"""
        
        # 确定刺激的身体部位
        if hasattr(somatosensory_input, 'spatial_location') and somatosensory_input.spatial_location is not None:
            body_part = self._determine_body_part(somatosensory_input.spatial_location)
        else:
            body_part = 'hand'  # 默认
        
        # 获取该身体部位的皮层表征
        if body_part in self.areas['S1']['somatotopic_map']:
            somatotopic_info = self.areas['S1']['somatotopic_map'][body_part]
            
            # 计算激活强度
            total_activity = sum(area['mean_activity'] for area in area_responses.values())
            magnification_factor = somatotopic_info['magnification_factor']
            
            activation_strength = total_activity * magnification_factor
        else:
            activation_strength = 0.0
            somatotopic_info = {}
        
        return {
            'body_part': body_part,
            'activation_strength': activation_strength,
            'somatotopic_info': somatotopic_info,
            'cortical_spread': self._compute_cortical_spread(activation_strength)
        }
    
    def _determine_body_part(self, spatial_location: np.ndarray) -> str:
        """确定身体部位"""
        
        # 简化的身体部位映射
        x, y = spatial_location[0], spatial_location[1]
        
        if x < 20:
            return 'face'
        elif x < 40:
            return 'hand'
        elif x < 60:
            return 'arm'
        elif x < 80:
            return 'trunk'
        else:
            return 'leg'
    
    def _compute_cortical_spread(self, activation_strength: float) -> Dict[str, float]:
        """计算皮层扩散"""
        
        # 激活扩散到邻近区域
        spread_radius = activation_strength * 5.0  # 扩散半径
        spread_strength = activation_strength * 0.3  # 扩散强度
        
        return {
            'spread_radius': spread_radius,
            'spread_strength': spread_strength,
            'lateral_facilitation': activation_strength * 0.1,
            'lateral_inhibition': activation_strength * 0.2
        }
    
    def _apply_lateral_inhibition(self, area_responses: Dict[str, Any]) -> Dict[str, Any]:
        """应用侧抑制"""
        
        # 计算各区域间的抑制
        inhibition_matrix = {}
        
        for area1, response1 in area_responses.items():
            inhibition_matrix[area1] = {}
            
            for area2, response2 in area_responses.items():
                if area1 != area2:
                    # 相互抑制强度
                    inhibition_strength = response1['mean_activity'] * 0.1
                    inhibition_matrix[area1][area2] = inhibition_strength
                else:
                    inhibition_matrix[area1][area2] = 0.0
        
        # 应用抑制
        inhibited_responses = {}
        for area, response in area_responses.items():
            total_inhibition = sum(inhibition_matrix[other_area][area] 
                                 for other_area in area_responses.keys() if other_area != area)
            
            inhibited_activity = max(0.0, response['mean_activity'] - total_inhibition)
            
            inhibited_responses[area] = {
                'original_activity': response['mean_activity'],
                'inhibited_activity': inhibited_activity,
                'inhibition_received': total_inhibition
            }
        
        return inhibited_responses
    
    def _create_feature_maps(self, area_responses: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """创建特征图"""
        
        feature_maps = {}
        
        # 为每个区域创建特征图
        for area_name, area_response in area_responses.items():
            # 简化的特征图（32x32）
            feature_map = np.zeros((32, 32))
            
            # 基于神经元响应填充特征图
            neuron_responses = area_response['neuron_responses']
            
            for neuron_id, response in neuron_responses.items():
                # 将神经元ID映射到空间位置
                x = neuron_id % 32
                y = neuron_id // 32
                
                if y < 32:
                    feature_map[y, x] = response
            
            feature_maps[area_name] = feature_map
        
        return feature_maps
    
    def _s2_processing(self, s1_response: Dict[str, Any]) -> Dict[str, Any]:
        """S2处理：复杂触觉特征"""
        
        # 双侧整合
        bilateral_integration = self._bilateral_integration(s1_response)
        
        # 复杂特征提取
        complex_features = self._extract_complex_tactile_features(s1_response)
        
        # 触觉记忆
        tactile_memory = self._tactile_working_memory(complex_features)
        
        # 更新S2神经元
        s2_neuron_responses = self._update_s2_neurons(bilateral_integration, complex_features)
        
        return {
            'bilateral_integration': bilateral_integration,
            'complex_features': complex_features,
            'tactile_memory': tactile_memory,
            'neuron_responses': s2_neuron_responses
        }
    
    def _bilateral_integration(self, s1_response: Dict[str, Any]) -> Dict[str, Any]:
        """双侧整合"""
        
        # 简化的双侧整合（假设有左右侧输入）
        area_responses = s1_response['area_responses']
        
        # 计算双侧一致性
        bilateral_consistency = {}
        
        for area_name, area_response in area_responses.items():
            # 假设左右侧活动相似
            left_activity = area_response['mean_activity']
            right_activity = area_response['mean_activity'] * (0.8 + np.random.random() * 0.4)  # 模拟右侧
            
            consistency = 1.0 - abs(left_activity - right_activity) / (left_activity + right_activity + 1e-6)
            bilateral_consistency[area_name] = consistency
        
        return {
            'bilateral_consistency': bilateral_consistency,
            'integration_strength': np.mean(list(bilateral_consistency.values())),
            'hemispheric_dominance': self._compute_hemispheric_dominance(bilateral_consistency)
        }
    
    def _compute_hemispheric_dominance(self, bilateral_consistency: Dict[str, float]) -> Dict[str, float]:
        """计算半球优势"""
        
        # 简化的半球优势计算
        left_dominance = np.mean(list(bilateral_consistency.values()))
        right_dominance = 1.0 - left_dominance
        
        return {
            'left_hemisphere': left_dominance,
            'right_hemisphere': right_dominance,
            'dominance_strength': abs(left_dominance - right_dominance)
        }
    
    def _extract_complex_tactile_features(self, s1_response: Dict[str, Any]) -> Dict[str, Any]:
        """提取复杂触觉特征"""
        
        area_responses = s1_response['area_responses']
        feature_maps = s1_response['feature_maps']
        
        # 纹理复杂度
        texture_complexity = self._compute_texture_complexity(feature_maps)
        
        # 形状特征
        shape_features = self._extract_shape_features(feature_maps)
        
        # 运动特征
        motion_features = self._extract_motion_features(area_responses)
        
        # 物体识别
        object_recognition = self._tactile_object_recognition(texture_complexity, shape_features)
        
        return {
            'texture_complexity': texture_complexity,
            'shape_features': shape_features,
            'motion_features': motion_features,
            'object_recognition': object_recognition
        }
    
    def _compute_texture_complexity(self, feature_maps: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算纹理复杂度"""
        
        complexity = {}
        
        for area_name, feature_map in feature_maps.items():
            if area_name == '1':  # 纹理处理区域
                # 计算纹理统计
                mean_intensity = np.mean(feature_map)
                std_intensity = np.std(feature_map)
                
                # 空间频率分析
                fft_map = np.fft.fft2(feature_map)
                power_spectrum = np.abs(fft_map)**2
                
                # 纹理复杂度指标
                complexity[area_name] = {
                    'roughness': std_intensity,
                    'regularity': 1.0 / (1.0 + std_intensity),
                    'spatial_frequency': np.mean(power_spectrum),
                    'overall_complexity': (std_intensity + np.mean(power_spectrum)) / 2
                }
        
        return complexity
    
    def _extract_shape_features(self, feature_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """提取形状特征"""
        
        shape_features = {}
        
        for area_name, feature_map in feature_maps.items():
            if area_name == '2':  # 形状处理区域
                # 边缘检测
                grad_x = np.gradient(feature_map, axis=1)
                grad_y = np.gradient(feature_map, axis=0)
                edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # 曲率计算
                curvature = self._compute_curvature(feature_map)
                
                # 形状描述符
                shape_features[area_name] = {
                    'edge_density': np.mean(edge_magnitude),
                    'curvature': curvature,
                    'compactness': self._compute_compactness(feature_map),
                    'elongation': self._compute_elongation(feature_map)
                }
        
        return shape_features
    
    def _compute_curvature(self, feature_map: np.ndarray) -> float:
        """计算曲率"""
        
        # 简化的曲率计算
        grad_x = np.gradient(feature_map, axis=1)
        grad_y = np.gradient(feature_map, axis=0)
        
        grad_xx = np.gradient(grad_x, axis=1)
        grad_yy = np.gradient(grad_y, axis=0)
        grad_xy = np.gradient(grad_x, axis=0)
        
        # 平均曲率
        curvature = np.mean(np.abs(grad_xx + grad_yy))
        
        return curvature
    
    def _compute_compactness(self, feature_map: np.ndarray) -> float:
        """计算紧致度"""
        
        # 二值化
        binary_map = feature_map > np.percentile(feature_map, 75)
        
        # 计算面积和周长
        area = np.sum(binary_map)
        
        if area == 0:
            return 0.0
        
        # 简化的周长计算
        perimeter = np.sum(np.abs(np.gradient(binary_map.astype(float), axis=0))) + \
                   np.sum(np.abs(np.gradient(binary_map.astype(float), axis=1)))
        
        # 紧致度 = 4π * 面积 / 周长²
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter**2)
        else:
            compactness = 0.0
        
        return min(1.0, compactness)
    
    def _compute_elongation(self, feature_map: np.ndarray) -> float:
        """计算伸长度"""
        
        # 二值化
        binary_map = feature_map > np.percentile(feature_map, 75)
        
        # 计算主轴
        y_coords, x_coords = np.where(binary_map)
        
        if len(x_coords) < 2:
            return 0.0
        
        # 协方差矩阵
        coords = np.column_stack([x_coords, y_coords])
        cov_matrix = np.cov(coords.T)
        
        # 特征值
        eigenvalues = np.linalg.eigvals(cov_matrix)
        
        if len(eigenvalues) == 2 and eigenvalues[1] > 0:
            elongation = eigenvalues[0] / eigenvalues[1]
        else:
            elongation = 1.0
        
        return min(10.0, elongation)  # 限制最大值
    
    def _extract_motion_features(self, area_responses: Dict[str, Any]) -> Dict[str, Any]:
        """提取运动特征"""
        
        motion_features = {}
        
        # 基于3a区域（本体感觉）的运动信息
        if '3a' in area_responses:
            proprioceptive_activity = area_responses['3a']['mean_activity']
            
            # 运动速度估计
            motion_speed = proprioceptive_activity
            
            # 运动方向（简化）
            motion_direction = np.random.uniform(0, 2*np.pi)  # 占位
            
            motion_features = {
                'speed': motion_speed,
                'direction': motion_direction,
                'acceleration': abs(motion_speed - 0.5),  # 简化
                'smoothness': 1.0 - abs(motion_speed - 0.5)
            }
        
        return motion_features
    
    def _tactile_object_recognition(self, texture_complexity: Dict[str, Any], 
                                  shape_features: Dict[str, Any]) -> Dict[str, Any]:
        """触觉物体识别"""
        
        # 简化的物体分类
        object_features = {}
        
        # 提取关键特征
        if '1' in texture_complexity:
            roughness = texture_complexity['1']['roughness']
            regularity = texture_complexity['1']['regularity']
        else:
            roughness = 0.0
            regularity = 0.5
        
        if '2' in shape_features:
            compactness = shape_features['2']['compactness']
            elongation = shape_features['2']['elongation']
        else:
            compactness = 0.5
            elongation = 1.0
        
        # 物体分类规则
        if roughness > 0.5 and compactness > 0.7:
            object_type = 'rough_sphere'
            confidence = min(1.0, (roughness + compactness) / 2)
        elif roughness < 0.2 and elongation > 3.0:
            object_type = 'smooth_rod'
            confidence = min(1.0, (1.0 - roughness + elongation/10.0) / 2)
        elif regularity > 0.8 and compactness > 0.8:
            object_type = 'smooth_sphere'
            confidence = min(1.0, (regularity + compactness) / 2)
        else:
            object_type = 'unknown'
            confidence = 0.3
        
        return {
            'object_type': object_type,
            'confidence': confidence,
            'feature_vector': [roughness, regularity, compactness, elongation]
        }
    
    def _tactile_working_memory(self, complex_features: Dict[str, Any]) -> Dict[str, Any]:
        """触觉工作记忆"""
        
        # 简化的触觉工作记忆
        memory_capacity = 3  # 触觉工作记忆容量较小
        
        # 选择最显著的特征进入记忆
        feature_saliency = {}
        
        if 'object_recognition' in complex_features:
            obj_conf = complex_features['object_recognition']['confidence']
            feature_saliency['object'] = obj_conf
        
        if 'texture_complexity' in complex_features:
            for area, texture_info in complex_features['texture_complexity'].items():
                feature_saliency[f'texture_{area}'] = texture_info['overall_complexity']
        
        # 按显著性排序
        sorted_features = sorted(feature_saliency.items(), key=lambda x: x[1], reverse=True)
        memory_contents = sorted_features[:memory_capacity]
        
        memory_load = len(memory_contents) / memory_capacity
        
        return {
            'memory_contents': memory_contents,
            'memory_load': memory_load,
            'memory_capacity': memory_capacity,
            'decay_rate': 0.1 + memory_load * 0.05  # 负载越高衰减越快
        }
    
    def _update_s2_neurons(self, bilateral_integration: Dict[str, Any], 
                          complex_features: Dict[str, Any]) -> Dict[int, float]:
        """更新S2神经元"""
        
        neuron_responses = {}
        
        for neuron_id, neuron in self.areas['S2']['neurons'].items():
            # 双侧输入整合
            bilateral_input = bilateral_integration['integration_strength']
            
            # 复杂特征输入
            feature_input = 0.0
            if 'object_recognition' in complex_features:
                feature_input = complex_features['object_recognition']['confidence']
            
            # 神经元特征选择性
            selectivity_match = np.dot(neuron['complex_feature_selectivity'], 
                                     np.random.random(10))  # 简化
            selectivity_match = selectivity_match / 10.0  # 归一化
            
            # 综合响应
            total_input = (bilateral_input + feature_input) / 2
            neuron_response = total_input * selectivity_match
            
            neuron['firing_rate'] = max(0, neuron_response)
            neuron_responses[neuron_id] = neuron['firing_rate']
        
        return neuron_responses
    
    def _posterior_parietal_processing(self, s1_response: Dict[str, Any], 
                                     s2_response: Dict[str, Any]) -> Dict[str, Any]:
        """后顶叶处理：身体图式和空间整合"""
        
        # 身体图式表征
        body_schema_representation = self._compute_body_schema_representation(s1_response, s2_response)
        
        # 空间注意力
        spatial_attention = self._compute_spatial_attention(s1_response)
        
        # 多模态整合准备
        multimodal_integration = self._prepare_multimodal_integration(s1_response, s2_response)
        
        return {
            'body_schema_representation': body_schema_representation,
            'spatial_attention': spatial_attention,
            'multimodal_integration': multimodal_integration
        }
    
    def _compute_body_schema_representation(self, s1_response: Dict[str, Any], 
                                          s2_response: Dict[str, Any]) -> Dict[str, Any]:
        """计算身体图式表征"""
        
        somatotopic_activation = s1_response['somatotopic_activation']
        body_part = somatotopic_activation['body_part']
        activation_strength = somatotopic_activation['activation_strength']
        
        # 身体部位置信度
        body_part_confidence = activation_strength
        
        # 身体图式一致性
        schema_consistency = s2_response['bilateral_integration']['integration_strength']
        
        # 身体所有权
        body_ownership = (body_part_confidence + schema_consistency) / 2
        
        return {
            'active_body_part': body_part,
            'body_part_confidence': body_part_confidence,
            'schema_consistency': schema_consistency,
            'body_ownership': body_ownership,
            'schema_update_strength': activation_strength * 0.1
        }
    
    def _compute_spatial_attention(self, s1_response: Dict[str, Any]) -> Dict[str, Any]:
        """计算空间注意力"""
        
        somatotopic_activation = s1_response['somatotopic_activation']
        
        # 注意力焦点
        attention_focus = somatotopic_activation['body_part']
        attention_strength = somatotopic_activation['activation_strength']
        
        # 注意力范围
        cortical_spread = somatotopic_activation['cortical_spread']
        attention_radius = cortical_spread['spread_radius']
        
        return {
            'attention_focus': attention_focus,
            'attention_strength': attention_strength,
            'attention_radius': attention_radius,
            'attention_selectivity': attention_strength * 0.8
        }
    
    def _prepare_multimodal_integration(self, s1_response: Dict[str, Any], 
                                      s2_response: Dict[str, Any]) -> Dict[str, Any]:
        """准备多模态整合"""
        
        # 触觉-视觉整合准备度
        tactile_visual_readiness = s2_response['bilateral_integration']['integration_strength']
        
        # 触觉-本体感觉整合
        if '3a' in s1_response['area_responses']:
            tactile_proprioceptive_integration = s1_response['area_responses']['3a']['mean_activity']
        else:
            tactile_proprioceptive_integration = 0.0
        
        return {
            'tactile_visual_readiness': tactile_visual_readiness,
            'tactile_proprioceptive_integration': tactile_proprioceptive_integration,
            'overall_integration_readiness': (tactile_visual_readiness + tactile_proprioceptive_integration) / 2
        }
    
    def _update_body_schema(self, s1_response: Dict[str, Any], 
                           parietal_response: Dict[str, Any]) -> Dict[str, Any]:
        """更新身体图式"""
        
        body_schema_repr = parietal_response['body_schema_representation']
        
        # 身体图式更新
        schema_update = {
            'body_part': body_schema_repr['active_body_part'],
            'update_strength': body_schema_repr['schema_update_strength'],
            'ownership_change': body_schema_repr['body_ownership'] - 0.5,  # 相对于基线
            'spatial_remapping': self._compute_spatial_remapping_update(s1_response)
        }
        
        return schema_update
    
    def _compute_spatial_remapping_update(self, s1_response: Dict[str, Any]) -> Dict[str, float]:
        """计算空间重映射更新"""
        
        somatotopic_activation = s1_response['somatotopic_activation']
        
        # 空间映射精度更新
        mapping_accuracy_change = somatotopic_activation['activation_strength'] * 0.05
        
        # 感受野可塑性
        receptive_field_plasticity = somatotopic_activation['activation_strength'] * 0.02
        
        return {
            'mapping_accuracy_change': mapping_accuracy_change,
            'receptive_field_plasticity': receptive_field_plasticity,
            'cortical_reorganization': mapping_accuracy_change * 0.5
        }
    
    def _extract_body_parts_info(self, s1_response: Dict[str, Any]) -> Dict[str, Any]:
        """提取身体部位信息"""
        
        somatotopic_activation = s1_response['somatotopic_activation']
        
        body_parts_info = {
            somatotopic_activation['body_part']: {
                'confidence': somatotopic_activation['activation_strength'],
                'position': np.array([50, 50, 0]),  # 占位位置
            }
        }
        
        return body_parts_info