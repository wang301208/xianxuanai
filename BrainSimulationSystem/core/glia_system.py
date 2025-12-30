"""
胶质细胞系统模块
负责胶质细胞的功能模拟，包括代谢支持、离子缓冲、神经递质摄取等
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from .enhanced_configs import GlialConfig

class GlialSystem:
    """胶质细胞系统"""
    
    def __init__(self, config: GlialConfig):
        self.config = config
        self.astrocyte_locations = []
        self.microglia_locations = []
        self.oligodendrocyte_locations = []
        
        # 代谢状态
        self.glucose_levels = {}
        self.lactate_levels = {}
        self.atp_levels = {}
        self.calcium_levels = {}
        
        # 活动状态
        self.astrocyte_activation = {}
        self.microglia_activation = {}
        
    def initialize_glial_cells(self, volume: Tuple[float, float, float]):
        """初始化胶质细胞分布"""
        vol_x, vol_y, vol_z = volume
        total_volume = vol_x * vol_y * vol_z  # mm³
        
        # 计算细胞数量
        n_astrocytes = int(total_volume * self.config.astrocyte_density)
        n_microglia = int(total_volume * self.config.microglia_density)
        n_oligodendrocytes = int(total_volume * self.config.oligodendrocyte_density)
        
        # 随机分布
        self.astrocyte_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_astrocytes)
        ]
        
        self.microglia_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_microglia)
        ]
        
        self.oligodendrocyte_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_oligodendrocytes)
        ]
        
        # 初始化代谢状态
        for i, loc in enumerate(self.astrocyte_locations):
            self.glucose_levels[f"astro_{i}"] = 5.0  # mM
            self.lactate_levels[f"astro_{i}"] = 1.0  # mM
            self.atp_levels[f"astro_{i}"] = 3.0  # mM
            self.calcium_levels[f"astro_{i}"] = 0.1  # μM
            self.astrocyte_activation[f"astro_{i}"] = 0.0
        
        for i, loc in enumerate(self.microglia_locations):
            self.microglia_activation[f"micro_{i}"] = 0.0
    
    def glutamate_uptake(self, location: Tuple[float, float, float], 
                        glutamate_concentration: float) -> float:
        """星形胶质细胞谷氨酸摄取"""
        # 找到最近的星形胶质细胞
        min_distance = float('inf')
        nearest_astrocyte = None
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < min_distance:
                min_distance = distance
                nearest_astrocyte = f"astro_{i}"
        
        if nearest_astrocyte is None or min_distance > 50.0:  # μm
            return glutamate_concentration
        
        # 米氏动力学
        km = 10.0  # μM
        vmax = self.config.glutamate_uptake_rate
        
        uptake_rate = vmax * glutamate_concentration / (km + glutamate_concentration)
        
        # 考虑距离衰减
        distance_factor = np.exp(-min_distance / 20.0)
        effective_uptake = uptake_rate * distance_factor
        
        return max(0.0, glutamate_concentration - effective_uptake)
    
    def potassium_buffering(self, location: Tuple[float, float, float], 
                           k_concentration: float) -> float:
        """钾离子缓冲"""
        # 找到附近的星形胶质细胞
        buffering_capacity = 0.0
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < 30.0:  # μm
                distance_factor = np.exp(-distance / 15.0)
                buffering_capacity += self.config.k_buffering_capacity * distance_factor
        
        # 缓冲效应
        baseline_k = 3.0  # mM
        excess_k = k_concentration - baseline_k
        buffered_k = baseline_k + excess_k * np.exp(-buffering_capacity)
        
        return max(baseline_k, buffered_k)
    
    def calcium_wave_propagation(self, trigger_location: Tuple[float, float, float], 
                               intensity: float, current_time: float):
        """钙波传播"""
        wave_speed = self.config.calcium_wave_speed
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(trigger_location, astro_loc)))
            arrival_time = distance / wave_speed
            
            if arrival_time < 1000.0:  # 1秒内到达
                # 计算钙浓度增加
                amplitude = intensity * np.exp(-distance / 100.0)  # 距离衰减
                
                astro_id = f"astro_{i}"
                if astro_id in self.calcium_levels:
                    self.calcium_levels[astro_id] += amplitude
                    
                    # 如果超过阈值，释放ATP
                    if self.calcium_levels[astro_id] > self.config.atp_release_threshold:
                        self._release_atp(astro_loc, amplitude)
    
    def _release_atp(self, location: Tuple[float, float, float], amount: float):
        """释放ATP"""
        # ATP可以激活附近的小胶质细胞
        for i, micro_loc in enumerate(self.microglia_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, micro_loc)))
            if distance < 50.0:  # μm
                activation = amount * np.exp(-distance / 25.0)
                micro_id = f"micro_{i}"
                if micro_id in self.microglia_activation:
                    self.microglia_activation[micro_id] += activation
    
    def metabolic_support(self, location: Tuple[float, float, float], 
                         energy_demand: float) -> float:
        """代谢支持"""
        # 找到最近的星形胶质细胞
        min_distance = float('inf')
        nearest_astrocyte = None
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < min_distance:
                min_distance = distance
                nearest_astrocyte = f"astro_{i}"
        
        if nearest_astrocyte is None or min_distance > 30.0:
            return 0.0
        
        # 检查葡萄糖和ATP水平
        glucose = self.glucose_levels.get(nearest_astrocyte, 0.0)
        atp = self.atp_levels.get(nearest_astrocyte, 0.0)
        
        # 计算可提供的能量
        available_energy = min(glucose * 0.1, atp * 0.2)  # 简化的能量转换
        provided_energy = min(energy_demand, available_energy)
        
        # 更新代谢状态
        if provided_energy > 0:
            self.glucose_levels[nearest_astrocyte] -= provided_energy / 0.1
            self.atp_levels[nearest_astrocyte] -= provided_energy / 0.2
            
            # 产生乳酸
            lactate_production = provided_energy * 0.5
            self.lactate_levels[nearest_astrocyte] += lactate_production
        
        return provided_energy
    
    def update_metabolism(self, dt: float):
        """更新代谢状态"""
        for astro_id in self.astrocyte_activation.keys():
            # 葡萄糖消耗
            glucose_consumption = self.config.glucose_consumption * dt
            if astro_id in self.glucose_levels:
                self.glucose_levels[astro_id] = max(0.0, 
                    self.glucose_levels[astro_id] - glucose_consumption)
            
            # ATP再生
            if astro_id in self.glucose_levels and self.glucose_levels[astro_id] > 0:
                atp_generation = glucose_consumption * 30  # ATP/glucose比例
                if astro_id in self.atp_levels:
                    self.atp_levels[astro_id] = min(5.0,  # 最大ATP浓度
                        self.atp_levels[astro_id] + atp_generation)
            
            # 钙离子衰减
            if astro_id in self.calcium_levels:
                decay_rate = 0.01  # 1/ms
                self.calcium_levels[astro_id] *= np.exp(-decay_rate * dt)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取胶质细胞系统统计信息"""
        return {
            'astrocytes': len(self.astrocyte_locations),
            'microglia': len(self.microglia_locations),
            'oligodendrocytes': len(self.oligodendrocyte_locations),
            'mean_glucose': float(np.mean(list(self.glucose_levels.values()))),
            'mean_atp': float(np.mean(list(self.atp_levels.values()))),
            'mean_calcium': float(np.mean(list(self.calcium_levels.values())))
        }