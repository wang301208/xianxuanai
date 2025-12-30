"""
体积传导系统模块
负责神经递质在细胞外空间的扩散和清除过程
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from .enhanced_configs import VolumeTransmissionConfig

class VolumeTransmissionSystem:
    """体积传导系统"""
    
    def __init__(self, config: VolumeTransmissionConfig):
        self.config = config
        self.transmitter_concentrations = {}
        self.diffusion_grid = None
        self.grid_size = (100, 100, 100)
        
        if config.enabled:
            self._initialize_diffusion_grid()
    
    def _initialize_diffusion_grid(self):
        """初始化扩散网格"""
        self.diffusion_grid = {}
        
        # 为每种神经递质创建3D浓度网格
        transmitters = ['glutamate', 'gaba', 'dopamine', 'acetylcholine', 
                       'serotonin', 'norepinephrine']
        
        for transmitter in transmitters:
            self.diffusion_grid[transmitter] = np.zeros(self.grid_size)
    
    def add_point_source(self, transmitter: str, location: Tuple[float, float, float], 
                        amount: float, molecular_weight: float = 150.0):
        """添加点源释放"""
        if not self.config.enabled or transmitter not in self.diffusion_grid:
            return
        
        # 转换为网格坐标
        x, y, z = [int(coord) for coord in location]
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        
        # 计算扩散系数
        diffusion_coeff = self._calculate_diffusion_coefficient(molecular_weight)
        
        # 添加到网格
        self.diffusion_grid[transmitter][x, y, z] += amount
    
    def _calculate_diffusion_coefficient(self, molecular_weight: float) -> float:
        """计算扩散系数"""
        # Stokes-Einstein方程的简化版本
        base_diffusion = 1e-6  # cm²/s
        
        # 分子量依赖性
        if self.config.molecular_weight_dependence:
            mw_factor = (150.0 / molecular_weight) ** 0.5  # 参考分子量150 Da
        else:
            mw_factor = 1.0
        
        # 迂曲度和细胞外空间分数
        effective_diffusion = (base_diffusion * mw_factor * 
                             self.config.diffusion_space_fraction / 
                             self.config.tortuosity)
        
        return effective_diffusion
    
    def update_diffusion(self, dt: float):
        """更新扩散过程"""
        if not self.config.enabled:
            return
        
        for transmitter in self.diffusion_grid.keys():
            # 3D扩散方程的简化数值解
            concentration = self.diffusion_grid[transmitter]
            
            # 拉普拉斯算子（6点模板）
            laplacian = np.zeros_like(concentration)
            
            # x方向
            laplacian[1:-1, :, :] += (concentration[2:, :, :] - 
                                     2*concentration[1:-1, :, :] + 
                                     concentration[:-2, :, :])
            
            # y方向
            laplacian[:, 1:-1, :] += (concentration[:, 2:, :] - 
                                     2*concentration[:, 1:-1, :] + 
                                     concentration[:, :-2, :])
            
            # z方向
            laplacian[:, :, 1:-1] += (concentration[:, :, 2:] - 
                                     2*concentration[:, :, 1:-1] + 
                                     concentration[:, :, :-2])
            
            # 扩散系数（假设所有递质相同）
            D = 1e-6  # cm²/s
            
            # 更新浓度
            diffusion_term = D * laplacian * dt
            clearance_term = -self.config.clearance_rate * concentration * dt
            
            self.diffusion_grid[transmitter] += diffusion_term + clearance_term
            
            # 确保非负
            self.diffusion_grid[transmitter] = np.maximum(
                self.diffusion_grid[transmitter], 0.0
            )
    
    def get_concentration(self, transmitter: str, 
                         location: Tuple[float, float, float]) -> float:
        """获取指定位置的递质浓度"""
        if (not self.config.enabled or 
            transmitter not in self.diffusion_grid):
            return 0.0
        
        x, y, z = [int(coord) for coord in location]
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        
        return float(self.diffusion_grid[transmitter][x, y, z])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取体积传导系统统计信息"""
        if not self.config.enabled:
            return {'enabled': False}
        
        stats = {'enabled': True}
        for transmitter, grid in self.diffusion_grid.items():
            stats[transmitter] = {
                'mean_concentration': float(np.mean(grid)),
                'max_concentration': float(np.max(grid)),
                'active_voxels': int(np.sum(grid > 0.01))
            }
        
        return stats