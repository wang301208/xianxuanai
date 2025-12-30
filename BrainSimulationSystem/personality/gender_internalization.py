"""
性别角色内化系统

实现基于社会认知理论和生物心理社会模型的性别发展过程
"""

import torch
from torch.nn import Parameter

class GenderInternalizer:
    def __init__(self):
        # 自我概念张量 (4D: 生物/心理/社会/文化)
        self.self_concept = Parameter(torch.rand(4) * 0.5 + 0.25)
        
        # 文化规范权重
        self.cultural_norms = {
            'masculinity': 0.6,
            'femininity': 0.7,
            'neutrality': 0.5
        }
    
    def internalize(self, strength: float = 0.1):
        """执行内化过程"""
        # 计算规范吸引力
        norm_attraction = torch.tensor([
            self.cultural_norms['masculinity'],
            self.cultural_norms['femininity'],
            self.cultural_norms['neutrality'],
            0.5  # 自由探索因子
        ])
        
        # 更新自我概念 (带动量)
        delta = (norm_attraction - self.self_concept) * strength
        self.self_concept.data += delta
        
        # 应用约束
        self.self_concept.data = torch.clamp(self.self_concept, 0, 1)
        
        return {
            'biological': self.self_concept[0].item(),
            'psychological': self.self_concept[1].item(),
            'social': self.self_concept[2].item(),
            'cultural': self.self_concept[3].item()
        }