"""
创造性问题解决引擎

实现基于概念组合与联想思维的新颖解决方案生成
"""

from typing import Dict, List
import random

class CreativeProblemSolver:
    def __init__(self):
        # 创造性思维参数
        self.creativity_parameters = {
            'associative_breadth': 0.7,  # 联想广度
            'conceptual_distance': 0.5,  # 概念跨度
            'novelty_bias': 0.6,         # 新颖性偏好
            'practicality_weight': 0.4    # 实用性考量
        }
        
        # 知识网络
        self.semantic_network = {}
    
    def generate_solutions(self, problem: Dict, num_solutions: int = 3) -> List[Dict]:
        """生成创造性解决方案"""
        # 激活相关概念
        activated_concepts = self._activate_related_concepts(
            problem['keywords'],
            depth=self.creativity_parameters['associative_breadth']
        )
        
        solutions = []
        for _ in range(num_solutions):
            # 随机概念组合
            combo = self._combine_concepts(
                activated_concepts,
                distance=self.creativity_parameters['conceptual_distance']
            )
            
            # 评估方案
            solutions.append({
                'solution': combo,
                'novelty': self._calculate_novelty(combo),
                'feasibility': random.uniform(0.3, 0.9)  # 简化可行性评估
            })
        
        # 按创新性排序
        return sorted(solutions, key=lambda x: -x['novelty'])
    
    def _activate_related_concepts(self, seeds: List[str], depth: float) -> List[str]:
        """激活语义网络中的相关概念"""
        related = []
        for seed in seeds:
            # 模拟概念激活扩散
            related.extend([
                f"{seed}_analog",
                f"{seed}_inverse",
                f"{seed}_hybrid"
            ][:int(3 * depth)])
        return list(set(related))
    
    def _combine_concepts(self, concepts: List[str], distance: float) -> str:
        """组合概念生成新方案"""
        # 按概念距离采样
        n = min(3, max(1, int(len(concepts) * distance)))
        sampled = random.sample(concepts, n)
        return " + ".join(sampled)
    
    def _calculate_novelty(self, concept_combo: str) -> float:
        """计算方案新颖度"""
        components = concept_combo.split(" + ")
        return min(1.0, 0.2 * len(components) + 0.3 * len(set(components)))