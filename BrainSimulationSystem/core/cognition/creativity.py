"""
创造性思维过程模块

实现发散思维、远程关联和顿悟生成
"""

import numpy as np
from typing import List, Dict

class CreativeProcessOrchestrator:
    def __init__(self):
        # 创造性思维参数
        self.divergence_threshold = 0.7  # 发散度阈值
        self.association_range = 3       # 关联跨度(1-5)
        self.insight_sensitivity = 0.6  # 顿悟敏感度
        
        # 知识库
        self.conceptual_space = {}  # 概念网络
        
    def generate_solutions(self, problem: Dict) -> List[Dict]:
        """生成创造性解决方案
        
        Args:
            problem: 包含constraints和resources的问题描述
            
        Returns:
            解决方案列表，按创造性评分排序
        """
        # 阶段1: 发散思维
        raw_ideas = self._divergent_thinking(
            constraints=problem['constraints'],
            resources=problem['resources']
        )
        
        # 阶段2: 远程关联
        combined_ideas = []
        for i in range(len(raw_ideas)):
            for j in range(i+1, min(i+self.association_range, len(raw_ideas))):
                combined = self._cross_domain_associate(raw_ideas[i], raw_ideas[j])
                if combined:
                    combined_ideas.append(combined)
        
        # 阶段3: 顿悟筛选
        solutions = []
        for idea in raw_ideas + combined_ideas:
            if self._insight_detection(idea):
                solutions.append({
                    'solution': idea,
                    'creativity_score': self._evaluate_creativity(idea)
                })
                
        return sorted(solutions, key=lambda x: -x['creativity_score'])
    
    def _divergent_thinking(self, constraints: List[str], resources: List[str]) -> List[str]:
        """发散思维阶段"""
        # 在实际实现中接入知识图谱查询
        return [
            f"{res}+{const}" 
            for res in resources 
            for const in constraints
        ]
    
    def _cross_domain_associate(self, ideaA: str, ideaB: str) -> str:
        """跨领域关联"""
        if len(set(ideaA.split('+')) & set(ideaB.split('+'))) == 0:
            return f"({ideaA})×({ideaB})"
        return None
        
    def _insight_detection(self, idea: str) -> bool:
        """顿悟检测"""
        novelty = len(set(idea.split('+')))/10  # 简化计算
        return novelty > self.insight_sensitivity
        
    def _evaluate_creativity(self, idea: str) -> float:
        """创造性评分(0-1)"""
        components = idea.split('+')
        return min(1.0, 0.2*len(components) + 0.8*self._conceptual_distance(components))
        
    def _conceptual_distance(self, concepts: List[str]) -> float:
        """计算概念间平均距离"""
        # 简化实现 - 实际应使用概念嵌入
        return np.mean([
            abs(hash(c1)%100 - hash(c2)%100)/100 
            for c1 in concepts 
            for c2 in concepts
        ])