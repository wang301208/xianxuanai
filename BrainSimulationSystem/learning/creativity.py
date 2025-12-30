"""
创造性决策系统
实现自主生成新型决策框架的能力
"""

from typing import Dict, List, Any, Callable
import numpy as np
import random
import copy

class CreativeDecisionSystem:
    def __init__(self):
        # 基础决策框架库
        self.framework_library = []
        
        # 创造性参数
        self.creativity_params = {
            'divergence': 0.7,      # 发散思维强度
            'combination': 0.6,     # 概念组合能力
            'abstraction': 0.5,     # 抽象化能力
            'constraint_relaxation': 0.4  # 约束放松程度
        }
        
        # 评估函数
        self.evaluation_functions = {}
        
        # 创新历史
        self.innovation_history = []
        
    def register_framework(self, framework: Dict) -> int:
        """注册基础决策框架"""
        # 框架结构
        if 'structure' not in framework or 'evaluation' not in framework:
            return -1
            
        # 添加框架ID
        framework['id'] = len(self.framework_library)
        self.framework_library.append(framework)
        
        return framework['id']
        
    def register_evaluation_function(self, name: str, func: Callable) -> bool:
        """注册评估函数"""
        if name in self.evaluation_functions:
            return False
            
        self.evaluation_functions[name] = func
        return True
        
    def generate_novel_framework(self, context: Dict) -> Dict:
        """生成新型决策框架"""
        if not self.framework_library:
            return {'status': 'no_frameworks'}
            
        # 选择基础框架
        base_frameworks = self._select_base_frameworks(context)
        
        # 创造性转换
        novel_framework = self._apply_creative_transformations(base_frameworks, context)
        
        # 评估新框架
        evaluation = self._evaluate_framework(novel_framework, context)
        
        # 记录创新
        self.innovation_history.append({
            'base_frameworks': [f['id'] for f in base_frameworks],
            'novel_framework': novel_framework,
            'evaluation': evaluation,
            'context': context
        })
        
        return {
            'framework': novel_framework,
            'evaluation': evaluation,
            'base_frameworks': [f['id'] for f in base_frameworks]
        }
        
    def _select_base_frameworks(self, context: Dict) -> List[Dict]:
        """选择基础框架"""
        # 计算每个框架与上下文的相关性
        relevance_scores = []
        for framework in self.framework_library:
            # 简化的相关性计算
            relevance = self._calculate_relevance(framework, context)
            relevance_scores.append((framework, relevance))
            
        # 按相关性排序
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前2个最相关的框架
        selected = [item[0] for item in relevance_scores[:2]]
        
        # 随机添加一个不太相关的框架以增加多样性
        if len(relevance_scores) > 3:
            divergent_idx = random.randint(2, min(5, len(relevance_scores)-1))
            selected.append(relevance_scores[divergent_idx][0])
            
        return selected
        
    def _apply_creative_transformations(self, base_frameworks: List[Dict], 
                                      context: Dict) -> Dict:
        """应用创造性转换"""
        # 创建新框架骨架
        novel_framework = {
            'id': 'novel_' + str(len(self.innovation_history)),
            'structure': {},
            'evaluation': {},
            'origin': 'creative_generation',
            'base_frameworks': [f['id'] for f in base_frameworks]
        }
        
        # 应用不同的创造性操作
        operations = [
            self._combine_frameworks,
            self._abstract_framework,
            self._relax_constraints,
            self._introduce_novelty
        ]
        
        # 随机选择操作顺序
        random.shuffle(operations)
        
        # 应用创造性操作
        current = copy.deepcopy(base_frameworks[0])
        for operation in operations:
            current = operation(current, base_frameworks, context)
            
        # 更新新框架
        novel_framework['structure'] = current['structure']
        novel_framework['evaluation'] = current['evaluation']
        
        return novel_framework
        
    def _combine_frameworks(self, current: Dict, base_frameworks: List[Dict], 
                          context: Dict) -> Dict:
        """组合多个框架"""
        if len(base_frameworks) < 2:
            return current
            
        # 创建组合框架
        combined = copy.deepcopy(current)
        
        # 从其他框架中整合结构元素
        for framework in base_frameworks[1:]:
            # 随机选择一些结构元素进行整合
            structure_keys = list(framework['structure'].keys())
            num_elements = max(1, int(len(structure_keys) * self.creativity_params['combination']))
            selected_keys = random.sample(structure_keys, min(num_elements, len(structure_keys)))
            
            # 整合选中的元素
            for key in selected_keys:
                if key not in combined['structure']:
                    combined['structure'][key] = framework['structure'][key]
                else:
                    # 如果元素已存在，尝试融合
                    combined['structure'][key] = self._merge_elements(
                        combined['structure'][key],
                        framework['structure'][key]
                    )
        
        return combined
        
    def _abstract_framework(self, current: Dict, base_frameworks: List[Dict], 
                          context: Dict) -> Dict:
        """抽象化框架"""
        abstracted = copy.deepcopy(current)
        
        # 抽象化程度
        abstraction_level = self.creativity_params['abstraction']
        
        # 识别共同模式
        common_patterns = self._identify_common_patterns(base_frameworks)
        
        # 应用抽象模式
        for pattern_name, pattern in common_patterns.items():
            if random.random() < abstraction_level:
                # 创建更抽象的元素
                abstracted['structure'][f'abstract_{pattern_name}'] = {
                    'type': 'abstract_pattern',
                    'components': pattern['components'],
                    'relations': pattern['relations']
                }
                
                # 移除被抽象的具体元素
                for component in pattern['components']:
                    if component in abstracted['structure']:
                        del abstracted['structure'][component]
        
        return abstracted
        
    def _relax_constraints(self, current: Dict, base_frameworks: List[Dict], 
                         context: Dict) -> Dict:
        """放松约束条件"""
        relaxed = copy.deepcopy(current)
        
        # 约束放松程度
        relaxation_level = self.creativity_params['constraint_relaxation']
        
        # 识别约束条件
        constraints = []
        for key, element in relaxed['structure'].items():
            if 'constraints' in element:
                constraints.append((key, element['constraints']))
                
        # 随机放松一些约束
        for key, constraint_list in constraints:
            if random.random() < relaxation_level:
                # 随机选择一个约束进行放松
                if constraint_list:
                    constraint_idx = random.randint(0, len(constraint_list)-1)
                    # 放松或移除约束
                    if random.random() < 0.5:
                        # 放松约束
                        constraint_list[constraint_idx]['strength'] *= 0.5
                    else:
                        # 移除约束
                        constraint_list.pop(constraint_idx)
        
        return relaxed
        
    def _introduce_novelty(self, current: Dict, base_frameworks: List[Dict], 
                         context: Dict) -> Dict:
        """引入新颖元素"""
        novel = copy.deepcopy(current)
        
        # 新颖性引入程度
        novelty_level = self.creativity_params['divergence']
        
        # 随机添加新元素
        if random.random() < novelty_level:
            # 生成新元素名称
            new_element_name = f"novel_{len(novel['structure'])}"
            
            # 创建新元素
            novel['structure'][new_element_name] = {
                'type': 'novel_component',
                'properties': {
                    'origin': 'generated',
                    'novelty': random.random()
                }
            }
            
            # 随机连接到现有元素
            if novel['structure']:
                target = random.choice(list(novel['structure'].keys()))
                if 'connections' not in novel['structure'][target]:
                    novel['structure'][target]['connections'] = []
                novel['structure'][target]['connections'].append(new_element_name)
        
        return novel
        
    def _evaluate_framework(self, framework: Dict, context: Dict) -> Dict:
        """评估决策框架"""
        evaluation = {
            'coherence': 0.0,    # 内部一致性
            'novelty': 0.0,      # 新颖性
            'utility': 0.0,      # 实用性
            'adaptability': 0.0  # 适应性
        }
        
        # 计算内部一致性
        evaluation['coherence'] = self._calculate_coherence(framework)
        
        # 计算新颖性
        evaluation['novelty'] = self._calculate_novelty(framework)
        
        # 计算实用性
        evaluation['utility'] = self._calculate_utility(framework, context)
        
        # 计算适应性
        evaluation['adaptability'] = self._calculate_adaptability(framework)
        
        return evaluation
        
    def _calculate_relevance(self, framework: Dict, context: Dict) -> float:
        """计算框架与上下文的相关性"""
        # 简化实现
        return random.random()
        
    def _merge_elements(self, element1: Dict, element2: Dict) -> Dict:
        """合并两个框架元素"""
        merged = copy.deepcopy(element1)
        
        # 合并属性
        if 'properties' in element2:
            if 'properties' not in merged:
                merged['properties'] = {}
            for k, v in element2['properties'].items():
                if k not in merged['properties']:
                    merged['properties'][k] = v
                else:
                    # 如果属性已存在，取平均值
                    if isinstance(v, (int, float)) and isinstance(merged['properties'][k], (int, float)):
                        merged['properties'][k] = (merged['properties'][k] + v) / 2
        
        # 合并连接
        if 'connections' in element2:
            if 'connections' not in merged:
                merged['connections'] = []
            for conn in element2['connections']:
                if conn not in merged['connections']:
                    merged['connections'].append(conn)
                    
        return merged
        
    def _identify_common_patterns(self, frameworks: List[Dict]) -> Dict:
        """识别框架间的共同模式"""
        patterns = {}
        
        if len(frameworks) < 2:
            return patterns
            
        # 比较前两个框架
        framework1 = frameworks[0]['structure']
        framework2 = frameworks[1]['structure']
        
        # 寻找共同元素
        common_keys = set(framework1.keys()) & set(framework2.keys())
        
        for key in common_keys:
            # 简化模式识别
            if framework1[key]['type'] == framework2[key]['type']:
                patterns[f"pattern_{key}"] = {
                    'components': [key],
                    'relations': []
                }
                
        return patterns
        
    def _calculate_coherence(self, framework: Dict) -> float:
        """计算框架内部一致性"""
        # 简化实现
        return 0.7 + random.random() * 0.3
        
    def _calculate_novelty(self, framework: Dict) -> float:
        """计算框架新颖性"""
        # 简化实现
        return min(1.0, len(framework['structure']) / 10.0)
        
    def _calculate_utility(self, framework: Dict, context: Dict) -> float:
        """计算框架实用性"""
        # 简化实现
        return 0.5 + random.random() * 0.5
        
    def _calculate_adaptability(self, framework: Dict) -> float:
        """计算框架适应性"""
        # 简化实现
        return 0.6 + random.random() * 0.4