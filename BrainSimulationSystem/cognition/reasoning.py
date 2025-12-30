"""
神经符号混合推理引擎
实现从神经活动到符号规则的提取与推理
"""

import numpy as np
from typing import Dict, List, Any

class NeuralRuleExtractor:
    """从神经活动中提取符号规则"""
    def __init__(self):
        self.feature_detectors = []
        self.rule_templates = {}
        
    def extract_rules(self, neural_activity: np.ndarray) -> List[Dict]:
        """从神经活动中提取规则"""
        # 特征检测
        features = self._detect_features(neural_activity)
        
        # 规则匹配
        rules = []
        for feature_set in features:
            for template_name, template in self.rule_templates.items():
                if self._matches_template(feature_set, template):
                    rules.append({
                        'type': template_name,
                        'params': self._extract_params(feature_set, template),
                        'confidence': self._calculate_confidence(feature_set, template)
                    })
        
        return rules
        
    def _detect_features(self, neural_activity: np.ndarray) -> List[Dict]:
        """检测神经活动中的特征"""
        features = []
        for detector in self.feature_detectors:
            response = detector.activate(neural_activity)
            if response > 0.7:  # 激活阈值
                features.append({
                    'type': detector.feature_type,
                    'value': response,
                    'location': detector.receptive_field
                })
        return self._group_features(features)
        
    def _group_features(self, features: List[Dict]) -> List[Dict]:
        """将特征分组"""
        # 简化实现
        return [{'features': features}]
        
    def _matches_template(self, feature_set: Dict, template: Dict) -> bool:
        """检查特征集是否匹配规则模板"""
        required = template.get('required_features', [])
        return all(self._has_feature(feature_set, req) for req in required)
        
    def _has_feature(self, feature_set: Dict, required: Dict) -> bool:
        """检查特征集是否包含所需特征"""
        for feature in feature_set.get('features', []):
            if feature['type'] == required['type']:
                return feature['value'] >= required.get('min_value', 0)
        return False
        
    def _extract_params(self, feature_set: Dict, template: Dict) -> Dict:
        """从特征集中提取规则参数"""
        params = {}
        for param_name, param_spec in template.get('params', {}).items():
            for feature in feature_set.get('features', []):
                if feature['type'] == param_spec['source']:
                    params[param_name] = feature['value']
                    break
        return params
        
    def _calculate_confidence(self, feature_set: Dict, template: Dict) -> float:
        """计算规则置信度"""
        # 简化实现
        return 0.8


class SymbolicReasoner:
    """符号逻辑推理引擎"""
    def __init__(self):
        self.knowledge_base = []
        self.inference_rules = []
        
    def add_fact(self, fact: Dict) -> None:
        """添加事实到知识库"""
        self.knowledge_base.append({
            'type': 'fact',
            'content': fact,
            'confidence': fact.get('confidence', 1.0)
        })
        
    def add_rule(self, rule: Dict) -> None:
        """添加推理规则"""
        self.inference_rules.append({
            'type': 'rule',
            'if': rule.get('if', {}),
            'then': rule.get('then', {}),
            'confidence': rule.get('confidence', 0.9)
        })
        
    def query(self, query: Dict) -> List[Dict]:
        """查询知识库"""
        # 直接匹配
        direct_matches = self._direct_match(query)
        
        # 规则推理
        inferred = self._apply_rules(query)
        
        # 合并结果
        results = direct_matches + inferred
        
        # 按置信度排序
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return results
        
    def _direct_match(self, query: Dict) -> List[Dict]:
        """直接匹配知识库"""
        matches = []
        for item in self.knowledge_base:
            if item['type'] == 'fact' and self._matches(item['content'], query):
                matches.append(item)
        return matches
        
    def _matches(self, fact: Dict, query: Dict) -> bool:
        """检查事实是否匹配查询"""
        for key, value in query.items():
            if key not in fact or fact[key] != value:
                return False
        return True
        
    def _apply_rules(self, query: Dict) -> List[Dict]:
        """应用推理规则"""
        results = []
        for rule in self.inference_rules:
            if self._matches(rule['then'], query):
                # 检查前提条件
                matches = self._direct_match(rule['if'])
                if matches:
                    # 计算置信度
                    confidence = rule['confidence'] * max(m['confidence'] for m in matches)
                    results.append({
                        'type': 'inferred',
                        'content': rule['then'],
                        'confidence': confidence,
                        'source': 'rule'
                    })
        return results


class HybridReasoningEngine:
    """神经符号混合推理引擎"""
    def __init__(self):
        self.neural_extractor = NeuralRuleExtractor()
        self.symbolic_reasoner = SymbolicReasoner()
        
    def process(self, neural_activity: np.ndarray, query: Dict = None) -> Dict:
        """处理神经活动并执行推理"""
        # 从神经活动中提取规则
        extracted_rules = self.neural_extractor.extract_rules(neural_activity)
        
        # 将提取的规则添加到符号推理器
        for rule in extracted_rules:
            if rule['type'] == 'fact':
                self.symbolic_reasoner.add_fact(rule['params'])
            else:
                self.symbolic_reasoner.add_rule({
                    'if': rule.get('if', {}),
                    'then': rule.get('then', {}),
                    'confidence': rule.get('confidence', 0.8)
                })
        
        # 执行查询
        if query:
            results = self.symbolic_reasoner.query(query)
        else:
            # 默认查询所有可能的结论
            results = self.symbolic_reasoner.query({})
            
        return {
            'results': results,
            'extracted_rules': extracted_rules,
            'confidence': self._aggregate_confidence(results)
        }
        
    def _aggregate_confidence(self, results: List[Dict]) -> float:
        """聚合结果置信度"""
        if not results:
            return 0.0
        return sum(r.get('confidence', 0) for r in results) / len(results)