"""
元决策系统
实现对决策过程本身的监控、评估和调整
"""

from typing import Dict, List, Any
from collections import Counter
import copy
import math
import time
from statistics import mean

class MetaDecisionSystem:
    def __init__(self):
        # 决策质量指标
        self.quality_metrics = {
            'consistency': 0.0,      # 决策一致性
            'adaptability': 0.0,     # 环境适应性
            'efficiency': 0.0,       # 决策效率
            'rationality': 0.0       # 理性程度
        }
        
        # 决策历史
        self.decision_history = []
        self.max_history = 1000
        
        # 决策过程监控
        self.process_monitors = {
            'time_spent': [],        # 决策时间
            'options_evaluated': [],  # 评估的选项数
            'confidence_levels': []   # 决策置信度
        }
        
        # 调整参数
        self.adjustment_factors = {
            'cognitive_weight': 0.0,  # 认知权重调整
            'emotional_weight': 0.0,  # 情感权重调整
            'social_weight': 0.0,     # 社会权重调整
            'risk_tolerance': 0.0     # 风险容忍调整
        }

        # 待应用的调整（分离决策引擎和决策过程）
        self._pending_engine_adjustments: Dict[str, Any] = {}
        self._pending_process_adjustments: Dict[str, Any] = {}
        
    def monitor_decision_process(self, process_data: Dict) -> Dict:
        """监控决策过程"""
        # 记录决策时间
        start_time = time.time()
        
        # 处理决策过程数据
        self.process_monitors['options_evaluated'].append(
            len(process_data.get('options', []))
        )
        
        # 记录决策结束时间
        time_spent = process_data.get('time_spent')
        if time_spent is None:
            time_spent = time.time() - start_time
        process_data['time_spent'] = time_spent
        self.process_monitors['time_spent'].append(process_data['time_spent'])
        
        # 记录置信度
        confidence = process_data.get('confidence', 0.5)
        self.process_monitors['confidence_levels'].append(confidence)
        
        # 保存决策历史
        self.decision_history.append({
            'timestamp': time.time(),
            'context': process_data.get('context', {}),
            'options': process_data.get('options', []),
            'selected': process_data.get('selected', None),
            'confidence': confidence,
            'time_spent': process_data['time_spent']
        })
        
        # 限制历史大小
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
            
        return {
            'monitored': True,
            'metrics_updated': self._update_quality_metrics()
        }
        
    def evaluate_decision_quality(self, decision_outcome: Dict) -> Dict:
        """评估决策质量"""
        if not self.decision_history:
            return {'status': 'no_history'}
            
        # 获取最近的决策
        recent_decision = self.decision_history[-1]
        
        # 计算决策质量指标
        quality = {}
        
        # 结果与预期的差距
        expected = recent_decision.get('expected_outcome')
        if not expected:
            expected = decision_outcome.get('expected_outcome', {})
        recent_decision['expected_outcome'] = expected
        actual = decision_outcome.get('actual_outcome', {})

        # 计算预期与实际的差距
        expectation_gap = self._calculate_outcome_gap(expected, actual)
        quality['expectation_gap'] = expectation_gap
        
        # 决策时间效率
        time_spent = recent_decision.get('time_spent', 0)
        quality['time_efficiency'] = 1.0 / (1.0 + time_spent)
        
        # 决策收益
        utility = decision_outcome.get('utility', 0)
        quality['utility'] = utility

        # 综合质量评分
        quality['overall'] = (
            (1 - expectation_gap) * 0.4 + 
            quality['time_efficiency'] * 0.2 + 
            min(1.0, utility / 10) * 0.4
        )
        
        # 更新决策历史
        recent_decision['outcome'] = decision_outcome
        recent_decision['quality'] = quality
        if 'selected_option' in decision_outcome:
            recent_decision['selected_option'] = decision_outcome['selected_option']

        return quality

    def suggest_adjustments(self) -> Dict:
        """建议决策参数调整"""
        if len(self.decision_history) < 5:
            return {'status': 'insufficient_history'}

        # 分析最近的决策
        recent_decisions = self.decision_history[-5:]

        # 计算平均质量
        avg_quality = self._safe_mean([
            d.get('quality', {}).get('overall', 0.5)
            for d in recent_decisions
            if 'quality' in d
        ])

        # 分析决策模式
        patterns = self._analyze_decision_patterns(recent_decisions)

        # 生成调整建议（区分决策引擎与决策过程）
        engine_adjustments: Dict[str, Any] = {}
        heuristic_deltas: Dict[str, float] = {}
        process_adjustments: Dict[str, float] = {}

        # 如果质量低于阈值，建议提升学习和探索
        if avg_quality < 0.6:
            quality_gap = max(0.0, 0.6 - float(avg_quality))
            process_adjustments['learning_rate'] = round(min(0.1, 0.05 + quality_gap * 0.2), 4)
            process_adjustments['exploration_rate'] = round(min(0.2, 0.1 + quality_gap * 0.3), 4)
        elif avg_quality > 0.8:
            quality_excess = float(avg_quality) - 0.8
            process_adjustments['learning_rate'] = round(-min(0.05, 0.02 + quality_excess * 0.1), 4)
            process_adjustments['exploration_rate'] = round(-min(0.1, 0.05 + quality_excess * 0.2), 4)

        # 认知-情感平衡调整
        if patterns.get('emotional_bias', 0) > 0.2:
            heuristic_deltas['cognitive'] = heuristic_deltas.get('cognitive', 0.0) + 0.08
            heuristic_deltas['emotional'] = heuristic_deltas.get('emotional', 0.0) - 0.08
        elif patterns.get('cognitive_bias', 0) > 0.2:
            heuristic_deltas['cognitive'] = heuristic_deltas.get('cognitive', 0.0) - 0.05
            heuristic_deltas['emotional'] = heuristic_deltas.get('emotional', 0.0) + 0.05

        # 风险调整
        risk_delta = 0.0
        if patterns.get('risk_aversion', 0) > 0.3:
            risk_delta = min(0.2, patterns['risk_aversion'])
        elif patterns.get('risk_seeking', 0) > 0.3:
            risk_delta = -min(0.2, patterns['risk_seeking'])
        if risk_delta:
            engine_adjustments['risk_tolerance_delta'] = round(float(risk_delta), 4)

        if heuristic_deltas:
            engine_adjustments['heuristic_weights'] = {
                k: round(float(v), 4) for k, v in heuristic_deltas.items()
            }

        # 更新调整因子以反映最新建议
        self.adjustment_factors['cognitive_weight'] = heuristic_deltas.get('cognitive', 0.0)
        self.adjustment_factors['emotional_weight'] = heuristic_deltas.get('emotional', 0.0)
        self.adjustment_factors['social_weight'] = heuristic_deltas.get('social', 0.0)
        self.adjustment_factors['risk_tolerance'] = engine_adjustments.get('risk_tolerance_delta', 0.0)

        self._pending_engine_adjustments = engine_adjustments
        self._pending_process_adjustments = process_adjustments

        return {
            'suggested_adjustments': {
                'engine': copy.deepcopy(engine_adjustments),
                'process': copy.deepcopy(process_adjustments)
            },
            'average_quality': avg_quality,
            'patterns': patterns
        }
        
    def get_meta_insights(self) -> Dict:
        """获取元决策洞察"""
        if len(self.decision_history) < 10:
            return {'status': 'insufficient_history'}
            
        # 计算决策时间趋势
        time_trend = self._calculate_trend(self.process_monitors['time_spent'])
        
        # 计算置信度趋势
        confidence_trend = self._calculate_trend(self.process_monitors['confidence_levels'])
        
        # 计算质量趋势
        quality_values = [
            d.get('quality', {}).get('overall', 0.5) 
            for d in self.decision_history 
            if 'quality' in d
        ]
        quality_trend = self._calculate_trend(quality_values)
        
        # 识别决策瓶颈
        bottlenecks = self._identify_bottlenecks()
        
        return {
            'trends': {
                'time': time_trend,
                'confidence': confidence_trend,
                'quality': quality_trend
            },
            'bottlenecks': bottlenecks,
            'quality_metrics': self.quality_metrics,
            'adjustment_factors': self.adjustment_factors
        }
    
    def _update_quality_metrics(self) -> bool:
        """更新决策质量指标"""
        if len(self.decision_history) < 5:
            return False

        recent = self.decision_history[-5:]

        # 计算一致性 (相似情境下决策的一致性)
        self.quality_metrics['consistency'] = self._calculate_consistency(recent)

        # 计算适应性 (对环境变化的响应)
        self.quality_metrics['adaptability'] = self._calculate_adaptability(recent)

        # 计算效率 (决策时间与质量的平衡)
        self.quality_metrics['efficiency'] = self._calculate_efficiency(recent)

        # 计算理性程度 (决策与最优选择的接近程度)
        self.quality_metrics['rationality'] = self._calculate_rationality(recent)

        return True
        
    def _calculate_outcome_gap(self, expected: Dict, actual: Dict) -> float:
        """计算预期与实际结果的差距"""
        if not expected or not actual:
            return 0.5
            
        # 简化实现，计算共有键的值差异
        common_keys = set(expected.keys()) & set(actual.keys())
        if not common_keys:
            return 0.5
            
        total_diff = 0
        for key in common_keys:
            exp_val = expected[key]
            act_val = actual[key]
            
            # 处理数值类型
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                # 归一化差异
                max_val = max(abs(exp_val), abs(act_val))
                if max_val > 0:
                    diff = abs(exp_val - act_val) / max_val
                else:
                    diff = 0
            else:
                # 非数值类型，相等为0，不等为1
                diff = 0 if exp_val == act_val else 1
                
            total_diff += diff
            
        return total_diff / len(common_keys)
        
    def _analyze_decision_patterns(self, decisions: List[Dict]) -> Dict:
        """分析决策模式"""
        patterns = {}
        
        # 情感偏差
        emotional_scores = []
        cognitive_scores = []
        
        for d in decisions:
            context = d.get('context', {})
            emotional_scores.append(context.get('emotional_factor', 0.5))
            cognitive_scores.append(context.get('cognitive_factor', 0.5))
            
        # 计算情感和认知偏差
        if emotional_scores and cognitive_scores:
            avg_emotional = self._safe_mean(emotional_scores, 0.5)
            avg_cognitive = self._safe_mean(cognitive_scores, 0.5)
            
            # 偏差计算 (与平衡点0.5的偏离)
            patterns['emotional_bias'] = avg_emotional - 0.5
            patterns['cognitive_bias'] = avg_cognitive - 0.5
            
        # 风险模式
        risk_scores = []
        for d in decisions:
            options = d.get('options', [])
            selected = d.get('selected')
            
            if options and selected is not None:
                # 找到选中的选项
                selected_option = options[selected] if selected < len(options) else None
                if selected_option:
                    risk_scores.append(selected_option.get('risk', 0.5))
                    
        if risk_scores:
            avg_risk = self._safe_mean(risk_scores, 0.5)
            # 风险偏好计算 (与中性点0.5的偏离)
            patterns['risk_aversion'] = 0.5 - avg_risk if avg_risk < 0.5 else 0
            patterns['risk_seeking'] = avg_risk - 0.5 if avg_risk > 0.5 else 0
            
        return patterns
        
    def _calculate_trend(self, values: List[float]) -> Dict:
        """计算数值序列的趋势"""
        if len(values) < 3:
            return {'direction': 'stable', 'magnitude': 0}
            
        # 简化线性趋势计算
        x = list(range(len(values)))
        y = [float(v) for v in values]

        # 线性回归
        slope = self._linear_slope(x, y)

        # 趋势方向和幅度
        direction = 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
        magnitude = abs(slope)
        
        return {
            'direction': direction,
            'magnitude': magnitude
        }
        
    def _identify_bottlenecks(self) -> List[Dict]:
        """识别决策过程中的瓶颈"""
        bottlenecks = []
        
        # 检查决策时间异常
        time_values = self.process_monitors['time_spent']
        if time_values and len(time_values) > 5:
            avg_time = self._safe_mean(time_values, 0.0)
            max_time = max(time_values)
            
            if max_time > avg_time * 2:
                bottlenecks.append({
                    'type': 'time_spike',
                    'severity': (max_time / avg_time) - 1,
                    'description': '决策时间出现显著峰值'
                })
                
        # 检查置信度异常
        conf_values = self.process_monitors['confidence_levels']
        if conf_values and len(conf_values) > 5:
            avg_conf = self._safe_mean(conf_values, 0.0)
            
            if avg_conf < 0.4:
                bottlenecks.append({
                    'type': 'low_confidence',
                    'severity': 0.4 - avg_conf,
                    'description': '决策置信度普遍较低'
                })
                
        return bottlenecks
        
    def consume_pending_adjustments(self, target: str = 'engine') -> Dict[str, Any]:
        """获取并清空指定目标的待应用调整"""
        if target == 'engine':
            adjustments = copy.deepcopy(self._pending_engine_adjustments)
            self._pending_engine_adjustments = {}
            return adjustments
        if target == 'process':
            adjustments = copy.deepcopy(self._pending_process_adjustments)
            self._pending_process_adjustments = {}
            return adjustments

        adjustments = {
            'engine': copy.deepcopy(self._pending_engine_adjustments),
            'process': copy.deepcopy(self._pending_process_adjustments)
        }
        self._pending_engine_adjustments = {}
        self._pending_process_adjustments = {}
        return adjustments

    def _calculate_consistency(self, decisions: List[Dict]) -> float:
        """计算决策一致性"""
        context_groups: Dict[str, List[Any]] = {}
        for decision in decisions:
            context = decision.get('context', {})
            context_key = self._context_signature(context)
            selected = decision.get('selected_option') or decision.get('outcome', {}).get('selected_option')
            if selected is None:
                continue
            context_groups.setdefault(context_key, []).append(self._serialize_option(selected))

        group_scores = []
        for selections in context_groups.values():
            if len(selections) < 2:
                continue
            counts = Counter(selections)
            most_common = counts.most_common(1)[0][1]
            group_scores.append(most_common / len(selections))

        if not group_scores:
            return 0.5

        return float(self._clip(self._safe_mean(group_scores, 0.5), 0.0, 1.0))

    def _calculate_adaptability(self, decisions: List[Dict]) -> float:
        """计算环境适应性"""
        utilities = []
        timestamps = []
        for decision in decisions:
            quality = decision.get('quality', {})
            utility = quality.get('utility')
            if utility is None:
                outcome = decision.get('outcome', {})
                utility = outcome.get('utility')
            if utility is None:
                continue
            utilities.append(float(utility))
            timestamps.append(decision.get('timestamp', time.time()))

        if len(utilities) < 2:
            return 0.5

        x = [float(t) - float(timestamps[0]) for t in timestamps]
        if all(abs(v) < 1e-9 for v in x):
            x = [float(i) for i in range(len(utilities))]
        y = [float(v) for v in utilities]
        slope = self._linear_slope(x, y)
        slope = max(min(slope, 20.0), -20.0)
        adaptability = 1.0 / (1.0 + math.exp(-slope))
        return float(self._clip(adaptability, 0.0, 1.0))

    def _calculate_efficiency(self, decisions: List[Dict]) -> float:
        """计算决策效率"""
        scores = []
        for decision in decisions:
            quality = decision.get('quality', {})
            overall = quality.get('overall')
            time_spent = decision.get('time_spent')
            if overall is None or time_spent is None:
                continue
            time_factor = 1.0 / (1.0 + max(0.0, float(time_spent)))
            scores.append(float(overall) * time_factor)

        if not scores:
            return 0.5

        return float(self._clip(self._safe_mean(scores, 0.5), 0.0, 1.0))

    def _calculate_rationality(self, decisions: List[Dict]) -> float:
        """计算决策理性程度"""
        ratios = []
        for decision in decisions:
            options = decision.get('options', [])
            if not options:
                continue
            selected = decision.get('selected_option') or decision.get('outcome', {}).get('selected_option')
            if selected is None:
                continue
            selected_score = self._evaluate_option_score(selected)
            option_scores = [self._evaluate_option_score(opt) for opt in options]
            if not option_scores:
                continue
            best_score = max(option_scores)
            if best_score > 0:
                ratio = selected_score / best_score
            else:
                ratio = 1.0 if selected_score >= best_score else 0.0
            ratios.append(self._clip(ratio, 0.0, 1.0))

        if not ratios:
            return 0.5

        return float(self._clip(self._safe_mean(ratios, 0.5), 0.0, 1.0))

    def _evaluate_option_score(self, option: Any) -> float:
        if not isinstance(option, dict):
            return float(option) if isinstance(option, (int, float)) else 0.0
        value = float(option.get('expected_value', 0.0))
        risk = float(option.get('risk', 0.0))
        cost = float(option.get('cost', 0.0))
        return (value * (1.0 - risk)) - cost

    @staticmethod
    def _context_signature(context: Dict[str, Any]) -> str:
        if not isinstance(context, dict):
            return str(context)
        return str(sorted(context.items()))

    @staticmethod
    def _serialize_option(option: Any) -> Any:
        if isinstance(option, dict):
            return tuple(sorted(option.items()))
        return option

    @staticmethod
    def _safe_mean(values: List[float], default: float = 0.0) -> float:
        data = [float(v) for v in values if v is not None]
        if not data:
            return float(default)
        return float(mean(data))

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _linear_slope(x_vals: List[float], y_vals: List[float]) -> float:
        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            return 0.0
        x_mean = mean(x_vals)
        y_mean = mean(y_vals)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        if denominator == 0:
            return 0.0
        return numerator / denominator