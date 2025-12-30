"""
非传统策略生成系统

实现突破常规的问题解决策略生成
"""

from typing import Dict, List
import random

class UnconventionalStrategist:
    def __init__(self):
        # 策略生成参数
        self.strategy_parameters = {
            'risk_tolerance': 0.7,       # 风险承受
            'rule_breaking': 0.6,        # 规则突破倾向
            'resourcefulness': 0.5,      # 资源利用创意
            'lateral_thinking': 0.8      # 横向思维强度
        }
        
        # 策略模板库
        self.strategy_templates = [
            "Reverse {standard_approach}",
            "Apply {domain_A} method in {domain_B}",
            "Minimize {key_factor} to extreme",
            "Combine {opposite_strategies}"
        ]
    
    def generate_unconventional_strategies(self, context: Dict) -> List[Dict]:
        """生成非传统策略列表"""
        strategies = []
        
        # 基础策略变形
        if context.get('standard_solution'):
            strategies.append(
                self._apply_template(
                    self.strategy_templates[0],
                    {'standard_approach': context['standard_solution']}
                )
            )
        
        # 跨领域策略
        if len(context.get('related_domains', [])) >= 2:
            strategies.append(
                self._apply_template(
                    self.strategy_templates[1],
                    {
                        'domain_A': context['related_domains'][0],
                        'domain_B': context['related_domains'][1]
                    }
                )
            )
        
        # 极端化策略
        if context.get('key_factors'):
            strategies.append(
                self._apply_template(
                    self.strategy_templates[2],
                    {'key_factor': random.choice(context['key_factors'])}
                )
            )
        
        # 评估策略特性
        evaluated = []
        for s in strategies:
            evaluated.append({
                'strategy': s,
                'risk_level': self._assess_risk(s),
                'innovativeness': self._assess_innovation(s)
            })
        
        return sorted(evaluated, key=lambda x: -x['innovativeness'])
    
    def _apply_template(self, template: str, bindings: Dict) -> str:
        """应用策略模板生成具体策略"""
        result = template
        for k, v in bindings.items():
            result = result.replace(f"{{{k}}}", str(v))
        return result
    
    def _assess_risk(self, strategy: str) -> float:
        """评估策略风险水平"""
        return min(1.0, 0.3 + self.strategy_parameters['risk_tolerance'] * 0.7)
    
    def _assess_innovation(self, strategy: str) -> float:
        """评估策略创新度"""
        components = len(strategy.split("_")) - 1
        return min(1.0, 0.1 * components + self.strategy_parameters['lateral_thinking'] * 0.6)