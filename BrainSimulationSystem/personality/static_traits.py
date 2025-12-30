"""
静态个性特质系统

实现不可修改的科学标准化个性参数
"""

class StaticPersonality:
    def __init__(self):
        # 固定五大人格特质 (科学平均值)
        self.traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.3
        }
    
    def get_traits(self):
        """获取不可修改的个性参数"""
        return self.traits.copy()