"""
大脑模拟系统枚举类型
Enums for the Brain Simulation System

定义了系统中使用的各种生理和认知相关的枚举类型。
- BrainRegion: 脑区类型
- CognitiveFunction: 认知功能类型
- OscillationBand: 脑电振荡频段
"""
from enum import Enum

class BrainRegion(Enum):
    """脑区类型"""
    # 皮层区域
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    MOTOR_CORTEX = "motor_cortex"
    SOMATOSENSORY_CORTEX = "somatosensory_cortex"
    VISUAL_CORTEX = "visual_cortex"
    AUDITORY_CORTEX = "auditory_cortex"
    PARIETAL_CORTEX = "parietal_cortex"
    TEMPORAL_CORTEX = "temporal_cortex"
    OCCIPITAL_CORTEX = "occipital_cortex"
    CINGULATE_CORTEX = "cingulate_cortex"
    INSULAR_CORTEX = "insular_cortex"
    
    # 皮层下结构
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"
    THALAMUS = "thalamus"
    HYPOTHALAMUS = "hypothalamus"
    BASAL_GANGLIA = "basal_ganglia"
    CEREBELLUM = "cerebellum"
    BRAINSTEM = "brainstem"
    
    # 特殊区域
    CORPUS_CALLOSUM = "corpus_callosum"
    FORNIX = "fornix"
    ANTERIOR_COMMISSURE = "anterior_commissure"

class CognitiveFunction(Enum):
    """认知功能类型"""
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    EXECUTIVE_CONTROL = "executive_control"
    LANGUAGE = "language"
    PERCEPTION = "perception"
    MOTOR_CONTROL = "motor_control"
    EMOTION = "emotion"
    CONSCIOUSNESS = "consciousness"
    DECISION_MAKING = "decision_making"

class OscillationBand(Enum):
    """脑电振荡频段"""
    DELTA = (0.5, 4.0)      # 深度睡眠
    THETA = (4.0, 8.0)      # 记忆编码
    ALPHA = (8.0, 13.0)     # 放松状态
    BETA = (13.0, 30.0)     # 活跃思维
    GAMMA = (30.0, 100.0)   # 意识绑定
    HIGH_GAMMA = (100.0, 200.0)  # 局部处理