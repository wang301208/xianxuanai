"""
后端配置管理
Backend Configuration Management
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..backends.framework_integration import FrameworkType, SimulationConfig
from ..core.backends import NeuromorphicPlatform, HardwareConfig

class BackendProfile(Enum):
    """后端配置档案"""
    DEVELOPMENT = "development"      # 开发测试
    RESEARCH = "research"           # 科研仿真
    PRODUCTION = "production"       # 生产部署
    REALTIME = "realtime"          # 实时应用
    LOWPOWER = "lowpower"          # 低功耗应用

@dataclass
class BackendPreferences:
    """后端偏好设置"""
    prefer_gpu: bool = True
    prefer_distributed: bool = False
    prefer_hardware: bool = False
    max_memory_gb: float = 16.0
    max_power_watts: float = 100.0
    target_latency_ms: float = 10.0
    precision_requirement: str = "float32"

def get_backend_config(profile: BackendProfile = BackendProfile.DEVELOPMENT) -> Dict[str, Any]:
    """获取后端配置"""
    
    base_config = {
        'profile': profile.value,
        'frameworks': {},
        'hardware': {},
        'preferences': {},
        'optimization': {},
        'monitoring': {}
    }
    
    if profile == BackendProfile.DEVELOPMENT:
        return _get_development_config(base_config)
    elif profile == BackendProfile.RESEARCH:
        return _get_research_config(base_config)
    elif profile == BackendProfile.PRODUCTION:
        return _get_production_config(base_config)
    elif profile == BackendProfile.REALTIME:
        return _get_realtime_config(base_config)
    elif profile == BackendProfile.LOWPOWER:
        return _get_lowpower_config(base_config)
    else:
        return base_config

def _get_development_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """开发配置"""
    
    config = base_config.copy()
    
    # 框架配置
    config['frameworks'] = {
        FrameworkType.NEST.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'dt': 0.1,
                'threads': 2,
                'gpu_enabled': False,
                'precision': 'float64',
                'recording': {
                    'spikes': True,
                    'voltages': True,
                    'currents': False,
                    'weights': False
                }
            }
        },
        FrameworkType.CARLSIM.value: {
            'enabled': False,  # 开发环境通常不需要GPU
            'priority': 2,
            'config': {
                'dt': 0.1,
                'gpu_enabled': False,
                'precision': 'float32'
            }
        }
    }
    
    # 硬件配置
    config['hardware'] = {
        NeuromorphicPlatform.SPINNAKER.value: {
            'enabled': False,  # 开发环境通常没有专用硬件
            'priority': 3
        }
    }
    
    # 偏好设置
    config['preferences'] = asdict(BackendPreferences(
        prefer_gpu=False,
        prefer_distributed=False,
        prefer_hardware=False,
        max_memory_gb=8.0,
        max_power_watts=50.0,
        precision_requirement="float64"
    ))
    
    # 优化设置
    config['optimization'] = {
        'auto_backend_selection': True,
        'memory_optimization': False,
        'speed_optimization': False,
        'debug_mode': True
    }
    
    # 监控设置
    config['monitoring'] = {
        'performance_tracking': True,
        'memory_tracking': True,
        'error_logging': True,
        'profiling': True
    }
    
    return config

def _get_research_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """科研配置"""
    
    config = base_config.copy()
    
    # 框架配置 - 支持多种框架
    config['frameworks'] = {
        FrameworkType.NEST.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'dt': 0.1,
                'threads': 8,
                'gpu_enabled': False,
                'precision': 'float64',
                'distributed': True,
                'recording': {
                    'spikes': True,
                    'voltages': True,
                    'currents': True,
                    'weights': True
                }
            }
        },
        FrameworkType.CARLSIM.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'dt': 0.1,
                'gpu_enabled': True,
                'precision': 'float32',
                'recording': {
                    'spikes': True,
                    'voltages': False
                }
            }
        },
        FrameworkType.GENN.value: {
            'enabled': True,
            'priority': 3,
            'config': {
                'dt': 0.1,
                'gpu_enabled': True,
                'precision': 'float32'
            }
        },
        FrameworkType.TVB.value: {
            'enabled': True,
            'priority': 4,
            'config': {
                'dt': 0.1,
                'precision': 'float64',
                'whole_brain': True
            }
        }
    }
    
    # 硬件配置
    config['hardware'] = {
        NeuromorphicPlatform.SPINNAKER.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'board_count': 1,
                'real_time': False
            }
        },
        NeuromorphicPlatform.LOIHI.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'chip_count': 1,
                'learning_enabled': True
            }
        }
    }
    
    # 偏好设置
    config['preferences'] = asdict(BackendPreferences(
        prefer_gpu=True,
        prefer_distributed=True,
        prefer_hardware=True,
        max_memory_gb=64.0,
        max_power_watts=500.0,
        precision_requirement="float64"
    ))
    
    # 优化设置
    config['optimization'] = {
        'auto_backend_selection': True,
        'memory_optimization': True,
        'speed_optimization': True,
        'debug_mode': False,
        'benchmark_mode': True
    }
    
    return config

def _get_production_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """生产配置"""
    
    config = base_config.copy()
    
    # 框架配置 - 稳定性优先
    config['frameworks'] = {
        FrameworkType.NEST.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'dt': 0.1,
                'threads': 16,
                'gpu_enabled': False,
                'precision': 'float32',
                'distributed': True,
                'recording': {
                    'spikes': True,
                    'voltages': False,
                    'currents': False,
                    'weights': False
                }
            }
        },
        FrameworkType.CARLSIM.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'dt': 0.1,
                'gpu_enabled': True,
                'precision': 'float32'
            }
        }
    }
    
    # 硬件配置
    config['hardware'] = {
        NeuromorphicPlatform.SPINNAKER.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'board_count': 4,
                'real_time': False,
                'redundancy': True
            }
        }
    }
    
    # 偏好设置
    config['preferences'] = asdict(BackendPreferences(
        prefer_gpu=True,
        prefer_distributed=True,
        prefer_hardware=True,
        max_memory_gb=128.0,
        max_power_watts=1000.0,
        precision_requirement="float32"
    ))
    
    # 优化设置
    config['optimization'] = {
        'auto_backend_selection': True,
        'memory_optimization': True,
        'speed_optimization': True,
        'debug_mode': False,
        'fault_tolerance': True,
        'load_balancing': True
    }
    
    # 监控设置
    config['monitoring'] = {
        'performance_tracking': True,
        'memory_tracking': True,
        'error_logging': True,
        'health_monitoring': True,
        'alerting': True
    }
    
    return config

def _get_realtime_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """实时配置"""
    
    config = base_config.copy()
    
    # 框架配置 - 实时性优先
    config['frameworks'] = {
        FrameworkType.CARLSIM.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'dt': 0.1,
                'gpu_enabled': True,
                'precision': 'float32',
                'real_time': True,
                'recording': {
                    'spikes': True,
                    'voltages': False
                }
            }
        },
        FrameworkType.GENN.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'dt': 0.1,
                'gpu_enabled': True,
                'precision': 'float32',
                'real_time': True
            }
        }
    }
    
    # 硬件配置 - 神经形态芯片优先
    config['hardware'] = {
        NeuromorphicPlatform.LOIHI.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'chip_count': 2,
                'real_time': True,
                'low_latency': True
            }
        },
        NeuromorphicPlatform.SPINNAKER.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'board_count': 1,
                'real_time': True
            }
        }
    }
    
    # 偏好设置
    config['preferences'] = asdict(BackendPreferences(
        prefer_gpu=True,
        prefer_distributed=False,
        prefer_hardware=True,
        max_memory_gb=32.0,
        max_power_watts=100.0,
        target_latency_ms=1.0,
        precision_requirement="float32"
    ))
    
    # 优化设置
    config['optimization'] = {
        'auto_backend_selection': True,
        'memory_optimization': False,
        'speed_optimization': True,
        'latency_optimization': True,
        'debug_mode': False
    }
    
    return config

def _get_lowpower_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """低功耗配置"""
    
    config = base_config.copy()
    
    # 框架配置 - 功耗优先
    config['frameworks'] = {
        FrameworkType.NEST.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'dt': 1.0,  # 较大时间步
                'threads': 1,
                'gpu_enabled': False,
                'precision': 'float32',
                'recording': {
                    'spikes': True,
                    'voltages': False
                }
            }
        }
    }
    
    # 硬件配置 - 神经形态芯片优先
    config['hardware'] = {
        NeuromorphicPlatform.LOIHI.value: {
            'enabled': True,
            'priority': 1,
            'config': {
                'chip_count': 1,
                'power_gating': True,
                'clock_gating': True
            }
        },
        NeuromorphicPlatform.TRUENORTH.value: {
            'enabled': True,
            'priority': 2,
            'config': {
                'chip_count': 1,
                'ultra_low_power': True
            }
        }
    }
    
    # 偏好设置
    config['preferences'] = asdict(BackendPreferences(
        prefer_gpu=False,
        prefer_distributed=False,
        prefer_hardware=True,
        max_memory_gb=4.0,
        max_power_watts=5.0,
        precision_requirement="fixed8"
    ))
    
    # 优化设置
    config['optimization'] = {
        'auto_backend_selection': True,
        'memory_optimization': True,
        'speed_optimization': False,
        'power_optimization': True,
        'debug_mode': False
    }
    
    return config

def get_framework_specific_config(framework: FrameworkType, 
                                 profile: BackendProfile = BackendProfile.DEVELOPMENT) -> Dict[str, Any]:
    """获取特定框架配置"""
    
    backend_config = get_backend_config(profile)
    framework_configs = backend_config.get('frameworks', {})
    
    return framework_configs.get(framework.value, {})

def get_hardware_specific_config(platform: NeuromorphicPlatform,
                                profile: BackendProfile = BackendProfile.DEVELOPMENT) -> Dict[str, Any]:
    """获取特定硬件配置"""
    
    backend_config = get_backend_config(profile)
    hardware_configs = backend_config.get('hardware', {})
    
    return hardware_configs.get(platform.value, {})

def validate_backend_config(config: Dict[str, Any]) -> List[str]:
    """验证后端配置"""
    
    errors = []
    
    # 检查必需字段
    required_fields = ['profile', 'frameworks', 'hardware', 'preferences']
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    # 检查框架配置
    frameworks = config.get('frameworks', {})
    if not frameworks:
        errors.append("至少需要配置一个仿真框架")
    
    for framework_name, framework_config in frameworks.items():
        if 'enabled' not in framework_config:
            errors.append(f"框架 {framework_name} 缺少 enabled 字段")
        
        if framework_config.get('enabled', False):
            if 'config' not in framework_config:
                errors.append(f"启用的框架 {framework_name} 缺少 config 字段")
    
    # 检查偏好设置
    preferences = config.get('preferences', {})
    if 'max_memory_gb' in preferences:
        if preferences['max_memory_gb'] <= 0:
            errors.append("最大内存必须大于0")
    
    if 'max_power_watts' in preferences:
        if preferences['max_power_watts'] <= 0:
            errors.append("最大功耗必须大于0")
    
    return errors

def save_backend_config(config: Dict[str, Any], filepath: str):
    """保存后端配置"""
    
    import json
    
    # 验证配置
    errors = validate_backend_config(config)
    if errors:
        raise ValueError(f"配置验证失败: {errors}")
    
    # 保存配置
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_backend_config(filepath: str) -> Dict[str, Any]:
    """加载后端配置"""
    
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 验证配置
    errors = validate_backend_config(config)
    if errors:
        raise ValueError(f"配置验证失败: {errors}")
    
    return config

def get_environment_config() -> Dict[str, Any]:
    """从环境变量获取配置"""
    
    config = {}
    
    # 从环境变量读取配置
    profile = os.getenv('BRAIN_SIM_PROFILE', 'development')
    config['profile'] = profile
    
    # GPU设置
    gpu_enabled = os.getenv('BRAIN_SIM_GPU', 'false').lower() == 'true'
    config['gpu_enabled'] = gpu_enabled
    
    # 分布式设置
    distributed = os.getenv('BRAIN_SIM_DISTRIBUTED', 'false').lower() == 'true'
    config['distributed'] = distributed
    
    # 内存限制
    max_memory = float(os.getenv('BRAIN_SIM_MAX_MEMORY_GB', '16.0'))
    config['max_memory_gb'] = max_memory
    
    # 功耗限制
    max_power = float(os.getenv('BRAIN_SIM_MAX_POWER_W', '100.0'))
    config['max_power_watts'] = max_power
    
    return config

# 预定义配置模板
CONFIG_TEMPLATES = {
    'small_network': {
        'n_neurons': 1000,
        'n_synapses': 10000,
        'recommended_frameworks': [FrameworkType.NEST],
        'recommended_hardware': []
    },
    'medium_network': {
        'n_neurons': 100000,
        'n_synapses': 10000000,
        'recommended_frameworks': [FrameworkType.NEST, FrameworkType.CARLSIM],
        'recommended_hardware': [NeuromorphicPlatform.SPINNAKER]
    },
    'large_network': {
        'n_neurons': 1000000,
        'n_synapses': 1000000000,
        'recommended_frameworks': [FrameworkType.NEST],
        'recommended_hardware': [NeuromorphicPlatform.SPINNAKER]
    },
    'realtime_network': {
        'n_neurons': 10000,
        'n_synapses': 1000000,
        'recommended_frameworks': [FrameworkType.CARLSIM, FrameworkType.GENN],
        'recommended_hardware': [NeuromorphicPlatform.LOIHI, NeuromorphicPlatform.SPINNAKER]
    }
}