"""
计算与工程支持基础设施

提供完整的大脑仿真系统基础设施支持：
- 自动化仿真流水线
- 大规模测试框架
- 可扩展记录与可视化系统
"""

from .experiment_storage import ExperimentCatalog, ExperimentDirs, ExperimentStorage
from .simulation_pipeline import (
    ExperimentConfig,
    OptimizationMethod,
    ParameterSpace,
    SimulationPipeline,
    create_experiment_config,
)

__all__ = [
    # 仿真流水线
    "SimulationPipeline",
    "ExperimentConfig",
    "ParameterSpace",
    "OptimizationMethod",
    "create_experiment_config",
    # 实验数据存储
    "ExperimentStorage",
    "ExperimentCatalog",
    "ExperimentDirs",
]

try:
    from .testing_framework import (
        BaseTestCase,
        TestDiscovery,
        TestExecutor,
        TestLevel,
        TestPriority,
        TestReporter,
        TestStatus,
        TestSuite,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            # 测试框架
            "TestLevel",
            "TestStatus",
            "TestPriority",
            "BaseTestCase",
            "TestSuite",
            "TestExecutor",
            "TestReporter",
            "TestDiscovery",
        ]
    )

try:
    from .recording_visualization import (
        DataRecorder,
        DataVisualizer,
        RealTimeVisualizer,
        RecordingConfig,
        RecordingManager,
        RecordingType,
        SpikeRecorder,
        StorageFormat,
        VisualizationType,
        VoltageRecorder,
        WeightRecorder,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            # 记录与可视化
            "RecordingType",
            "VisualizationType",
            "StorageFormat",
            "RecordingConfig",
            "DataRecorder",
            "SpikeRecorder",
            "VoltageRecorder",
            "WeightRecorder",
            "DataVisualizer",
            "RealTimeVisualizer",
            "RecordingManager",
        ]
    )
