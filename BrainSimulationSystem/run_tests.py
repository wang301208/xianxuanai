"""
运行分层神经网络测试
Run Hierarchical Neural Network Tests
"""

import unittest
import sys
import os
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_all_tests():
    """运行所有测试"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("分层神经网络架构测试")
    print("="*60)
    
    # 发现并运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests')
    
    if not os.path.exists(start_dir):
        print(f"测试目录不存在: {start_dir}")
        return False
    
    # 加载所有测试
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
    
    print("="*60)
    
    return success

def run_specific_test(test_module):
    """运行特定测试模块"""
    
    print(f"运行测试模块: {test_module}")
    print("="*60)
    
    # 导入测试模块
    try:
        module = __import__(f'tests.{test_module}', fromlist=[test_module])
    except ImportError as e:
        print(f"无法导入测试模块 {test_module}: {e}")
        return False
    
    # 运行测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def validate_installation():
    """验证安装和依赖"""
    
    print("验证安装和依赖...")
    print("-" * 40)
    
    # 检查核心模块
    core_modules = [
        'BrainSimulationSystem.core.hierarchical_structure',
        'BrainSimulationSystem.core.multi_neuron_models',
        'BrainSimulationSystem.core.enhanced_connectivity',
        'BrainSimulationSystem.config.hierarchical_network_config'
    ]
    
    missing_modules = []
    
    for module_name in core_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            missing_modules.append(module_name)
    
    # 检查可选依赖
    optional_deps = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'h5py': 'HDF5 支持',
        'neo4j': 'Neo4j 图数据库',
        'networkx': 'NetworkX 图分析'
    }
    
    print("\n可选依赖:")
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {desc}")
        except ImportError:
            print(f"⚠️  {desc} (可选)")
    
    if missing_modules:
        print(f"\n❌ 缺少必需模块: {', '.join(missing_modules)}")
        return False
    else:
        print("\n✅ 所有核心模块可用")
        return True

def run_configuration_test():
    """运行配置测试"""
    
    print("测试配置文件...")
    print("-" * 40)
    
    try:
        from BrainSimulationSystem.config.hierarchical_network_config import (
            get_config, validate_config
        )
        
        # 加载配置
        config = get_config()
        print("✅ 配置文件加载成功")
        
        # 验证配置
        validate_config(config)
        print("✅ 配置验证通过")
        
        # 打印配置摘要
        print(f"\n配置摘要:")
        print(f"  总神经元数: {config['structure']['total_neurons']:,}")
        print(f"  脑区数量: {len(config['structure']['brain_regions'])}")
        print(f"  神经元类型数: {len(config['cellular']['neuron_parameters'])}")
        print(f"  连接类型数: {len(config['connectivity']['connection_parameters'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def run_quick_functionality_test():
    """运行快速功能测试"""
    
    print("快速功能测试...")
    print("-" * 40)
    
    try:
        # 测试结构层
        from BrainSimulationSystem.core.hierarchical_structure import (
            create_hierarchical_structure, NeuronDensity
        )
        
        test_config = {
            'total_neurons': 1000,
            'brain_regions': [
                {
                    'name': 'test_region',
                    'neurons': 1000,
                    'volume': 100.0
                }
            ],
            'columns_per_subregion': 2,
            'microcircuits_per_column': 2
        }
        
        hierarchy = create_hierarchical_structure(test_config)
        print("✅ 结构层创建成功")
        
        # 测试细胞层
        from BrainSimulationSystem.core.multi_neuron_models import (
            create_neuron, NeuronType, get_default_parameters
        )
        
        params = get_default_parameters(NeuronType.LIF)
        neuron = create_neuron(NeuronType.LIF, neuron_id=1, params=params)
        print("✅ 神经元创建成功")
        
        # 测试连接层
        from BrainSimulationSystem.core.enhanced_connectivity import (
            create_enhanced_connectivity_manager
        )
        
        conn_config = {
            'connector': {'seed': 42},
            'graph_database': {'enabled': False}
        }
        
        conn_manager = create_enhanced_connectivity_manager(conn_config)
        conn_manager.initialize(100, ['test_region'])
        print("✅ 连接管理器创建成功")
        
        print("\n✅ 所有核心功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'validate':
            validate_installation()
        elif command == 'config':
            run_configuration_test()
        elif command == 'quick':
            run_quick_functionality_test()
        elif command == 'all':
            if validate_installation() and run_configuration_test():
                run_all_tests()
        elif command.startswith('test_'):
            run_specific_test(command)
        else:
            print(f"未知命令: {command}")
            print("可用命令: validate, config, quick, all, test_<module_name>")
    else:
        # 默认运行完整测试流程
        print("分层神经网络架构测试套件")
        print("="*60)
        
        # 1. 验证安装
        if not validate_installation():
            print("❌ 安装验证失败，请检查依赖")
            return
        
        # 2. 测试配置
        if not run_configuration_test():
            print("❌ 配置测试失败")
            return
        
        # 3. 快速功能测试
        if not run_quick_functionality_test():
            print("❌ 功能测试失败")
            return
        
        # 4. 运行完整测试套件
        print("\n" + "="*60)
        success = run_all_tests()
        
        if success:
            print("\n🎉 所有测试通过！分层神经网络架构已就绪。")
        else:
            print("\n⚠️  部分测试失败，请检查相关模块。")

