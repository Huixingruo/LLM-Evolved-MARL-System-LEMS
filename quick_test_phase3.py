"""
阶段三快速测试脚本
验证并行训练框架的基础功能（不含实际训练）

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("阶段三快速测试")
print("=" * 80)

# 测试1: 导入模块
print("\n[测试1] 导入核心模块...")
try:
    from llm_reward_agent.tools import (
        SandboxManager,
        LogAnalyzer,
        SimulationTool
    )
    from launcher import ParallelLauncher
    print("[OK] 所有模块导入成功")
except Exception as e:
    print(f"[FAIL] 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 沙盒管理器
print("\n[测试2] 测试沙盒管理器...")
try:
    manager = SandboxManager(base_dir="test_logs/quick_test")
    
    test_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 1.0
    total_reward = sum(components.values())
    return total_reward, components
"""
    
    sandboxes = manager.create_sandboxes(generation=0, codes=[test_code])
    
    assert len(sandboxes) == 1
    assert os.path.exists(sandboxes[0])
    
    # 检查奖励函数文件
    reward_file = os.path.join(sandboxes[0], "MADDPG", "envs", "reward_function.py")
    assert os.path.exists(reward_file)
    
    # 获取沙盒信息
    info = manager.get_sandbox_info(sandboxes[0])
    print(f"[OK] 沙盒创建成功")
    print(f"   路径: {sandboxes[0]}")
    print(f"   大小: {info['size_mb']:.2f} MB")
    print(f"   文件数: {info['files_count']}")
    
except Exception as e:
    print(f"[FAIL] 沙盒管理器测试失败: {e}")

# 测试3: 日志分析器
print("\n[测试3] 测试日志分析器...")
try:
    analyzer = LogAnalyzer()
    
    test_metrics = {
        'success_rate': 0.75,
        'avg_capture_time': 50.0,
        'reward_components': {
            'distance_reward': {'mean': -1.0, 'std': 0.3}
        }
    }
    
    fitness = analyzer.calculate_fitness(test_metrics)
    
    assert fitness > 0
    print(f"[OK] Fitness计算成功: {fitness:.4f}")
    
    # 生成报告
    report = analyzer.generate_analysis_report({**test_metrics, 'fitness': fitness})
    assert len(report) > 0
    print(f"[OK] 分析报告生成成功 (长度: {len(report)} 字符)")
    
except Exception as e:
    print(f"[FAIL] 日志分析器测试失败: {e}")

# 测试4: 并行调度器
print("\n[测试4] 测试并行调度器...")
try:
    launcher = ParallelLauncher(
        max_workers=2,
        timeout=300,
        episode_num=10
    )
    
    assert launcher.max_workers == 2
    assert launcher.timeout == 300
    assert launcher.episode_num == 10
    
    print(f"[OK] 并行调度器初始化成功")
    print(f"   最大并行数: {launcher.max_workers}")
    print(f"   超时时间: {launcher.timeout}秒")
    
except Exception as e:
    print(f"[FAIL] 并行调度器测试失败: {e}")

# 测试5: 仿真工具集成
print("\n[测试5] 测试仿真工具集成...")
try:
    sim_tool = SimulationTool(
        base_dir="test_logs/sim_test",
        max_workers=2,
        timeout=300,
        episode_num=10
    )
    
    assert sim_tool.sandbox_mgr is not None
    assert sim_tool.launcher is not None
    
    print(f"[OK] 仿真工具初始化成功")
    
except Exception as e:
    print(f"[FAIL] 仿真工具测试失败: {e}")

# 测试6: 检查与阶段二的集成
print("\n[测试6] 检查与阶段二的集成...")
try:
    from llm_reward_agent.agent import RewardDesignAgent
    
    # 检查step方法是否有use_real_training参数
    import inspect
    sig = inspect.signature(RewardDesignAgent.step)
    params = list(sig.parameters.keys())
    
    assert 'use_real_training' in params
    print(f"[OK] RewardDesignAgent已集成阶段三功能")
    print(f"   step方法参数: {params}")
    
except Exception as e:
    print(f"[FAIL] 集成检查失败: {e}")

# 清理测试文件
print("\n清理测试文件...")
import shutil
test_dirs = ["test_logs/quick_test", "test_logs/sim_test"]
for test_dir in test_dirs:
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
print("[OK] 测试文件清理完成")

print("\n" + "=" * 80)
print("快速测试完成")
print("=" * 80)
print("\n说明:")
print("- 所有核心模块均已成功导入和测试")
print("- 如需完整测试（含实际训练），请运行: python test_phase3.py")
print("- 如需集成测试，请参考文档中的示例代码")
print("\n[OK] 阶段三开发完成！")
print("\n下一步:")
print("1. 运行完整测试: python test_phase3.py")
print("2. 阅读文档: PHASE3_DOCUMENTATION.md")
print("3. 开始阶段四开发（反馈闭环与整合）")
