"""
simulation_tool.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import shutil
from llm_reward_agent.tools.simulation_tool import SimulationTool


def test_simulation_tool_initialization():
    """测试 SimulationTool 初始化"""
    base_dir = "test_logs/sim_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    sim_tool = SimulationTool(
        base_dir=base_dir,
        max_workers=2,
        timeout=60,
        episode_num=5,
        use_gpu=False
    )

    assert sim_tool is not None
    assert sim_tool.sandbox_mgr is not None
    assert sim_tool.launcher is not None

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_simulation_tool_initialization")


def test_print_summary():
    """测试摘要打印功能"""
    sim_tool = SimulationTool(
        base_dir="test_logs/summary_test",
        max_workers=2,
        timeout=60,
        episode_num=5,
        use_gpu=False
    )

    test_results = [
        {
            'id': 'candidate_0',
            'status': 'success',
            'fitness': 0.75,
            'metrics': {
                'success_rate': 0.80,
                'avg_capture_time': 45.0
            }
        },
        {
            'id': 'candidate_1',
            'status': 'error',
            'fitness': 0.0,
            'metrics': {}
        }
    ]

    # 测试摘要打印
    sim_tool._print_summary(test_results)
    print("[OK] test_print_summary")

    import shutil
    if os.path.exists("test_logs/summary_test"):
        shutil.rmtree("test_logs/summary_test")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 SimulationTool 模块")
    print("=" * 80)
    print()

    tests = [
        test_simulation_tool_initialization,
        test_print_summary,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
