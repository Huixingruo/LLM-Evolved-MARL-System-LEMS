"""
launcher.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from llm_reward_agent.tools.launcher import ParallelLauncher


def test_launcher_initialization():
    """测试 ParallelLauncher 初始化"""
    launcher = ParallelLauncher(
        max_workers=2,
        timeout=60,
        episode_num=5,
        use_gpu=False
    )

    assert launcher is not None
    assert launcher.max_workers == 2
    assert launcher.timeout == 60
    assert launcher.episode_num == 5

    print("[OK] test_launcher_initialization")


def test_gpu_check():
    """测试 GPU 检查"""
    has_gpu = torch.cuda.is_available()
    print(f"[INFO] CUDA 可用: {has_gpu}")

    if has_gpu:
        print(f"[INFO] GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")

    print("[OK] test_gpu_check (信息性测试)")


def test_launcher_with_gpu_ids():
    """测试指定 GPU IDs"""
    launcher = ParallelLauncher(
        max_workers=2,
        timeout=60,
        episode_num=5,
        use_gpu=False,
        gpu_ids=[0]
    )

    assert launcher.gpu_ids == [0]

    print("[OK] test_launcher_with_gpu_ids")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 ParallelLauncher 模块")
    print("=" * 80)
    print()

    tests = [
        test_launcher_initialization,
        test_gpu_check,
        test_launcher_with_gpu_ids,
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
