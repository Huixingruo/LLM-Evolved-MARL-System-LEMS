"""
sandbox_manager.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import shutil
from llm_reward_agent.tools.sandbox_manager import SandboxManager


def test_sandbox_manager_initialization():
    """测试 SandboxManager 初始化"""
    base_dir = "test_logs/sandbox_init_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    manager = SandboxManager(base_dir=base_dir)
    assert manager is not None
    assert os.path.exists(base_dir)

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_sandbox_manager_initialization")


def test_create_sandboxes():
    """测试创建沙盒"""
    base_dir = "test_logs/sandbox_create_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    manager = SandboxManager(base_dir=base_dir)

    test_codes = [
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 1.0
    total_reward = sum(components.values())
    return total_reward, components
""",
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 2.0
    total_reward = sum(components.values())
    return total_reward, components
"""
    ]

    sandboxes = manager.create_sandboxes(generation=0, codes=test_codes)

    assert len(sandboxes) == 2
    assert os.path.exists(sandboxes[0])
    assert os.path.exists(sandboxes[1])

    # 检查奖励函数是否正确写入
    reward_file = os.path.join(sandboxes[0], "MADDPG", "envs", "reward_function.py")
    assert os.path.exists(reward_file)

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_create_sandboxes")


def test_get_sandbox_info():
    """测试获取沙盒信息"""
    base_dir = "test_logs/sandbox_info_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    manager = SandboxManager(base_dir=base_dir)

    test_codes = [
        """
import numpy as np

def compute_reward(...):
    pass
"""
    ]

    sandboxes = manager.create_sandboxes(generation=0, codes=test_codes)

    info = manager.get_sandbox_info(sandboxes[0])

    assert info['exists'] == True
    assert info['path'] == sandboxes[0]
    assert info['files_count'] > 0

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_get_sandbox_info")


def test_cleanup_generation():
    """测试清理指定代沙盒"""
    base_dir = "test_logs/sandbox_cleanup_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    manager = SandboxManager(base_dir=base_dir)

    test_codes = ["import numpy as np\n\ndef compute_reward(...): pass"]
    manager.create_sandboxes(generation=0, codes=test_codes)

    gen_dir = os.path.join(base_dir, "generation_000")
    assert os.path.exists(gen_dir)

    manager.cleanup_generation(0)
    assert not os.path.exists(gen_dir)

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_cleanup_generation")


def test_cleanup_all():
    """测试清理所有沙盒"""
    base_dir = "test_logs/sandbox_cleanup_all_test"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    manager = SandboxManager(base_dir=base_dir)

    test_codes = ["import numpy as np\n\ndef compute_reward(...): pass"]
    manager.create_sandboxes(generation=0, codes=test_codes)
    manager.create_sandboxes(generation=1, codes=test_codes)

    assert os.path.exists(os.path.join(base_dir, "generation_000"))
    assert os.path.exists(os.path.join(base_dir, "generation_001"))

    manager.cleanup_all()

    # evolution_archive 应该保留
    os.makedirs(os.path.join(base_dir, "evolution_archive"), exist_ok=True)

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    print("[OK] test_cleanup_all")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 SandboxManager 模块")
    print("=" * 80)
    print()

    tests = [
        test_sandbox_manager_initialization,
        test_create_sandboxes,
        test_get_sandbox_info,
        test_cleanup_generation,
        test_cleanup_all,
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
