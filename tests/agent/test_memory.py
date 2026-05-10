"""
memory.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import shutil
from llm_reward_agent.agent.memory import EvolutionaryMemory


def test_memory_initialization():
    """测试记忆初始化"""
    save_dir = "test_logs/memory_test"

    # 清理可能存在的目录
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)
    assert memory is not None
    assert len(memory.history) == 0
    assert memory.metadata['total_generations'] == 0

    print("[OK] test_memory_initialization")


def test_memory_save():
    """测试记忆保存"""
    save_dir = "test_logs/memory_save_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)

    test_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    return 0, components
"""

    test_results = [
        {'id': 0, 'fitness': 0.6, 'success_rate': 0.7},
        {'id': 1, 'fitness': 0.65, 'success_rate': 0.75},
    ]

    memory.save(
        generation=0,
        best_code=test_code,
        reflection="第0代反思",
        all_results=test_results
    )

    assert len(memory.history) == 1
    assert memory.metadata['total_generations'] == 1

    # 检查文件是否保存
    assert os.path.exists(os.path.join(save_dir, "generation_000.json"))
    assert os.path.exists(os.path.join(save_dir, "metadata.json"))

    print("[OK] test_memory_save")


def test_memory_get_best_code():
    """测试获取最优代码"""
    save_dir = "test_logs/memory_get_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)

    test_code = """
import numpy as np
def compute_reward(...):
    pass
"""

    test_results = [{'id': 0, 'fitness': 0.6}]

    memory.save(
        generation=0,
        best_code=test_code,
        reflection="反思",
        all_results=test_results
    )

    retrieved_code = memory.get_best_code(0)
    assert retrieved_code == test_code

    print("[OK] test_memory_get_best_code")


def test_memory_fitness_history():
    """测试获取fitness历史"""
    save_dir = "test_logs/memory_fitness_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)

    # 保存3代
    for gen in range(3):
        memory.save(
            generation=gen,
            best_code=f"code_{gen}",
            reflection=f"反思_{gen}",
            all_results=[{'id': 0, 'fitness': 0.6 + gen * 0.1}]
        )

    history = memory.get_fitness_history()
    assert len(history) == 3
    assert history == [0.6, 0.7, 0.8]

    print("[OK] test_memory_fitness_history")


def test_memory_load_from_disk():
    """测试从磁盘加载"""
    save_dir = "test_logs/memory_load_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # 创建并保存
    memory1 = EvolutionaryMemory(save_dir=save_dir)
    memory1.save(
        generation=0,
        best_code="best_code_v1",
        reflection="反思1",
        all_results=[{'id': 0, 'fitness': 0.7}]
    )

    # 重新加载
    memory2 = EvolutionaryMemory(save_dir=save_dir)
    memory2.load_from_disk()

    assert len(memory2.history) == 1
    assert memory2.history[0]['best_code'] == "best_code_v1"

    print("[OK] test_memory_load_from_disk")


def test_memory_export_summary():
    """测试导出摘要"""
    save_dir = "test_logs/memory_summary_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)

    for gen in range(2):
        memory.save(
            generation=gen,
            best_code=f"code_{gen}",
            reflection=f"反思_{gen}",
            all_results=[{'id': 0, 'fitness': 0.6 + gen * 0.1}]
        )

    summary = memory.export_summary()
    assert "进化过程摘要" in summary
    assert "Fitness进化曲线" in summary

    print("[OK] test_memory_export_summary")


def test_memory_clear():
    """测试清空记忆"""
    save_dir = "test_logs/memory_clear_test"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    memory = EvolutionaryMemory(save_dir=save_dir)

    memory.save(
        generation=0,
        best_code="test_code",
        reflection="反思",
        all_results=[{'id': 0, 'fitness': 0.6}]
    )

    assert len(memory.history) == 1

    memory.clear()
    assert len(memory.history) == 0

    print("[OK] test_memory_clear")


def cleanup():
    """清理测试目录"""
    import shutil
    test_dirs = [
        "test_logs/memory_test",
        "test_logs/memory_save_test",
        "test_logs/memory_get_test",
        "test_logs/memory_fitness_test",
        "test_logs/memory_load_test",
        "test_logs/memory_summary_test",
        "test_logs/memory_clear_test",
    ]
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 EvolutionaryMemory 模块")
    print("=" * 80)
    print()

    tests = [
        test_memory_initialization,
        test_memory_save,
        test_memory_get_best_code,
        test_memory_fitness_history,
        test_memory_load_from_disk,
        test_memory_export_summary,
        test_memory_clear,
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

    # 清理
    cleanup()

    print()
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
