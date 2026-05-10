"""
prompt_templates.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_reward_agent.agent.prompt_templates import (
    PromptTemplates,
    PREDEFINED_ENV_CONTEXT,
    PREDEFINED_TASK_DESCRIPTION
)


def test_predefined_context_exists():
    """测试预定义环境上下文是否存在"""
    assert PREDEFINED_ENV_CONTEXT is not None
    assert 'env_name' in PREDEFINED_ENV_CONTEXT
    assert 'agent_info' in PREDEFINED_ENV_CONTEXT
    assert 'code_snippet' in PREDEFINED_ENV_CONTEXT
    print("[OK] test_predefined_context_exists")


def test_predefined_task_description_exists():
    """测试预定义任务描述是否存在"""
    assert PREDEFINED_TASK_DESCRIPTION is not None
    assert len(PREDEFINED_TASK_DESCRIPTION) > 0
    print("[OK] test_predefined_task_description_exists")


def test_prompt_templates_initialization():
    """测试 PromptTemplates 类初始化"""
    builder = PromptTemplates()
    assert builder is not None
    assert hasattr(builder, 'SYSTEM_MESSAGE')
    print("[OK] test_prompt_templates_initialization")


def test_system_message():
    """测试系统提示词"""
    assert len(PromptTemplates.SYSTEM_MESSAGE) > 0
    assert '奖励函数' in PromptTemplates.SYSTEM_MESSAGE
    print("[OK] test_system_message")


def test_initial_generation_prompt_with_predefined_context():
    """测试使用预定义上下文的方法"""
    predefined_prompt = PromptTemplates.initial_generation_prompt_with_predefined_context()
    assert predefined_prompt is not None
    assert len(predefined_prompt) > 0
    # 检查关键内容
    assert '任务描述' in predefined_prompt
    assert 'compute_reward' in predefined_prompt
    print(f"[OK] test_initial_generation_prompt_with_predefined_context (长度: {len(predefined_prompt)} 字符)")


def test_evolution_prompt_with_predefined_context():
    """测试进化提示词"""
    test_parent_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    dist = np.linalg.norm(global_state['agent_positions'][0] - global_state['prey_position'])
    components['distance_reward'] = -dist
    total_reward = sum(components.values())
    return total_reward, components
"""
    test_reflection = """
成功率较低（60%），主要问题是智能体扎堆。
建议：增加角度排斥力，鼓励均匀分布。
"""
    evolution_prompt = PromptTemplates.evolution_prompt_with_predefined_context(
        parent_code=test_parent_code,
        reflection=test_reflection
    )
    assert evolution_prompt is not None
    assert len(evolution_prompt) > 0
    assert '改进' in evolution_prompt
    print(f"[OK] test_evolution_prompt_with_predefined_context (长度: {len(evolution_prompt)} 字符)")


def test_env_context_agent_info():
    """测试环境上下文中的智能体信息"""
    agent_info = PREDEFINED_ENV_CONTEXT.get('agent_info', {})
    assert agent_info.get('num_adversaries') == 3
    assert agent_info.get('num_good') == 1
    assert agent_info.get('num_obstacles') == 0
    print("[OK] test_env_context_agent_info")


def test_code_snippet_contains_key_classes():
    """测试代码片段包含关键类"""
    code_snippet = PREDEFINED_ENV_CONTEXT.get('code_snippet', '')
    assert 'CoreEnvLogic' in code_snippet
    assert 'is_collision' in code_snippet
    assert 'world_size' in code_snippet
    assert 'capture_threshold' in code_snippet
    print("[OK] test_code_snippet_contains_key_classes")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 PromptTemplates 模块")
    print("=" * 80)
    print()

    tests = [
        test_predefined_context_exists,
        test_predefined_task_description_exists,
        test_prompt_templates_initialization,
        test_system_message,
        test_initial_generation_prompt_with_predefined_context,
        test_evolution_prompt_with_predefined_context,
        test_env_context_agent_info,
        test_code_snippet_contains_key_classes,
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
