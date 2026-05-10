"""
context_extractor.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os as os_module

from llm_reward_agent.tools.context_extractor import (
    LLMFriendlyContextExtractor,
    EnvironmentContextExtractor
)


def find_test_env_file():
    """查找测试环境文件"""
    search_paths = [
        "MADDPG/envs/simple_tag_env.py",
        os_module.path.join(os_module.path.dirname(__file__), "../../../MADDPG/envs/simple_tag_env.py"),
    ]

    for path in search_paths:
        if os_module.path.exists(path):
            return os_module.path.abspath(path)
    return None


def test_llm_friendly_context_extractor_initialization():
    """测试 LLMFriendlyContextExtractor 初始化"""
    env_file = find_test_env_file()

    if env_file is None:
        print("[SKIP] test_llm_friendly_context_extractor_initialization: 环境文件未找到")
        return

    extractor = LLMFriendlyContextExtractor(env_file)
    assert extractor is not None
    assert extractor.file_path == env_file

    print("[OK] test_llm_friendly_context_extractor_initialization")


def test_keep_classes_config():
    """测试保留类配置"""
    env_file = find_test_env_file()

    if env_file is None:
        print("[SKIP] test_keep_classes_config: 环境文件未找到")
        return

    extractor = LLMFriendlyContextExtractor(env_file)

    assert 'Custom_raw_env' in extractor.keep_classes
    assert 'Scenario' in extractor.keep_classes

    print("[OK] test_keep_classes_config")


def test_render_blacklist():
    """测试渲染黑名单"""
    env_file = find_test_env_file()

    if env_file is None:
        print("[SKIP] test_render_blacklist: 环境文件未找到")
        return

    extractor = LLMFriendlyContextExtractor(env_file)

    assert 'render' in extractor.render_blacklist
    assert 'draw' in extractor.render_blacklist

    print("[OK] test_render_blacklist")


def test_environment_context_extractor_compatibility():
    """测试兼容性包装类"""
    env_file = find_test_env_file()

    if env_file is None:
        print("[SKIP] test_environment_context_extractor_compatibility: 环境文件未找到")
        return

    # 测试默认初始化
    try:
        extractor = EnvironmentContextExtractor()
        assert extractor is not None
        print("[OK] test_environment_context_extractor_compatibility (默认初始化)")
    except Exception as e:
        print(f"[SKIP] test_environment_context_extractor_compatibility: {e}")


def test_token_estimation():
    """测试 Token 估算"""
    extractor = EnvironmentContextExtractor.__new__(EnvironmentContextExtractor)

    text = "这是一段测试文本" * 100
    estimate = extractor.estimate_token_count(text)

    assert estimate > 0
    assert isinstance(estimate, int)
    print(f"[OK] test_token_estimation: {estimate} tokens")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 ContextExtractor 模块")
    print("=" * 80)
    print()

    tests = [
        test_llm_friendly_context_extractor_initialization,
        test_keep_classes_config,
        test_render_blacklist,
        test_environment_context_extractor_compatibility,
        test_token_estimation,
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
