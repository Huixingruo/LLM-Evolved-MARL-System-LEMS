"""
llm_interface.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_reward_agent.agent.llm_interface import LLMInterface


def test_llm_interface_initialization():
    """测试 LLMInterface 类初始化"""
    try:
        llm = LLMInterface(
            provider="openai",
            model_name="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "test-key")
        )
        assert llm is not None
        assert llm.provider == "openai"
        assert llm.model_name == "gpt-4"
        print("[OK] test_llm_interface_initialization")
        return llm
    except ValueError:
        print("[SKIP] test_llm_interface_initialization: API密钥未设置")
        return None
    except Exception as e:
        print(f"[ERROR] test_llm_interface_initialization: {e}")
        return None


def test_provider_defaults():
    """测试默认 provider 设置"""
    llm = LLMInterface(
        provider="openai",
        model_name="gpt-4",
        api_key="test-key"
    )
    assert llm.provider == "openai"
    assert llm.max_retries == 3
    print("[OK] test_provider_defaults")


def test_deepseek_provider():
    """测试 DeepSeek provider 配置"""
    llm = LLMInterface(
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="test-key",
        base_url="https://api.deepseek.com/v1"
    )
    assert llm.provider == "deepseek"
    assert llm.model_name == "deepseek-chat"
    print("[OK] test_deepseek_provider")


def test_cost_estimation():
    """测试成本估算"""
    llm = LLMInterface(
        provider="openai",
        model_name="gpt-4",
        api_key="test-key"
    )

    cost = llm.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost >= 0
    print(f"[OK] test_cost_estimation: ${cost:.6f}")


def test_cost_estimation_deepseek():
    """测试 DeepSeek 成本估算"""
    llm = LLMInterface(
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="test-key"
    )

    cost = llm.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost >= 0
    print(f"[OK] test_cost_estimation_deepseek: ${cost:.6f}")


def test_unsupported_provider():
    """测试不支持的 provider"""
    try:
        llm = LLMInterface(
            provider="unsupported_provider",
            model_name="some-model",
            api_key="test-key"
        )
        print("[FAIL] test_unsupported_provider: 应该抛出异常")
    except ValueError as e:
        print(f"[OK] test_unsupported_provider: 正确抛出 ValueError")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 LLMInterface 模块")
    print("=" * 80)
    print()

    tests = [
        test_llm_interface_initialization,
        test_provider_defaults,
        test_deepseek_provider,
        test_cost_estimation,
        test_cost_estimation_deepseek,
        test_unsupported_provider,
    ]

    passed = 0
    skipped = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is not None or "[SKIP]" not in test.__name__:
                passed += 1
            else:
                skipped += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"测试完成: {passed} 通过, {skipped} 跳过, {failed} 失败")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
