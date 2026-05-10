"""
LLM接口可用性快速检查
用于验证 LLM API 是否可以正常连接和响应

使用方法:
    python tests/check_llm_connection.py
    python tests/check_llm_connection.py --config llm_reward_agent/config/llm_config.yaml
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_reward_agent.agent.llm_interface import LLMInterface


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def check_environment_variables():
    """检查必要的环境变量"""
    print("\n[1/5] 检查环境变量...")

    required_vars = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "ZHIPU_API_KEY": "智谱AI",
        "DEEPSEEK_API_KEY": "DeepSeek",
    }

    found = []
    for var, name in required_vars.items():
        value = os.getenv(var)
        if value:
            # 只显示前4位和后4位，中间用***隐藏
            masked = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
            print(f"  ✅ {name} ({var}): {masked}")
            found.append((var, value))

    if not found:
        print("  ❌ 未找到任何 LLM API Key 环境变量")
        print("  请设置以下环境变量之一:")
        for var, name in required_vars.items():
            print(f"    set {var}=your_key_here")
        return None

    # 优先使用第一个找到的
    print(f"\n  将使用: {found[0][0]}")
    return found[0]


def check_llm_connection(provider: str, model: str, api_key: str, base_url: str = None, timeout: int = 120):
    """测试LLM连接"""
    print(f"\n[2/5] 初始化 LLM 接口...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    if base_url:
        print(f"  Base URL: {base_url}")
    print(f"  Timeout: {timeout}s")

    try:
        llm = LLMInterface(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1
        )
        print("  ✅ LLM接口初始化成功\n")
        return llm
    except Exception as e:
        print(f"  ❌ LLM接口初始化失败: {e}")
        return None


def test_simple_generation(llm: LLMInterface):
    """测试简单文本生成"""
    print("[3/5] 测试简单文本生成...")

    test_prompt = "请回复：你好，这是一条测试消息。请简短回复。"

    start_time = time.time()
    try:
        results = llm.generate(
            prompt=test_prompt,
            n=1,
            temperature=0.7,
            max_tokens=500
        )
        elapsed = time.time() - start_time

        if results and len(results) > 0:
            response = results[0]
            print(f"  ✅ 生成成功 (耗时: {elapsed:.2f}s)")
            print(f"\n  模型回复:")
            print("  " + "-" * 60)
            # 限制显示长度
            display_text = response[:300] + "..." if len(response) > 300 else response
            for line in display_text.split('\n'):
                print(f"  {line}")
            print("  " + "-" * 60)
            return True
        else:
            print(f"  ❌ 生成失败: 返回结果为空")
            return False

    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_generation(llm: LLMInterface):
    """测试代码生成"""
    print("\n[4/5] 测试代码生成...")

    test_prompt = """请用Python写一个简单的加法函数，函数名为 add，接受两个参数 a 和 b，返回它们的和。
只输出代码，不需要解释。"""

    start_time = time.time()
    try:
        results = llm.generate(
            prompt=test_prompt,
            n=1,
            temperature=0.3,
            max_tokens=500
        )
        elapsed = time.time() - start_time

        if results and len(results) > 0:
            response = results[0]
            print(f"  ✅ 代码生成成功 (耗时: {elapsed:.2f}s)")
            print(f"\n  生成的代码:")
            print("  " + "-" * 60)
            # 限制显示长度
            display_text = response[:300] + "..." if len(response) > 300 else response
            for line in display_text.split('\n'):
                print(f"  {line}")
            print("  " + "-" * 60)
            return True
        else:
            print(f"  ❌ 代码生成失败: 返回结果为空")
            return False

    except Exception as e:
        print(f"  ❌ 代码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze(llm: LLMInterface):
    """测试分析功能"""
    print("\n[5/5] 测试分析功能 (analyze方法)...")

    test_prompt = """请分析以下奖励函数设计的优缺点，并给出改进建议：

def compute_reward(agent_name, observation, global_state, actions, world):
    if global_state.get('is_adversary', False):
        return 1.0, {}
    return 0.0, {}

请用简洁的语言分析。"""

    start_time = time.time()
    try:
        result = llm.analyze(
            prompt=test_prompt,
            temperature=0.3,
            max_tokens=5000
        )
        elapsed = time.time() - start_time

        if result:
            print(f"  ✅ 分析成功 (耗时: {elapsed:.2f}s)")
            print(f"\n  分析结果:")
            print("  " + "-" * 60)
            # 限制显示长度
            display_text = result[:300] + "..." if len(result) > 300 else result
            for line in display_text.split('\n'):
                print(f"  {line}")
            print("  " + "-" * 60)
            return True
        else:
            print(f"  ❌ 分析失败: 返回结果为空")
            return False

    except Exception as e:
        print(f"  ❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LLM接口可用性检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='llm_reward_agent/config/llm_config.yaml',
        help='LLM配置文件路径'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LLM 接口可用性检查工具")
    print("=" * 70)

    # 1. 检查环境变量
    env_result = check_environment_variables()
    if env_result is None:
        print("\n" + "=" * 70)
        print("❌ 检查失败: 未找到API Key")
        print("=" * 70)
        return 1

    env_var, api_key = env_result

    # 2. 尝试加载配置文件获取模型信息
    config_path = args.config
    if os.path.exists(config_path):
        print(f"\n  正在读取配置文件: {config_path}")
        try:
            config = load_config(config_path)
            llm_config = config.get('llm', {})
            provider = llm_config.get('provider', 'openai')
            model = llm_config.get('model', 'gpt-4')
            base_url = llm_config.get('base_url')
            timeout = llm_config.get('timeout', 120)
        except Exception as e:
            print(f"  ⚠️ 配置文件读取失败，将使用默认值: {e}")
            provider = 'openai'
            model = 'gpt-4'
            base_url = None
            timeout = 120
    else:
        print(f"\n  ⚠️ 配置文件不存在: {config_path}，使用默认值")
        provider = 'openai'
        model = 'gpt-4'
        base_url = None
        timeout = 120

    # 3. 测试连接
    llm = check_llm_connection(provider, model, api_key, base_url, timeout)
    if llm is None:
        print("\n" + "=" * 70)
        print("❌ 检查失败: 无法初始化LLM接口")
        print("=" * 70)
        return 1

    # 4. 运行测试
    results = []
    results.append(("简单文本生成", test_simple_generation(llm)))
    results.append(("代码生成", test_code_generation(llm)))
    results.append(("分析功能", test_analyze(llm)))

    # 5. 输出总结
    print("\n" + "=" * 70)
    print("检查结果总结")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}  {name}")

    print()
    if passed == total:
        print("🎉 所有检查通过！LLM接口工作正常。")
    else:
        print(f"⚠️  {total - passed}/{total} 项检查失败，请检查配置和网络。")

    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
