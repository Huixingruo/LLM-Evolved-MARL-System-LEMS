"""
LEMS 测试套件运行器
运行所有模块的测试

使用方法:
    python tests/run_all_tests.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_agent_tests():
    """运行 agent 模块测试"""
    print("\n" + "=" * 80)
    print("测试 llm_reward_agent.agent 模块")
    print("=" * 80 + "\n")

    from tests.agent import test_prompt_templates
    from tests.agent import test_llm_interface
    from tests.agent import test_reward_design_agent
    from tests.agent import test_memory

    modules = [
        test_prompt_templates,
        test_llm_interface,
        test_reward_design_agent,
        test_memory,
    ]

    for module in modules:
        module.run_all_tests()


def run_tools_tests():
    """运行 tools 模块测试"""
    print("\n" + "=" * 80)
    print("测试 llm_reward_agent.tools 模块")
    print("=" * 80 + "\n")

    from tests.tools import test_simulation_tool
    from tests.tools import test_log_analyzer
    from tests.tools import test_sandbox_manager
    from tests.tools import test_context_extractor
    from tests.tools import test_launcher

    modules = [
        test_simulation_tool,
        test_log_analyzer,
        test_sandbox_manager,
        test_context_extractor,
        test_launcher,
    ]

    for module in modules:
        module.run_all_tests()


def main():
    """主函数"""
    print("=" * 80)
    print("LEMS 测试套件")
    print("=" * 80)

    run_agent_tests()
    run_tools_tests()

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
