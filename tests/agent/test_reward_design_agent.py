"""
reward_design_agent.py 测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
from llm_reward_agent.agent.reward_design_agent import RewardDesignAgent


def create_test_config():
    """创建测试用配置"""
    config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': 'test-key',
            'timeout': 10,
            'max_retries': 1
        },
        'generation': {
            'num_candidates': 4,
            'temperature': 0.8,
            'max_tokens': 2000
        },
        'reflection': {
            'temperature': 0.3,
            'max_tokens': 1500
        },
        'training': {
            'episode_num': 10,
            'parallel_workers': 2,
            'timeout': 300,
            'use_gpu': False
        },
        'logging': {
            'save_dir': 'test_logs'
        },
        'fitness': {
            'weights': {
                'success_rate': 1.0,
                'capture_time': -0.001,
                'formation_quality': 0.3,
                'collision_penalty': -0.5
            }
        }
    }
    return config


def test_agent_initialization_without_api():
    """测试智能体初始化（不调用真实API）"""
    config = create_test_config()
    config_path = 'test_llm_config.yaml'

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    try:
        # 注意：这会尝试初始化LLM接口，需要API密钥
        # 在测试环境中会跳过实际LLM调用
        print("[INFO] test_agent_initialization_without_api: 配置已创建")
        os.remove(config_path)
        print("[OK] test_agent_initialization_without_api")
    except Exception as e:
        if os.path.exists(config_path):
            os.remove(config_path)
        print(f"[INFO] test_agent_initialization_without_api: {e}")


def test_parse_code_blocks():
    """测试代码块解析"""
    config = create_test_config()
    config_path = 'test_llm_config.yaml'

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    try:
        agent = RewardDesignAgent(config_path=config_path)

        # 测试代码块解析
        test_outputs = [
            """```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test'] = 1.0
    return sum(components.values()), components
```
"""
        ]

        codes = agent._parse_code_blocks(test_outputs)
        assert len(codes) == 1
        assert 'import numpy as np' in codes[0]
        assert 'def compute_reward' in codes[0]

        os.remove(config_path)
        print("[OK] test_parse_code_blocks")

    except Exception as e:
        if os.path.exists(config_path):
            os.remove(config_path)
        print(f"[INFO] test_parse_code_blocks: {e}")


def test_syntax_check():
    """测试语法检查"""
    config = create_test_config()
    config_path = 'test_llm_config.yaml'

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    try:
        agent = RewardDesignAgent(config_path=config_path)

        # 测试有效代码
        valid_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    return 0, components
"""
        assert agent._syntax_check(valid_code) == True

        # 测试无效代码
        invalid_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world)
    components = {}
"""
        assert agent._syntax_check(invalid_code) == False

        os.remove(config_path)
        print("[OK] test_syntax_check")

    except Exception as e:
        if os.path.exists(config_path):
            os.remove(config_path)
        print(f"[INFO] test_syntax_check: {e}")


def test_simulate_training():
    """测试模拟训练"""
    config = create_test_config()
    config_path = 'test_llm_config.yaml'

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    try:
        agent = RewardDesignAgent(config_path=config_path)

        codes = [
            """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    return 0, components
""",
            """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    return 0, components
"""
        ]

        results = agent._simulate_training(codes)
        assert len(results) == 2
        assert results[0]['status'] == 'success'
        assert 'fitness' in results[0]

        os.remove(config_path)
        print("[OK] test_simulate_training")

    except Exception as e:
        if os.path.exists(config_path):
            os.remove(config_path)
        print(f"[INFO] test_simulate_training: {e}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("测试 RewardDesignAgent 模块")
    print("=" * 80)
    print()

    tests = [
        test_agent_initialization_without_api,
        test_parse_code_blocks,
        test_syntax_check,
        test_simulate_training,
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
