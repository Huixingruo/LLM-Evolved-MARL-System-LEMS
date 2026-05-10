"""
测试夹具（共享的测试数据和辅助函数）
"""

import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 标准奖励函数代码示例（用于测试）
SAMPLE_REWARD_CODES = [
    """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}

    if global_state.get('is_adversary', False):
        agent_idx = int(agent_name.split('_')[1])
        agent_pos = global_state['agent_positions'][agent_idx]
        prey_pos = global_state['prey_position']
        dist = np.linalg.norm(agent_pos - prey_pos)

        components['distance_reward'] = -0.1 * dist
        components['boundary_penalty'] = 0.0
    else:
        components['escape_reward'] = 0.1
        components['boundary_penalty'] = 0.0

    total_reward = sum(components.values())
    return total_reward, components
""",
    """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}

    if global_state.get('is_adversary', False):
        agent_idx = int(agent_name.split('_')[1])
        agent_pos = global_state['agent_positions'][agent_idx]
        prey_pos = global_state['prey_position']
        dist = np.linalg.norm(agent_pos - prey_pos)

        # 增强距离奖励
        components['distance_reward'] = -0.2 * dist
        components['boundary_penalty'] = 0.0
    else:
        components['escape_reward'] = 0.1
        components['boundary_penalty'] = 0.0

    total_reward = sum(components.values())
    return total_reward, components
""",
]


# 标准模拟训练结果（用于测试）
SAMPLE_TRAINING_RESULTS = [
    {
        'id': 0,
        'code': SAMPLE_REWARD_CODES[0],
        'status': 'success',
        'fitness': 0.75,
        'metrics': {
            'success_rate': 0.80,
            'avg_capture_time': 45.0,
            'reward_components': {
                'distance_reward': {'mean': -1.23, 'std': 0.45},
                'collision_penalty': {'mean': -0.15, 'std': 0.12},
                'formation_reward': {'mean': 0.34, 'std': 0.08}
            },
            'collaboration_metrics': {
                'formation_quality': {'mean': 0.56}
            }
        }
    },
    {
        'id': 1,
        'code': SAMPLE_REWARD_CODES[1],
        'status': 'success',
        'fitness': 0.68,
        'metrics': {
            'success_rate': 0.72,
            'avg_capture_time': 52.0,
            'reward_components': {
                'distance_reward': {'mean': -1.45, 'std': 0.50},
                'collision_penalty': {'mean': -0.10, 'std': 0.08}
            },
            'collaboration_metrics': {
                'formation_quality': {'mean': 0.48}
            }
        }
    },
]


# 测试用 LLM 配置
TEST_LLM_CONFIG = {
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


# 模拟 LLM 响应
class MockLLMInterface:
    """模拟 LLM 接口，用于测试"""

    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or []
        self.call_count = 0

    def generate(self, prompt, n=1, temperature=0.7, max_tokens=2000, system_message=None):
        self.call_count += 1
        if self.mock_responses:
            idx = min(self.call_count - 1, len(self.mock_responses) - 1)
            return [self.mock_responses[idx]] * n if n > 1 else [self.mock_responses[idx]]
        return ["# Mock response\nimport numpy as np\n\ndef compute_reward(...):\n    pass"] * n

    def analyze(self, prompt, temperature=0.3, max_tokens=1500):
        return "Mock reflection: Test successful."


def cleanup_test_logs():
    """清理测试日志目录"""
    import shutil
    test_dirs = ['test_logs', 'experiments/test']
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
