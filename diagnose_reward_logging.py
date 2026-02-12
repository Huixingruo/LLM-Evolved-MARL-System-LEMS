"""
诊断奖励分量记录问题

检查：
1. parallel_env是否保留了last_reward_components属性
2. reward方法是否正确调用
3. 奖励分量是否正确保存
"""

import sys
import os

# 设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("诊断：奖励分量记录问题")
print("=" * 80)

try:
    from MADDPG.envs import simple_tag_env
    import numpy as np
    
    # 使用parallel_env（与训练一致）
    print("\n1. 创建parallel_env...")
    env = simple_tag_env.parallel_env(
        render_mode=None,
        num_good=1,
        num_adversaries=3,
        num_obstacles=0,
        max_cycles=50,
        continuous_actions=True
    )
    
    print("2. 重置环境...")
    obs, info = env.reset()
    
    print("3. 检查环境类型和属性...")
    print(f"   - Environment type: {type(env).__name__}")
    print(f"   - Has unwrapped: {hasattr(env, 'unwrapped')}")
    print(f"   - Has aec_env: {hasattr(env, 'aec_env')}")
    
    # 尝试访问底层的raw_env
    if hasattr(env, 'aec_env'):
        print(f"   - AEC env type: {type(env.aec_env).__name__}")
        print(f"   - AEC has last_reward_components: {hasattr(env.aec_env, 'last_reward_components')}")
        print(f"   - AEC has current_actions: {hasattr(env.aec_env, 'current_actions')}")
        if hasattr(env.aec_env, 'world'):
            print(f"   - AEC.world has _env_instance: {hasattr(env.aec_env.world, '_env_instance')}")
    
    print("\n4. 执行一轮完整的动作...")
    # 收集所有智能体的动作
    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    
    print(f"   - Actions keys: {list(actions.keys())}")
    
    # 执行step（parallel_env会同时执行所有智能体的动作）
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print(f"   - Rewards received: {list(rewards.keys())}")
    print(f"   - Reward values: {[f'{r:.4f}' for r in rewards.values()]}")
    
    print("\n5. 检查奖励分量记录...")
    # parallel_env的底层是aec_env
    if hasattr(env, 'aec_env'):
        aec_env = env.aec_env
        if hasattr(aec_env, 'last_reward_components'):
            print(f"   - last_reward_components存在")
            print(f"   - Keys: {list(aec_env.last_reward_components.keys())}")
            
            # 检查每个智能体的分量
            for agent_id in aec_env.last_reward_components.keys():
                components = aec_env.last_reward_components[agent_id]
                print(f"\n   {agent_id}:")
                print(f"     - 分量数: {len(components)}")
                if len(components) > 0:
                    print(f"     - 分量名称: {list(components.keys())}")
                    for key, val in list(components.items())[:3]:
                        print(f"       * {key}: {val:.6f}")
                else:
                    print(f"     - [WARNING] 分量为空！")
        else:
            print("   [ERROR] aec_env没有last_reward_components属性")
    else:
        print("   [ERROR] env没有aec_env属性")
    
    print("\n6. 检查current_actions...")
    if hasattr(env, 'aec_env') and hasattr(env.aec_env, 'current_actions'):
        current_actions = env.aec_env.current_actions
        print(f"   - current_actions type: {type(current_actions)}")
        print(f"   - current_actions keys: {list(current_actions.keys()) if isinstance(current_actions, dict) else 'Not a dict'}")
        if isinstance(current_actions, dict) and len(current_actions) > 0:
            first_key = list(current_actions.keys())[0]
            first_value = current_actions[first_key]
            print(f"   - Sample action (key={first_key}): type={type(first_value)}, shape={first_value.shape if hasattr(first_value, 'shape') else 'N/A'}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
except Exception as e:
    print(f"\n[ERROR] 诊断失败: {e}")
    import traceback
    traceback.print_exc()
