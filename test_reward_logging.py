"""
测试奖励分量是否正确记录

Author: LEMS Project
Date: 2026-02-03
"""

import sys
import os
import numpy as np

# 设置UTF-8输出编码（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_reward_component_logging():
    """测试奖励分量记录是否正常工作"""
    print("=" * 80)
    print("测试：奖励分量记录功能")
    print("=" * 80)
    
    try:
        from MADDPG.envs.simple_tag_env import Custom_raw_env
        
        # 创建环境
        print("\n1. 创建环境...")
        env = Custom_raw_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            continuous_actions=True,
            render_mode=None,
            max_cycles=50
        )
        
        # 重置环境
        print("2. 重置环境...")
        obs, info = env.reset()
        
        # 检查初始化
        print("3. 检查环境属性...")
        print(f"   - Has last_reward_components: {hasattr(env, 'last_reward_components')}")
        print(f"   - Has current_actions: {hasattr(env, 'current_actions')}")
        print(f"   - Has world._env_instance: {hasattr(env.world, '_env_instance')}")
        print(f"   - world._env_instance is env: {env.world._env_instance is env}")
        
        # 执行几步并检查奖励分量
        print("\n4. 执行10步交互...")
        for step_num in range(10):
            # 收集所有智能体的动作
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = env.action_space(agent_id).sample()
            
            # 执行step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # 检查奖励分量
            if step_num == 0:  # 第一步详细检查
                print(f"\n   Step {step_num} 检查:")
                print(f"   - current_actions keys: {list(env.current_actions.keys())}")
                print(f"   - last_reward_components keys: {list(env.last_reward_components.keys())}")
                
                for agent_id in env.agents:
                    if agent_id in env.last_reward_components:
                        components = env.last_reward_components[agent_id]
                        print(f"\n   {agent_id}:")
                        print(f"     - 奖励分量数: {len(components)}")
                        print(f"     - 分量名称: {list(components.keys())}")
                        print(f"     - 总奖励: {rewards.get(agent_id, 0):.4f}")
                        
                        # 显示前3个分量的值
                        for i, (key, val) in enumerate(list(components.items())[:3]):
                            print(f"       * {key}: {val:.4f}")
            
            obs = next_obs
            
            # 如果环境结束了，退出
            if not env.agents:
                break
        
        # 验证
        print("\n5. 验证结果...")
        if len(env.last_reward_components) > 0:
            has_data = False
            for agent_id, components in env.last_reward_components.items():
                if len(components) > 0:
                    has_data = True
                    break
            
            if has_data:
                print("   [OK] 奖励分量已成功记录！")
                return True
            else:
                print("   [ERROR] last_reward_components存在但为空")
                return False
        else:
            print("   [ERROR] last_reward_components为空字典")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_reward_component_logging()
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] 奖励分量记录功能正常工作！")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("[FAILED] 奖励分量记录功能存在问题，需要修复")
        print("="*80)
        sys.exit(1)
