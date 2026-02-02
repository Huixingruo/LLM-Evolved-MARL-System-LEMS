"""
带调试信息的评估脚本
显示每个智能体的详细行为和奖励信息
"""
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from main_parameters import main_parameters
from utils.runner import RUNNER
from agents.maddpg.MADDPG_agent import MADDPG
import torch
from envs import simple_tag_env
import os
import numpy as np

def get_env(env_name, ep_len=50, render_mode = "None"):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_tag_env':
        new_env = simple_tag_env.parallel_env(render_mode = render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    new_env.reset()
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:",agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action,  hign action]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)

    return new_env, _dim_info, action_bound


def evaluate_with_debug(agent, env, args, device, num_episodes=5):
    """带调试信息的评估函数"""
    print("\n" + "="*60)
    print("开始调试评估")
    print("="*60)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"第 {episode + 1}/{num_episodes} 回合")
        print(f"{'='*60}")
        
        obs, _ = env.reset()
        step = 0
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        
        # 记录初始位置
        print("\n初始位置:")
        try:
            # 尝试不同的方式访问底层环境
            if hasattr(env, 'aec_env'):
                world_agents = env.aec_env.env.world.agents
            elif hasattr(env, 'unwrapped'):
                world_agents = env.unwrapped.world.agents
            else:
                world_agents = env.env.world.agents
            
            for agent_name in world_agents:
                pos = agent_name.state.p_pos
                agent_type = "追捕者" if agent_name.adversary else "逃跑者"
                print(f"  {agent_name.name} ({agent_type}): [{pos[0]:.2f}, {pos[1]:.2f}]")
        except Exception as e:
            print(f"  无法访问智能体位置: {e}")
        
        while env.agents:
            step += 1
            
            # 选择动作
            action = agent.select_action(obs)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 累积奖励
            for agent_id, r in reward.items():
                episode_rewards[agent_id] += r
            
            # 每10步打印一次详细信息
            if step % 10 == 0:
                print(f"\n--- Step {step} ---")
                
                # 计算追捕者与目标的距离
                target_agent = None
                adversaries = []
                try:
                    # 尝试访问底层环境
                    if hasattr(env, 'aec_env'):
                        world_agents = env.aec_env.env.world.agents
                    elif hasattr(env, 'unwrapped'):
                        world_agents = env.unwrapped.world.agents
                    else:
                        world_agents = env.env.world.agents
                    
                    for agent_obj in world_agents:
                        if not agent_obj.adversary:
                            target_agent = agent_obj
                        else:
                            adversaries.append(agent_obj)
                except Exception as e:
                    print(f"  无法访问世界状态: {e}")
                    continue
                
                if target_agent is not None:
                    print(f"\n目标位置 (逃跑者): [{target_agent.state.p_pos[0]:.2f}, {target_agent.state.p_pos[1]:.2f}]")
                    print(f"目标速度: [{target_agent.state.p_vel[0]:.2f}, {target_agent.state.p_vel[1]:.2f}]")
                    
                    print("\n追捕者状态:")
                    distances = []
                    for adv in adversaries:
                        dist = np.linalg.norm(target_agent.state.p_pos - adv.state.p_pos)
                        distances.append(dist)
                        print(f"  {adv.name}:")
                        print(f"    位置: [{adv.state.p_pos[0]:.2f}, {adv.state.p_pos[1]:.2f}]")
                        print(f"    距离目标: {dist:.3f}")
                        print(f"    速度: [{adv.state.p_vel[0]:.2f}, {adv.state.p_vel[1]:.2f}]")
                        print(f"    动作: [{action[adv.name][0]:.2f}, {action[adv.name][1]:.2f}]")
                        print(f"    当前奖励: {reward.get(adv.name, 0):.3f}")
                    
                    # 检查是否形成围捕
                    avg_dist = np.mean(distances)
                    max_dist = np.max(distances)
                    min_dist = np.min(distances)
                    print(f"\n围捕分析:")
                    print(f"  平均距离: {avg_dist:.3f}")
                    print(f"  最大距离: {max_dist:.3f} (最远的追捕者)")
                    print(f"  最小距离: {min_dist:.3f} (最近的追捕者)")
                    
                    # 判断协作状态
                    try:
                        if hasattr(env, 'aec_env'):
                            world_size = env.aec_env.env.world.world_size
                        elif hasattr(env, 'unwrapped'):
                            world_size = env.unwrapped.world.world_size
                        else:
                            world_size = env.env.world.world_size
                    except:
                        world_size = 2.5  # 默认值
                    
                    capture_threshold = world_size * 0.2
                    if all(d < capture_threshold for d in distances):
                        print(f"  [成功] 围捕成功！所有追捕者都在 {capture_threshold:.2f} 范围内")
                    elif max_dist > capture_threshold * 2:
                        print(f"  [警告] 有追捕者远离目标 (距离 > {capture_threshold*2:.2f})")
                    
            obs = next_obs
        
        # 回合结束统计
        print(f"\n{'='*60}")
        print(f"第 {episode + 1} 回合结束 - 总步数: {step}")
        print(f"{'='*60}")
        print("\n各智能体累积奖励:")
        total_reward = 0
        adversary_rewards = []
        for agent_id, rew in episode_rewards.items():
            agent_type = "追捕者" if "adversary" in agent_id else "逃跑者"
            print(f"  {agent_id} ({agent_type}): {rew:.2f}")
            total_reward += rew
            if "adversary" in agent_id:
                adversary_rewards.append(rew)
        
        print(f"\n追捕者平均奖励: {np.mean(adversary_rewards):.2f}")
        print(f"总奖励: {total_reward:.2f}")
        
        # 分析追捕者表现差异
        if len(adversary_rewards) > 0:
            reward_std = np.std(adversary_rewards)
            if reward_std > 5.0:
                print(f"\n[警告] 追捕者奖励差异较大 (标准差: {reward_std:.2f})")
                print("    可能原因: 某些智能体没有有效参与围捕")


if __name__ == '__main__':
    # device ='cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)
    
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    
    # 加载模型的时间戳
    load_timestamp = "2026-01-28_14-02"  # 请修改为你的模型时间戳
    model_timestamp = None if load_timestamp == '' else load_timestamp
    
    # 定义参数
    args = main_parameters()
    args.render_mode = "human"  # 可视化
    args.episode_num = 5  # 评估5个回合
    
    print(f"\n加载模型时间戳: {load_timestamp}")
    
    # 创建环境
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode)
    
    # 创建MA-DDPG智能体
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, 
                   action_bound, _chkpt_dir = chkpt_dir, _model_timestamp = model_timestamp)
    
    # 检查模型是否存在
    model_dir = os.path.join(chkpt_dir, load_timestamp) if load_timestamp else chkpt_dir
    if not os.path.exists(model_dir):
        print(f"\n[错误] 模型文件夹不存在: {model_dir}")
        print("\n建议:")
        print("1. 先运行 main_train.py 训练模型")
        print("2. 或者修改 load_timestamp 为已有的模型时间戳")
        print("\n当前可用的模型时间戳:")
        if os.path.exists(chkpt_dir):
            timestamps = [d for d in os.listdir(chkpt_dir) if os.path.isdir(os.path.join(chkpt_dir, d))]
            if timestamps:
                for ts in timestamps:
                    print(f"  - {ts}")
            else:
                print("  (无可用模型)")
        exit(1)
    
    print("\n--- 加载模型 ---")
    agent.load_model()
    
    print('\n--- 开始调试评估 ---')
    env.reset()
    evaluate_with_debug(agent, env, args, device, num_episodes=5)
    
    print('\n--- 评估完成 ---')
