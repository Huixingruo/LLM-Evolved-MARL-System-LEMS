"""
使用matplotlib渲染的评估脚本 - 支持实时渲染和生成GIF
展示如何使用新添加的render_matplotlib()方法

运行模式：
1. 实时渲染模式（display_mode='live'）：边评估边显示动画
2. 保存图片模式（display_mode='save'）：保存PNG序列
3. 生成GIF模式（display_mode='gif'）：直接生成GIF动画
4. 混合模式（display_mode='both'）：实时显示+保存GIF
"""
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from main_parameters import main_parameters
from agents.maddpg.MADDPG_agent import MADDPG
import torch
from envs import simple_tag_env
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

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


def evaluate_with_matplotlib_live(agent, env, args, device, num_episodes=3, 
                                   display_mode='both', save_gif=True, fps=10):
    """
    使用matplotlib实时渲染评估
    
    Args:
        agent: 训练好的MADDPG智能体
        env: 环境
        args: 参数
        device: 设备
        num_episodes: 评估回合数
        display_mode: 显示模式
            - 'live': 仅实时显示，不保存
            - 'save': 仅保存图片
            - 'gif': 仅生成GIF
            - 'both': 实时显示+生成GIF
        save_gif: 是否保存为GIF
        fps: GIF帧率
    """
    print("\n" + "="*60)
    print("使用Matplotlib实时渲染评估")
    print(f"显示模式: {display_mode}")
    print("="*60)
    
    # 创建保存目录
    render_dir = 'renders_matplotlib'
    os.makedirs(render_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"第 {episode + 1}/{num_episodes} 回合")
        print(f"{'='*60}")
        
        obs, _ = env.reset()
        step = 0
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        
        # 用于存储帧数据
        frames = []
        
        # 如果需要实时显示，创建figure和axis
        if display_mode in ['live', 'both']:
            plt.ion()  # 开启交互模式
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.canvas.manager.set_window_title(f'Episode {episode+1} - MADDPG Evaluation')
        
        while env.agents:
            step += 1
            
            # 选择动作（评估模式，无探索噪声）
            action = agent.select_action(obs, evaluate=True)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 累积奖励
            for agent_id, r in reward.items():
                episode_rewards[agent_id] += r
            
            # 渲染当前帧
            if display_mode in ['live', 'both']:
                # 实时渲染
                ax.clear()
                render_frame_on_axis(ax, env, step, episode_rewards)
                plt.pause(0.01)  # 短暂暂停以更新显示
            
            # 保存帧用于GIF
            if display_mode in ['gif', 'both', 'save']:
                img_array = env.unwrapped.render_matplotlib()
                frames.append(img_array)
                
                # 如果是save模式，每5步保存一张
                if display_mode == 'save' and step % 5 == 0:
                    img = Image.fromarray(img_array, 'RGBA')
                    filename = f'{render_dir}/episode_{episode+1:02d}_step_{step:04d}.png'
                    img.save(filename)
            
            obs = next_obs
        
        # 关闭实时显示窗口
        if display_mode in ['live', 'both']:
            plt.ioff()
            plt.close(fig)
        
        # 回合结束统计
        print(f"\n第 {episode + 1} 回合结束 - 总步数: {step}")
        print("各智能体累积奖励:")
        total_reward = 0
        adversary_rewards = []
        for agent_id, rew in episode_rewards.items():
            agent_type = "追捕者" if "adversary" in agent_id else "逃跑者"
            print(f"  {agent_id} ({agent_type}): {rew:.2f}")
            total_reward += rew
            if "adversary" in agent_id:
                adversary_rewards.append(rew)
        
        print(f"追捕者平均奖励: {np.mean(adversary_rewards):.2f}")
        print(f"总奖励: {total_reward:.2f}")
        
        # 生成GIF
        if save_gif and display_mode in ['gif', 'both'] and len(frames) > 0:
            gif_filename = f'{render_dir}/episode_{episode+1:02d}_animation.gif'
            print(f"\n正在生成GIF动画: {gif_filename}")
            save_frames_as_gif(frames, gif_filename, fps=fps)
            print(f"✓ GIF已保存")
    
    print(f"\n{'='*60}")
    print("评估完成！")
    if display_mode == 'save':
        print(f"图片已保存到: {render_dir}/")
    if display_mode in ['gif', 'both']:
        print(f"GIF动画已保存到: {render_dir}/")
    print(f"{'='*60}")


def render_frame_on_axis(ax, env, step, episode_rewards):
    """在给定的axis上渲染一帧"""
    # 设置坐标轴
    cam_range = env.unwrapped.world_size
    ax.set_xlim(-cam_range, cam_range)
    ax.set_ylim(-cam_range, cam_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # 显示步数和奖励
    reward_text = f"Step: {step}\n"
    for agent_id, rew in episode_rewards.items():
        agent_type = "追" if "adversary" in agent_id else "逃"
        reward_text += f"{agent_type}: {rew:.1f}\n"
    ax.text(0.02, 0.98, reward_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 获取环境中的智能体和场景
    scenario = env.unwrapped.scenario
    world = env.unwrapped.world
    
    # 绘制捕获圈
    for agent_obj in scenario.good_agents(world):
        circle = Circle(agent_obj.state.p_pos, env.unwrapped.capture_threshold,
                       color='green', fill=False, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    # 绘制轨迹
    for agent_obj in world.agents:
        if len(env.unwrapped.history_positions[agent_obj.name]) >= 2:
            trajectory = np.array(env.unwrapped.history_positions[agent_obj.name])
            color = 'blue' if agent_obj.adversary else 'red'
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, alpha=0.3, linewidth=1.5)
    
    # 绘制智能体
    for entity in world.entities:
        from pettingzoo.mpe._mpe_utils.core import Agent
        pos = entity.state.p_pos
        if isinstance(entity, Agent):
            color = 'blue' if entity.adversary else 'red'
            circle = Circle(pos, entity.size, color=color, alpha=0.7)
            ax.add_patch(circle)
            circle_border = Circle(pos, entity.size, color='white', 
                                 fill=False, linewidth=1.5)
            ax.add_patch(circle_border)
            # 添加速度箭头
            vel = entity.state.p_vel
            if np.linalg.norm(vel) > 0.01:
                ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5,
                        head_width=0.05, head_length=0.03, fc=color, ec=color, alpha=0.6)
        else:  # Landmark
            circle = Circle(pos, entity.size, color='gray', alpha=0.5)
            ax.add_patch(circle)


def save_frames_as_gif(frames, filename, fps=10):
    """将帧序列保存为GIF"""
    if len(frames) == 0:
        print("警告: 没有帧可保存")
        return
    
    # 转换为PIL图像列表
    images = []
    for frame in frames:
        img = Image.fromarray(frame, 'RGBA')
        # 转换为RGB（GIF不支持RGBA）
        img_rgb = Image.new('RGB', img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
        images.append(img_rgb)
    
    # 保存为GIF
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # 0表示无限循环
        optimize=False
    )


def evaluate_with_matplotlib(agent, env, args, device, num_episodes=3, save_images=True):
    """原版评估函数（仅保存图片，保持向后兼容）"""
    print("\n" + "="*60)
    print("使用Matplotlib渲染评估（保存图片模式）")
    print("="*60)
    
    # 创建保存目录
    if save_images:
        render_dir = 'renders_matplotlib'
        os.makedirs(render_dir, exist_ok=True)
        print(f"图片将保存到: {render_dir}/")
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"第 {episode + 1}/{num_episodes} 回合")
        print(f"{'='*60}")
        
        obs, _ = env.reset()
        step = 0
        episode_rewards = {agent_id: 0 for agent_id in env.agents}
        
        while env.agents:
            step += 1
            
            # 选择动作（评估模式，无探索噪声）
            action = agent.select_action(obs, evaluate=True)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 累积奖励
            for agent_id, r in reward.items():
                episode_rewards[agent_id] += r
            
            # 使用matplotlib渲染并保存（每5步保存一次）
            if save_images and step % 5 == 0:
                try:
                    # 调用新的matplotlib渲染方法
                    img_array = env.unwrapped.render_matplotlib()
                    
                    # 保存为PNG
                    img = Image.fromarray(img_array, 'RGBA')
                    filename = f'{render_dir}/episode_{episode+1:02d}_step_{step:04d}.png'
                    img.save(filename)
                    
                    if step % 20 == 0:  # 每20步打印一次
                        print(f"  已保存: {filename}")
                except Exception as e:
                    print(f"  渲染出错: {e}")
            
            obs = next_obs
        
        # 回合结束统计
        print(f"\n第 {episode + 1} 回合结束 - 总步数: {step}")
        print("各智能体累积奖励:")
        total_reward = 0
        adversary_rewards = []
        for agent_id, rew in episode_rewards.items():
            agent_type = "追捕者" if "adversary" in agent_id else "逃跑者"
            print(f"  {agent_id} ({agent_type}): {rew:.2f}")
            total_reward += rew
            if "adversary" in agent_id:
                adversary_rewards.append(rew)
        
        print(f"追捕者平均奖励: {np.mean(adversary_rewards):.2f}")
        print(f"总奖励: {total_reward:.2f}")
    
    if save_images:
        print(f"\n所有图片已保存到: {render_dir}/")
        print("提示: 可以使用以下命令制作GIF动画:")
        print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{render_dir}/*.png' -vf 'palettegen' palette.png")
        print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{render_dir}/*.png' -i palette.png -filter_complex 'paletteuse' output.gif")


if __name__ == '__main__':
    # 设置渲染模式
    # 'live' - 仅实时显示
    # 'save' - 仅保存图片
    # 'gif' - 仅生成GIF
    # 'both' - 实时显示+生成GIF（推荐）
    DISPLAY_MODE = 'live'  # 可以修改这里选择模式
    
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 模型存储路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    
    # 加载模型的时间戳（修改为你的模型时间戳）
    load_timestamp = "2026-01-29_21-20" 
    model_timestamp = None if load_timestamp == '' else load_timestamp
    
    # 定义参数
    args = main_parameters()
    args.render_mode = "None"  # 不使用pygame渲染
    args.episode_num = 5  # 评估3个回合
    
    print(f"\n加载模型时间戳: {load_timestamp}")
    print(f"渲染模式: {DISPLAY_MODE}")
    print("="*60)
    print("模式说明:")
    print("  'live' - 实时动画显示（不保存文件）")
    print("  'save' - 保存PNG图片序列")
    print("  'gif'  - 直接生成GIF动画")
    print("  'both' - 实时显示 + 生成GIF（推荐）")
    print("="*60)
    
    # 创建环境
    env, dim_info, action_bound = get_env(args.env_name, args.episode_length, args.render_mode)
    
    # 创建MA-DDPG智能体
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, 
                   action_bound, _chkpt_dir=chkpt_dir, _device=device, _model_timestamp=model_timestamp)
    
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
    
    print('\n--- 开始评估（使用matplotlib渲染）---')
    env.reset()
    
    # 使用新的实时渲染评估函数
    evaluate_with_matplotlib_live(
        agent, env, args, device, 
        num_episodes=5, 
        display_mode=DISPLAY_MODE,  # 使用设定的模式
        save_gif=True,  # 是否保存GIF
        fps=10  # GIF帧率
    )
    
    print('\n--- 评估完成 ---')
    
    if DISPLAY_MODE in ['gif', 'both']:
        print("\n提示: GIF文件已生成，可以直接打开查看！")
    if DISPLAY_MODE == 'live':
        print("\n提示: 如需保存GIF，请将DISPLAY_MODE改为'gif'或'both'")
    if DISPLAY_MODE == 'save':
        print("\n提示: 可以使用图片查看器查看保存的PNG序列")
