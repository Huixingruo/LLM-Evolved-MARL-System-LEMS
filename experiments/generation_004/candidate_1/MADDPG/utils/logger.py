import os
import json
import torch
from datetime import datetime



"""
1. 在../logs/下保存了一个 training_log.json 文件，它包含了训练的所有参数和日志信息。
2. 保存的 plot_data_{current_time.replace(':', '-')}.pkl 是一个 PyTorch 保存的文件，它并不包含模型本身，而是 训练过程中的奖励数据。

"""
class TrainingLogger:
    def __init__(self, log_dir="../logs"):
        # 使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(current_dir,'..', 'logs')
        
        # 确保目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
    def save_training_log(self, args, device, training_duration, runner):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算最终的围捕统计
        final_success_rate = (runner.capture_success_count / runner.total_episodes * 100) if runner.total_episodes > 0 else 0
        avg_episode_steps = sum(runner.episode_steps_record) / len(runner.episode_steps_record) if runner.episode_steps_record else 0

        # 构建每回合的时序数据，供log_analyzer.py的收敛拦截器使用
        episodes_data = []
        for i, reward_sum in enumerate(runner.all_sum_rewards):
            episode_info = {
                "episode": i,
                "total_reward": float(reward_sum)
            }
            # 如果有捕获记录，也一并写入
            if i < len(runner.capture_success_record):
                episode_info["capture_success"] = bool(runner.capture_success_record[i])
            if i < len(runner.episode_steps_record):
                episode_info["steps"] = int(runner.episode_steps_record[i])
            episodes_data.append(episode_info)

        # 准备训练日志信息
        log_info = {
            "训练时间": current_time,
            "训练设备": str(device),
            "训练用时": training_duration,
            "环境名称": args.env_name,
            "渲染模式": args.render_mode,
            "总回合数": args.episode_num,
            "每回合步数": args.episode_length,
            "学习间隔": args.learn_interval,
            "随机步数": args.random_steps,
            "tau": args.tau,
            "gamma": args.gamma,
            "buffer容量": args.buffer_capacity,
            "batch_size": args.batch_size,
            "actor学习率": args.actor_lr,
            "critic学习率": args.critic_lr,
            "是否使用visdom": args.visdom,
            "visdom窗口大小": args.size_win,
            # ========== 新增：围捕统计 ==========
            "成功围捕次数": runner.capture_success_count,
            "成功围捕率": f"{final_success_rate:.2f}%",
            "平均回合步数": f"{avg_episode_steps:.1f}",
            # ========== 核心修复：写入每回合时序数据，供收敛拦截器使用 ==========
            "episodes": episodes_data
        }

        # 保存训练日志
        log_file = os.path.join(self.log_dir, "training_log.json")
        
        # 打印当前目录和目标目录
        print(f"Current directory: {os.getcwd()}")
        print(f"Saving training log to: {log_file}")

        # 确保目录存在并且具有写权限
        if os.path.exists(self.log_dir):
            print(f"Log directory exists: {self.log_dir}")
        else:
            print(f"Log directory does not exist. Trying to create it...")
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 读取现有的日志文件，如果存在的话
        existing_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
        
        existing_logs.append(log_info)
        
        # 保存更新后的日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, ensure_ascii=False, indent=4)

        # 保存训练曲线数据
        plot_data = {
            "all_sum_rewards": runner.all_sum_rewards,  # 所有智能体的总奖励
            "all_adversary_avg_rewards": runner.all_adversary_avg_rewards,  # 追捕者的平均奖励
            "episode_rewards": runner.episode_rewards,  # 每个智能体的奖励历史
            "running_rewards": runner.get_running_reward(runner.reward_sum_record),  # 平滑后的奖励
            "timestamps": current_time,
            # ========== 新增：围捕统计数据 ==========
            "capture_success_record": runner.capture_success_record,  # 每个回合是否成功围捕
            "episode_steps_record": runner.episode_steps_record,  # 每个回合的步数
            "capture_success_count": runner.capture_success_count,  # 总成功次数
            "capture_success_rate": final_success_rate  # 总成功率
        }
        
        plot_file = os.path.join(self.log_dir, f"plot_data_{current_time.replace(':', '-')}.pkl")
        torch.save(plot_data, plot_file)

        return current_time
