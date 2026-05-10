import numpy as np
import visdom
import csv
import os
import threading
from datetime import datetime
from .reward_logger import (
    RewardComponentLogger, 
    compute_encirclement_angle_std,
    compute_formation_quality
)

class RUNNER:

    # 这部分负责把外部传进来的 Agent 和 Env 绑定到 Runner 上，并做一些准备工作。

    def __init__(self, agent, env, par, device, mode = 'evaluate'):
        self.agent = agent
        self.env = env
        self.par = par

        # [重点讲解 1] 静态智能体列表

        # 这里为什么新建而不是直接使用用agent.agents.keys()？
        # 因为pettingzoo中智能体死亡，这个字典就没有了，会导致 td target更新出错。所以这里维护一个不变的字典。
        # PettingZoo特性：如果一个智能体在仿真中“死亡”或“被移除”，它会从 env.agents 中消失。
        # 但我们需要计算它的损失函数或更新它的网络，所以必须维护一个包含所有（活着+死了）智能体的固定列表。
        self.env_agents = [agent_id for agent_id in self.agent.agents.keys()]
        self.done = {agent_id: False for agent_id in self.agent.agents.keys()}

        # 添加奖励记录相关的属性
        self.reward_sum_record = []  # 用于平滑的奖励记录
        self.all_reward_record = []  # 保存所有奖励记录，用于最终统计
        self.all_adversary_avg_rewards = []  # 追捕者平均奖励
        self.all_sum_rewards = []  # 所有智能体总奖励
        self.episode_rewards = {}  # 每个智能体的奖励历史
        
        # ========== 新增：成功围捕统计 ==========
        self.capture_success_count = 0  # 成功围捕的回合数
        self.total_episodes = 0  # 总回合数
        self.capture_success_record = []  # 每个回合是否成功围捕（True/False）
        self.episode_steps_record = []  # 每个回合的步数
        
        # ========== 新增：奖励分量日志记录器 ==========
        self.reward_logger = RewardComponentLogger(log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs'))

        # [重点讲解 2] 模型设备迁移

        # 将 agent 的模型放到指定设备上
        # 确保 Actor(策略网络) 和 Critic(价值网络) 都在 GPU/CPU 上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.target_actor.to(device)
            agent.critic.to(device)
            agent.target_critic.to(device)
        '''
        解决使用visdom过程中，输出控制台阻塞的问题。
        ''' #TODO

        # [辅助工具] Visdom 是一个可视化工具，类似 TensorBoard，用于实时画图

        if mode == 'train' and self.par.visdom:
            self.viz = visdom.Visdom()
            self.viz.close()
        else: # evaluate模式下不需要visdom
            pass


    def train(self):
        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        # reward_sum_record = []
        # # 存储csv数据
        # all_adversary_avg_rewards = []  # 记录每轮episode的追捕者的平均奖励
        # all_sum_rewards = []  # 记录每轮episode的所有智能体的奖励和

        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}

        # ========== 新增：提前终止机制相关变量 ==========
        early_stop_window = 200  # 每次计算平均围捕率的窗口大小
        early_stop_threshold = 0.90  # 围捕率阈值（90%）
        early_stop_consecutive = 3  # 需要连续达标的次数
        consecutive_count = 0  # 当前连续达标次数
        early_stopped = False  # 是否提前终止标志
        
        # 记录初始学习率
        initial_actor_lr = self.par.actor_lr
        initial_critic_lr = self.par.critic_lr

        # [大循环] 遍历每一个 Episode (例如训练 10000 轮)
        # episode循环
        for episode in range(self.par.episode_num):
            # ========== 新增：学习率衰减（每1000轮衰减5%）==========
            if episode > 0 and episode % 1000 == 0:
                decay_factor = 0.95 ** (episode // 1000)
                current_actor_lr = initial_actor_lr * decay_factor
                current_critic_lr = initial_critic_lr * decay_factor
                
                # 更新所有智能体的学习率
                for agent_id, agent in self.agent.agents.items():
                    for param_group in agent.actor_optimizer.param_groups:
                        param_group['lr'] = current_actor_lr
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] = current_critic_lr
                
                print(f"\n[学习率衰减] Episode {episode}: Actor LR = {current_actor_lr:.6f}, Critic LR = {current_critic_lr:.6f}")
            
            # print(f"This is episode {episode}")
            # 1. 重置环境，拿到初始观测值 obs
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}
            
            # ========== 新增：记录当前episode的步数 ==========
            episode_step_count = 0

            # [内循环] 只要环境里还有智能体（游戏没结束），就一直循环步数 (Step)
            # 每个智能体与环境进行交互
            while self.env.agents:  #  加入围捕判断
                # print(f"While num:{step}")
                step += 1
                episode_step_count += 1  # 记录当前episode的步数

                # 2. 动作选择策略 (Exploration vs Exploitation)
                # 收集经验。未到学习阶段 所有智能体随机选择动作 动作同样为字典 键为智能体名字 值为对应的动作 这里为随机选择动作
                # 热身阶段 (Random Steps)：为了填满经验池，先随机乱跑，不通过网络决策
                if step < self.par.random_steps:
                    action = {agent_id: self.env.action_space(agent_id).sample() for agent_id in self.env.agents}
                # 开始学习 根据策略选择动作
                # 正式学习阶段：使用 Actor 网络选择动作（带探索噪声）
                else:
                    action = self.agent.select_action(obs, training_step=step, evaluate=False)  # 添加training_step和evaluate参数

                # 3. 执行动作，环境反馈
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                # next_obs: 新的状态
                # reward: 获得的奖励
                # terminated/truncated: 是否结束/截断
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 4. [关键] 存储经验 (Experience Replay)
                # 将 (s, a, r, s', done) 存入 ReplayBuffer
                self.agent.add(obs, action, reward, next_obs, self.done)

                # 计算当前episode每个智能体的奖励 每个step求和
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # ========== 新增：记录奖励分量（每步都记录）==========
                # 从环境中获取奖励分量并记录
                # 注意：parallel_env包装器需要通过aec_env访问底层环境
                raw_env = getattr(self.env, 'aec_env', self.env)  # 获取底层环境
                
                if hasattr(raw_env, 'last_reward_components'):
                    for agent_id in self.env_agents:
                        if agent_id in raw_env.last_reward_components:
                            components = raw_env.last_reward_components[agent_id]
                            if len(components) > 0:  # 确保有数据
                                self.reward_logger.record_step(agent_id, components)
                
                # ========== 新增：记录协同行为指标 ==========
                # 计算并记录协同指标
                if step % 10 == 0:  # 每10步记录一次协同指标，避免过于频繁
                    self._record_collaboration_metrics()
                
                # 5. 模型更新
                # 不是每一步都更新，而是每隔 learn_interval 步更新一次，且要在热身结束后
                # 开始学习 有学习开始条件 有学习频率
                if step >= self.par.random_steps and step % self.par.learn_interval == 0:
                    # # 学习
                    # 从 Buffer 采样并进行梯度下降
                    self.agent.learn(self.par.batch_size, self.par.gamma)
                    # 更新网络
                    # 软更新 Target Network (这是 DDPG/MADDPG 稳定的关键)
                    self.agent.update_target(self.par.tau)
                
                # 状态更新，准备进入下一步
                obs = next_obs

            # ========== 新增：检测是否是围捕成功 ==========
            # terminated: 围捕成功（所有追捕者都在范围内）
            # truncated: 达到最大步数
            is_capture_success = False
            if len(self.env_agents) > 0:
                # 检查是否有任何一个智能体是因为terminated而结束的（围捕成功）
                # 注意：在PettingZoo中，当环境结束时，self.env.agents会变空
                # 我们需要检查最后一步的terminated状态
                is_capture_success = any(terminated.get(agent_id, False) for agent_id in self.env_agents)
            
            # 记录统计信息
            self.total_episodes += 1
            self.capture_success_record.append(is_capture_success)
            self.episode_steps_record.append(episode_step_count)
            if is_capture_success:
                self.capture_success_count += 1
            
            # 计算当前的成功率
            current_success_rate = (self.capture_success_count / self.total_episodes) * 100

            # 记录、绘制每个智能体在当前episode中的和奖励
            sum_reward = 0
            for agent_id, r in agent_reward.items():
                sum_reward += r
                if self.par.visdom:
                    self.viz.line(X=[episode + 1], Y=[r], win='sum reward of the agent ' + str(agent_id),
                             opts={'title': 'reward of the agent ' + str(agent_id) + ' in all episode'},
                             update='append')

            '''
                adversary_x:追捕者 
                agent_x:逃跑者
            '''# 绘制追捕者在当前episode的奖励和
            adversary_rewards_list = []
            for agent_id, r in agent_reward.items():
                if agent_id.startswith('adversary_'):        
                    adversary_rewards_list.append(r)
            # 计算围捕者的平均奖励
            avg_adversary_reward  =  np.mean(adversary_rewards_list)
            if self.par.visdom:
                self.viz.line(X=[episode + 1], Y=[avg_adversary_reward], win='adversary average reward',
                         opts={'title': 'Average reward of adversaries'},
                         update='append')
                
            # 记录当前episode围捕者的平均奖励
            self.all_adversary_avg_rewards.append(avg_adversary_reward)

            # 绘制所有智能体在当前episode的和奖励
            if self.par.visdom:
                self.viz.line(X=[episode + 1], Y=[sum_reward], win='Sum reward of all agents',
                         opts={'title': 'Sum reward of all agents in all episode'},
                         update='append')
                
            # 记录当前episode的所有智能体和奖励 存储到csv中
            self.all_sum_rewards.append(sum_reward)
            # 记录当前episode的所有智能体和奖励 为奖励平滑做准备
            self.reward_sum_record.append(sum_reward)

            self.all_reward_record.append(sum_reward)  # 保存完整记录
            # 保存当前智能体在当前episode的奖励
            for agent_id, r in agent_reward.items():
                self.episode_rewards[agent_id][episode] = r  #  episode_rewards  字典： {agent_id:[episoed1_reward, episode2_reward,...]}
            # 根据平滑窗口确定打印间隔 并进行平滑
            if (episode + 1) % self.par.size_win == 0:  #  500 步平滑一次
                message = f'episode {episode + 1}, '
                sum_reward = 0
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:>4f}; ' # r:>4f 是格式化字符串，用于保留四位小数。
                    sum_reward += r
                message += f'sum reward: {sum_reward}'
                print(message)
                
                # ========== 新增：打印成功围捕率统计 ==========
                # 计算最近size_win个回合的成功率
                recent_window = min(self.par.size_win, len(self.capture_success_record))
                recent_successes = sum(self.capture_success_record[-recent_window:])
                recent_success_rate = (recent_successes / recent_window) * 100 if recent_window > 0 else 0
                
                # 计算平均步数
                recent_avg_steps = np.mean(self.episode_steps_record[-recent_window:]) if recent_window > 0 else 0
                
                print(f"  [围捕统计] 总成功率: {current_success_rate:.1f}% ({self.capture_success_count}/{self.total_episodes}), "
                      f"最近{recent_window}回合: {recent_success_rate:.1f}% ({recent_successes}次), "
                      f"平均步数: {recent_avg_steps:.1f}")
                if self.par.visdom:
                    epi = np.linspace(episode - (self.par.size_win - 2),
                                      episode - (self.par.size_win - 2) + (self.par.size_win - 1), self.par.size_win,
                                      dtype=int)
                    self.viz.line(X=epi, Y=self.get_running_reward(self.reward_sum_record), win='Average sum reward',
                             opts={'title': 'Average sum reward'},
                             update='append')
                self.reward_sum_record = []
            
            # ========== 新增：提前终止检测（每200个episode检测一次）==========
            if (episode + 1) >= early_stop_window and (episode + 1) % early_stop_window == 0:
                # 计算最近200个episode的围捕成功率
                window_start = episode + 1 - early_stop_window
                window_successes = sum(self.capture_success_record[window_start:episode + 1])
                window_success_rate = window_successes / early_stop_window
                
                print(f"\n[提前终止检测] Episode {episode + 1}: 最近{early_stop_window}回合成功率 = {window_success_rate*100:.1f}% ({window_successes}/{early_stop_window})")
                
                if window_success_rate >= early_stop_threshold:
                    consecutive_count += 1
                    print(f"  达标次数: {consecutive_count}/{early_stop_consecutive}")
                    
                    if consecutive_count >= early_stop_consecutive:
                        early_stopped = True
                        print(f"\n{'='*60}")
                        print(f"[提前终止] 连续{early_stop_consecutive}次{early_stop_window}回合平均围捕率 >= {early_stop_threshold*100:.0f}%")
                        print(f"  训练提前结束于 Episode {episode + 1}")
                        print(f"  最终成功率: {current_success_rate:.1f}%")
                        print(f"{'='*60}\n")
                        break
                else:
                    consecutive_count = 0
                    print(f"  未达标，重置计数器")

        # 保存数据到文件（CSV格式）
        self.save_rewards_to_csv(self.all_adversary_avg_rewards, self.all_sum_rewards, early_stopped=early_stopped)
        
        # ========== 新增：保存奖励分量统计 ==========
        self.reward_logger.episode_count = self.total_episodes
        self.reward_logger.save_statistics()
        self.reward_logger.save_summary_report()
        print("\n" + "="*80)
        self.reward_logger.print_summary()
        print("="*80)

    def get_running_reward(self, arr):

        if len(arr) == 0:  # 如果传入空数组，使用完整记录
            arr = self.all_reward_record

        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.par.size_win
        running_reward = np.zeros_like(arr)

        # for i in range(window - 1):
        #     running_reward[i] = np.mean(arr[:i + 1])
        # for i in range(window - 1, len(arr)):
        #     running_reward[i] = np.mean(arr[i - window + 1:i + 1])
            # 确保不会访问超出数组范围的位置
        for i in range(len(arr)):
            # 对每个i，确保窗口大小不会超出数组的实际大小
            start_idx = max(0, i - window + 1)
            running_reward[i] = np.mean(arr[start_idx:i + 1])
        # print(f"running_reward{running_reward}")
        return running_reward

    @staticmethod
    def exponential_moving_average(rewards, alpha=0.1):
        """计算指数移动平均奖励"""
        ema_rewards = np.zeros_like(rewards)
        ema_rewards[0] = rewards[0]
        for t in range(1, len(rewards)):
            ema_rewards[t] = alpha * rewards[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards

    def moving_average(self, rewards):
        """计算简单移动平均奖励"""
        window_size = self.par.size_win
        sma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return sma_rewards
    
    """保存围捕者平均奖励和所有智能体总奖励到 CSV 文件"""
    def save_rewards_to_csv(self, adversary_rewards, sum_rewards, filename = None, early_stopped=False): # filename="data_rewards.csv"
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            # 如果提前终止，在文件名中标注
            early_stop_tag = "_early_stopped" if early_stopped else ""
            filename = f"data_rewards_{timestamp}{early_stop_tag}.csv"
        # 获取 runner.py 所在目录，并生成与 utils 同级的 plot 目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件（runner.py）的绝对路径
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')  # 获取与 utils 同级的 plot 文件夹
        os.makedirs(plot_dir, exist_ok=True)  # 创建 plot 目录（如果不存在）

        # 构造完整的 CSV 文件路径
        full_filename = os.path.join(plot_dir, filename)

        # ========== 新增：添加围捕统计列 ==========
        header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents', 
                  'Capture Success', 'Episode Steps']
        
        # 确保capture_success_record和episode_steps_record的长度与奖励记录一致
        capture_data = self.capture_success_record if len(self.capture_success_record) == len(adversary_rewards) else [False] * len(adversary_rewards)
        steps_data = self.episode_steps_record if len(self.episode_steps_record) == len(adversary_rewards) else [0] * len(adversary_rewards)
        
        data = list(zip(
            range(1, len(adversary_rewards) + 1), 
            adversary_rewards, 
            sum_rewards,
            capture_data,  # 是否成功围捕（True/False）
            steps_data  # 该回合的步数
        ))
        
        # 将数据写入 CSV 文件
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)  # 写入数据

        print(f"Rewards data saved to {full_filename}")
        print(f"  Total episodes: {len(adversary_rewards)}")
        print(f"  Successful captures: {sum(capture_data)} ({sum(capture_data)/len(capture_data)*100:.1f}%)")
        print(f"  Average episode steps: {np.mean(steps_data):.1f}")
    
    def _record_collaboration_metrics(self):
        """
        计算并记录协同行为指标
        """
        try:
            # 获取底层环境
            raw_env = getattr(self.env, 'aec_env', self.env)
            
            # 获取所有智能体的位置
            if not hasattr(raw_env, 'world') or not hasattr(raw_env.world, 'agents'):
                return
            
            world = raw_env.world
            
            # 分离追捕者和逃跑者
            adversaries = [agent for agent in world.agents if agent.adversary]
            preys = [agent for agent in world.agents if not agent.adversary]
            
            if len(adversaries) < 2 or len(preys) < 1:
                return
            
            # 获取位置
            adversary_positions = np.array([agent.state.p_pos for agent in adversaries])
            prey_position = preys[0].state.p_pos
            
            # 获取捕获阈值
            capture_threshold = getattr(raw_env, 'capture_threshold', world.world_size * 0.2)
            
            # 计算协同指标
            metrics = {}
            
            # 1. 围捕角度标准差
            metrics['encirclement_angle_std'] = compute_encirclement_angle_std(
                adversary_positions, prey_position
            )
            
            # 2. 智能体间最小距离
            min_dist = float('inf')
            for i in range(len(adversaries)):
                for j in range(i+1, len(adversaries)):
                    dist = np.linalg.norm(adversaries[i].state.p_pos - adversaries[j].state.p_pos)
                    min_dist = min(min_dist, dist)
            metrics['min_agent_distance'] = min_dist if min_dist != float('inf') else 0.0
            
            # 3. 到猎物的平均距离
            distances_to_prey = [np.linalg.norm(adv.state.p_pos - prey_position) for adv in adversaries]
            metrics['avg_distance_to_prey'] = np.mean(distances_to_prey)
            
            # 4. 队形质量
            metrics['formation_quality'] = compute_formation_quality(
                adversary_positions, prey_position, capture_threshold
            )
            
            # 记录指标
            self.reward_logger.record_collaboration_metrics(metrics)
            
        except Exception as e:
            # 静默失败，避免影响训练
            pass

#============================================================================================================

    def evaluate(self):

        # [差异点]
        # 1. 直接用 select_action，没有随机动作
        # 2. 没有 self.agent.add (不存数据)
        # 3. 没有 self.agent.learn (不更新网络)

        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        # step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        # episode循环
        for episode in range(self.par.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 每个智能体与环境进行交互
            while self.env.agents:
                # print(f"While num:{step}")
                step += 1
                # 使用训练好的智能体选择动作（没有随机探索）
                action = self.agent.select_action(obs, evaluate=True)  # 评估模式不添加噪声
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs

                
                if step % 10 == 0:
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)