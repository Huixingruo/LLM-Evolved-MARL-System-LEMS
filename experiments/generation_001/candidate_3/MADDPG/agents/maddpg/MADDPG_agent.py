import os

import numpy as np
import torch
import torch.nn.functional as F
from agents.maddpg.DDPG_agent import DDPG
from agents.maddpg.buffer import BUFFER

class MADDPG():
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, _chkpt_dir, _device = 'cuda', _model_timestamp = None):
        # 确保模型保存路径存在
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)

        self.device = _device
        self.model_timestamp = _model_timestamp
        # [核心概念：上帝视角]
        # MADDPG 的精髓在于 Critic (评论家) 能看到所有人的状态和动作。
        # global_obs_act_dim 就是 Critic 的输入维度 = Sum(所有agent的obs) + Sum(所有agent的act)
        # 状态（全局观测）与所有智能体动作维度的和 即critic网络的输入维度  dim_info =  [obs_dim, act_dim]
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # 创建智能体与buffer，每个智能体有自己的buffer, actor, critic
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            # print("dim_info -> agent_id:",agent_id)
            # 每一个智能体都是一个DDPG智能体
            # 每一个智能体内部其实是一个 DDPG 算法
            # 注意参数：它不仅传了自己的 obs_dim，还传了 global_obs_act_dim 给 Critic 用
            self.agents[agent_id] = DDPG(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, action_bound[agent_id], chkpt_name = (agent_id + '_'), chkpt_dir = _chkpt_dir)
            # buffer均只是存储自己的观测与动作
            # Buffer 是独立的，每个智能体存自己的观察
            self.buffers[agent_id] = BUFFER(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info
        self.batch_size = batch_size

    # 经验存储与采样 (add & sample)

    def add(self, obs, action, reward, next_obs, done):
        #NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):  #返回值为True or False, 判断a是否为int类型，是，返回True。
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                # 神经网络无法直接处理整数类别(如动作0,1,2)，必须转为 One-Hot 编码 (如 [1,0,0], [0,1,0])
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)
    
    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        # 我们必须保证取出的数据是“同一时刻”发生的。
        # 不能 Agent A 取第1帧的数据，Agent B 取第10帧的数据，那样因果关系就乱了。
        indices = np.random.choice(total_num, size = batch_size, replace = False)
        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            # [关键] 计算 Target Action
            # 为了计算目标 Q 值，我们需要知道“下一步动作”是什么。
            # 这是通过 Target Actor 网络预测出来的。
            next_act[agent_id], _ = self.agents[agent_id].target_action(n_o)
        
        return obs, act, reward, next_obs, done, next_act
    
    def select_action(self, obs, training_step=0, evaluate=False):
        """
        选择动作（带探索噪声）
        Args:
            obs: 观测字典
            training_step: 当前训练步数（用于噪声衰减）
            evaluate: 是否为评估模式（评估时不添加噪声）
        """
        action = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            a, _ = self.agents[agent].action(o)   # torch.Size([1, action_size])
            a = a.squeeze(0).detach().cpu().numpy()
            
            # 添加探索噪声（参考UAV项目的噪声策略）
            if not evaluate:
                # 计算当前噪声幅度（指数衰减）
                agent_obj = self.agents[agent]
                noise_scale = max(agent_obj.min_noise, 
                                agent_obj.max_noise * (agent_obj.noise_decay_rate ** training_step))
                
                # 生成[-1, 1)范围的随机噪声
                
                noise = np.random.randn(agent_obj.act_dim)
                a = a + noise_scale * noise
                
                # 裁剪动作到环境允许的范围 [-1.0, 1.0]（修复：防止超出动作空间）
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
            
            # NOTE that the output is a tensor, convert it to int before input to the environment
            action[agent] = a
        return action

    def learn(self, batch_size, gamma):
        # 遍历每一个智能体，依次进行更新
        for agent_id, agent in self.agents.items():
            # 1. 获取所有人的数据 (因为 Critic 需要看所有人)
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # upate critic

            # ---------------- 更新 Critic (价值网络) ----------------
            # Critic 评估当前局面：输入所有人的 obs 和 act，输出 Q 值
            critic_value = agent.critic_value( list(obs.values()), list(act.values()) )

            # 计算 Target Q 值 (贝尔曼方程: y = r + gamma * Q_next)
            # 注意：这里用的也是所有人的 next_obs 和 next_act
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value* (1-done[agent_id])

            # 计算 Loss 并更新
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction = 'mean')
            agent.update_critic(critic_loss)

            #update actor
            # ---------------- 更新 Actor (策略网络) ----------------
            # 这里的目的是让 Critic 打分越高越好

            # 让当前智能体根据最新策略产生动作 (action)
            action, logits = agent.action(obs[agent_id], model_out = True)

            # 更新动作列表 (把旧的 sample 出来的动作换成最新的计算结果)
            act[agent_id] = action

            # Actor Loss = -Q (负号是因为我们要最大化 Q，而优化器是做梯度下降)
            actor_loss = - agent.critic_value( list(obs.values()), list(act.values()) ).mean()

            actor_loss_pse = torch.pow(logits, 2).mean()  #这个是干嘛的？

            # 更新 Actor，加上了正则化项
            agent.update_actor(actor_loss + 1e-3 *actor_loss_pse)


    # 目标网络软更新 (update_target)
    def update_target(self, tau): #  嵌套函数定义
        # 嵌套函数：专门用于把 src 网络的参数 慢慢 复制给 dest 网络
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)  #体现使用嵌套函数的作用！ 易于维护和使用
            soft_update(agent.critic, agent.target_critic)

    @classmethod
    def load( cls, dim_info, file):
        """ init maddpg using the model saved in `file` """
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location=instance.device)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
    
    def save_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_actor.save_checkpoint(is_target = True, timestamp = True)
            self.agents[agent_id].critic.save_checkpoint(is_target = False, timestamp = True)
            self.agents[agent_id].target_critic.save_checkpoint(is_target = True, timestamp = True)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组


    def load_model(self):
        for agent_id in self.dim_info.keys():
            self.agents[agent_id].actor.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_actor.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)
            self.agents[agent_id].critic.load_checkpoint(device = self.device, is_target = False, timestamp = self.model_timestamp)
            self.agents[agent_id].target_critic.load_checkpoint(device = self.device, is_target = True, timestamp = self.model_timestamp)

        agent_id = list(self.dim_info.keys())[0]  # 获取第一个代理的 ID
        agent = self.agents[agent_id]
        for name, param in agent.actor.state_dict().items():
        # 仅打印前几个值（例如前5个）
            print(f"Layer: {name}, Shape: {param.shape}, Values: {param.flatten()[:5]}")  # flatten() 展开参数为一维数组
  
