import os
from copy import deepcopy
from typing import List

import torch 
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from agents.maddpg.NN_actor import MLPNetworkActor
from agents.maddpg.NN_critic import MLPNetworkCritic

class DDPG():
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, action_bound,  chkpt_dir, chkpt_name):

        # 1. 实例化 Actor（策略网络）：输入局部观测 obs，输出动作 action
        self.actor = MLPNetworkActor(in_dim=obs_dim, out_dim=act_dim, hidden_dim = 64, action_bound=action_bound, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'actor.pth')).to(device)

        # 2. 实例化 Critic（价值网络）：输入全局信息（所有人的obs+所有人的act），输出评分 Q值
        self.critic = MLPNetworkCritic(in_dim=global_obs_dim, out_dim=1, hidden_dim = 64, chkpt_dir = chkpt_dir, chkpt_name = (chkpt_name + 'critic.pth')).to(device)

        # 3. 定义优化器 (Adam)
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = critic_lr)
        
        # # 3.1 添加学习率调度器（参考UAV项目的优化策略）
        # # Actor: 每1000步衰减到原来的80%，帮助训练后期稳定
        # # Critic: 每5000步衰减到原来的33%，价值网络更新更保守
        # from torch.optim.lr_scheduler import StepLR
        # self.actor_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=0.8)
        # self.critic_scheduler = StepLR(self.critic_optimizer, step_size=5000, gamma=0.33)

        # 4. 创建 Target Network (目标网络)
        """
        使用 deepcopy 创建 target 网络是一个更好的选择，原因如下：
        初始化一致性：
            - deepcopy 确保 target 网络和原网络完全相同的初始参数
            - 重新创建网络可能因为随机初始化导致参数不一致
        """
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        
        # 5. 探索噪声参数（参考UAV项目）
        self.device = device
        self.act_dim = act_dim
        self.max_noise = 0.75  # 初始噪声幅度
        self.min_noise = 0.01  # 最小噪声幅度
        self.noise_decay_rate = 0.999995  # 噪声衰减率

    # 前向传播接口 (action & critic_value)
    def action(self, obs, model_out = False):
        # 其中没有用到logi, 接受其返回值第二项为 '_' 具体地:  a, _ = self.agents[agent].action(o)
        # 调用 Actor 网络，返回动作和 logits
        action, logi = self.actor(obs)
        return action, logi

    def target_action(self,obs):
        # 调用 Target Actor 网络（用于计算 next_action，计算 TD Error 用）
        action, logi = self.target_actor(obs)
        return action, logi
    
    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):  # 包含Tensor对象的列表
        # [核心] 拼接操作
        # 将所有智能体的状态列表和动作列表拼接在一起
        # dim=1 表示在特征维度拼接 (Batch, Feature_1 + Feature_2 + ...)
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # # 压缩维度，确保输出是 [batch_size]
    
    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length
    
    def update_actor(self, loss):
        self.actor_optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播计算梯度
        # [关键技术] 梯度裁剪 (Gradient Clipping)
        # 防止梯度爆炸，数值设为 0.5
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # clip_grad_norm_ ：带有下划线后缀，表示这是一个就地操作，会直接修改传入的参数梯度。
        self.actor_optimizer.step() # 更新参数
        # self.actor_scheduler.step() # 学习率调度
    
    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # clip_grad_norm_ ：带有下划线后缀，表示这是一个就地操作，会直接修改传入的参数梯度。
        self.critic_optimizer.step()
        # self.critic_scheduler.step() # 学习率调度
