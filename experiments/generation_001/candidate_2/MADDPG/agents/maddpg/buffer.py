import numpy as np
import torch

class BUFFER():
    
    # 内存预分配 (__init__)
    def __init__(self,capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        # [优点] 直接在内存中开辟好固定大小的数组，避免动态 append 带来的内存重新分配开销
        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype = bool)
        self._index = 0
        self._size = 0
        self.device = device

    # 环形存储机制(add)
    def add(self,obs, action, reward, next_obs, done):
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        # [核心] 环形指针：如果存满了，新的数据会覆盖最旧的数据
        self._index = (self._index +1) % self.capacity
        if self._size < self.capacity:
            self._size += 1


    # 采样逻辑 (sample)
    def sample(self, indices):
        # 直接使用传入的 indices 进行切片索引
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # [关键] 转换为 Tensor 并移动到 GPU
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size
        
        return obs, action, reward, next_obs, done

    def __len__(self):  #保留方法
        return self._size
        