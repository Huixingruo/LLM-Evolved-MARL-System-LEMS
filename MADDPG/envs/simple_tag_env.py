"""
追捕逃逸环境 (Simple Tag Environment)
"""

import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld
from . import reward_function  # 可插拔的奖励函数（仅包含追捕者奖励）


class Custom_raw_env(SimpleEnv, EzPickle):

    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=100,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        world_size=2.5,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, _world_size=world_size)

        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

        # 核心物理属性定义（LLM需要理解）

        self.world_size = world_size
        self.max_force = 1.0
        self.capture_threshold = self.world_size * 0.2

        # 轨迹记录（用于日志分析，不用于渲染）
        self.history_positions = {agent.name: [] for agent in world.agents}

        # 奖励分量记录
        self.last_reward_components = {agent.name: {} for agent in world.agents}
        self.current_actions = {}

        # 关联world和环境实例
        world._env_instance = self

        # 初始化空间
        self._init_spaces()

    def _init_spaces(self):
        """
        初始化动作空间和观测空间

        空间结构说明：
        - 动作空间：Box(2,) for continuous, Discrete(5) for discrete
        - 观测空间：Box(n,) 包含自身状态、相对位置等
        """
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0

        for agent in self.world.agents:
            # 计算动作空间维度
            if agent.movable:
                if self.continuous_actions:
                    space_dim = self.world.dim_p  # 2
                else:
                    space_dim = self.world.dim_p * 2 + 1  # 5
            else:
                space_dim = 1

            # 通信动作
            if agent.silent == False:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            # 观测空间维度
            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim

            # 定义动作空间
            if self.continuous_actions:
                self.action_spaces[agent.name] = gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(space_dim,), dtype=np.float32
                )
            else:
                self.action_spaces[agent.name] = gymnasium.spaces.Discrete(space_dim)

            # 定义观测空间
            self.observation_spaces[agent.name] = gymnasium.spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # 定义状态空间
        self.state_space = gymnasium.spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """重置环境状态"""
        super().reset(seed=seed, options=options)
        self.history_positions = {agent.name: [] for agent in self.world.agents}
        self.last_reward_components = {agent.name: {} for agent in self.world.agents}
        self.current_actions = {}

    def reset_world(self, world, np_random):
        """重置世界状态"""
        self.history_positions = {agent.name: [] for agent in self.world.agents}
        self.last_reward_components = {agent.name: {} for agent in self.world.agents}
        self.current_actions = {}
        super().scenario.reset_world(world, np_random)

    def _execute_world_step(self):
        """
        执行物理步进

        物理引擎接口说明：
        1. 接收所有智能体的动作
        2. 将动作转换为物理力
        3. 调用World.step()执行动力学积分
        4. 计算奖励

        动作空间结构：
        - 连续动作: Box(2,) -> [Fx, Fy]
        - 离散动作: Discrete(5) -> [静止, 左, 右, 下, 上]
        """
        # 构建以agent.name为键的动作字典
        actions_by_name = {}

        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            mdim = self.world.dim_p if self.continuous_actions else self.world.dim_p * 2 + 1

            # 提取物理动作向量（用于奖励函数）
            if agent.movable:
                if self.continuous_actions:
                    actions_by_name[agent.name] = action[0:mdim]
                else:
                    action_vec = np.zeros(self.world.dim_p)
                    discrete_action = action % mdim
                    if discrete_action == 1:
                        action_vec[0] = -1.0
                    elif discrete_action == 2:
                        action_vec[0] = +1.0
                    elif discrete_action == 3:
                        action_vec[1] = -1.0
                    elif discrete_action == 4:
                        action_vec[1] = +1.0
                    actions_by_name[agent.name] = action_vec
            else:
                actions_by_name[agent.name] = np.zeros(self.world.dim_p)

            # 构建scenario_action
            if agent.movable:
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)

            self._set_action(scenario_action, agent, self.action_spaces[agent.name], time=None)

        # 保存动作字典
        self.current_actions = actions_by_name

        self.world.step()

        # 计算奖励
        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward
            self.rewards[agent.name] = reward

    def _set_action(self, action, agent, action_space, time=None):
        """
        设置智能体动作

        将动作转换为物理力：
        - 连续动作：直接作为力向量
        - 离散动作：转换为方向向量
        """
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                agent.action.u[0] = action[0][0]
                agent.action.u[1] = action[0][1]
            else:
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0

        # 力限幅
        agent.action.u = np.clip(agent.action.u, -self.max_force, self.max_force)

    def step(self, action):
        """
        环境步进

        流程：
        1. 接收智能体动作
        2. 累积动作，当所有智能体动作齐备时执行物理步进
        3. 执行物理引擎
        4. 检查捕获条件
        5. 计算奖励
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()

            # 记录轨迹（用于日志分析）
            for agent in self.world.agents:
                self.history_positions[agent.name].append(agent.state.p_pos.copy())

            self.steps += 1
            self.check_capture_condition(threshold=self.capture_threshold)

            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()


    def check_capture_condition(self, threshold=None):
        """
        检查捕获条件

        判定规则：
        当所有追捕者都进入逃跑者的捕获范围内时，判定捕获成功
        """
        if threshold is None:
            threshold = self.world_size * 0.2

        agents = self.scenario.good_agents(self.world)
        adversaries = self.scenario.adversaries(self.world)

        for agent in agents:
            captured = all(
                np.linalg.norm(agent.state.p_pos - adv.state.p_pos) < threshold
                for adv in adversaries
            )
            if captured:
                for a in self.agents:
                    self.terminations[a] = True



    def render(self, mode=None):
        """
        渲染方法（matplotlib支持）

        Args:
            mode: 渲染模式（支持 None 和 "human"）

        Returns:
            numpy.ndarray: RGB图像数组（如果需要）
        """
        if mode is None:
            mode = self.render_mode

        if mode is None:
            return None

        # 返回简单的状态信息
        if mode == "human":
            print(f"Step: {self.steps}, Positions: {[a.state.p_pos for a in self.world.agents]}")
            return None

        return None

    def render_matplotlib(self):
        """
        使用matplotlib渲染环境

        Returns:
            numpy.ndarray: RGBA图像数组，可用于保存或显示
        """
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as agg
        from matplotlib.patches import Circle

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(8, 8)
        ax = plt.gca()

        # 设置坐标轴范围
        cam_range = self.world_size
        ax.set_xlim(-cam_range, cam_range)
        ax.set_ylim(-cam_range, cam_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 绘制捕获圈
        for agent in self.scenario.good_agents(self.world):
            circle = Circle(agent.state.p_pos, self.capture_threshold,
                          color='green', fill=False, linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(circle)

        # 绘制轨迹
        for agent in self.world.agents:
            if len(self.history_positions[agent.name]) >= 2:
                trajectory = np.array(self.history_positions[agent.name])
                color = 'blue' if agent.adversary else 'red'
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=color, alpha=0.3, linewidth=1.5)

        # 绘制智能体
        for entity in self.world.entities:
            pos = entity.state.p_pos
            if isinstance(entity, Agent):
                color = 'blue' if entity.adversary else 'red'
                circle = Circle(pos, entity.size, color=color, alpha=0.7)
                ax.add_patch(circle)
                circle_border = Circle(pos, entity.size, color='white',
                                     fill=False, linewidth=1.5)
                ax.add_patch(circle_border)
            else:
                circle = Circle(pos, entity.size, color='gray', alpha=0.5)
                ax.add_patch(circle)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Multi-Agent Pursuit - Step: {self.steps}')

        # 转换为RGBA数组
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)

        return image


env = make_env(Custom_raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2, _world_size=2.5):
        """
        核心物理常量（LLM需要理解）
        | 常量名           | 默认值      | 物理含义                    |
        |-----------------|-------------|----------------------------|
        | world_size      | 2.5         | 世界边界 (±2.5)            |
        | dim_p           | 2           | 位置空间维度 (x, y)         |
        | dim_c           | 0           | 通信通道维度（无通信）       |
        | dt              | 0.1         | 物理模拟时间步长            |
        | damping         | 0.2         | 速度阻尼系数                |
        | capture_threshold | 0.5       | 围捕判定距离 (world_size*0.2) |
        | max_force       | 1.0         | 最大作用力                  |
        | max_speed       | 1.0或1.3     | 智能体最大速度              |
        | agent_size      | 0.15        | 智能体半径                  |
        
        创建世界

        智能体配置：
        - 追捕者 (adversary): num_adversaries 个
        - 逃跑者 (agent): num_good 个
        """
        world = CustomWorld()
        world.world_size = _world_size
        world.dim_c = 0
        world.dim_p = 2
        world.dt = 0.1
        world.damping = 0.2

        num_agents = num_adversaries + num_good
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True

            base_size = _world_size * 0.1
            agent.size = base_size * 0.6
            agent.initial_mass = 0.8
            agent.accel = None
            agent.max_speed = 1.0 if agent.adversary else 1.3

        world.landmarks = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        return world

    def reset_world(self, world, np_random):
        """重置世界状态"""
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(
                -world.world_size * 0.9, +world.world_size * 0.9, world.dim_p
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(
                    -world.world_size * 0.8, +world.world_size * 0.8, world.dim_p
                )
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        """基准数据（用于评估）"""
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        return 0

    def is_collision(self, agent1, agent2):
        """检测碰撞"""
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        """返回逃跑者列表"""
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """返回追捕者列表"""
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        """
        奖励函数入口

        奖励设计说明：
        - 追捕者奖励：使用可插拔的 reward_function.py
        - 逃跑者奖励：在此文件中实现
        """
        global_state = self._build_global_state(agent, world)

        env_instance = getattr(world, '_env_instance', None)
        actions = getattr(env_instance, 'current_actions', {}) if env_instance is not None else {}

        if agent.adversary:
            # 追捕者奖励：使用可插拔奖励函数
            total_reward, components = reward_function.compute_reward(
                agent.name,
                None,
                global_state,
                actions,
                world
            )
        else:
            # 逃跑者奖励：在环境中实现
            total_reward, components = self._agent_reward(agent, global_state)

        if env_instance is not None:
            env_instance.last_reward_components[agent.name] = components

        return total_reward

    def _build_global_state(self, agent, world):
        """
        构建全局状态信息

        global_state 包含：
        - agent_positions: 所有智能体位置 (n_agents, 2)
        - agent_velocities: 所有智能体速度 (n_agents, 2)
        - prey_position: 逃跑者位置 (2,)
        - prey_velocity: 逃跑者速度 (2,)
        - distances_to_prey: 每个追捕者到逃跑者的距离
        - inter_agent_distances: 智能体间距离矩阵
        - is_adversary: 当前是否为追捕者
        - adversary_indices: 追捕者索引
        - prey_indices: 逃跑者索引
        - world_size: 世界大小
        - capture_threshold: 围捕阈值
        """
        adversaries = self.adversaries(world)
        preys = self.good_agents(world)
        all_agents = world.agents

        agent_positions = np.array([a.state.p_pos for a in all_agents])
        agent_velocities = np.array([a.state.p_vel for a in all_agents])

        if len(preys) > 0:
            prey_position = preys[0].state.p_pos
            prey_velocity = preys[0].state.p_vel
        else:
            prey_position = np.zeros(2)
            prey_velocity = np.zeros(2)

        distances_to_prey = np.array([
            np.linalg.norm(adv.state.p_pos - prey_position)
            for adv in adversaries
        ])

        n_agents = len(all_agents)
        inter_agent_distances = np.zeros((n_agents, n_agents))
        for i, agent_i in enumerate(all_agents):
            for j, agent_j in enumerate(all_agents):
                if i != j:
                    inter_agent_distances[i][j] = np.linalg.norm(
                        agent_i.state.p_pos - agent_j.state.p_pos
                    )

        adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
        prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

        global_state = {
            'agent_positions': agent_positions,
            'agent_velocities': agent_velocities,
            'prey_position': prey_position,
            'prey_velocity': prey_velocity,
            'distances_to_prey': distances_to_prey,
            'inter_agent_distances': inter_agent_distances,
            'is_adversary': agent.adversary,
            'adversary_indices': adversary_indices,
            'prey_indices': prey_indices,
            'world_size': world.world_size,
            'capture_threshold': world.world_size * 0.2,
        }

        return global_state

    def _agent_reward(self, agent, global_state):
        """
        逃跑者奖励函数（固定不变）

        核心策略：最大化与最近追捕者的距离 + 避免出界

        奖励分量：
        - escape_reward: 逃逸奖励（远离威胁）
        - capture_penalty: 被捕获惩罚
        - boundary_penalty: 边界惩罚
        """
        rew = 0
        components = {}

        world_size = global_state['world_size']
        capture_threshold = global_state['capture_threshold']

        agent_positions = global_state['agent_positions']
        adversary_indices = global_state['adversary_indices']
        prey_indices = global_state['prey_indices']

        agent_idx = prey_indices[0]
        agent_pos = agent_positions[agent_idx]

        adversary_positions = agent_positions[adversary_indices]

        # 1. 逃逸奖励
        escape_reward = 0.0
        capture_penalty = 0.0

        if len(adversary_indices) > 0:
            dists = [np.linalg.norm(adversary_positions[i] - agent_pos) for i in range(len(adversary_indices))]
            min_dist = min(dists)

            fear_radius = world_size * 0.5

            if min_dist < fear_radius:
                escape_reward = -2.0 * (fear_radius - min_dist)
            else:
                escape_reward = 0.1

            rew += escape_reward
            components['escape_reward'] = escape_reward

            # 2. 被捕获惩罚
            if min_dist < capture_threshold:
                capture_penalty = -10.0
                rew += capture_penalty
                components['capture_penalty'] = capture_penalty

        # 3. 边界惩罚
        boundary_penalty_total = 0.0
        for p in range(2):
            x = abs(agent_pos[p])
            boundary_penalty = self._calculate_bound_penalty(x, world_size)
            boundary_penalty_total += boundary_penalty

        rew -= boundary_penalty_total
        components['boundary_penalty'] = -boundary_penalty_total

        return rew, components

    def _calculate_bound_penalty(self, x, world_size):
        """计算边界惩罚"""
        boundary_start = world_size * 0.96
        full_boundary = world_size

        if x < boundary_start:
            return 0
        if x < full_boundary:
            return (x - boundary_start) * 10
        return min(np.exp(2 * x - 2 * full_boundary), 10)

    def observation(self, agent, world):
        """
        观测函数

        观测空间结构：
        [自身速度(2), 自身位置(2), 地标相对位置(2*num_obstacles),
         其他智能体相对位置(2*n_others), 逃跑者速度(2)]

        归一化说明：
        - 位置归一化到 [-1, 1]
        - 速度归一化到 [-1, 1]
        """
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                relative_entity_pos = (entity.state.p_pos - agent.state.p_pos) / world.world_size
                entity_pos.append(relative_entity_pos)

        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
            other_pos.append(relative_pos)
            if not other.adversary:
                norm_vel = other.state.p_vel / other.max_speed
                other_vel.append(norm_vel)

        norm_self_vel = agent.state.p_vel / world.world_size
        norm_self_pos = agent.state.p_pos / world.world_size

        return np.concatenate(
            [norm_self_vel]
            + [norm_self_pos]
            + entity_pos
            + other_pos
            + other_vel
        )


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Custom_raw_env (无渲染版本)")
    print("=" * 60)

    num_good = 1
    num_adversaries = 3
    num_obstacles = 0

    env = Custom_raw_env(
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        continuous_actions=True,
        render_mode=None
    )

    env.reset()

    print(f"\n环境名称: {env.metadata['name']}")
    print(f"智能体数量: {len(env.agents)}")
    print(f"世界大小: {env.world_size}")
    print(f"捕获阈值: {env.capture_threshold}")
    print(f"最大力: {env.max_force}")

    for agent_name in env.agents:
        obs_space = env.observation_space(agent_name)
        action_space = env.action_space(agent_name)
        observation = env.observe(agent_name)

        print(f"\n==== {agent_name} ====")
        print(f"观测空间: {obs_space.shape}")
        print(f"动作空间: {action_space.shape}")

    # 测试一步
    print("\n" + "-" * 40)
    print("测试环境步进...")
    for agent_name in env.agents:
        action = env.action_space(agent_name).sample()
        env.step(action)

    print(f"完成1步，位置: {[a.state.p_pos for a in env.agents]}")

    print("\n" + "=" * 60)
    print("✅ 环境测试通过！")
    print("=" * 60)
