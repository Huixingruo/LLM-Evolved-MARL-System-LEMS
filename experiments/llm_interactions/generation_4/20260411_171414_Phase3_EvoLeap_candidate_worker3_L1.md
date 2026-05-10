# LLM Interaction Log

- **Generation**: 4
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260411_171414

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
你是一位专业的强化学习奖励工程师。请基于上一代的诊断反馈执行特定的变异操作。

# 环境基座

import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld
from . import reward_function  # 可插拔的奖励函数（仅包含追捕者奖励）

class CoreEnvLogic:
    """
    环境核心逻辑
    用于辅助设计 Reward Function
    """
    def __init__(self):
        # 核心物理常量
        self.world_size = 2.5          # 地图范围 (-2.5, 2.5)
        self.max_force = 1.0             # 动作最大值
        self.capture_threshold = 0.5     # 围捕判定阈值 (world_size * 0.2)
        
        # 智能体参数
        # size: 碰撞体积半径
        # max_speed: 逃跑者 1.3 > 追捕者 1.0
        self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
        self.agent_params = {'size': 0.050, 'max_speed': 1.3}

    def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
        """碰撞检测：欧氏距离 < 半径之和"""
        delta_pos = agent1_pos - agent2_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1_size + agent2_size
        return dist < dist_min

    def _build_global_state(self, agent, world):
        """
        【重要】传给 compute_reward 的 global_state 结构
        """
        all_agents = world.agents
        adversaries = [a for a in all_agents if a.adversary]
        preys = [a for a in all_agents if not a.adversary]
        
        agent_positions = np.array([a.state.p_pos for a in all_agents])
        agent_velocities = np.array([a.state.p_vel for a in all_agents])
        prey_pos = preys[0].state.p_pos if preys else np.zeros(2)
        prey_vel = preys[0].state.p_vel if preys else np.zeros(2)
        
        # 每个追捕者到猎物的距离
        distances_to_prey = np.array([np.linalg.norm(adv.state.p_pos - prey_pos) for adv in adversaries])
        
        # 智能体间距离矩阵 (用于防撞)
        n_agents = len(all_agents)
        inter_agent_distances = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                inter_agent_distances[i][j] = np.linalg.norm(agent_positions[i] - agent_positions[j])

        return {
            'agent_positions': agent_positions,
            'agent_velocities': agent_velocities,
            'prey_position': prey_pos,
            'prey_velocity': prey_vel,
            'distances_to_prey': distances_to_prey,
            'inter_agent_distances': inter_agent_distances,
            'is_adversary': agent.adversary,
            'world_size': self.world_size,
            'capture_threshold': self.capture_threshold
        }

    def observation(self, agent, world):
        """
        【重要】观测向量结构
        Return: np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
        """
        # 自身状态
        norm_self_vel = agent.state.p_vel / agent.max_speed
        norm_self_pos = agent.state.p_pos / self.world_size
        
        # 其他智能体相对位置
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
            other_pos.append(rel_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel / other.max_speed)
        
        return np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)


# 上一代最优代码
```python
import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # 物理与任务常量（硬编码）
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075

    # 奖励权重（可调）
    w_approach = 1.0           # 接近猎物
    w_center_radius = 1.0      # 整体圈半径小
    w_radius_var = 0.5         # 半径均匀性
    w_angle_uniform = 0.5      # 角度均匀性
    w_inside = 1.0             # 猎物在多边形内部
    w_time = 0.01              # 时间惩罚
    w_capture_bonus = 5.0      # 成功围捕 bonus
    w_too_close = 0.5          # 追捕者间过近惩罚

    # 从 global_state 中提取信息
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"],
        dtype=float,
    )

    all_agents = world.agents
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) == 0 or len(prey_indices) == 0:
        return 0.0, {}

    prey_idx = prey_indices[0]

    this_idx = None
    for i, a in enumerate(all_agents):
        if a.name == agent_name:
            this_idx = i
            break
    if this_idx is None:
        return 0.0, {}

    if not all_agents[this_idx].adversary:
        return 0.0, {}

    this_pos = agent_positions[this_idx]
    this_vel = agent_velocities[this_idx]

    # 1. 接近目标 & 包围整体收缩
    this_dist_to_prey = np.linalg.norm(this_pos - prey_pos)
    norm_dist = this_dist_to_prey / world_size
    approach_reward = -norm_dist
    components["approach_reward"] = w_approach * approach_reward

    if distances_to_prey.size > 0:
        max_team_dist = np.max(distances_to_prey)
        center_radius_reward = -(max_team_dist / world_size)
    else:
        center_radius_reward = 0.0
    components["center_radius_reward"] = w_center_radius * center_radius_reward

    # 2. 包围圈几何结构（半径方差 + 角度均匀）
    target_radius = capture_threshold
    if len(adversary_indices) >= 3:
        adv_positions = agent_positions[adversary_indices]

        radii = np.linalg.norm(adv_positions - prey_pos, axis=1)
        mean_radius = np.mean(radii)
        radius_var = np.var(radii)
        radius_uniform_reward = -radius_var / (world_size ** 2 + 1e-8)
        radius_scale_reward = -abs(mean_radius - target_radius) / (
            world_size + 1e-8
        )
        components["radius_uniform_reward"] = w_radius_var * radius_uniform_reward
        components["radius_scale_reward"] = 0.5 * w_radius_var * radius_scale_reward

        rel_pos = adv_positions - prey_pos
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.empty_like(angles_sorted)
        angle_diffs[:-1] = angles_sorted[1:] - angles_sorted[:-1]
        angle_diffs[-1] = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        n_adv = len(adversary_indices)
        ideal_diff = 2 * np.pi / n_adv
        angle_uniformity = np.mean((angle_diffs - ideal_diff) ** 2)
        angle_uniform_reward = -angle_uniformity / (np.pi ** 2 + 1e-8)
        components["angle_uniform_reward"] = w_angle_uniform * angle_uniform_reward

        def point_in_polygon(point, poly):
            x, y = point
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                cond1 = (y1 > y) != (y2 > y)
                if not cond1:
                    continue
                t = (y - y1) / (y2 - y1 + 1e-12)
                x_intersect = x1 + t * (x2 - x1)
                if x_intersect > x:
                    inside = not inside
            return inside

        prey_inside = point_in_polygon(prey_pos, adv_positions)
        inside_reward = 1.0 if prey_inside else 0.0
        components["inside_polygon_reward"] = w_inside * inside_reward
    else:
        components["radius_uniform_reward"] = 0.0
        components["radius_scale_reward"] = 0.0
        components["angle_uniform_reward"] = 0.0
        components["inside_polygon_reward"] = 0.0

    # 3. 防碰撞与队形安全距离
    too_close_penalty = 0.0

    for i in adversary_indices:
        if i == this_idx:
            continue
        dist_ij = inter_agent_distances[this_idx, i]
        min_safe_dist = 2 * adv_size

        margin = 0.2
        min_soft_dist = min_safe_dist + margin
        if dist_ij < min_soft_dist:
            too_close_penalty -= (min_soft_dist - dist_ij) / (
                min_soft_dist + 1e-8
            )

    components["too_close_penalty"] = w_too_close * too_close_penalty

    # 4. 时间惩罚 + 围捕成功 Bonus
    components["time_penalty"] = -w_time

    close_radius = capture_threshold
    capture_bonus = 0.0
    if len(adversary_indices) >= 3:
        all_close = np.all(distances_to_prey < close_radius)
        inside_polygon = components["inside_polygon_reward"] > 0.0
        if all_close and inside_polygon:
            capture_bonus = w_capture_bonus
    components["capture_bonus"] = capture_bonus

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
1. 分量贡献度  
- 完全为0：无（所有分量均有非零均值或方差）。  
- 方差过大：boundary_penalty（std 1.88）、escape_reward（0.75）、center_radius_reward（0.54）、approach_reward（0.46）、inside_polygon_reward（0.34）、capture_bonus（0.19）。这些分量符号不一、波动大，易在策略切换时引起震荡。  
- 主导作用：capture_penalty 均值 -10 远大于其他分量，整体奖励被强烈惩罚主导；同时多个空间相关惩罚（center_radius、escape、boundary、approach）整体为负，指向“求稳不敢压缩”的趋势。

2. 任务瓶颈  
- 成功率 44.7%，平均 81.8 步，说明能学会捕获但效率偏低，尚未形成稳定高效策略。  
- 多个空间惩罚长期为负且逃逸相关奖励为负，暗示智能体偏保守，可能在安全边界附近反复试探，难以果断收缩，存在“半包围、易被突破”的局部最优。

3. 协同缺陷  
- encirclement_angle_std=1.70、formation_quality=0.30 偏差大且队形质量较低，包围圈形状不稳定、难以形成均匀封锁。  
- min_agent_distance=0.87 不算极近，扎堆碰撞不是主问题，更像队形松散、有人落后。  
- avg_distance_to_prey=1.32 较大，结合低队形质量，疑似存在个体停滞或行动消极（搭便车），整体无法形成有序收缩包围圈。

# 强制变异指令
【Reward Paradigm Leap (范式跃迁)】
彻底抛弃原代码的设计思路！请从零开始构建一个全新的奖励函数（例如尝试全局势场、相对距离极坐标系等与原先完全不同的视角）。

# 接口规范
请实现 compute_reward 函数，保持接口签名不变：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state['is_adversary']: return 0.0, {}
    components = {}
    # [根据变异指令修改这里的核心逻辑]
    total_reward = sum(components.values())
    return total_reward, components
```

只允许输出1个Python代码块，严禁任何解释性文字，直接以 ```python 开头。

# 致命红线约束 (Anti-Hallucination Guardrails)
- **禁止引入外部依赖或未定义的类**：绝不可实例化 `CoreEnvLogic()` 或尝试访问 `world.logic`、`world.adversary_params` 等不存在属性。
- **物理常量硬编码**：若要新增依赖于物理常量的逻辑（如体积、速度），必须直接在函数内写死数值常量（如 `adv_size = 0.075`）。
- **严格保持接口签名**：必须保留 `components` 字典收集机制并返回 `total_reward, components`。
- 只允许输出1个修改后的Python代码块，严禁任何解释性文字，直接以 ```python 开头。
```

================================================================================
## Response (From LLM)
================================================================================

```text
```python
import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 基础常量（硬编码）
    # -----------------------
    world_radius = 2.5
    capture_radius = 0.5
    adv_size = 0.075
    prey_size = 0.05

    # 势场权重（可调）
    w_radial_attract = 1.5      # 径向吸引到目标圈
    w_tangential_align = 0.8    # 切向环绕/夹击
    w_team_ring = 1.0           # 团队环形成型
    w_gap_closure = 1.0         # 封堵最大缺口
    w_velocity_focus = 0.8      # 速度朝向目标圈
    w_collision_avoid = 1.0     # 追捕者间防撞
    w_boundary_safe = 0.5       # 地图边界安全
    w_time = 0.01               # 时间惩罚
    w_capture_bonus = 5.0       # 捕获成功奖励

    # -----------------------
    # 1. 从 global_state 中提取信息
    # -----------------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    all_agents = world.agents
    n_agents = len(all_agents)
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) == 0 or len(prey_indices) == 0:
        return 0.0, {}

    prey_idx = prey_indices[0]

    this_idx = None
    for i, a in enumerate(all_agents):
        if a.name == agent_name:
            this_idx = i
            break
    if this_idx is None:
        return 0.0, {}

    if not all_agents[this_idx].adversary:
        return 0.0, {}

    this_pos = agent_positions[this_idx]
    this_vel = agent_velocities[this_idx]

    # 相对猎物的极坐标
    rel_pos = this_pos - prey_pos
    dist_to_prey = np.linalg.norm(rel_pos) + 1e-8
    dir_radial = rel_pos / dist_to_prey

    # 切向方向（顺时针/逆时针都视为好，只看速度分量大小）
    dir_tangential = np.array([-dir_radial[1], dir_radial[0]])

    # -----------------------
    # 2. 个体势场：径向吸引 + 切向环绕
    # -----------------------
    # (a) 径向势场：鼓励靠近 capture_radius
    # 目标环：capture_radius ~ capture_radius + 0.2
    ring_center = capture_radius + 0.1
    ring_width = 0.2
    radial_deviation = np.abs(dist_to_prey - ring_center)
    radial_score = np.exp(-(radial_deviation ** 2) / (2 * ring_width ** 2))
    radial_reward = radial_score - 0.5  # 居中到 [-0.5, 0.5]
    components["radial_attract"] = w_radial_attract * radial_reward

    # (b) 切向速度奖励：鼓励环绕/合围
    vel_norm = np.linalg.norm(this_vel) + 1e-8
    vel_dir = this_vel / vel_norm
    tangential_component = np.abs(np.dot(vel_dir, dir_tangential))
    tangential_reward = tangential_component  # [0,1]
    components["tangential_align"] = w_tangential_align * tangential_reward

    # (c) 速度对准目标环（径向方向 + 控制强度）
    desired_radius = ring_center
    radial_sign = -1.0 if dist_to_prey > desired_radius else 1.0
    desired_vel_dir = radial_sign * dir_radial
    align_score = np.dot(vel_dir, desired_vel_dir)
    align_score = float(np.clip(align_score, -1.0, 1.0))
    # 只在距离目标环外一定范围时施加强约束
    radial_band = 0.3
    dist_from_band = np.maximum(0.0, np.abs(dist_to_prey - desired_radius) - radial_band)
    scale = 1.0 - np.tanh(dist_from_band)
    velocity_focus_reward = align_score * scale
    components["velocity_focus"] = w_velocity_focus * velocity_focus_reward

    # -----------------------
    # 3. 团队几何：环形与缺口封堵（全局基于极坐标）
    # -----------------------
    if len(adversary_indices) >= 2:
        adv_positions = agent_positions[adversary_indices]
        rel_adv = adv_positions - prey_pos
        radii = np.linalg.norm(rel_adv, axis=1) + 1e-8
        angles = np.arctan2(rel_adv[:, 1], rel_adv[:, 0])

        # (a) 环形质量：所有半径向目标环收缩
        mean_radius = float(np.mean(radii))
        ring_deviation = np.mean(np.abs(radii - ring_center))
        ring_quality = np.exp(-ring_deviation / (ring_width + 1e-4))
        ring_center_penalty = np.abs(mean_radius - ring_center) / (world_radius + 1e-8)
        team_ring_reward = ring_quality - ring_center_penalty
        components["team_ring"] = w_team_ring * team_ring_reward

        # (b) 最大角度缺口（封堵逃跑方向）
        sorted_angles = np.sort(angles)
        n_adv = len(sorted_angles)
        angle_diffs = np.empty_like(sorted_angles)
        angle_diffs[:-1] = sorted_angles[1:] - sorted_angles[:-1]
        angle_diffs[-1] = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
        max_gap = float(np.max(angle_diffs))
        # 目标：最大缺口接近 2π / n_adv
        ideal_gap = 2.0 * np.pi / n_adv
        gap_deviation = max_gap - ideal_gap
        # 大缺口强烈惩罚，小于 ideal_gap 略惩罚以避免过度聚集
        gap_penalty = gap_deviation / (2.0 * np.pi)
        gap_penalty = np.clip(gap_penalty, -0.5, 1.0)
        gap_closure_reward = -gap_penalty
        components["gap_closure"] = w_gap_closure * gap_closure_reward
    else:
        components["team_ring"] = 0.0
        components["gap_closure"] = 0.0

    # -----------------------
    # 4. 防碰撞势场：短程强斥力 + 中程轻斥力
    # -----------------------
    collision_avoid_reward = 0.0
    soft_radius = 2.5 * adv_size
    hard_radius = 2.0 * adv_size

    for j in adversary_indices:
        if j == this_idx:
            continue
        d = inter_agent_distances[this_idx, j]
        if d <= 1e-8:
            continue

        if d < hard_radius:
            # 强惩罚（几乎碰撞）
            collision_avoid_reward -= 1.0
        elif d < soft_radius:
            # 平滑惩罚
            t = (soft_radius - d) / (soft_radius - hard_radius + 1e-8)
            collision_avoid_reward -= t

    components["collision_avoid"] = w_collision_avoid * collision_avoid_reward

    # -----------------------
    # 5. 边界安全势场
    # -----------------------
    # 地图为 [-world_radius, world_radius]^2，鼓励远离边界
    margin = 0.3
    abs_pos = np.abs(this_pos)
    dist_to_bound_x = world_radius - abs_pos[0]
    dist_to_bound_y = world_radius - abs_pos[1]

    bound_penalty = 0.0
    for d in (dist_to_bound_x, dist_to_bound_y):
        if d < margin:
            t = (margin - d) / margin
            bound_penalty -= t

    components["boundary_safe"] = w_boundary_safe * bound_penalty

    # -----------------------
    # 6. 时间代价
    # -----------------------
    components["time_penalty"] = -w_time

    # -----------------------
    # 7. 捕获判定：基于环与距离的势场终态
    # -----------------------
    capture_bonus = 0.0
    if len(adversary_indices) >= 3 and distances_to_prey.size == len(adversary_indices):
        close_enough = distances_to_prey[adversary_indices] < (capture_radius + prey_size)
        all_close = bool(np.all(close_enough))
        # 若团队平均半径在目标环附近，视为围捕成功
        mean_adv_radius = float(np.mean(distances_to_prey[adversary_indices]))
        ring_ok = np.abs(mean_adv_radius - capture_radius) < 0.2
        if all_close and ring_ok:
            capture_bonus = w_capture_bonus

    components["capture_bonus"] = capture_bonus

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
