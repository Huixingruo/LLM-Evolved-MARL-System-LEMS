# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260411_142947

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
    # 非追捕者不奖励
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理与任务常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075
    prey_size = 0.050

    # 目标包围半径（略大于碰撞和捕获阈值，用于形成包围圈）
    target_radius = capture_threshold
    # 阈值：认为"足够接近"用于时间奖励和终局奖励
    close_radius = capture_threshold

    # 奖励权重（可调）
    w_approach = 1.0           # 接近猎物
    w_center_radius = 1.0      # 整体圈半径小
    w_radius_var = 0.5         # 半径均匀性
    w_angle_uniform = 0.5      # 角度均匀性
    w_inside = 1.0             # 猎物在三角形内部
    w_time = 0.01              # 时间惩罚
    w_capture_bonus = 5.0      # 成功围捕 bonus
    w_collision = 2.0          # 追捕者间碰撞惩罚
    w_too_close = 0.5          # 追捕者间过近惩罚

    # -----------------------
    # 从 global_state 中提取信息
    # -----------------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(global_state["inter_agent_distances"], dtype=float)

    # world.agents 顺序与 global_state 对齐：追捕者 + 猎物，共 4 个
    all_agents = world.agents
    num_agents = len(all_agents)

    # 索引映射
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) == 0 or len(prey_indices) == 0:
        return 0.0, {}

    prey_idx = prey_indices[0]

    # 当前智能体索引
    this_idx = None
    for i, a in enumerate(all_agents):
        if a.name == agent_name:
            this_idx = i
            break
    if this_idx is None:
        return 0.0, {}

    this_is_adv = all_agents[this_idx].adversary
    if not this_is_adv:
        return 0.0, {}

    # 只对追捕者计算奖励
    this_pos = agent_positions[this_idx]
    this_vel = agent_velocities[this_idx]

    # -----------------------
    # 1. 接近目标 & 包围整体收缩
    # -----------------------
    # a) 个体距离猎物的奖励（距离越小越好）
    this_dist_to_prey = np.linalg.norm(this_pos - prey_pos)
    # 归一化距离（相对 world_size），避免数值过大
    norm_dist = this_dist_to_prey / world_size
    approach_reward = -norm_dist  # 距离越小奖励越高（负距离）
    components["approach_reward"] = w_approach * approach_reward

    # b) 团队整体到猎物的最大距离（希望最大距离也尽量小，整体收缩）
    if distances_to_prey.size > 0:
        max_team_dist = np.max(distances_to_prey)
        center_radius_reward = -(max_team_dist / world_size)
    else:
        center_radius_reward = 0.0
    components["center_radius_reward"] = w_center_radius * center_radius_reward

    # -----------------------
    # 2. 包围圈几何结构（半径方差 + 角度均匀）
    # -----------------------
    # 仅在追捕者数量 >= 3 时计算包围几何
    if len(adversary_indices) >= 3:
        adv_positions = agent_positions[adversary_indices]

        # 半径
        radii = np.linalg.norm(adv_positions - prey_pos, axis=1)
        mean_radius = np.mean(radii)
        radius_var = np.var(radii)
        # 惩罚半径方差，并鼓励整体半径接近 target_radius
        radius_uniform_reward = -radius_var / (world_size ** 2 + 1e-8)
        radius_scale_reward = -abs(mean_radius - target_radius) / (world_size + 1e-8)
        components["radius_uniform_reward"] = w_radius_var * radius_uniform_reward
        components["radius_scale_reward"] = 0.5 * w_radius_var * radius_scale_reward

        # 角度均匀性：希望 3 个追捕者在猎物周围角度差接近 120°
        rel_pos = adv_positions - prey_pos
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        angles_sorted = np.sort(angles)
        # 计算环状角度差
        angle_diffs = np.empty_like(angles_sorted)
        angle_diffs[:-1] = angles_sorted[1:] - angles_sorted[:-1]
        angle_diffs[-1] = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        # 理想值 2*pi / n
        n_adv = len(adversary_indices)
        ideal_diff = 2 * np.pi / n_adv
        angle_uniformity = np.mean((angle_diffs - ideal_diff) ** 2)
        angle_uniform_reward = -angle_uniformity / (np.pi ** 2 + 1e-8)
        components["angle_uniform_reward"] = w_angle_uniform * angle_uniform_reward

        # 判断猎物是否在追捕者形成的多边形内部（三角形）
        def point_in_polygon(point, poly):
            # 射线法判断点在多边形内部
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

    # -----------------------
    # 3. 防碰撞与队形安全距离
    # -----------------------
    # 追捕者之间避免碰撞，惩罚过近
    collision_penalty = 0.0
    too_close_penalty = 0.0

    for i in adversary_indices:
        if i == this_idx:
            continue
        dist_ij = inter_agent_distances[this_idx, i]
        min_safe_dist = 2 * adv_size

        # 碰撞惩罚
        if dist_ij < min_safe_dist:
            collision_penalty -= 1.0
        # 即使尚未碰撞，但太近也惩罚（平滑）
        margin = 0.2  # 额外缓冲区
        min_soft_dist = min_safe_dist + margin
        if dist_ij < min_soft_dist:
            # 线性惩罚：距离越小惩罚越大
            too_close_penalty -= (min_soft_dist - dist_ij) / (min_soft_dist + 1e-8)

    components["collision_penalty"] = w_collision * collision_penalty
    components["too_close_penalty"] = w_too_close * too_close_penalty

    # -----------------------
    # 4. 时间惩罚 + 围捕成功 Bonus
    # -----------------------
    # 时间惩罚：每一步小负奖励，鼓励尽快完成
    components["time_penalty"] = -w_time

    # 围捕成功条件（简单版）：
    # 1) 所有追捕者到猎物距离 < close_radius
    # 2) 猎物在追捕者多边形内部（上面已计算）
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
- 完全为0：collision_penalty（均值和方差都为0，几乎未触发）。  
- 方差极大：boundary_penalty（std=1.50）、escape_reward（0.78）、center_radius_reward（0.54）、approach_reward（0.46），这些会在轨迹中造成强烈震荡与策略不稳定。  
- 主导作用：capture_penalty（-10 极大负向牵引），escape_reward、center_radius_reward、approach_reward和boundary_penalty共同主导整体梯度方向，正向奖励（inside_polygon、capture_bonus等）影响很弱。

2. 任务瓶颈  
- 成功率≈58%，平均捕获时间74.2步，说明能找到策略但效率较低。  
- 大量负向“逃逸/距离”与“中心半径”惩罚叠加，容易驱动保守、畏缩策略；同时boundary_penalty波动大，暗示智能体在边界附近反复试探，存在“在安全边缘犹豫、收缩不足”的局部最优倾向。

3. 协同缺陷  
- formation_quality仅0.30，encirclement_angle_std高（1.71），说明包围角度分布不均，难以形成稳定合围。  
- avg_distance_to_prey=1.31、min_agent_distance=0.87，表明整体距离偏大、彼此间不够紧密，收缩包围圈能力不足，更像松散跟随而非有序协同；未见碰撞惩罚触发，也可能意味着“扎堆/逼近不够”，甚至部分个体保持安全距离、存在轻度“搭便车”倾向。

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
    # 硬编码环境常量
    # -----------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050

    # 目标势场与协同结构参数
    target_ring_radius = 0.7          # 期望围捕环半径（相对地图尺度中等偏近）
    ring_tolerance = 0.25             # 半径容忍带
    min_pair_distance = 2 * adv_size  # 硬碰撞半径
    soft_pair_distance = min_pair_distance + 0.15  # 软排斥开始距离
    ideal_angle_diff = None           # 将在运行时根据追捕者数计算

    # 权重（可调）
    w_radial_potential = 1.5      # 环状势场：拉到目标半径
    w_angular_potential = 1.0     # 角度均匀势场
    w_center_pull = 0.75          # 过远时整体拉近势场
    w_velocity_alignment = 0.5    # 朝向猎物方向的速度对齐奖励
    w_pair_repulsion = 1.0        # 追捕者之间软排斥
    w_collision_hard = 3.0        # 硬碰撞惩罚
    w_boundary_soft = 0.5         # 靠近边界的轻微惩罚
    w_time_step = 0.01            # 时间步惩罚
    w_capture_bonus = 4.0         # 成功围捕 bonus（局部条件）
    w_local_shaping = 0.5         # 本智能体局部极坐标 shaping

    # -----------------------
    # 提取全局状态
    # -----------------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"],
        dtype=float,
    )

    all_agents = world.agents
    num_agents = len(all_agents)

    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) == 0 or len(prey_indices) == 0:
        return 0.0, {}

    prey_idx = prey_indices[0]

    # 当前智能体索引
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

    # 追捕者数量
    n_adv = len(adversary_indices)
    if n_adv > 0:
        ideal_angle_diff = 2 * np.pi / n_adv
    else:
        ideal_angle_diff = 2 * np.pi

    # -----------------------
    # 1. 基于势场的环状围捕结构
    # -----------------------

    # a) 环状径向势场（所有追捕者共享）
    adv_positions = agent_positions[adversary_indices]
    rel_adv = adv_positions - prey_pos
    radii = np.linalg.norm(rel_adv, axis=1) + 1e-8

    # 目标半径附近是势场最小值：使用平滑的“盆形”势函数
    # U_r(r) = (max(0, |r - R*| - tol))^2 / world_size^2
    radial_deviation = np.abs(radii - target_ring_radius)
    radial_excess = np.maximum(0.0, radial_deviation - ring_tolerance)
    radial_potential = (radial_excess ** 2) / (world_size ** 2 + 1e-8)
    # 对本智能体取对应值
    this_radial_potential = radial_potential[
        adversary_indices.index(this_idx)
    ]
    # 势能越低越好 -> 奖励为负势能
    components["radial_potential_reward"] = -w_radial_potential * float(
        this_radial_potential,
    )

    # b) 全局角度均匀势场（追捕者相对猎物的角度差）
    if n_adv >= 3:
        adv_rel_angles = np.arctan2(rel_adv[:, 1], rel_adv[:, 0])
        angles_sorted = np.sort(adv_rel_angles)
        angle_diffs = np.empty_like(angles_sorted)
        angle_diffs[:-1] = angles_sorted[1:] - angles_sorted[:-1]
        angle_diffs[-1] = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]

        # 角度差与理想差之间的平方偏差
        angle_deviation = angle_diffs - ideal_angle_diff
        angle_potential_all = (angle_deviation ** 2) / (np.pi ** 2 + 1e-8)

        # 将角度势场平均分摊给所有追捕者
        mean_angle_potential = float(np.mean(angle_potential_all))
        components["angular_potential_reward"] = (
            -w_angular_potential * mean_angle_potential
        )
    else:
        components["angular_potential_reward"] = 0.0

    # c) 中心拉近势场：如果整体平均半径过大，则产生向内拉近的势场
    mean_radius = float(np.mean(radii))
    center_pull_term = np.maximum(0.0, mean_radius - (target_ring_radius + 0.3))
    center_pull_potential = (center_pull_term ** 2) / (world_size ** 2 + 1e-8)
    components["center_pull_reward"] = -w_center_pull * center_pull_potential

    # -----------------------
    # 2. 极坐标局部 shaping（本智能体）
    # -----------------------
    rel_this = this_pos - prey_pos
    r_this = float(np.linalg.norm(rel_this) + 1e-8)
    theta_this = float(np.arctan2(rel_this[1], rel_this[0]))

    # 在目标半径附近鼓励稳定（对小偏移给轻微惩罚，大偏移惩罚加大）
    local_radial_diff = r_this - target_ring_radius
    local_radial_potential = (local_radial_diff ** 2) / (world_size ** 2 + 1e-8)

    # 猎物运动方向（若速度很小，则弱化此项）
    prey_speed = float(np.linalg.norm(prey_vel))
    if prey_speed > 1e-3:
        prey_dir = prey_vel / prey_speed
        # 期望：围捕处于猎物运动方向的侧后方 -> 鼓励与径向方向正交（即切向运动）
        radial_dir = rel_this / (r_this + 1e-8)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])
        tangential_alignment = float(np.dot(this_vel, tangential_dir))
        tangential_norm = float(np.linalg.norm(this_vel) + 1e-8)
        tangential_cos = tangential_alignment / tangential_norm
        # 使其在 -0.5 ~ 1 范围内平滑，留最大奖励给切向运动
        tangential_reward = (tangential_cos + 0.5) / 1.5
    else:
        tangential_reward = 0.0

    components["local_radial_shaping"] = -w_local_shaping * local_radial_potential
    components["tangential_motion_reward"] = w_local_shaping * tangential_reward

    # -----------------------
    # 3. 速度对齐与追猎行为
    # -----------------------
    # 希望追捕者朝向猎物所在方向或包围环切向方向移动，而非远离
    if r_this > 1e-3:
        direction_to_prey = -rel_this / r_this  # 从本智能体指向猎物
    else:
        direction_to_prey = np.zeros(2, dtype=float)
    this_speed = float(np.linalg.norm(this_vel))
    if this_speed > 1e-6:
        vel_dir = this_vel / this_speed
        cos_to_prey = float(np.dot(vel_dir, direction_to_prey))
        # 将 -1~1 压缩后移到 0~1，再映射到奖励
        align_reward = (cos_to_prey + 1.0) / 2.0
    else:
        align_reward = 0.0

    components["velocity_alignment_reward"] = (
        w_velocity_alignment * align_reward
    )

    # -----------------------
    # 4. 追捕者之间的软排斥与硬碰撞
    # -----------------------
    pair_repulsion = 0.0
    collision_hard = 0.0

    for j in adversary_indices:
        if j == this_idx:
            continue
        dist_ij = inter_agent_distances[this_idx, j]
        # 硬碰撞：距离小于物理碰撞半径之和
        hard_threshold = min_pair_distance
        if dist_ij < hard_threshold:
            collision_hard -= 1.0
        # 软排斥：在 soft_pair_distance 范围内使用 1/d 型或线性势场
        if dist_ij < soft_pair_distance:
            # 距离越小惩罚越大，平滑归一化
            repulse = (soft_pair_distance - dist_ij) / (
                soft_pair_distance - hard_threshold + 1e-8
            )
            pair_repulsion -= repulse

    components["pair_repulsion_penalty"] = w_pair_repulsion * pair_repulsion
    components["hard_collision_penalty"] = w_collision_hard * collision_hard

    # -----------------------
    # 5. 边界势场（温和约束）
    # -----------------------
    # 越靠近边界，势能越大，防止个体被吸附在边缘
    # 使用独立于原方案的平滑“外墙”势场
    wall_margin = 0.5  # 距离边界 0.5 内开始惩罚
    abs_pos = np.abs(this_pos)
    dist_to_wall = world_size - abs_pos  # x,y 方向距离边界
    # 在两个方向上分别计算势场并取最大
    wall_potential_x = max(0.0, wall_margin - dist_to_wall[0])
    wall_potential_y = max(0.0, wall_margin - dist_to_wall[1])
    wall_potential = max(wall_potential_x, wall_potential_y)
    wall_potential = (wall_potential ** 2) / (wall_margin ** 2 + 1e-8)
    components["boundary_soft_penalty"] = -w_boundary_soft * wall_potential

    # -----------------------
    # 6. 捕获完成 Bonus（几何+距离）
    # -----------------------
    capture_bonus = 0.0
    if n_adv >= 3:
        # 所有追捕者到猎物的距离
        dists_all = np.linalg.norm(
            agent_positions[adversary_indices] - prey_pos,
            axis=1,
        )
        # 条件：整体半径小且各追捕者都在一定范围以内
        all_close = np.all(dists_all < (target_ring_radius + 0.2))
        # 利用环状结构：平均半径小于目标半径 + 容忍带
        if mean_radius < (target_ring_radius + 0.2) and all_close:
            # 再检查是否形成较均匀角度
            # 用前面计算的角度势场，如果已经较小，则视为成功
            if n_adv >= 3:
                # 简单度量：半径方差与角度势场综合
                radius_var = float(np.var(radii))
                radial_ok = radius_var < 0.1
                if radial_ok:
                    capture_bonus = w_capture_bonus

    components["capture_bonus"] = capture_bonus

    # -----------------------
    # 7. 时间步惩罚（鼓励尽快完成）
    # -----------------------
    components["time_penalty"] = -w_time_step

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
