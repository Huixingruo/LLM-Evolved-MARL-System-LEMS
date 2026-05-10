# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260411_152433

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

    # -----------------------
    # 物理与任务常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075

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
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(global_state["inter_agent_distances"], dtype=float)

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

    # -----------------------
    # 1. 接近目标 & 包围整体收缩
    # -----------------------
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

    # -----------------------
    # 2. 包围圈几何结构（半径方差 + 角度均匀）
    # -----------------------
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

    # -----------------------
    # 3. 防碰撞与队形安全距离
    # -----------------------
    collision_penalty = 0.0
    too_close_penalty = 0.0

    for i in adversary_indices:
        if i == this_idx:
            continue
        dist_ij = inter_agent_distances[this_idx, i]
        min_safe_dist = 2 * adv_size

        if dist_ij < min_safe_dist:
            collision_penalty -= 1.0

        margin = 0.2
        min_soft_dist = min_safe_dist + margin
        if dist_ij < min_soft_dist:
            too_close_penalty -= (min_soft_dist - dist_ij) / (
                min_soft_dist + 1e-8
            )

    components["collision_penalty"] = w_collision * collision_penalty
    components["too_close_penalty"] = w_too_close * too_close_penalty

    # -----------------------
    # 4. 时间惩罚 + 围捕成功 Bonus
    # -----------------------
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
- 完全为0：collision_penalty（均值、方差均为0，几乎无碰撞惩罚信号）。  
- 方差极大：boundary_penalty（std≈1.19）、escape_reward（≈0.75）、center_radius_reward（≈0.46）、approach_reward（≈0.41）、inside_polygon_reward（≈0.37）等，易产生策略震荡，且方向多为负向。  
- 主导作用：capture_penalty（-10，刚性极强）、escape_reward、center_radius_reward、approach_reward、boundary_penalty的负值占主导；正向激励主要来自inside_polygon_reward且幅度偏小，难以抵消大额惩罚。

2. 任务瓶颈  
- 成功率约50%，平均捕获时间78.6步，说明能偶尔成功但效率偏低。  
- 多数空间相关奖励为负（radius_scale/center_radius/angle_uniform均为负），暗示智能体更倾向于“避免惩罚”而非积极围捕，可能停在较安全但非最优位置，存在防守型局部最优。

3. 协同缺陷  
- formation_quality仅0.308，encirclement_angle_std较大（1.61），说明包围角度分布不均匀，难以形成稳定合围圈。  
- min_agent_distance≈0.91且collision_penalty为0，说明没有明显扎堆撞击信号，但也不够紧密配合。  
- avg_distance_to_prey≈1.26，结合负向approach_reward，推断存在“靠得太近就被罚”的行为抑制，一些智能体可能在外围徘徊、协同收缩能力弱，存在轻微“搭便车”倾向。

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
    # 物理与任务常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075

    # 势场与协同权重（可调）
    w_radial_field = 1.0          # 径向势场：拉到目标环形带
    w_tangential_field = 0.5      # 切向势场：驱动沿环切向分布
    w_velocity_align = 0.3        # 速度方向与势场方向对齐
    w_ring_consistency = 0.5      # 团队整体在统一环带上
    w_angle_spread = 0.5          # 团队角度尽量均匀占据一圈
    w_gap_closure = 0.8           # 闭合最大角缺口
    w_min_separation = 0.8        # 最小安全间距（防堆叠）
    w_soft_separation = 0.4       # 柔性分离（防过近但允许靠拢）
    w_time = 0.01                 # 时间惩罚
    w_capture_bonus = 6.0         # 成功围捕 bonus
    w_center_stay = 0.3           # 防止全体跑远
    w_escape_penalty = 0.5        # 猎物靠近边界惩罚（防逃脱）

    # 目标环带参数：在猎物周围的“理想围捕环”
    target_radius = capture_threshold * 0.9
    radius_tolerance = capture_threshold * 0.6  # 允许的环宽
    min_radius = adv_size * 3.0                 # 防止太靠近猎物
    max_radius = world_size * 0.9               # 防止跑太远

    # 追捕者之间安全距离
    min_safe_dist = 2.5 * adv_size
    soft_safe_margin = 0.3

    # -----------------------
    # 从 global_state 中提取信息
    # -----------------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.asarray(global_state["inter_agent_distances"],
                                       dtype=float)

    all_agents = world.agents
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) == 0 or len(prey_indices) == 0:
        return 0.0, {}

    # 当前智能体索引
    this_idx = None
    for i, a in enumerate(all_agents):
        if a.name == agent_name:
            this_idx = i
            break
    if this_idx is None or not all_agents[this_idx].adversary:
        return 0.0, {}

    this_pos = agent_positions[this_idx]
    this_vel = agent_velocities[this_idx]

    # -----------------------
    # 坐标变换：相对猎物的极坐标
    # -----------------------
    rel_pos = this_pos - prey_pos
    rel_dist = np.linalg.norm(rel_pos) + 1e-8
    rel_unit = rel_pos / rel_dist

    # 角度 [-pi, pi)
    rel_angle = np.arctan2(rel_pos[1], rel_pos[0])

    # -----------------------
    # 1. 径向势场：环带吸引 + 内外壁排斥
    # -----------------------
    # 理想距离：target_radius
    radial_error = rel_dist - target_radius
    # 在容忍范围内给小惩罚，外侧/内侧偏离更大
    band_half_width = radius_tolerance
    normalized_error = radial_error / (band_half_width + 1e-8)

    # 平滑双向势场：|normalized_error| 越小越好
    radial_field_reward = - (normalized_error ** 2)

    # 内侧硬排斥与外侧软拉回
    if rel_dist < min_radius:
        inside_penalty = -((min_radius - rel_dist) / (min_radius + 1e-8)) ** 2
        radial_field_reward += inside_penalty
    if rel_dist > max_radius:
        outside_penalty = -((rel_dist - max_radius) /
                            (world_size + 1e-8)) ** 2
        radial_field_reward += outside_penalty

    components["radial_field_reward"] = w_radial_field * radial_field_reward

    # -----------------------
    # 2. 切向势场：驱动围绕猎物转动（统一旋向）
    # -----------------------
    # 选择固定旋向：逆时针
    tangential_dir = np.array([-rel_unit[1], rel_unit[0]])
    tangential_speed = np.dot(this_vel, tangential_dir)
    # 奖励沿切向运动，但限制极值，防止无意义高速绕圈
    desired_tangential = 0.6  # 目标切向速度（相对 max_speed≈1）
    tangential_error = tangential_speed - desired_tangential
    tangential_reward = - (tangential_error ** 2)
    components["tangential_field_reward"] = w_tangential_field * tangential_reward

    # -----------------------
    # 3. 速度对齐：速度方向对齐合力方向（径向 + 切向）
    # -----------------------
    desired_dir = rel_unit * (-normalized_error) + tangential_dir
    desired_norm = np.linalg.norm(desired_dir)
    if desired_norm > 1e-8:
        desired_dir /= desired_norm
    else:
        desired_dir = tangential_dir

    vel_norm = np.linalg.norm(this_vel)
    if vel_norm > 1e-8:
        vel_dir = this_vel / vel_norm
        cos_sim = np.clip(np.dot(vel_dir, desired_dir), -1.0, 1.0)
        velocity_align_reward = cos_sim
    else:
        velocity_align_reward = 0.0

    components["velocity_align_reward"] = w_velocity_align * \
        velocity_align_reward

    # -----------------------
    # 4. 团队层面：统一环带与角度分布
    # -----------------------
    team_ring_consistency = 0.0
    team_angle_spread = 0.0
    team_gap_closure = 0.0

    if len(adversary_indices) >= 2:
        adv_positions = agent_positions[adversary_indices]
        rel_team = adv_positions - prey_pos
        radii = np.linalg.norm(rel_team, axis=1)
        mean_radius = np.mean(radii)
        # 统一环带：方差越小越好，同时接近 target_radius
        radius_var = np.var(radii)
        ring_var_term = -radius_var / (world_size ** 2 + 1e-8)
        ring_scale_term = -((mean_radius - target_radius) ** 2) / \
            (world_size ** 2 + 1e-8)
        team_ring_consistency = ring_var_term + 0.5 * ring_scale_term

        # 角度
        angles = np.arctan2(rel_team[:, 1], rel_team[:, 0])
        angles_sorted = np.sort(angles)
        # 邻角差 + 闭合差
        diffs = np.diff(angles_sorted)
        wrap_diff = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([diffs, [wrap_diff]])

        n_adv = len(adversary_indices)
        ideal_diff = 2 * np.pi / n_adv
        # 角度均匀性
        angle_spread_var = np.mean((angle_diffs - ideal_diff) ** 2)
        team_angle_spread = -angle_spread_var / (np.pi ** 2 + 1e-8)

        # 最大缺口越小越好
        max_gap = np.max(angle_diffs)
        team_gap_closure = -((max_gap - ideal_diff) ** 2) / \
            (np.pi ** 2 + 1e-8)

    components["ring_consistency_reward"] = w_ring_consistency * \
        team_ring_consistency
    components["angle_spread_reward"] = w_angle_spread * team_angle_spread
    components["gap_closure_reward"] = w_gap_closure * team_gap_closure

    # -----------------------
    # 5. 局部相邻协同：角邻居保持适中距离
    # -----------------------
    local_spacing_reward = 0.0
    if len(adversary_indices) >= 3:
        adv_positions = agent_positions[adversary_indices]
        rel_team = adv_positions - prey_pos
        angles = np.arctan2(rel_team[:, 1], rel_team[:, 0])

        # 按角度排序
        sort_idx = np.argsort(angles)
        sorted_indices = np.array(adversary_indices)[sort_idx]
        sorted_angles = angles[sort_idx]

        # 找到当前智能体在该序列中的位置
        if this_idx in sorted_indices:
            pos = np.where(sorted_indices == this_idx)[0][0]
            left_idx = sorted_indices[(pos - 1) % len(sorted_indices)]
            right_idx = sorted_indices[(pos + 1) % len(sorted_indices)]

            left_dist = np.linalg.norm(agent_positions[this_idx] -
                                       agent_positions[left_idx])
            right_dist = np.linalg.norm(agent_positions[this_idx] -
                                        agent_positions[right_idx])

            ideal_neighbor_dist = 2.0 * target_radius * np.sin(
                np.pi / len(adversary_indices)
            )

            def spacing_term(d):
                if d < min_safe_dist:
                    return -((min_safe_dist - d) /
                             (min_safe_dist + 1e-8)) ** 2
                return -((d - ideal_neighbor_dist) ** 2) / \
                    (world_size ** 2 + 1e-8)

            local_spacing_reward = 0.5 * (spacing_term(left_dist) +
                                          spacing_term(right_dist))

    components["local_spacing_reward"] = local_spacing_reward

    # -----------------------
    # 6. 防碰撞与柔性分离（纯基于距离）
    # -----------------------
    min_separation_penalty = 0.0
    soft_separation_penalty = 0.0

    for i in adversary_indices:
        if i == this_idx:
            continue
        dij = inter_agent_distances[this_idx, i]
        if dij < 1e-8:
            continue

        if dij < min_safe_dist:
            # 硬惩罚：平方型
            min_separation_penalty -= ((min_safe_dist - dij) /
                                       (min_safe_dist + 1e-8)) ** 2

        soft_limit = min_safe_dist + soft_safe_margin
        if dij < soft_limit:
            soft_separation_penalty -= ((soft_limit - dij) /
                                        (soft_limit + 1e-8)) ** 2

    components["min_separation_penalty"] = w_min_separation * \
        min_separation_penalty
    components["soft_separation_penalty"] = w_soft_separation * \
        soft_separation_penalty

    # -----------------------
    # 7. 全局布局：防止整体外逃与猎物接近边界
    # -----------------------
    # 团队质心相对猎物不要太远
    team_center_reward = 0.0
    if len(adversary_indices) > 0:
        adv_positions = agent_positions[adversary_indices]
        team_center = np.mean(adv_positions, axis=0)
        center_rel = team_center - prey_pos
        center_dist = np.linalg.norm(center_rel)
        # 希望团队质心也在 target_radius 附近
        center_error = (center_dist - target_radius) / (world_size + 1e-8)
        team_center_reward = - (center_error ** 2)

    components["center_stay_reward"] = w_center_stay * team_center_reward

    # 猎物接近边界 -> 对追捕者惩罚（说明控制不力）
    prey_dist_center = np.linalg.norm(prey_pos)
    boundary_limit = world_size * 0.9
    escape_penalty = 0.0
    if prey_dist_center > boundary_limit:
        escape_penalty = -((prey_dist_center - boundary_limit) /
                           (world_size + 1e-8)) ** 2
    components["escape_penalty"] = w_escape_penalty * escape_penalty

    # -----------------------
    # 8. 时间惩罚 + 捕获 Bonus
    # -----------------------
    components["time_penalty"] = -w_time

    capture_bonus = 0.0
    if len(adversary_indices) >= 3:
        adv_positions = agent_positions[adversary_indices]
        rel_team = adv_positions - prey_pos
        radii = np.linalg.norm(rel_team, axis=1)
        angles = np.arctan2(rel_team[:, 1], rel_team[:, 0])
        angles_sorted = np.sort(angles)
        diffs = np.diff(angles_sorted)
        wrap_diff = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([diffs, [wrap_diff]])

        # 条件：所有人在一个相对紧的环带内 + 最大角缺口足够小
        ring_ok = np.all(np.abs(radii - target_radius) <= radius_tolerance)
        max_gap = np.max(angle_diffs)
        gap_ok = max_gap <= (np.pi / 2.0)

        if ring_ok and gap_ok:
            capture_bonus = w_capture_bonus

    components["capture_bonus"] = capture_bonus

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
