# LLM Interaction Log

- **Generation**: 3
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260411_161905

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
    w_collision = 2.0          # 追捕者间碰撞惩罚
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

    # 3. 防碰撞与队形安全距离（移除始终为0的刚性碰撞惩罚项）
    collision_penalty = 0.0
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
1. **分量贡献度**
   - 完全为0：无（各分量均有非零均值或方差），但`time_penalty`、`capture_penalty`方差为0，说明恒定给罚，不随行为区分。
   - 方差极大：`escape_reward`(0.76)、`boundary_penalty`(1.76) 波动大，易诱发策略震荡；`center_radius_reward`、`approach_reward` 方差中等偏大。
   - 主导作用：`capture_penalty=-10`远大于其他分量，且恒定；其次是持续性的负向：`escape_reward`、`center_radius_reward`、`approach_reward`、`boundary_penalty`，整体负激励居多，正向捕获奖励很弱（`capture_bonus`均值接近0）。

2. **任务瓶颈**
   - 成功率46.77%、平均捕获78.4步，说明尚未形成稳定、高效的捕获策略。
   - 大量持续负奖励与较高失败惩罚，易形成保守或犹豫行为；`boundary_penalty`和`center_radius_reward`负值较大，暗示智能体在边缘/不理想半径反复试探，存在“徘徊型”局部最优倾向。

3. **协同缺陷**
   - `formation_quality=0.2988`偏低，`encirclement_angle_std=1.6853`较大，表明包围角度分布不均，难以形成完整包围圈。
   - `min_agent_distance=0.9142`不算极小，但结合低队形质量，说明既未有效收缩包围圈，也未出现高质量协同挤压，多为松散、各自为战。
   - `avg_distance_to_prey=1.3442`偏大，配合负向`approach_reward`，显示存在“靠不近、不敢收缩”的协同收缩失败，而非明显扎堆或大规模碰撞；更像部分个体拖后腿、队形拉长的“伪协同”而非静止搭便车。

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

    # --------------------
    # 硬编码物理与任务常量
    # --------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.05

    # --------------------
    # 奖励权重（可调）
    # --------------------
    # 势场 & 协同
    w_radial_potential = 1.5     # 围捕势场：推向捕获半径
    w_angular_spread = 1.0       # 角度均匀分布
    w_orbit_motion = 0.5         # 沿切线(绕圈)运动
    w_center_alignment = 0.8     # 队形中心贴近猎物
    w_velocity_coherence = 0.5   # 速度方向协同（环形协同）

    # 安全与形态约束
    w_safe_distance = 0.8        # 追捕者间距离-软约束
    w_boundary_potential = 0.5   # 远离边界势场
    w_cross_through_prey = 0.8   # 禁止穿越猎物直线的惩罚

    # 任务驱动
    w_capture_shell = 2.0        # 所有追捕者进入捕获壳层
    w_capture_final = 8.0        # 最终围捕成功奖励
    w_step_cost = 0.005          # 温和时间成本

    # ---------------
    # 解析 global_state
    # ---------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
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

    # 将坐标转换到以猎物为原点的极坐标系
    rel_pos = this_pos - prey_pos
    r = np.linalg.norm(rel_pos) + 1e-8
    theta = np.arctan2(rel_pos[1], rel_pos[0])

    # ---------------
    # 1. 径向势场：鼓励进入捕获“壳层”
    # ---------------
    # 目标壳层半径：稍微大于精确捕获阈值，避免过度挤压
    target_shell = capture_threshold * 1.1
    # 构造单峰势场：在 target_shell 附近奖励最高，远离则衰减
    radial_error = (r - target_shell) / world_size
    radial_potential = -radial_error**2
    components["radial_potential"] = w_radial_potential * radial_potential

    # -----------------------
    # 2. 角度均匀 & 编队势场
    # -----------------------
    if len(adversary_indices) >= 2:
        adv_positions = agent_positions[adversary_indices]
        rel_adv = adv_positions - prey_pos
        adv_radii = np.linalg.norm(rel_adv, axis=1) + 1e-8
        adv_angles = np.arctan2(rel_adv[:, 1], rel_adv[:, 0])

        # 计算该追捕者在 adversary list 中的索引
        try:
            adv_local_idx = adversary_indices.index(this_idx)
        except ValueError:
            adv_local_idx = 0

        # 排序后的角度，用于构造局部角间距势场
        sort_idx = np.argsort(adv_angles)
        sorted_angles = adv_angles[sort_idx]
        n_adv = len(sorted_angles)

        # 理想角间隔
        ideal_gap = 2.0 * np.pi / float(n_adv)

        # 找到当前追捕者在排序后列表中的位置
        sorted_pos = int(np.where(sort_idx == adv_local_idx)[0][0])
        left_idx = (sorted_pos - 1) % n_adv
        right_idx = (sorted_pos + 1) % n_adv

        theta_left = sorted_angles[left_idx]
        theta_right = sorted_angles[right_idx]
        theta_self = sorted_angles[sorted_pos]

        # 归一化到 [0, 2π)
        def wrap_angle(x):
            return (x + 2.0 * np.pi) % (2.0 * np.pi)

        theta_self_n = wrap_angle(theta_self)
        theta_left_n = wrap_angle(theta_left)
        theta_right_n = wrap_angle(theta_right)

        # 角间距
        gap_left = wrap_angle(theta_self_n - theta_left_n)
        gap_right = wrap_angle(theta_right_n - theta_self_n)

        # 与理想间隔的偏差（仅对本体角度）
        gap_left_err = (gap_left - ideal_gap) / np.pi
        gap_right_err = (gap_right - ideal_gap) / np.pi
        angular_spread_reward = -0.5 * (gap_left_err**2 + gap_right_err**2)
        components["angular_spread"] = w_angular_spread * angular_spread_reward

        # -----------------------------
        # 3. 环绕式（轨道）运动协同势场
        # -----------------------------
        # 理想速度方向：沿当前相对位置切线方向（逆时针）
        radial_unit = rel_pos / r
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
        if np.linalg.norm(this_vel) > 1e-6:
            vel_dir = this_vel / (np.linalg.norm(this_vel) + 1e-8)
            orbit_alignment = np.dot(vel_dir, tangent_unit)
        else:
            orbit_alignment = 0.0
        components["orbit_motion"] = w_orbit_motion * orbit_alignment

        # --------------------------------
        # 4. 队形中心对齐：中心靠近猎物
        # --------------------------------
        center_pos = np.mean(adv_positions, axis=0)
        center_dist = np.linalg.norm(center_pos - prey_pos)
        center_alignment = - (center_dist / world_size) ** 2
        components["center_alignment"] = w_center_alignment * center_alignment

        # --------------------------------
        # 5. 速度协同（沿局部环方向）
        # --------------------------------
        # 参考方向：根据邻居的切线方向平均
        neighbor_dirs = []
        for j_idx in [left_idx, right_idx]:
            neighbor_global_idx = adversary_indices[sort_idx[j_idx]]
            n_rel = agent_positions[neighbor_global_idx] - prey_pos
            n_r = np.linalg.norm(n_rel) + 1e-8
            n_radial_unit = n_rel / n_r
            n_tangent = np.array([-n_radial_unit[1], n_radial_unit[0]])
            neighbor_dirs.append(n_tangent)
        if neighbor_dirs:
            ref_dir = np.mean(neighbor_dirs, axis=0)
            ref_dir_norm = np.linalg.norm(ref_dir)
            if ref_dir_norm > 1e-6 and np.linalg.norm(this_vel) > 1e-6:
                ref_dir = ref_dir / ref_dir_norm
                vel_dir = this_vel / (np.linalg.norm(this_vel) + 1e-8)
                velocity_coherence = np.dot(vel_dir, ref_dir)
            else:
                velocity_coherence = 0.0
        else:
            velocity_coherence = 0.0

        components["velocity_coherence"] = (
            w_velocity_coherence * velocity_coherence
        )
    else:
        components["angular_spread"] = 0.0
        components["orbit_motion"] = 0.0
        components["center_alignment"] = 0.0
        components["velocity_coherence"] = 0.0

    # -------------------------
    # 6. 安全距离软约束（势场）
    # -------------------------
    safe_dist = 2.5 * adv_size
    soft_margin = 0.25
    soft_safe_dist = safe_dist + soft_margin
    safe_distance_penalty = 0.0

    for j in adversary_indices:
        if j == this_idx:
            continue
        d_ij = inter_agent_distances[this_idx, j]
        if d_ij < soft_safe_dist:
            diff = (soft_safe_dist - d_ij) / soft_safe_dist
            safe_distance_penalty -= diff**2

    components["safe_distance"] = w_safe_distance * safe_distance_penalty

    # -------------------------
    # 7. 边界势场：远离边缘
    # -------------------------
    # 越接近边界（world_size），势场越负
    # 使用坐标绝对值的最大分量构造距离边界的度量
    abs_pos = np.abs(this_pos)
    margin_to_border = world_size - np.max(abs_pos)
    margin_to_border = max(margin_to_border, 1e-3)
    # 在 margin_to_border 小时给予强烈负势场
    boundary_potential = - (1.0 / margin_to_border) * (0.1 / world_size)
    components["boundary_potential"] = w_boundary_potential * boundary_potential

    # -------------------------------------
    # 8. 禁止 "穿过猎物" 的轨迹几何惩罚
    # -------------------------------------
    # 近似：当前相对位置与下一步相对位置夹角过大且速度径向分量过强时惩罚
    rel_vel = this_vel - prey_vel
    radial_unit = rel_pos / r
    radial_speed = np.dot(rel_vel, radial_unit)
    # 若径向速度指向猎物且 r 小于阈值，则视为可能穿越
    cross_penalty = 0.0
    cross_threshold = capture_threshold * 1.3
    if (r < cross_threshold) and (radial_speed < 0.0):
        # 越快向内冲刺，惩罚越大
        cross_penalty = - (abs(radial_speed)) / (1.0 + np.linalg.norm(rel_vel))
    components["cross_through_prey"] = w_cross_through_prey * cross_penalty

    # -----------------------------------
    # 9. 壳层一致性 & 最终捕获奖励
    # -----------------------------------
    capture_shell_reward = 0.0
    capture_final_reward = 0.0

    if len(adversary_indices) >= 3 and distances_to_prey.size > 0:
        # 壳层一致性：所有追捕者都在 [target_shell - δ, target_shell + δ] 内
        shell_width = 0.2 * capture_threshold
        lower = target_shell - shell_width
        upper = target_shell + shell_width

        in_shell = np.logical_and(
            distances_to_prey >= lower,
            distances_to_prey <= upper,
        )
        shell_ratio = np.mean(in_shell.astype(float))
        capture_shell_reward = shell_ratio

        # 最终捕获判定：所有追捕者在捕获阈值内，并形成基本包围
        all_inside_capture = np.all(distances_to_prey <= capture_threshold * 1.05)

        # 简化包围判定：猎物相对追捕者的质心偏移较小
        adv_positions = agent_positions[adversary_indices]
        center_pos = np.mean(adv_positions, axis=0)
        center_dist = np.linalg.norm(center_pos - prey_pos)
        encirclement_ok = center_dist < capture_threshold * 0.6

        if all_inside_capture and encirclement_ok:
            capture_final_reward = 1.0

    components["capture_shell"] = w_capture_shell * capture_shell_reward
    components["capture_final"] = w_capture_final * capture_final_reward

    # -------------------------
    # 10. 轻微时间成本
    # -------------------------
    components["step_cost"] = -w_step_cost

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
