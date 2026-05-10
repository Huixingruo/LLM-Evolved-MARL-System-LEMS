# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker1_F2
- **Timestamp**: 20260411_152419

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
【Reward Prune Refinement (修剪精炼)】
请精简原代码，直接删除诊断报告中指出失效的、一直为0的或起反作用的奖励分量。不要增加任何新逻辑。

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
```
