# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker0_F1
- **Timestamp**: 20260410_161613

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
    # 逃跑者奖励为 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5  # world_size * 0.2
    adv_size = 0.075
    prey_size = 0.050

    # 参数与权重（可调）
    w_approach = 1.0
    w_capture_bonus = 20.0
    w_time_penalty = -0.05
    w_collision_penalty = -5.0
    w_prey_collision_penalty = -2.0
    w_radius_uniformity = 0.5
    w_angle_uniformity = 0.5
    w_spread_penalty = -0.5

    # 安全距离（防止追捕者之间碰撞）
    safety_margin = 0.01
    min_pair_distance = 2.0 * adv_size + safety_margin

    # 理想包围半径（略小于 capture_threshold，避免太松散）
    ideal_radius = capture_threshold * 0.7

    # -----------------------
    # 从 global_state 解析数据
    # -----------------------
    agent_positions = global_state["agent_positions"]  # shape (n_agents, 2)
    agent_velocities = global_state["agent_velocities"]
    prey_position = global_state["prey_position"]
    prey_velocity = global_state["prey_velocity"]
    distances_to_prey = global_state["distances_to_prey"]
    inter_agent_distances = global_state["inter_agent_distances"]

    # 区分追捕者与猎物
    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    # 映射 agent_name -> index in agent_positions
    name_to_index = {a.name: i for i, a in enumerate(all_agents)}
    self_idx = name_to_index[agent_name]

    # 当前 agent 是否为追捕者（上面已经过滤，但这里保持一致性）
    self_agent = all_agents[self_idx]
    if not self_agent.adversary:
        return 0.0, {}

    # 追捕者索引列表（在 agent_positions 中的索引）
    adversary_indices = [name_to_index[a.name] for a in adversaries]

    # 该追捕者在 adversaries 列表中的索引，用于 distances_to_prey
    adv_index_map = {idx: i for i, idx in enumerate(adversary_indices)}
    self_adv_idx = adv_index_map[self_idx]

    self_pos = agent_positions[self_idx]
    self_vel = agent_velocities[self_idx]

    # -----------------------
    # 1. 接近目标的距离型奖励
    # -----------------------
    # 使用当前距离的负值作为 shaping
    self_dist_to_prey = np.linalg.norm(self_pos - prey_position)
    distance_reward = -self_dist_to_prey / world_size
    components["distance_reward"] = w_approach * distance_reward

    # 成功围捕 / 靠近额外加成：当所有追捕者都在 capture_threshold 内
    all_adv_distances = distances_to_prey
    all_inside_capture = bool(
        (len(all_adv_distances) == len(adversaries))
        and np.all(all_adv_distances < capture_threshold)
    )
    capture_bonus = 0.0
    if all_inside_capture:
        capture_bonus = w_capture_bonus
    components["capture_bonus"] = capture_bonus

    # 每步时间惩罚，鼓励尽快完成
    components["time_penalty"] = w_time_penalty

    # -----------------------
    # 2. 防碰撞奖励（追捕者-追捕者、追捕者-猎物）
    # -----------------------
    # 追捕者-追捕者碰撞惩罚
    collision_penalty = 0.0
    for other_idx in adversary_indices:
        if other_idx == self_idx:
            continue
        dist_aa = inter_agent_distances[self_idx, other_idx]
        if dist_aa < adv_size + adv_size:
            collision_penalty += 1.0
    components["adv_collision_penalty"] = w_collision_penalty * collision_penalty

    # 与猎物发生硬碰撞的轻微惩罚（鼓励围而不撞）
    prey_collision_penalty = 0.0
    if preys:
        prey_idx = name_to_index[preys[0].name]
        dist_ap = inter_agent_distances[self_idx, prey_idx]
        if dist_ap < adv_size + prey_size:
            prey_collision_penalty = 1.0
    components["prey_collision_penalty"] = (
        w_prey_collision_penalty * prey_collision_penalty
    )

    # -----------------------
    # 3. 包围队形：半径均匀 + 角度均匀
    # -----------------------
    # 以猎物为中心计算追捕者的极坐标
    adv_positions = agent_positions[adversary_indices]
    rel_pos = adv_positions - prey_position  # shape (n_adv, 2)
    radii = np.linalg.norm(rel_pos, axis=1)  # (n_adv,)
    angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])  # [-pi, pi]

    # 3.1 半径均匀性与接近理想半径
    #   - 惩罚半径的方差
    #   - 惩罚与 ideal_radius 的偏差
    if len(radii) > 1:
        radius_var = np.var(radii)
    else:
        radius_var = 0.0
    radius_deviation = np.mean(np.abs(radii - ideal_radius))

    radius_uniformity_reward = -radius_var - radius_deviation
    components["radius_uniformity_reward"] = (
        w_radius_uniformity * radius_uniformity_reward
    )

    # 3.2 角度均匀性（对于 3 个追捕者期望约 120 度均分）
    angle_uniformity_reward = 0.0
    if len(angles) >= 2:
        # 归一化到 [0, 2pi)，排序后计算相邻角差，包括首尾闭合
        angles_norm = (angles + 2.0 * np.pi) % (2.0 * np.pi)
        angles_sorted = np.sort(angles_norm)
        diffs = np.diff(angles_sorted)
        # 闭环差值
        last_gap = 2.0 * np.pi - np.sum(diffs)
        angle_gaps = np.concatenate([diffs, np.array([last_gap])])

        ideal_gap = 2.0 * np.pi / len(angles)
        angle_gap_deviation = np.mean(np.abs(angle_gaps - ideal_gap))

        angle_uniformity_reward = -angle_gap_deviation
    components["angle_uniformity_reward"] = (
        w_angle_uniformity * angle_uniformity_reward
    )

    # -----------------------
    # 4. 防止围捕圈过于松散 / 出现大缺口
    # -----------------------
    # 使用追捕者间距离的最大值（大于某值则惩罚）
    adv_pair_dists = []
    for i in range(len(adversary_indices)):
        for j in range(i + 1, len(adversary_indices)):
            idx_i = adversary_indices[i]
            idx_j = adversary_indices[j]
            adv_pair_dists.append(inter_agent_distances[idx_i, idx_j])
    if adv_pair_dists:
        max_pair_dist = max(adv_pair_dists)
    else:
        max_pair_dist = 0.0

    # 期望追捕者间距离与 (ideal_radius * 2) 同量级
    desired_pair_dist = 2.0 * ideal_radius
    spread_penalty = max(0.0, max_pair_dist - desired_pair_dist)
    components["spread_penalty"] = w_spread_penalty * spread_penalty

    # -----------------------
    # 5. 安全间距正向奖励（不过近）
    # -----------------------
    # 若和所有其他追捕者的距离都大于最小安全距离，则给一个小正奖励
    safe_spacing_bonus = 0.0
    all_safe = True
    for other_idx in adversary_indices:
        if other_idx == self_idx:
            continue
        dist_aa = inter_agent_distances[self_idx, other_idx]
        if dist_aa < min_pair_distance:
            all_safe = False
            break
    if all_safe:
        safe_spacing_bonus = 0.1
    components["safe_spacing_bonus"] = safe_spacing_bonus

    # -----------------------
    # 总奖励
    # -----------------------
    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
1. **分量贡献度**
   - 完全为0：`adv_collision_penalty` 为 0；`prey_collision_penalty` 近似 0，几乎不起作用。
   - 方差极大：`capture_bonus`（std≈1.23）、`boundary_penalty`（std≈2.37）波动大，易诱发策略震荡；`escape_reward`、`radius_uniformity_reward`、`spread_penalty` 也有较大方差。
   - 主导作用：`capture_penalty=-10` 且无方差，成为绝对主导负项；其余奖励普遍为负（特别是 `angle_uniformity_reward`、`boundary_penalty`、`radius_uniformity_reward`、`spread_penalty`、`distance_reward`），整体强烈惩罚驱动，只靠偶发的 `capture_bonus`、`safe_spacing_bonus` 抵消。

2. **任务瓶颈**
   - 成功率 33.13% 偏低，平均 86.7 步说明捕获效率差、探索周期长。
   - 强惩罚主导+稳定的大额捕获惩罚，暗示智能体在“避免失败”而非积极压缩包围，容易卡在保守局部最优（谨慎拖延、试探为主）。

3. **协同缺陷**
   - `formation_quality=0.2689` 偏低，`angle_uniformity_reward` 长期为负，表明包围圈结构松散，存在方向集中或空档。
   - `min_agent_distance=0.8369` 配合较低的 formation quality，可能出现局部扎堆、侧翼空缺，而非整体环形收缩。
   - `avg_distance_to_prey=1.4285` 较大，说明整体无法有效收缩包围圈，部分智能体疑似在远处“观望”式搭便车，未形成紧逼协同。

# 强制变异指令
【Reward Branch Augmentation (分支扩充)】
请完全保留原代码的现有逻辑和权重，新增一个（且仅新增一个）奖励或惩罚分量，用于解决诊断报告中缺失的协同行为引导。

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
    # 逃跑者奖励为 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5  # world_size * 0.2
    adv_size = 0.075
    prey_size = 0.050

    # 参数与权重（可调）
    w_approach = 1.0
    w_capture_bonus = 20.0
    w_time_penalty = -0.05
    w_collision_penalty = -5.0
    w_prey_collision_penalty = -2.0
    w_radius_uniformity = 0.5
    w_angle_uniformity = 0.5
    w_spread_penalty = -0.5
    w_cohesive_closure = 1.0  # 新增：协同闭合奖励权重

    # 安全距离（防止追捕者之间碰撞）
    safety_margin = 0.01
    min_pair_distance = 2.0 * adv_size + safety_margin

    # 理想包围半径（略小于 capture_threshold，避免太松散）
    ideal_radius = capture_threshold * 0.7

    # -----------------------
    # 从 global_state 解析数据
    # -----------------------
    agent_positions = global_state["agent_positions"]  # shape (n_agents, 2)
    agent_velocities = global_state["agent_velocities"]
    prey_position = global_state["prey_position"]
    prey_velocity = global_state["prey_velocity"]
    distances_to_prey = global_state["distances_to_prey"]
    inter_agent_distances = global_state["inter_agent_distances"]

    # 区分追捕者与猎物
    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    # 映射 agent_name -> index in agent_positions
    name_to_index = {a.name: i for i, a in enumerate(all_agents)}
    self_idx = name_to_index[agent_name]

    # 当前 agent 是否为追捕者（上面已经过滤，但这里保持一致性）
    self_agent = all_agents[self_idx]
    if not self_agent.adversary:
        return 0.0, {}

    # 追捕者索引列表（在 agent_positions 中的索引）
    adversary_indices = [name_to_index[a.name] for a in adversaries]

    # 该追捕者在 adversaries 列表中的索引，用于 distances_to_prey
    adv_index_map = {idx: i for i, idx in enumerate(adversary_indices)}
    self_adv_idx = adv_index_map[self_idx]

    self_pos = agent_positions[self_idx]
    self_vel = agent_velocities[self_idx]

    # -----------------------
    # 1. 接近目标的距离型奖励
    # -----------------------
    # 使用当前距离的负值作为 shaping
    self_dist_to_prey = np.linalg.norm(self_pos - prey_position)
    distance_reward = -self_dist_to_prey / world_size
    components["distance_reward"] = w_approach * distance_reward

    # 成功围捕 / 靠近额外加成：当所有追捕者都在 capture_threshold 内
    all_adv_distances = distances_to_prey
    all_inside_capture = bool(
        (len(all_adv_distances) == len(adversaries))
        and np.all(all_adv_distances < capture_threshold)
    )
    capture_bonus = 0.0
    if all_inside_capture:
        capture_bonus = w_capture_bonus
    components["capture_bonus"] = capture_bonus

    # 每步时间惩罚，鼓励尽快完成
    components["time_penalty"] = w_time_penalty

    # -----------------------
    # 2. 防碰撞奖励（追捕者-追捕者、追捕者-猎物）
    # -----------------------
    # 追捕者-追捕者碰撞惩罚
    collision_penalty = 0.0
    for other_idx in adversary_indices:
        if other_idx == self_idx:
            continue
        dist_aa = inter_agent_distances[self_idx, other_idx]
        if dist_aa < adv_size + adv_size:
            collision_penalty += 1.0
    components["adv_collision_penalty"] = w_collision_penalty * collision_penalty

    # 与猎物发生硬碰撞的轻微惩罚（鼓励围而不撞）
    prey_collision_penalty = 0.0
    if preys:
        prey_idx = name_to_index[preys[0].name]
        dist_ap = inter_agent_distances[self_idx, prey_idx]
        if dist_ap < adv_size + prey_size:
            prey_collision_penalty = 1.0
    components["prey_collision_penalty"] = (
        w_prey_collision_penalty * prey_collision_penalty
    )

    # -----------------------
    # 3. 包围队形：半径均匀 + 角度均匀
    # -----------------------
    # 以猎物为中心计算追捕者的极坐标
    adv_positions = agent_positions[adversary_indices]
    rel_pos = adv_positions - prey_position  # shape (n_adv, 2)
    radii = np.linalg.norm(rel_pos, axis=1)  # (n_adv,)
    angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])  # [-pi, pi]

    # 3.1 半径均匀性与接近理想半径
    #   - 惩罚半径的方差
    #   - 惩罚与 ideal_radius 的偏差
    if len(radii) > 1:
        radius_var = np.var(radii)
    else:
        radius_var = 0.0
    radius_deviation = np.mean(np.abs(radii - ideal_radius))

    radius_uniformity_reward = -radius_var - radius_deviation
    components["radius_uniformity_reward"] = (
        w_radius_uniformity * radius_uniformity_reward
    )

    # 3.2 角度均匀性（对于 3 个追捕者期望约 120 度均分）
    angle_uniformity_reward = 0.0
    if len(angles) >= 2:
        # 归一化到 [0, 2pi)，排序后计算相邻角差，包括首尾闭合
        angles_norm = (angles + 2.0 * np.pi) % (2.0 * np.pi)
        angles_sorted = np.sort(angles_norm)
        diffs = np.diff(angles_sorted)
        # 闭环差值
        last_gap = 2.0 * np.pi - np.sum(diffs)
        angle_gaps = np.concatenate([diffs, np.array([last_gap])])

        ideal_gap = 2.0 * np.pi / len(angles)
        angle_gap_deviation = np.mean(np.abs(angle_gaps - ideal_gap))

        angle_uniformity_reward = -angle_gap_deviation
    components["angle_uniformity_reward"] = (
        w_angle_uniformity * angle_uniformity_reward
    )

    # -----------------------
    # 4. 防止围捕圈过于松散 / 出现大缺口
    # -----------------------
    # 使用追捕者间距离的最大值（大于某值则惩罚）
    adv_pair_dists = []
    for i in range(len(adversary_indices)):
        for j in range(i + 1, len(adversary_indices)):
            idx_i = adversary_indices[i]
            idx_j = adversary_indices[j]
            adv_pair_dists.append(inter_agent_distances[idx_i, idx_j])
    if adv_pair_dists:
        max_pair_dist = max(adv_pair_dists)
    else:
        max_pair_dist = 0.0

    # 期望追捕者间距离与 (ideal_radius * 2) 同量级
    desired_pair_dist = 2.0 * ideal_radius
    spread_penalty = max(0.0, max_pair_dist - desired_pair_dist)
    components["spread_penalty"] = w_spread_penalty * spread_penalty

    # -----------------------
    # 5. 安全间距正向奖励（不过近）
    # -----------------------
    # 若和所有其他追捕者的距离都大于最小安全距离，则给一个小正奖励
    safe_spacing_bonus = 0.0
    all_safe = True
    for other_idx in adversary_indices:
        if other_idx == self_idx:
            continue
        dist_aa = inter_agent_distances[self_idx, other_idx]
        if dist_aa < min_pair_distance:
            all_safe = False
            break
    if all_safe:
        safe_spacing_bonus = 0.1
    components["safe_spacing_bonus"] = safe_spacing_bonus

    # -----------------------
    # 6. 新增：协同逼近闭合奖励
    # -----------------------
    # 只在有多个追捕者时计算
    cohesive_closure_reward = 0.0
    n_adv = len(adversary_indices)
    if n_adv > 1 and preys:
        # 当前追捕者与其他追捕者到猎物的平均距离差异
        other_dists = []
        for idx in adversary_indices:
            if idx == self_idx:
                continue
            d = np.linalg.norm(
                agent_positions[idx] - prey_position
            )
            other_dists.append(d)
        if other_dists:
            avg_other_dist = float(np.mean(other_dists))
            # 鼓励自己与队友保持相近距离且整体靠近目标
            dist_gap = abs(self_dist_to_prey - avg_other_dist)
            # 基础项：与队友平均距离差越小越好
            base_term = -dist_gap / world_size
            # 额外项：只有当全体在捕获阈值附近时才强化
            group_mean_dist = float(
                np.mean(distances_to_prey)
            ) if len(distances_to_prey) > 0 else 0.0
            proximity_factor = np.exp(
                -max(0.0, group_mean_dist - capture_threshold)
            )
            cohesive_closure_reward = base_term * proximity_factor
    components["cohesive_closure_reward"] = (
        w_cohesive_closure * cohesive_closure_reward
    )

    # -----------------------
    # 总奖励
    # -----------------------
    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
