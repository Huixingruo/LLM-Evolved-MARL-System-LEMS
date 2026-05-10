# LLM Interaction Log

- **Generation**: 3
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker0_F2
- **Timestamp**: 20260414_155216

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
    # 非追捕者奖励恒为 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ---------------------------
    # 物理常量（硬编码）
    # ---------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 形状与索引辅助
    agent_positions = np.array(global_state["agent_positions"])
    prey_pos = np.array(global_state["prey_position"])
    distances_to_prey = np.array(global_state["distances_to_prey"])
    inter_agent_distances = np.array(global_state["inter_agent_distances"])

    # 追捕者与猎物数量
    n_agents = agent_positions.shape[0]
    # 在 MPE 中 adversary 一般在列表前部；这里我们通过距离数组大小推断追捕者数量
    n_adv = len(distances_to_prey)

    # 找到当前 agent 在 world.agents 中的索引
    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break
    if agent_index is None:
        return 0.0, components

    agent = world.agents[agent_index]

    # ---------------------------
    # 基本几何量构造
    # ---------------------------
    # 追捕者索引列表（假定为前 n_adv 个）
    adversary_indices = list(range(n_adv))

    # 当前追捕者在 adversary 索引中的位置（若不在其中，则直接返回 0 奖励）
    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    # 当前追捕者与猎物的距离
    d_ap = np.linalg.norm(agent.state.p_pos - prey_pos)

    # ---------------------------
    # 1. 距离引导奖励（靠近并保持合适半径）
    # ---------------------------
    # 目标半径：略小于 capture_threshold，鼓励形成稳定包围圈而不是贴脸碰撞
    target_radius = capture_threshold * 0.9
    # 当前追捕者到猎物距离
    distance_error = d_ap - target_radius
    # 使用负的绝对误差作为 shaping，使距离逼近 target_radius
    distance_reward = -abs(distance_error)
    # 额外的全局靠近进度奖励：使用所有追捕者平均距离
    mean_dist_to_prey = float(np.mean(distances_to_prey)) if n_adv > 0 else d_ap
    global_distance_reward = -mean_dist_to_prey

    components["distance_shaping_self"] = 0.7 * distance_reward
    components["distance_shaping_global"] = 0.3 * global_distance_reward

    # ---------------------------
    # 2. 防碰撞奖励（追捕者-追捕者）
    # ---------------------------
    # 安全距离 = 两追捕者半径之和再加一点 margin
    safe_margin = 0.02
    min_safe_dist = 2 * adv_size + safe_margin

    collision_penalty = 0.0
    near_collision_penalty = 0.0

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j]
        # 严重碰撞
        if d_aa < 2 * adv_size:
            collision_penalty -= 5.0
        # 接近碰撞区域
        elif d_aa < min_safe_dist:
            near_collision_penalty -= (min_safe_dist - d_aa)

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ---------------------------
    # 3. 队形奖励：均匀包围圈
    # ---------------------------
    formation_angle_reward = 0.0
    formation_radius_reward = 0.0
    containment_bonus = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        # 追捕者相对于猎物的向量与角度
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos  # shape: (n_adv, 2)
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-6
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        # 角度均匀性：希望相邻角度差接近 2π/n_adv
        target_delta = 2.0 * np.pi / float(n_adv)
        angles_sorted = np.sort(rel_angles)
        deltas = []
        for i in range(n_adv - 1):
            deltas.append(angles_sorted[i + 1] - angles_sorted[i])
        # 闭合差值
        deltas.append((angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        deltas = np.array(deltas)
        angle_error = np.mean(np.abs(deltas - target_delta))
        # 负误差作为奖励
        formation_angle_reward = -angle_error

        # 半径均匀性：希望所有追捕者距离猎物相近
        radius_spread = np.max(rel_dists) - np.min(rel_dists)
        formation_radius_reward = -radius_spread

        # 包含性：若猎物在追捕者形成的多边形内则给额外奖励
        # 仅在 n_adv == 3 时使用三角形包含测试
        if n_adv == 3:
            p = prey_pos
            a, b, c = adv_positions

            def _sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
                       (p2[0] - p3[0]) * (p1[1] - p3[1])

            b1 = _sign(p, a, b) < 0.0
            b2 = _sign(p, b, c) < 0.0
            b3 = _sign(p, c, a) < 0.0
            is_inside = (b1 == b2) and (b2 == b3)
            if is_inside:
                containment_bonus = 2.0

    components["formation_angle_reward"] = 0.5 * formation_angle_reward
    components["formation_radius_reward"] = 0.5 * formation_radius_reward
    components["containment_bonus"] = containment_bonus

    # ---------------------------
    # 4. 时间惩罚（鼓励尽快完成）
    # ---------------------------
    # 每一步给一个小的时间成本
    time_penalty = -0.01
    components["time_penalty"] = time_penalty

    # ---------------------------
    # 5. 围捕完成奖励
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        # 所有追捕者都在 capture_threshold 内，且角度/半径相对合理
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            # 使用当前构造的 formation 指标判断质量
            quality_score = (
                -abs(formation_angle_reward) -
                abs(formation_radius_reward)
            )
            capture_bonus = 10.0 + 5.0 * quality_score
    components["capture_bonus"] = capture_bonus

    # ---------------------------
    # 6. 新增协同收缩奖励：鼓励整体包围圈半径缩小
    # ---------------------------
    cooperative_shrink_reward = 0.0
    if n_adv > 1:
        # 使用所有追捕者到猎物的均值作为团队半径，负号鼓励整体靠近
        team_mean_radius = float(np.mean(distances_to_prey))
        cooperative_shrink_reward = -0.1 * team_mean_radius
    components["cooperative_shrink_reward"] = cooperative_shrink_reward

    # ---------------------------
    # 7. 新增协同对称收缩奖励：鼓励队友在同一尺度上压缩
    # ---------------------------
    symmetric_shrink_reward = 0.0
    if n_adv > 1:
        # 当前追捕者距离与团队平均距离的差值越小越好
        team_mean_radius = float(np.mean(distances_to_prey))
        local_distance = float(distances_to_prey[local_adv_idx])
        symmetric_shrink_reward = -0.05 * abs(local_distance - team_mean_radius)
    components["symmetric_shrink_reward"] = symmetric_shrink_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
主导信号为强烈的capture_penalty与持续time_penalty，使策略偏向保守拖时而非高效围捕。formation_angle/formation_radius/distance_shaping系列长期为负，说明队形结构与收缩策略未奏效甚至误导，cooperative_shrink_reward同样偏负。containment_bonus和capture_bonus偏小且方差大，正向信号不足。near_collision/collision几乎为0，说明无明显碰撞问题，但encirclement_angle_std较大、formation_quality偏低，体现协同围捕松散，存在“各打各的”式协作缺陷而非明显搭便车。

[算子分配]
Candidate 0: F2
Candidate 1: F3
Candidate 2: F1
Candidate 3: L1

# 强制变异指令
【Reward Component Reconstruction (失效分量重构)】
请定位诊断报告中指出的"失效"或"起反作用"的分量。不要直接删除它们，而是重构其数学逻辑（例如：将线性惩罚改为指数惩罚、引入平滑阈值或改变距离函数的计算方式）。保持其他有效分量不变。

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

    # ---------------------------
    # 物理常量（硬编码）
    # ---------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 形状与索引辅助
    agent_positions = np.array(global_state["agent_positions"])
    prey_pos = np.array(global_state["prey_position"])
    distances_to_prey = np.array(global_state["distances_to_prey"])
    inter_agent_distances = np.array(global_state["inter_agent_distances"])

    n_agents = agent_positions.shape[0]
    n_adv = len(distances_to_prey)

    # 找到当前 agent 在 world.agents 中的索引
    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break
    if agent_index is None:
        return 0.0, components

    agent = world.agents[agent_index]

    # 追捕者索引列表（假定为前 n_adv 个）
    adversary_indices = list(range(n_adv))

    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    d_ap = np.linalg.norm(agent.state.p_pos - prey_pos)

    # ---------------------------
    # 1. 距离引导奖励（重构：从线性绝对误差到平滑阈值 + 指数靠近）
    # ---------------------------
    # 将目标距离区间设置为 [r_min, r_max]，在区间内给予平坦高奖励，
    # 区间外采用指数衰减的 shaping，避免长期强负值
    r_min = capture_threshold * 0.6
    r_max = capture_threshold * 1.0

    if d_ap < r_min:
        # 离得过近：平滑惩罚，避免贴脸撞击，但不过分拉走
        distance_shaping_self = -0.3 * np.exp(-(d_ap / r_min))
    elif d_ap > r_max:
        # 离得偏远：指数靠近奖励，高阶靠近时梯度增强
        gap = d_ap - r_max
        distance_shaping_self = -0.5 * gap ** 2
    else:
        # 区间内给一个小正奖励，鼓励停留在合适包围圈
        distance_shaping_self = 0.2

    # 全局距离重构：对团队平均距离使用软阈值惩罚，避免长期大负值
    mean_dist_to_prey = float(np.mean(distances_to_prey)) if n_adv > 0 else d_ap
    if mean_dist_to_prey <= r_max:
        global_distance_reward = 0.2
    else:
        global_distance_reward = -0.3 * (mean_dist_to_prey - r_max)

    components["distance_shaping_self"] = 0.7 * distance_shaping_self
    components["distance_shaping_global"] = 0.3 * global_distance_reward

    # ---------------------------
    # 2. 防碰撞奖励（保持不变：无明显失效）
    # ---------------------------
    safe_margin = 0.02
    min_safe_dist = 2 * adv_size + safe_margin

    collision_penalty = 0.0
    near_collision_penalty = 0.0

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j]
        if d_aa < 2 * adv_size:
            collision_penalty -= 5.0
        elif d_aa < min_safe_dist:
            near_collision_penalty -= (min_safe_dist - d_aa)

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ---------------------------
    # 3. 队形奖励重构：从纯负差值转为“基线 + 误差扣减”
    # ---------------------------
    formation_angle_reward = 0.0
    formation_radius_raw_reward = 0.0
    containment_bonus = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-6
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        target_delta = 2.0 * np.pi / float(n_adv)
        angles_sorted = np.sort(rel_angles)
        deltas = []
        for i in range(n_adv - 1):
            deltas.append(angles_sorted[i + 1] - angles_sorted[i])
        deltas.append((angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        deltas = np.array(deltas)

        angle_error = np.mean(np.abs(deltas - target_delta))
        angle_error_clipped = np.clip(angle_error, 0.0, np.pi)
        # 新逻辑：给一个常数基线，再按归一化误差扣减，保证总体可为正
        formation_angle_reward = 1.0 - (angle_error_clipped / np.pi)

        radius_spread = np.max(rel_dists) - np.min(rel_dists)
        # 归一化半径差，以 capture_threshold 为尺度
        radius_error_norm = np.clip(radius_spread / (capture_threshold + 1e-6),
                                    0.0, 2.0)
        formation_radius_raw_reward = 1.0 - 0.5 * radius_error_norm

        # 包含性判定保持不变
        if n_adv == 3:
            p = prey_pos
            a, b, c = adv_positions

            def _sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
                       (p2[0] - p3[0]) * (p1[1] - p3[1])

            b1 = _sign(p, a, b) < 0.0
            b2 = _sign(p, b, c) < 0.0
            b3 = _sign(p, c, a) < 0.0
            is_inside = (b1 == b2) and (b2 == b3)
            if is_inside:
                containment_bonus = 3.0

    # 将重构后的 formation 奖励以缩放形式合入
    components["formation_angle_reward"] = 0.5 * formation_angle_reward
    components["formation_radius_reward"] = 0.5 * formation_radius_raw_reward
    components["containment_bonus"] = containment_bonus

    # ---------------------------
    # 4. 时间惩罚重构：由固定负值改为进度敏感
    # ---------------------------
    # 若队伍平均距离较近则减小时间惩罚，甚至给予微小时间奖金
    # 以 r_progress 为尺度衡量进展
    r_progress = capture_threshold * 1.5
    if mean_dist_to_prey <= capture_threshold:
        time_penalty = +0.02
    elif mean_dist_to_prey <= r_progress:
        frac = (mean_dist_to_prey - capture_threshold) / \
               (r_progress - capture_threshold + 1e-6)
        time_penalty = -0.01 * frac
    else:
        time_penalty = -0.02

    components["time_penalty"] = time_penalty

    # ---------------------------
    # 5. 围捕完成奖励（保持结构，仅稍作平滑）
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            quality_score = (
                (components["formation_angle_reward"] +
                 components["formation_radius_reward"])
            )
            capture_bonus = 8.0 + 6.0 * quality_score
    components["capture_bonus"] = capture_bonus

    # ---------------------------
    # 6. 协同收缩奖励重构：从线性负距到缓和的指数靠近
    # ---------------------------
    cooperative_shrink_reward = 0.0
    if n_adv > 1:
        # 对团队平均距离使用指数靠近：远时弱信号，近时增强
        # 将距离归一化到 world_size
        norm_rad = np.clip(mean_dist_to_prey / world_size, 0.0, 1.0)
        cooperative_shrink_reward = 0.5 * (1.0 - np.exp(-4.0 * (1.0 - norm_rad)))
    components["cooperative_shrink_reward"] = cooperative_shrink_reward

    # ---------------------------
    # 7. 协同对称收缩奖励重构：从线性误差到容忍区间 + 平滑惩罚
    # ---------------------------
    symmetric_shrink_reward = 0.0
    if n_adv > 1:
        team_mean_radius = float(np.mean(distances_to_prey))
        local_distance = float(distances_to_prey[local_adv_idx])
        diff = abs(local_distance - team_mean_radius)

        tol = capture_threshold * 0.2
        if diff <= tol:
            symmetric_shrink_reward = 0.05
        else:
            # 超出容忍区间的差值使用平滑平方惩罚
            excess = diff - tol
            symmetric_shrink_reward = -0.05 * (excess ** 2)
    components["symmetric_shrink_reward"] = symmetric_shrink_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
