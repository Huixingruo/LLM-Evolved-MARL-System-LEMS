# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker1_F2
- **Timestamp**: 20260414_145443

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

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
capture_penalty恒为-10主导整体收益，成功率仅41%，说明失败惩罚远大于成功收益，策略偏向“保守但不高效”。escape_reward与多项形态奖励(mean均为负)共同拉低回报，表明队形/包围相关设计当前多为噪声或反向信号。containment_bonus略有正向但波动大，合作收缩(cooperative_shrink_reward)略负，协同未形成稳定收益结构。碰撞相关惩罚全为0，未触发但也未体现“近距离压迫”的精细引导。协同指标显示包围角度波动大、队形质量低，存在明显协同缺陷但非由碰撞失衡导致，而是队形奖励设计失效。

[算子分配]
Candidate 0: F3
Candidate 1: F2
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

    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break
    if agent_index is None:
        return 0.0, components

    agent = world.agents[agent_index]

    adversary_indices = list(range(n_adv))
    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    d_ap = np.linalg.norm(agent.state.p_pos - prey_pos)

    # ---------------------------
    # 1. 距离引导奖励（重构：平滑带状目标 + 归一化全局 shaping）
    # ---------------------------
    target_radius = capture_threshold * 0.9
    # 在 [target_radius - band, target_radius + band] 内视为基本可接受
    band = 0.15 * capture_threshold
    distance_error = d_ap - target_radius
    # 使用平滑带状惩罚：带内误差权重减弱，带外逐渐增强
    abs_err = abs(distance_error)
    if abs_err <= band:
        # 轻微误差采用软惩罚
        distance_reward = -0.2 * (abs_err ** 2) / (band ** 2 + 1e-6)
    else:
        # 带外使用加权二次惩罚，避免过大负值拖垮整体回报
        excess = abs_err - band
        distance_reward = -0.2 - 0.8 * (excess ** 2) / \
            ((world_size - target_radius) ** 2 + 1e-6)

    # 全局均值距离归一化（重构：相对于 world_size 进行缩放）
    if n_adv > 0:
        mean_dist_to_prey = float(np.mean(distances_to_prey))
    else:
        mean_dist_to_prey = d_ap
    # 将距离归一化到 [0,1] 区间，再映射到 [-1,0]
    norm_mean_dist = np.clip(mean_dist_to_prey / world_size, 0.0, 1.0)
    global_distance_reward = -norm_mean_dist

    components["distance_shaping_self"] = 0.7 * distance_reward
    components["distance_shaping_global"] = 0.3 * global_distance_reward

    # ---------------------------
    # 2. 防碰撞奖励（保持不变）
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
    # 3. 队形奖励：均匀包围圈（重构：有界化和软窗口）
    # ---------------------------
    formation_angle_reward = 0.0
    formation_radius_reward = 0.0
    containment_bonus = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-6
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        # 角度均匀性：重构为有界奖励，避免长尾负值
        target_delta = 2.0 * np.pi / float(n_adv)
        angles_sorted = np.sort(rel_angles)
        deltas = []
        for i in range(n_adv - 1):
            deltas.append(angles_sorted[i + 1] - angles_sorted[i])
        deltas.append((angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        deltas = np.array(deltas)
        angle_error = np.mean(np.abs(deltas - target_delta))

        # 归一化误差：最大误差不会超过 π
        norm_angle_error = np.clip(angle_error / np.pi, 0.0, 1.0)
        # 使用平滑形状：1 - norm_error^2，范围 [0,1]
        angle_quality = 1.0 - norm_angle_error ** 2
        # 映射到 [-1,0]，使其成为惩罚但不会过分拉低回报
        formation_angle_reward = angle_quality - 1.0

        # 半径均匀性：重构为相对方差形式，避免大尺度依赖
        radius_spread = np.max(rel_dists) - np.min(rel_dists)
        mean_radius = float(np.mean(rel_dists))
        if mean_radius > 1e-6:
            rel_spread = np.clip(radius_spread / mean_radius, 0.0, 2.0)
        else:
            rel_spread = 0.0
        radius_quality = 1.0 - 0.5 * (rel_spread ** 2)
        radius_quality = max(radius_quality, -1.0)
        formation_radius_reward = radius_quality - 1.0

        # 包含性：重构为夺标式 sigmoid 奖励，弱化波动
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

            # 使用软 bonus，缩小幅度并使用 sigmoidal 变换
            if is_inside:
                # 根据三角形平均半径，内圈更紧给略高奖励
                tri_radius = float(np.mean(np.linalg.norm(
                    adv_positions - prey_pos, axis=1
                )))
                # 期望三角半径接近 capture_threshold
                r_err = abs(tri_radius - capture_threshold)
                # 将误差映射到 [0,1]
                r_scale = capture_threshold
                norm_r_err = np.clip(r_err / (r_scale + 1e-6), 0.0, 2.0)
                tightness = 1.0 / (1.0 + norm_r_err)
                containment_bonus = 0.5 * tightness

    components["formation_angle_reward"] = 0.5 * formation_angle_reward
    components["formation_radius_reward"] = 0.5 * formation_radius_reward
    components["containment_bonus"] = containment_bonus

    # ---------------------------
    # 4. 时间惩罚（保持不变）
    # ---------------------------
    time_penalty = -0.01
    components["time_penalty"] = time_penalty

    # ---------------------------
    # 5. 围捕完成奖励（重构：基于质量的分段奖励）
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            # 使用重构后的 formation 指标合成质量得分，范围在 [0,1]
            # formation_*_reward 在 [-1,0] 左右，先映射回 [0,1]
            angle_score = 1.0 + formation_angle_reward
            radius_score = 1.0 + formation_radius_reward
            angle_score = np.clip(angle_score, 0.0, 1.0)
            radius_score = np.clip(radius_score, 0.0, 1.0)

            quality_score = 0.5 * angle_score + 0.5 * radius_score
            # 分段奖励：基础成功 + 质量附加（总量适当）
            base_success = 6.0
            extra = 4.0 * quality_score
            capture_bonus = base_success + extra
    components["capture_bonus"] = capture_bonus

    # ---------------------------
    # 6. 协同收缩奖励（重构：相对缩放 + 软饱和）
    # ---------------------------
    cooperative_shrink_reward = 0.0
    if n_adv > 1:
        team_mean_radius = float(np.mean(distances_to_prey))
        # 相对 world_size 归一化
        norm_team_radius = np.clip(team_mean_radius / world_size, 0.0, 1.5)
        # 使用 1 - exp(-x) 形状，再映射到 [-0.2, 0]
        shrink_factor = 1.0 - np.exp(-norm_team_radius)
        cooperative_shrink_reward = -0.2 * shrink_factor
    components["cooperative_shrink_reward"] = cooperative_shrink_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
