# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260414_145451

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

    # 基本状态
    agent_positions = np.array(global_state["agent_positions"])
    agent_velocities = np.array(global_state["agent_velocities"])
    prey_pos = np.array(global_state["prey_position"])
    prey_vel = np.array(global_state["prey_velocity"])
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

    # 当前是否属于追捕者索引范围（默认前 n_adv 为追捕者）
    adversary_indices = list(range(n_adv))
    if agent_index not in adversary_indices:
        return 0.0, components

    # ---------------------------
    # 极坐标势场视角构建
    # ---------------------------
    agent_pos = agent_positions[agent_index]
    agent_vel = agent_velocities[agent_index]

    # 追捕者相对猎物向量与极坐标
    rel_vec = agent_pos - prey_pos
    dist_to_prey = np.linalg.norm(rel_vec) + 1e-8
    angle_to_prey = np.arctan2(rel_vec[1], rel_vec[0])

    # 全体追捕者的极坐标
    if n_adv > 0:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs_all = adv_positions - prey_pos
        dists_all = np.linalg.norm(rel_vecs_all, axis=1) + 1e-8
        angles_all = np.arctan2(rel_vecs_all[:, 1], rel_vecs_all[:, 0])
    else:
        dists_all = np.array([dist_to_prey])
        angles_all = np.array([angle_to_prey])

    # ---------------------------
    # 1. 径向势场：环形“捕圈”势能
    # ---------------------------
    # 目标半径为捕获阈值附近，使用平滑凸函数构造势能
    target_radius = capture_threshold * 0.9
    radius_width = capture_threshold * 0.5  # 势阱宽度

    radius_error = (dist_to_prey - target_radius) / (radius_width + 1e-8)
    # 采用 -exp(-x^2) 型势阱，使在目标环附近奖励接近 0，上下偏离都为负
    ring_potential_self = -np.exp(-radius_error**2)

    # 全局团队半径势场：鼓励所有追捕者围绕目标半径收缩
    if n_adv > 0:
        radius_error_team = (dists_all - target_radius) / (radius_width + 1e-8)
        ring_potential_team = -np.mean(np.exp(-radius_error_team**2))
    else:
        ring_potential_team = ring_potential_self

    components["ring_potential_self"] = 1.0 * ring_potential_self
    components["ring_potential_team"] = 0.5 * ring_potential_team

    # ---------------------------
    # 2. 角度势场：极坐标均匀覆盖
    # ---------------------------
    # 强制全新的视角：在角度空间建立“电荷斥力”，鼓励追捕者在 0~2π 上均匀排布
    angle_repulsion_self = 0.0
    angle_repulsion_team = 0.0

    if n_adv > 1:
        # 归一化到 [0, 2π)
        base_angle = angle_to_prey % (2.0 * np.pi)
        all_angles = angles_all % (2.0 * np.pi)

        # 当前追捕者在 adversary_indices 中的局部索引
        local_idx = adversary_indices.index(agent_index)

        # 与其他追捕者的角度距离（在圆环上最小弧长）
        for k, ang in enumerate(all_angles):
            if k == local_idx:
                continue
            diff = np.abs(base_angle - ang)
            diff = min(diff, 2.0 * np.pi - diff)  # 圆上的最小角差

            # 角度斥力：距离越近，惩罚越大；超过一定阈值趋近于 0
            angle_sigma = (2.0 * np.pi / n_adv) * 0.7
            repulsion = np.exp(-(diff**2) / (2.0 * angle_sigma**2))
            angle_repulsion_self -= repulsion

        # 团队级角度熵：鼓励角度分布接近均匀
        # 将 [0, 2π) 划分为 n_adv 个扇区，计算直方图熵
        num_bins = max(4, n_adv)
        hist, _ = np.histogram(all_angles, bins=num_bins, range=(0.0, 2.0 * np.pi))
        prob = hist.astype(np.float32) / (np.sum(hist) + 1e-8)
        # 避免 log(0)
        non_zero = prob > 0
        entropy = -np.sum(prob[non_zero] * np.log(prob[non_zero] + 1e-8))
        # 最大熵对应均匀分布，奖励归一化到 [0, 1] 左右
        max_entropy = np.log(num_bins + 1e-8)
        angle_entropy_reward = entropy / (max_entropy + 1e-8)
        angle_repulsion_team += angle_entropy_reward

    components["angle_repulsion_self"] = 0.5 * angle_repulsion_self
    components["angle_uniformity_team"] = 0.5 * angle_repulsion_team

    # ---------------------------
    # 3. 逃逸抑制势场：锁定猎物运动方向
    # ---------------------------
    # 在极坐标中，希望追捕者在猎物速度方向前后形成“闸门”
    gate_reward = 0.0
    prey_speed = np.linalg.norm(prey_vel)

    if prey_speed > 1e-4 and n_adv > 0:
        prey_dir = prey_vel / prey_speed
        prey_angle = np.arctan2(prey_dir[1], prey_dir[0])

        # 当前追捕者相对猎物的极角与猎物运动方向的角差
        diff = np.abs((angle_to_prey % (2.0 * np.pi)) - (prey_angle % (2.0 * np.pi)))
        diff = min(diff, 2.0 * np.pi - diff)

        # 希望在猎物前后方（0 和 π 附近）有追捕者
        # 对当前追捕者，给出两个“目标角”：0 和 π（相对于猎物速度方向）
        target_diffs = [0.0, np.pi]
        window = np.pi / 3.0  # 宽容角
        # 对每个目标角都给一个高斯势阱，取最大值
        gate_score = 0.0
        for td in target_diffs:
            d = min(abs(diff - td), 2.0 * np.pi - abs(diff - td))
            gate_score = max(gate_score, np.exp(-(d**2) / (2.0 * window**2)))
        gate_reward += gate_score

    components["prey_direction_gate"] = 0.5 * gate_reward

    # ---------------------------
    # 4. 速度对齐与协同缩放（完全不同范式）
    # ---------------------------
    # 面向猎物的径向速度分量：鼓励沿径向靠近而不是切线乱跑
    radial_velocity_reward = 0.0
    tangential_slowdown_reward = 0.0

    if dist_to_prey > 1e-8:
        radial_dir = rel_vec / dist_to_prey
        radial_speed = -np.dot(agent_vel, radial_dir)  # 朝向猎物为正
        tangential_vec = agent_vel - radial_speed * radial_dir
        tangential_speed = np.linalg.norm(tangential_vec)

        # 在环外（> target_radius）时，鼓励正向径向速度
        if dist_to_prey > target_radius:
            radial_velocity_reward += radial_speed
        else:
            # 在环内时，鼓励减小径向速度，避免撞进中心
            radial_velocity_reward -= abs(radial_speed)

        # 一般情况下抑制过大的切向速度，但不强行为 0
        tangential_slowdown_reward -= 0.2 * tangential_speed

    components["radial_velocity_reward"] = 0.3 * radial_velocity_reward
    components["tangential_slowdown"] = 0.3 * tangential_slowdown_reward

    # 团队尺度统一：所有追捕者距离的标准差越小越好
    team_scale_reward = 0.0
    if n_adv > 1:
        scale_std = float(np.std(dists_all))
        team_scale_reward = -scale_std
    components["team_scale_uniformity"] = 0.5 * team_scale_reward

    # ---------------------------
    # 5. 局部排斥势场：避免追捕者之间碰撞但允许压迫
    # ---------------------------
    # 基于距离的软势场，不直接依赖碰撞事件
    safe_margin = 0.02
    min_safe_dist = 2 * adv_size + safe_margin
    repulsion_strength = 0.5

    soft_repulsion = 0.0
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j] + 1e-8
        if d_aa < min_safe_dist:
            # 使用 1/d^2 型软斥力，近距离强，远处几乎无影响
            soft_repulsion -= repulsion_strength * (1.0 / (d_aa**2))

    components["soft_repulsion"] = soft_repulsion

    # ---------------------------
    # 6. 捕获势场终端奖励（环带内且角度覆盖良好）
    # ---------------------------
    capture_shape_bonus = 0.0
    if n_adv > 0:
        # 所有追捕者均进入捕获阈值以内
        in_cap = distances_to_prey < capture_threshold
        all_in_cap = bool(np.all(in_cap))

        if all_in_cap:
            # 使用角度均匀性熵 + 半径一致性构造终端奖励
            # 半径一致性
            radius_spread = float(np.max(dists_all) - np.min(dists_all))
            radius_score = np.exp(-radius_spread**2 / (0.1**2 + 1e-8))

            # 角度均匀性（若前面没有算出则重算）
            num_bins_cap = max(4, n_adv)
            hist_cap, _ = np.histogram(
                angles_all % (2.0 * np.pi),
                bins=num_bins_cap,
                range=(0.0, 2.0 * np.pi),
            )
            prob_cap = hist_cap.astype(np.float32) / (np.sum(hist_cap) + 1e-8)
            non_zero_cap = prob_cap > 0
            entropy_cap = -np.sum(
                prob_cap[non_zero_cap] * np.log(prob_cap[non_zero_cap] + 1e-8)
            )
            max_entropy_cap = np.log(num_bins_cap + 1e-8)
            angle_score_cap = entropy_cap / (max_entropy_cap + 1e-8)

            capture_shape_bonus = 5.0 * (0.5 * radius_score + 0.5 * angle_score_cap)

    components["capture_shape_bonus"] = capture_shape_bonus

    # ---------------------------
    # 7. 轻微时间成本（避免无限拖延）
    # ---------------------------
    components["time_cost"] = -0.005

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
