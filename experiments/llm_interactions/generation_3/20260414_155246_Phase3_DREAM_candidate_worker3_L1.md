# LLM Interaction Log

- **Generation**: 3
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260414_155246

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

    # ---------------------------
    # 物理与环境常量（硬编码）
    # ---------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_threshold = 0.5

    # 势场尺度参数
    # 距离势场衰减尺度（相对于 world_size）
    prey_attract_scale = 0.7 * world_size
    formation_shell_radius = 0.9 * capture_threshold
    # 角度协同尺度（角差正则化）
    angle_std_target = 0.5  # 角度标准差的目标值（弧度）

    # 安全距离参数
    safe_margin = 0.02
    min_safe_dist = 2 * adv_size + safe_margin

    # 时间成本权重
    step_cost = -0.005

    # ---------------------------
    # 几何与索引信息
    # ---------------------------
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

    # 追捕者索引假设为前 n_adv 个
    adversary_indices = list(range(n_adv))
    if agent_index not in adversary_indices:
        return 0.0, components

    local_adv_idx = adversary_indices.index(agent_index)

    # 当前追捕者的状态
    p_pos = agent.state.p_pos
    p_vel = agent.state.p_vel

    # 相对猎物向量（极坐标视角）
    rel_vec = p_pos - prey_pos
    dist_to_prey = np.linalg.norm(rel_vec) + 1e-8
    rel_dir = rel_vec / dist_to_prey
    angle_to_prey = np.arctan2(rel_vec[1], rel_vec[0])

    # ---------------------------
    # 1. 全局势场：围捕势壳与径向控制
    #    视角：猎物周围存在一条目标“势壳”，追捕者应聚集其上。
    # ---------------------------
    # (1) 径向势：鼓励距离落在 formation_shell_radius 附近
    radial_error = dist_to_prey - formation_shell_radius
    # 使用平滑二次势场：越接近半径，势能越低（奖励越高）
    radial_potential = - (radial_error ** 2) / (formation_shell_radius ** 2 + 1e-8)

    # (2) 全局压缩势：团队平均半径越小势能越低，引导整体向内收缩
    if n_adv > 0:
        team_mean_radius = float(np.mean(distances_to_prey))
    else:
        team_mean_radius = dist_to_prey
    # 指数衰减：远处梯度大、近处梯度小，避免过度挤压
    global_radial_potential = -np.exp(team_mean_radius / prey_attract_scale)

    components["radial_shell_potential"] = 1.0 * radial_potential
    components["global_compression_potential"] = 0.5 * global_radial_potential

    # ---------------------------
    # 2. 极坐标队形势：角度协同与对称包围
    #    视角：在极坐标中，追捕者应在猎物周围角度上均匀铺开。
    # ---------------------------
    angle_alignment_potential = 0.0
    angular_spread_potential = 0.0
    encirclement_potential = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        adv_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
        adv_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        # 标准化角度到 [0, 2π)
        adv_angles = (adv_angles + 2.0 * np.pi) % (2.0 * np.pi)

        # (1) 当前追捕者与队友角度相对均匀：
        #     使用局部角度间隔：最近邻角差越接近目标值越好。
        target_delta = 2.0 * np.pi / float(n_adv)
        # 计算所有角度排序后的间隔
        angles_sorted = np.sort(adv_angles)
        deltas = np.empty_like(angles_sorted)
        deltas[:-1] = angles_sorted[1:] - angles_sorted[:-1]
        deltas[-1] = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]

        # 找到当前追捕者在排序后对应的间隔误差均值
        # 为简化：使用全局平均间隔误差作为代理势
        interval_error = np.mean(np.abs(deltas - target_delta))
        angle_alignment_potential = - (interval_error ** 2) / (
            (target_delta ** 2) + 1e-8
        )

        # (2) 角度标准差控制：希望分布覆盖整个 2π 而不过度聚集
        # 将角度映射到单位圆，使用向量和长度衡量扩散程度
        unit_vectors = np.stack(
            [np.cos(adv_angles), np.sin(adv_angles)], axis=-1
        )
        mean_vec = np.mean(unit_vectors, axis=0)
        mean_len = np.linalg.norm(mean_vec)
        # mean_len 接近 0 表示分布均匀；接近 1 表示集中
        # 将其视为“角度集中度”，鼓励中等集中（不过度聚集亦不过度稀疏）
        # 使用一个目标集中度对应 angle_std_target 的柔和势
        # 简化：目标集中度 target_concentration 在 (0,1) 内
        target_concentration = np.exp(-angle_std_target)
        concentration_error = mean_len - target_concentration
        angular_spread_potential = - (concentration_error ** 2)

        # (3) 环绕度势：利用猎物到各追捕者连线的方向相对猎物速度方向
        # 若从猎物视角看，追捕者分布在其运动方向两侧，环绕程度更好。
        prey_speed = np.linalg.norm(prey_vel)
        if prey_speed > 1e-6:
            prey_dir = prey_vel / prey_speed
            # 各追捕者单位方向向量（从猎物指向追捕者）
            unit_dirs = rel_vecs / adv_dists[:, None]
            # 与猎物运动方向投影
            projections = unit_dirs @ prey_dir
            # 希望有正有负（两侧都有追捕者），则投影方差越大越好
            proj_var = np.var(projections)
            encirclement_potential = proj_var
        else:
            # 猎物不动时，鼓励覆盖整个圆周：使用 1 - mean_len
            encirclement_potential = 1.0 - mean_len

    components["angle_alignment_potential"] = 0.7 * angle_alignment_potential
    components["angular_spread_potential"] = 0.5 * angular_spread_potential
    components["encirclement_potential"] = 0.8 * encirclement_potential

    # ---------------------------
    # 3. 局部协同：相对极坐标同步压缩
    #    视角：每个追捕者的半径与团队平均半径一致，避免“跑前头”或“落太后”。
    # ---------------------------
    symmetric_radius_potential = 0.0
    if n_adv > 1:
        local_radius = float(distances_to_prey[local_adv_idx])
        # 将团队平均半径作为动态势场“等位线”
        radius_deviation = local_radius - team_mean_radius
        symmetric_radius_potential = - (radius_deviation ** 2) / (
            (team_mean_radius ** 2) + 1e-8
        )

    components["symmetric_radius_potential"] = 0.6 * symmetric_radius_potential

    # ---------------------------
    # 4. 防碰撞势：短程排斥场
    #    视角：追捕者之间存在强短程斥力，避免靠得过近。
    # ---------------------------
    collision_repulsive_potential = 0.0
    near_repulsive_potential = 0.0

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j] + 1e-8

        # 使用 1/r 型斥力，在阈值内才有效
        if d_aa < 2 * adv_size:
            collision_repulsive_potential -= 5.0 / d_aa
        elif d_aa < min_safe_dist:
            # 软斥力：越接近安全边界，势能越高（奖励越低）
            gap = min_safe_dist - d_aa
            near_repulsive_potential -= gap ** 2 / (min_safe_dist ** 2)

    components["collision_repulsive_potential"] = collision_repulsive_potential
    components["near_repulsive_potential"] = near_repulsive_potential

    # ---------------------------
    # 5. 动态协同势：速度与极坐标运动方向的一致性
    #    视角：鼓励沿着“正确的极坐标方向”移动，而不是盲目靠近或远离。
    # ---------------------------
    velocity_radial_alignment = 0.0
    velocity_tangential_alignment = 0.0

    speed = np.linalg.norm(p_vel)
    if speed > 1e-6:
        vel_dir = p_vel / speed
        # 径向方向（指向猎物的反方向：从猎物向追捕者是 rel_dir）
        radial_dir_outward = rel_dir
        radial_dir_inward = -rel_dir
        # 切向方向：在极坐标中绕猎物旋转
        tangential_dir = np.array([-rel_dir[1], rel_dir[0]])

        # 当前若在壳外：主导径向内收；若在壳内：主导径向外扩
        if dist_to_prey > formation_shell_radius:
            radial_target = radial_dir_inward
        else:
            radial_target = radial_dir_outward

        # 径向对齐：速度在目标径向方向的投影
        radial_alignment = float(np.dot(vel_dir, radial_target))
        velocity_radial_alignment = radial_alignment

        # 切向对齐：鼓励具备一定“旋转”成分，避免单线靠近
        tangential_alignment = float(np.abs(np.dot(vel_dir, tangential_dir)))
        velocity_tangential_alignment = tangential_alignment

    components["velocity_radial_alignment"] = 0.2 * velocity_radial_alignment
    components["velocity_tangential_alignment"] = 0.1 * velocity_tangential_alignment

    # ---------------------------
    # 6. 围捕完成势：基于半径与角度势的软判定
    #    视角：当所有追捕者都处于内势壳附近且角度协同度高时，给予一次性势谷奖励。
    # ---------------------------
    capture_soft_bonus = 0.0
    if n_adv > 0:
        all_in_shell = bool(np.all(distances_to_prey < capture_threshold))
        if all_in_shell:
            # 使用角度集中度与半径均匀度作为质量指标
            adv_positions = agent_positions[adversary_indices]
            rel_vecs = adv_positions - prey_pos
            adv_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
            adv_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
            adv_angles = (adv_angles + 2.0 * np.pi) % (2.0 * np.pi)

            unit_vectors = np.stack(
                [np.cos(adv_angles), np.sin(adv_angles)], axis=-1
            )
            mean_vec = np.mean(unit_vectors, axis=0)
            mean_len = np.linalg.norm(mean_vec)
            radius_spread = float(np.max(adv_dists) - np.min(adv_dists))

            angle_quality = 1.0 - mean_len  # 越均匀越接近 1
            radius_quality = np.exp(-radius_spread / (capture_threshold + 1e-8))

            capture_soft_bonus = 5.0 * angle_quality * radius_quality

    components["capture_soft_bonus"] = capture_soft_bonus

    # ---------------------------
    # 7. 时间成本：轻微惩罚拖延
    # ---------------------------
    components["step_cost"] = step_cost

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
