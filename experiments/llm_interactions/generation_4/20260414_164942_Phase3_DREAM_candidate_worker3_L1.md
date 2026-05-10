# LLM Interaction Log

- **Generation**: 4
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260414_164942

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

    # ---------------------------
    # 8. 新增协同进攻方向一致性奖励
    #    鼓励所有追捕者的速度方向指向猎物且彼此对齐，减少“各打各的”
    # ---------------------------
    cooperative_direction_reward = 0.0
    if n_adv > 1:
        # 收集追捕者的速度和指向猎物的方向
        adv_vels = []
        adv_dir_to_prey = []
        for idx in adversary_indices:
            vel = np.array(world.agents[idx].state.p_vel, dtype=float)
            pos = np.array(world.agents[idx].state.p_pos, dtype=float)
            to_prey = prey_pos - pos
            vel_norm = np.linalg.norm(vel) + 1e-8
            dir_norm = np.linalg.norm(to_prey) + 1e-8
            adv_vels.append(vel / vel_norm)
            adv_dir_to_prey.append(to_prey / dir_norm)

        adv_vels = np.stack(adv_vels, axis=0)
        adv_dir_to_prey = np.stack(adv_dir_to_prey, axis=0)

        # 每个追捕者速度与自身指向猎物方向的对齐程度（余弦相似度）
        to_prey_align = np.einsum("ij,ij->i", adv_vels, adv_dir_to_prey)
        # 当前追捕者的该项
        local_to_prey_align = float(to_prey_align[local_adv_idx])

        # 追捕者之间速度方向的一致性（与团队平均方向对齐）
        mean_vel_dir = np.mean(adv_vels, axis=0)
        mean_vel_dir_norm = mean_vel_dir / (np.linalg.norm(mean_vel_dir) + 1e-8)
        team_align = float(np.dot(adv_vels[local_adv_idx], mean_vel_dir_norm))

        # 奖励当前追捕者既朝向猎物又与队友方向一致
        cooperative_direction_reward = 0.1 * local_to_prey_align + \
            0.1 * team_align

    components["cooperative_direction_reward"] = cooperative_direction_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
主导作用：capture_penalty（-10固定）、escape_reward（强负）、distance_shaping_self/global、formation_angle/radius_reward、cooperative_shrink_reward共同拉低总回报；containment_bonus和cooperative_direction_reward为主要正向分量，但幅度有限。near_collision与collision为零，说明安全约束未触发，非主导。boundary_penalty波动大，可能在少数局面强烈惩罚。整体表现为：能一定概率围捕，但队形散乱（formation_quality低、角度奖励强负），包围半径偏大，平均距猎物较远，靠“耗时间+偶然合围”完成捕获，无明显碰撞或搭便车，但协同质量不足。

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
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ---------------------------
    # 物理与环境常量（硬编码）
    # ---------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_radius = 0.5  # 约为 world_size * 0.2
    formation_inner_radius = 0.25
    formation_outer_radius = 0.6
    safe_margin = 0.03
    min_safe_dist = 2 * adv_size + safe_margin

    # ---------------------------
    # 基本索引与状态提取
    # ---------------------------
    agent_positions = np.array(global_state["agent_positions"], dtype=float)
    agent_velocities = np.array(global_state["agent_velocities"], dtype=float)
    prey_pos = np.array(global_state["prey_position"], dtype=float)
    prey_vel = np.array(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.array(
        global_state["inter_agent_distances"], dtype=float
    )
    distances_to_prey = np.array(global_state["distances_to_prey"], dtype=float)

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

    # 追捕者索引假定为前 n_adv 个
    adversary_indices = list(range(n_adv))
    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    self_pos = agent_positions[agent_index]
    self_vel = agent_velocities[agent_index]
    d_ap = float(np.linalg.norm(self_pos - prey_pos))

    # 若无人类猎物（理论上不会发生），直接返回
    if n_adv <= 0:
        return 0.0, components

    # ---------------------------
    # 1. 全局势场：围捕势能 V_total
    #
    # 我们构造一个“理想环带势场”：
    #   - 半径在 [formation_inner_radius, formation_outer_radius] 内势能最低
    #   - 半径过大或过小势能增加
    #   - 同时增加追捕者之间的径向与角度均匀势能
    # 奖励 = - (局部势能 + 适度的全局势能)
    # ---------------------------
    adv_positions = agent_positions[adversary_indices]
    rel_vecs = adv_positions - prey_pos  # shape: (n_adv, 2)
    rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

    # --- 1.1 径向环带势能（团队）
    # 目标环带：inner <= r <= outer
    def ring_potential(r, r_in, r_out):
        # 三段式：内侧、环带、外侧
        pot = np.zeros_like(r)
        # 内侧：鼓励向外推
        inside_mask = r < r_in
        pot[inside_mask] = (r_in - r[inside_mask]) ** 2
        # 外侧：鼓励向内收缩
        outside_mask = r > r_out
        pot[outside_mask] = (r[outside_mask] - r_out) ** 2
        # 环带内近似平坦（势能 ~ 0）
        return pot

    team_ring_potentials = ring_potential(
        rel_dists,
        formation_inner_radius,
        formation_outer_radius,
    )
    local_ring_potential = float(team_ring_potentials[local_adv_idx])
    global_ring_potential = float(np.mean(team_ring_potentials))

    components["local_ring_potential"] = -0.8 * local_ring_potential
    components["global_ring_potential"] = -0.4 * global_ring_potential

    # --- 1.2 角度均匀势能（团队）
    angle_uniform_potential = 0.0
    if n_adv >= 2:
        angles_sorted = np.sort(rel_angles)
        deltas = np.empty(n_adv, dtype=float)
        for i in range(n_adv - 1):
            deltas[i] = angles_sorted[i + 1] - angles_sorted[i]
        deltas[-1] = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]

        target_delta = 2.0 * np.pi / float(n_adv)
        angle_uniform_potential = float(np.mean((deltas - target_delta) ** 2))

    components["angle_uniform_potential"] = -0.3 * angle_uniform_potential

    # --- 1.3 径向均匀势能（团队）
    radial_uniform_potential = float(np.var(rel_dists))
    components["radial_uniform_potential"] = -0.3 * radial_uniform_potential

    # ---------------------------
    # 2. 相对极坐标协同收缩与锁定
    #
    # 使用“相对猎物的极坐标轨道”视角：
    #   - 奖励：在角度上分布均匀时，进一步缩小平均半径
    #   - 奖励：相对于猎物速度方向形成一个“挡截扇区”
    # ---------------------------
    cooperative_orbit_shrink_reward = 0.0
    interception_arc_reward = 0.0

    if n_adv >= 2:
        mean_radius = float(np.mean(rel_dists))

        # 根据角度均匀性定义一个“队形质量系数” in [0, 1]
        # angle_uniform_potential 越小，quality 越高
        angle_quality = 1.0 / (1.0 + angle_uniform_potential)
        # 半径均匀性质量
        radial_quality = 1.0 / (1.0 + radial_uniform_potential)
        formation_quality = 0.5 * angle_quality + 0.5 * radial_quality

        # 协同收缩：队形越好，越鼓励减小平均半径
        cooperative_orbit_shrink_reward = -0.5 * formation_quality * mean_radius

        # 猎物速度方向上的“挡截扇区”
        prey_speed = float(np.linalg.norm(prey_vel))
        if prey_speed > 1e-6:
            prey_dir = prey_vel / prey_speed
            # 所有追捕者在猎物速度方向上的角度
            prey_angle = np.arctan2(prey_dir[1], prey_dir[0])
            angle_diff = np.abs(
                (rel_angles - prey_angle + np.pi) % (2.0 * np.pi) - np.pi
            )
            # 统计有多少追捕者位于猎物速度前方的 +/- 90 度扇区内
            front_mask = angle_diff < (0.5 * np.pi)
            front_ratio = float(np.mean(front_mask.astype(float)))

            # 若在前方扇区内的追捕者足够多，则奖励
            interception_arc_reward = 0.5 * formation_quality * front_ratio

    components["cooperative_orbit_shrink_reward"] = (
        cooperative_orbit_shrink_reward
    )
    components["interception_arc_reward"] = interception_arc_reward

    # ---------------------------
    # 3. 局部“轨道锁定”奖励：沿角向运动优于径向乱动
    #
    # 在极坐标中，速度可以分解为径向 vr 和角向 vt：
    #   - 希望在理想环带内以角向运动为主（锁定轨道）
    #   - 在环带外时，以合适方向的径向运动为主（向内或向外）
    # ---------------------------
    rel_vec_self = self_pos - prey_pos
    r_self = float(np.linalg.norm(rel_vec_self) + 1e-8)
    e_r = rel_vec_self / r_self
    e_t = np.array([-e_r[1], e_r[0]])  # 正向角向单位向量

    v_self = self_vel
    v_r = float(np.dot(v_self, e_r))
    v_t = float(np.dot(v_self, e_t))

    orbit_lock_reward = 0.0

    # 在环带内：鼓励角向速度，惩罚强烈径向扰动
    if formation_inner_radius <= r_self <= formation_outer_radius:
        orbit_lock_reward += 0.1 * (v_t / (np.linalg.norm(v_self) + 1e-8))
        orbit_lock_reward -= 0.05 * (v_r**2)
    # 在环带外：鼓励向内收缩
    elif r_self > formation_outer_radius:
        if v_r < 0.0:
            orbit_lock_reward += 0.1 * (-v_r)
        else:
            orbit_lock_reward -= 0.05 * v_r
    # 在环带内侧：鼓励向外扩张
    else:
        if v_r > 0.0:
            orbit_lock_reward += 0.1 * v_r
        else:
            orbit_lock_reward -= 0.05 * (-v_r)

    components["orbit_lock_reward"] = orbit_lock_reward

    # ---------------------------
    # 4. 去中心化的局部密度调控（反自杀 / 防搭便车）
    #
    #   - 惩罚与最近队友过近（防碰撞与团簇）
    #   - 惩罚与团队重心过远（防单飞/搭便车）
    # ---------------------------
    # 最近追捕者距离（仅在追捕者集合内）
    local_collision_penalty = 0.0
    local_spacing_penalty = 0.0
    local_team_anchor_reward = 0.0

    if n_adv > 1:
        # 追捕者间距离子矩阵
        adv_dist_matrix = inter_agent_distances[np.ix_(
            adversary_indices, adversary_indices
        )]
        # 当前追捕者到其他追捕者的距离
        self_adv_idx = local_adv_idx
        dists_to_adv = adv_dist_matrix[self_adv_idx]

        # 防碰撞 + 防极度靠近团簇
        for j, d_aa in enumerate(dists_to_adv):
            if j == self_adv_idx:
                continue
            if d_aa < 2 * adv_size:
                local_collision_penalty -= 3.0
            elif d_aa < min_safe_dist:
                local_spacing_penalty -= 0.2 * (min_safe_dist - d_aa)

        # 防单飞/搭便车：与追捕者重心距离
        team_center = np.mean(adv_positions, axis=0)
        dist_to_center = float(np.linalg.norm(self_pos - team_center))
        # 允许一定的分散度，但太远惩罚
        if dist_to_center > 0.5 * world_size:
            local_team_anchor_reward -= 0.3 * (dist_to_center - 0.5 * world_size)
        else:
            # 适度靠近重心给予少量奖励
            local_team_anchor_reward += 0.05 * (
                0.5 * world_size - dist_to_center
            )

    components["local_collision_penalty"] = local_collision_penalty
    components["local_spacing_penalty"] = local_spacing_penalty
    components["local_team_anchor_reward"] = local_team_anchor_reward

    # ---------------------------
    # 5. 空间约束：边界势场（防止全队贴边绕圈）
    #
    # 采用“软势墙”：越接近边界势能越大
    # ---------------------------
    def boundary_potential(pos, size):
        # 与四个边界的最小距离
        margin = world_size - size
        d_left = pos[0] + margin
        d_right = margin - pos[0]
        d_bottom = pos[1] + margin
        d_top = margin - pos[1]
        d_min = min(d_left, d_right, d_bottom, d_top)
        if d_min >= 0.3:
            return 0.0
        if d_min <= 0.0:
            return 1.0
        # 二次势能
        return (0.3 - d_min) ** 2 / (0.3**2)

    boundary_pot = float(boundary_potential(self_pos, adv_size))
    components["boundary_potential"] = -0.2 * boundary_pot

    # ---------------------------
    # 6. 捕获势阱：多智能体围捕完成奖励
    #
    #   - 当所有追捕者进入 capture_radius 内，并且队形质量较高时，
    #     给予强正奖励；反之在捕获区但队形差不额外奖励。
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        all_in_capture = bool(np.all(distances_to_prey < capture_radius))
        if n_adv >= 2:
            mean_radius = float(np.mean(rel_dists))
            angle_quality = 1.0 / (1.0 + angle_uniform_potential)
            radial_quality = 1.0 / (1.0 + radial_uniform_potential)
            formation_quality = 0.5 * angle_quality + 0.5 * radial_quality
        else:
            mean_radius = d_ap
            formation_quality = 0.5

        if all_in_capture:
            # 捕获质量从 0 到 1，奖金在 [5, 15] 左右
            capture_bonus = 5.0 + 10.0 * formation_quality
            # 越靠近猎物（半径越小）稍微增加奖励
            capture_bonus += 2.0 * max(0.0, capture_radius - mean_radius)

    components["capture_bonus"] = capture_bonus

    # ---------------------------
    # 7. 时间与进度：避免“耗时间+偶然合围”
    #
    #   - 小的时间惩罚
    #   - 以平均半径为进度指标（更靠近 -> 减少势能）
    # ---------------------------
    time_penalty = -0.01
    mean_team_radius = float(np.mean(rel_dists))
    progress_reward = -0.2 * mean_team_radius

    components["time_penalty"] = time_penalty
    components["progress_reward"] = progress_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
