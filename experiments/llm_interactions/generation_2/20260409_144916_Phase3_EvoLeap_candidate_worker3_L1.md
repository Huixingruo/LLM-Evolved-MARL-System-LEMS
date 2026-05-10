# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260409_144916

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
你是一位专业的强化学习奖励工程师。请基于上一代的诊断反馈执行特定的变异操作。

# 环境基座

import numpy as np

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
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -------------------------
    # 物理与任务常量（硬编码）
    # -------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_threshold = 0.5
    capture_radius = 0.35
    time_penalty = -0.01
    collision_penalty = -5.0
    prey_collision_reward = 5.0
    formation_radius_weight = 1.0
    angle_uniformity_weight = 1.0
    triangle_containment_weight = 1.0
    approach_weight = 1.0
    capture_bonus = 10.0

    # -------------------------
    # 从 global_state 中提取信息
    # -------------------------
    agent_positions = global_state['agent_positions']
    prey_position = global_state['prey_position']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']

    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    if len(adversaries) == 0 or len(preys) == 0:
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    agent_index_global = None
    for i, ag in enumerate(all_agents):
        if getattr(ag, 'name', None) == agent_name:
            agent_index_global = i
            break

    if agent_index_global is None:
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    adv_global_indices = [all_agents.index(a) for a in adversaries]
    if agent_index_global not in adv_global_indices:
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    # -------------------------
    # 1. 距离引导：接近猎物
    # -------------------------
    adv_to_prey_dists = distances_to_prey
    mean_dist = np.mean(adv_to_prey_dists)
    distance_reward = -mean_dist / world_size
    components['distance_reward'] = approach_weight * distance_reward

    # -------------------------
    # 2. 防碰撞：追捕者之间以及与猎物
    # -------------------------
    self_collisions = 0
    for i in adv_global_indices:
        for j in adv_global_indices:
            if j <= i:
                continue
            if inter_agent_distances[i, j] < 2 * adv_size:
                self_collisions += 1

    avoid_collision_reward = -collision_penalty * self_collisions
    components['avoid_collision_reward'] = avoid_collision_reward

    prey_global_index = all_agents.index(preys[0])
    hit_prey = False
    for i in adv_global_indices:
        if inter_agent_distances[i, prey_global_index] < (adv_size + prey_size):
            hit_prey = True
            break

    prey_collision_component = prey_collision_reward if hit_prey else 0.0
    components['prey_collision_reward'] = prey_collision_component

    # -------------------------
    # 3. 队形：环形半径一致性 + 角度均匀性 + 包含性
    # -------------------------
    adv_positions = agent_positions[adv_global_indices]
    prey_pos = prey_position

    rel_vecs = adv_positions - prey_pos
    rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    mean_radius = np.mean(rel_dists)
    radius_deviation = np.abs(mean_radius - capture_radius)
    radius_variance = np.var(rel_dists)

    radius_deviation_norm = radius_deviation / capture_threshold
    radius_variance_norm = radius_variance / (capture_threshold ** 2)

    radius_reward = -(0.5 * radius_deviation_norm + 0.5 * radius_variance_norm)
    components['formation_radius_reward'] = formation_radius_weight * radius_reward

    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    angles_sorted = np.sort(angles)
    angle_diffs = np.diff(angles_sorted, append=angles_sorted[0] + 2 * np.pi)
    target_diff = 2 * np.pi / 3.0
    angle_uniformity_loss = np.mean((angle_diffs - target_diff) ** 2)
    angle_uniformity_norm = angle_uniformity_loss / (4 * np.pi ** 2)
    angle_uniformity_reward = -angle_uniformity_norm
    components['angle_uniformity_reward'] = (
        angle_uniformity_weight * angle_uniformity_reward
    )

    def point_in_triangle(p, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        denom = dot00 * dot11 - dot01 * dot01
        if np.abs(denom) < 1e-8:
            return False
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)

    a_pos, b_pos, c_pos = adv_positions[0], adv_positions[1], adv_positions[2]
    prey_inside = point_in_triangle(prey_pos, a_pos, b_pos, c_pos)

    triangle_containment_reward = 1.0 if prey_inside else 0.0
    components['triangle_containment_reward'] = (
        triangle_containment_weight * triangle_containment_reward
    )

    # -------------------------
    # 4. 围捕成功奖励
    # -------------------------
    all_close = np.all(adv_to_prey_dists < capture_threshold)
    all_tight = np.all(adv_to_prey_dists < capture_radius)
    formation_good = (
        radius_deviation_norm < 0.3
        and radius_variance_norm < 0.3
        and angle_uniformity_norm < 0.1
        and prey_inside
    )

    tight_capture_success = all_tight and formation_good

    capture_reward = 0.0
    if tight_capture_success:
        capture_reward = capture_bonus

    components['capture_success_reward'] = capture_reward

    # -------------------------
    # 5. 时间步惩罚
    # -------------------------
    components['time_penalty'] = time_penalty

    total_reward = sum(components.values())
    return total_reward, components
```

# 客观诊断反馈
1. 分量贡献度  
   - Fitness 为负且绝对值极小，说明总奖励接近零：  
     - 可能存在大块奖励分量长期为 0（如协同/包围度量、个体贡献奖励），整体学习信号非常稀疏。  
     - 成功相关奖励在少数 episode 才触发，方差大、占主导，易引起训练震荡。  
     - 步数惩罚或存活奖励如存在，很可能与成功奖励量级不匹配，产生“拉锯”，削弱梯度方向感。

2. 任务瓶颈  
   - 成功率仅 7.2%，平均捕获时间接近最大步数上限，说明大量 episode 无法有效收敛到目标。  
   - 典型指标：智能体多半在“安全但无效”的策略附近徘徊（如远距离跟随、不敢贴近），呈现明显局部最优——活得久但几乎不完成抓捕。

3. 协同缺陷  
   - 低成功率+长时间拖延高度暗示：  
     - 部分追捕者存在“搭便车”，跟随队友或几乎不改变相对位置，对整体包围贡献为零。  
     - 队形难以收缩成有效包围圈，常见现象包括：追捕者排成一线追逐、在局部反复绕圈，无法形成多方向封锁。  
     - 协同奖励若存在，可能未能区分个体贡献，导致集体行为懒散、缺乏主动收缩驱动。

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
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -------------------------
    # 物理与任务常量（硬编码）
    # -------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_threshold = 0.5
    safe_dist_min = 2.5 * adv_size       # 追捕者之间的安全距离下限
    safe_dist_max = 1.2 * capture_threshold  # 追捕者之间安全距离上限（避免太松散）
    prey_close_dist = capture_threshold  # 对猎物的接近距离
    prey_very_close_dist = 0.3          # 激励局部收缩
    max_action = 1.0

    # 权重
    w_potential = 2.0
    w_radial_in = 0.8
    w_tangential = 0.8
    w_alignment = 0.5
    w_spacing = 1.0
    w_coverage = 1.2
    w_capture = 10.0
    w_time = -0.01

    # -------------------------
    # 从 global_state 中提取信息
    # -------------------------
    agent_positions = global_state['agent_positions']
    agent_velocities = global_state['agent_velocities']
    prey_position = global_state['prey_position']
    prey_velocity = global_state['prey_velocity']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']

    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    if len(adversaries) == 0 or len(preys) == 0:
        components['time_penalty'] = w_time
        return sum(components.values()), components

    # 找到当前智能体索引
    agent_index_global = None
    for i, ag in enumerate(all_agents):
        if getattr(ag, 'name', None) == agent_name:
            agent_index_global = i
            break

    if agent_index_global is None:
        components['time_penalty'] = w_time
        return sum(components.values()), components

    adv_global_indices = [all_agents.index(a) for a in adversaries]
    if agent_index_global not in adv_global_indices:
        components['time_penalty'] = w_time
        return sum(components.values()), components

    # 当前智能体局部变量
    self_pos = agent_positions[agent_index_global]
    self_vel = agent_velocities[agent_index_global]
    prey_pos = prey_position
    prey_vel = prey_velocity

    # 相对向量
    rel_vec = prey_pos - self_pos
    rel_dist = np.linalg.norm(rel_vec) + 1e-8
    rel_dir = rel_vec / rel_dist

    # -------------------------
    # 1. 势场视角：双层势场 (吸引 + 抑制过近)
    #    U(r) = a * r - b / (r + eps)
    #    奖励 = -ΔU，鼓励势能下降
    # -------------------------
    # 计算势能
    a_attr = 1.0 / world_size
    b_rep = 0.04
    potential = a_attr * rel_dist - b_rep / (rel_dist + 1e-3)

    # 为了形成密集但不过近的围捕区域，对过远距离增加额外吸引
    far_bonus = 0.0
    if rel_dist > prey_close_dist:
        far_bonus = -0.2 * (rel_dist - prey_close_dist) / world_size

    # 使用速度在势场梯度方向的投影来近似势能下降
    # grad(U) ~ dU/dr * rel_dir
    dU_dr = a_attr + b_rep / (rel_dist + 1e-3) ** 2
    gradU = dU_dr * rel_dir
    # 速度沿 -gradU 的投影越大，势能下降越快
    vel_proj = np.dot(self_vel, -gradU)
    potential_reward = w_potential * vel_proj + far_bonus
    components['potential_field_reward'] = potential_reward

    # -------------------------
    # 2. 极坐标分解：径向压缩 + 切向包围
    # -------------------------
    # 径向分量
    v_radial = np.dot(self_vel, rel_dir)
    # 切向分量
    v_tangent_vec = self_vel - v_radial * rel_dir
    v_tangent = np.linalg.norm(v_tangent_vec)

    # 径向压缩：鼓励在中远距离时向内收缩，在非常近时减弱压缩
    radial_gain = np.clip((rel_dist - prey_very_close_dist) /
                          (prey_close_dist + 1e-6), 0.0, 1.0)
    radial_reward = w_radial_in * radial_gain * v_radial
    components['radial_compression_reward'] = radial_reward

    # 切向运动：在接近捕获区时鼓励绕圈而不是直线冲撞
    tangent_gain = np.clip((prey_close_dist - rel_dist) /
                           (prey_close_dist + 1e-6), 0.0, 1.0)
    tangential_reward = w_tangential * tangent_gain * v_tangent
    components['tangential_orbit_reward'] = tangential_reward

    # -------------------------
    # 3. 速度对齐：与围捕整体速度场对齐
    #    目标方向 = 环绕方向 + 适度内收
    # -------------------------
    # 构造局部期望速度场：在近距离以切向为主，带少量内收
    # 构造正交切线单位向量
    if rel_dist > 1e-6:
        # 两个可能的切向方向，选择与当前速度更接近的
        t1 = np.array([-rel_dir[1], rel_dir[0]])
        t2 = -t1
        if np.dot(self_vel, t1) >= np.dot(self_vel, t2):
            t_dir = t1
        else:
            t_dir = t2
    else:
        t_dir = np.zeros_like(rel_dir)

    orbit_weight = tangent_gain
    inward_weight = 1.0 - orbit_weight
    desired_dir = orbit_weight * t_dir + inward_weight * rel_dir
    norm_desired = np.linalg.norm(desired_dir) + 1e-8
    desired_dir = desired_dir / norm_desired

    norm_self_vel = np.linalg.norm(self_vel)
    if norm_self_vel > 1e-8:
        vel_dir = self_vel / norm_self_vel
        alignment = np.dot(vel_dir, desired_dir)
    else:
        alignment = 0.0

    alignment_reward = w_alignment * alignment
    components['velocity_field_alignment_reward'] = alignment_reward

    # -------------------------
    # 4. 多智能体协同：间距势场 + 角度覆盖
    # -------------------------
    # 4.1 追捕者间距势场（避免重叠 + 避免过散）
    spacing_energy = 0.0
    self_coll_risk = 0.0
    for j in adv_global_indices:
        if j == agent_index_global:
            continue
        d = inter_agent_distances[agent_index_global, j]
        # 过近：强烈惩罚
        if d < safe_dist_min:
            self_coll_risk += (safe_dist_min - d) / safe_dist_min
        # 过远：轻微惩罚，鼓励收拢
        if d > safe_dist_max:
            spacing_energy += (d - safe_dist_max) / safe_dist_max

    spacing_reward = -w_spacing * (spacing_energy + 3.0 * self_coll_risk)
    components['inter_agent_spacing_reward'] = spacing_reward

    # 4.2 覆盖角度增益：对具有独特视角的追捕者给奖励
    adv_positions = agent_positions[adv_global_indices]
    rel_vecs_all = adv_positions - prey_pos
    rel_dists_all = np.linalg.norm(rel_vecs_all, axis=1) + 1e-8
    rel_dirs_all = rel_vecs_all / rel_dists_all[:, None]

    # 当前智能体相对方向
    self_rel_dir = -rel_dir  # 从猎物看追捕者方向
    # 统计其他追捕者的方向
    directions_other = []
    for idx, g_i in enumerate(adv_global_indices):
        if g_i == agent_index_global:
            continue
        directions_other.append(-rel_dirs_all[idx])
    directions_other = np.array(directions_other) if directions_other else None

    coverage_gain = 0.0
    if directions_other is not None and len(directions_other) > 0:
        # 计算与最近方向的夹角，角度越大说明覆盖更“独特”
        dots = directions_other @ self_rel_dir
        dots = np.clip(dots, -1.0, 1.0)
        min_angle = np.arccos(np.max(dots))
        # 归一化到 [0, 1] 范围，目标约为 2π/len(adversaries)
        ideal_angle = 2 * np.pi / max(len(adversaries), 2)
        coverage_gain = np.clip(min_angle / ideal_angle, 0.0, 1.5)

    coverage_reward = w_coverage * coverage_gain
    components['angular_coverage_reward'] = coverage_reward

    # -------------------------
    # 5. 捕获与成功信号（稠密 + 终端）
    # -------------------------
    # 5.1 局部接近奖励：距离越小奖励越大（平滑）
    dist_norm = rel_dist / world_size
    proximity_reward = -1.5 * dist_norm
    components['local_proximity_reward'] = proximity_reward

    # 5.2 全局围捕势：所有追捕者接近时增加共享奖励
    adv_to_prey_dists = distances_to_prey
    mean_dist = np.mean(adv_to_prey_dists)
    global_potential = -mean_dist / world_size
    components['global_capture_potential_reward'] = global_potential

    # 5.3 捕获判定：多追捕者同时接近 + 围圈
    close_mask = adv_to_prey_dists < prey_close_dist
    very_close_mask = adv_to_prey_dists < prey_very_close_dist

    num_close = np.sum(close_mask)
    num_very_close = np.sum(very_close_mask)

    # 基础捕获密度奖励：鼓励更多追捕者进入捕获圈
    density_term = 0.0
    if len(adversaries) > 0:
        density_term = (num_close / len(adversaries))

    # 如果多数追捕者非常接近，给出一次性较大奖励
    capture_signal = 0.0
    if num_very_close >= max(2, len(adversaries) - 1):
        capture_signal = w_capture

    components['capture_density_reward'] = 2.0 * density_term
    components['capture_bonus_reward'] = capture_signal

    # -------------------------
    # 6. 时间步惩罚（轻微）
    # -------------------------
    components['time_penalty'] = w_time

    total_reward = sum(components.values())
    return total_reward, components
```
```
