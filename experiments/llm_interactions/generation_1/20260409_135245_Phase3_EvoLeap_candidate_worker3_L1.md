# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260409_135245

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
    # 只对追捕者计算奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -------------------------
    # 物理与任务常量（硬编码）
    # -------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_threshold = 0.5  # 包围半径阈值
    capture_radius = 0.35    # 更紧的围捕半径
    time_penalty = -0.01     # 步长惩罚，鼓励尽快完成
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

    # world.agents 中 adversary / prey 布局假定：
    # 3 个 adversaries + 1 个 prey，总共 4 个
    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    # 防御性编程：若数量异常则直接返回时间惩罚
    if len(adversaries) == 0 or len(preys) == 0:
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    # 找到当前 agent 在 adversaries 列表中的索引
    # agent_name 与 world.agents[i].name 绑定
    agent_index_global = None
    agent_index_adv = None
    for i, ag in enumerate(all_agents):
        if getattr(ag, 'name', None) == agent_name:
            agent_index_global = i
            break

    if agent_index_global is None:
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    # adversary 索引映射
    adv_global_indices = [all_agents.index(a) for a in adversaries]
    if agent_index_global not in adv_global_indices:
        # 如果 global_state 说是 adversary 但不在 adversaries 中，则按时间惩罚
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    agent_index_adv = adv_global_indices.index(agent_index_global)

    # -------------------------
    # 1. 距离引导：接近猎物
    # -------------------------
    # 对 3 个追捕者的距离
    adv_to_prey_dists = distances_to_prey  # shape (3,)
    # 奖励使用负的平均距离，归一化到世界尺度
    mean_dist = np.mean(adv_to_prey_dists)
    distance_reward = -mean_dist / world_size
    components['distance_reward'] = approach_weight * distance_reward

    # -------------------------
    # 2. 防碰撞：追捕者之间以及与猎物
    # -------------------------
    n_agents = len(all_agents)
    self_collisions = 0
    # 追捕者之间防撞
    for i in adv_global_indices:
        for j in adv_global_indices:
            if j <= i:
                continue
            if inter_agent_distances[i, j] < 2 * adv_size:
                self_collisions += 1

    avoid_collision_reward = -collision_penalty * self_collisions
    components['avoid_collision_reward'] = avoid_collision_reward

    # 与猎物的碰撞（视作捕获事件的一部分）
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
    # 3.1 获取追捕者相对猎物的位置
    adv_positions = agent_positions[adv_global_indices]  # shape (3, 2)
    prey_pos = prey_position

    rel_vecs = adv_positions - prey_pos  # shape (3, 2)
    rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8  # 避免零除
    # 半径在 capture_threshold 内，并尽量接近 capture_radius
    mean_radius = np.mean(rel_dists)
    radius_deviation = np.abs(mean_radius - capture_radius)
    # 半径一致性：方差越小越好
    radius_variance = np.var(rel_dists)

    # 归一化半径相关量
    radius_deviation_norm = radius_deviation / capture_threshold
    radius_variance_norm = radius_variance / (capture_threshold ** 2)

    # 半径奖励：鼓励在目标半径附近且互相距离相似
    radius_reward = - (0.5 * radius_deviation_norm + 0.5 * radius_variance_norm)
    components['formation_radius_reward'] = formation_radius_weight * radius_reward

    # 3.2 角度均匀性（希望约 120 度均分）
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])  # [-pi, pi]
    # 排序并展开到 [0, 2pi)
    angles = np.mod(angles, 2 * np.pi)
    angles_sorted = np.sort(angles)
    # 环状差分
    angle_diffs = np.diff(angles_sorted, append=angles_sorted[0] + 2 * np.pi)
    target_diff = 2 * np.pi / 3.0
    # 均方差度量
    angle_uniformity_loss = np.mean((angle_diffs - target_diff) ** 2)
    # 归一化（最大差异约 (2pi)^2）
    angle_uniformity_norm = angle_uniformity_loss / (4 * np.pi ** 2)
    angle_uniformity_reward = -angle_uniformity_norm
    components['angle_uniformity_reward'] = angle_uniformity_weight * angle_uniformity_reward

    # 3.3 猎物是否在追捕者形成的三角形内部（包含性）
    # 使用重心坐标法判定点在三角形内
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
    # 4. 围捕成功奖励（所有追捕者足够接近并形成包围）
    # -------------------------
    all_close = np.all(adv_to_prey_dists < capture_threshold)
    all_tight = np.all(adv_to_prey_dists < capture_radius)
    formation_good = (
        radius_deviation_norm < 0.3
        and radius_variance_norm < 0.3
        and angle_uniformity_norm < 0.1
        and prey_inside
    )

    capture_success = all_close and formation_good
    tight_capture_success = all_tight and formation_good

    capture_reward = 0.0
    if tight_capture_success:
        capture_reward = capture_bonus
    elif capture_success:
        capture_reward = 0.5 * capture_bonus

    components['capture_success_reward'] = capture_reward

    # -------------------------
    # 5. 时间步惩罚（鼓励快速完成）
    # -------------------------
    components['time_penalty'] = time_penalty

    total_reward = sum(components.values())
    return total_reward, components
```

# 客观诊断反馈
**1. 分量贡献度**

- 从整体 Fitness 为负、成功率极低推断：  
  - 奖励中与“成功捕获”直接相关的分量几乎长期为 0（成功太少，信号极稀疏）；  
  - 若存在“移动代价/时间惩罚”类分量，其方差相对最大、占主导，容易压制探索；  
  - 任意与“逐步逼近目标”相关且连续可获得的密集奖励，显然不足以主导行为（否则成功率不会如此低）。

**2. 任务瓶颈**

- 成功率 4.53%，平均捕获时间接近上限（98.3 steps）表明：  
  - 大部分 episode 以“超时失败”结束，智能体无法稳定形成有效追捕策略；  
  - 有高概率陷入“保守但无效”的局部最优：靠时间拖延而非主动高效逼近，被动游走或边缘徘徊。

**3. 协同缺陷**

- 在协作追捕设定下，以下问题高度可疑：  
  - 搭便车：部分追捕者几乎不移动，依赖少数个体承担追击；  
  - 扎堆：追捕者倾向聚集于少数方向，形成自相阻挡或碰撞，无法多面夹击；  
  - 包围圈收缩失败：缺乏对“合围形态”的显式激励，导致围堵结构松散、漏人率高。

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
    # 硬编码物理与任务常量
    # -------------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    safety_distance = 2.5 * adv_size      # 追捕者间安全距离
    time_penalty = -0.002                 # 轻微时间惩罚（防止压制探索）
    max_potential_radius = world_size     # 势场归一化半径
    capture_distance = 0.35               # 捕获视作成功的距离
    surround_distance_min = 0.25          # 合围壳层内半径
    surround_distance_max = 0.55          # 合围壳层外半径
    surround_angle_margin = np.pi / 4.0   # 极角扇区范围（45度）

    # -------------------------
    # 提取全局状态
    # -------------------------
    agent_positions = global_state['agent_positions']
    prey_position = global_state['prey_position']
    inter_agent_distances = global_state['inter_agent_distances']

    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    if len(adversaries) == 0 or len(preys) == 0:
        components["time_penalty"] = time_penalty
        return sum(components.values()), components

    # 当前智能体的索引
    agent_index_global = None
    for i, ag in enumerate(all_agents):
        if getattr(ag, "name", None) == agent_name:
            agent_index_global = i
            break

    if agent_index_global is None:
        components["time_penalty"] = time_penalty
        return sum(components.values()), components

    adv_global_indices = [all_agents.index(a) for a in adversaries]
    if agent_index_global not in adv_global_indices:
        components["time_penalty"] = time_penalty
        return sum(components.values()), components

    prey_index_global = all_agents.index(preys[0])

    # =========================
    # 1. 全局势场：追捕者对猎物的“吸引 + 包围壳层”
    # =========================
    # 吸引：距离越近越好（在全局尺度上平滑）
    adv_positions = agent_positions[adv_global_indices]
    vecs_to_prey = prey_position - adv_positions
    dists_to_prey = np.linalg.norm(vecs_to_prey, axis=1) + 1e-8
    mean_dist_to_prey = np.mean(dists_to_prey)

    # 1.1 吸引势：在 [0, max_potential_radius] 上线性归一
    attraction_potential = mean_dist_to_prey / max_potential_radius
    attraction_reward = -attraction_potential
    components["attraction_field"] = attraction_reward

    # 1.2 壳层势：目标是在 [surround_distance_min, surround_distance_max] 的环带
    # 将每个追捕者到猎物的距离映射到壳层损失
    def shell_loss(dist, r_min, r_max):
        if dist < r_min:
            return (r_min - dist) / r_min
        if dist > r_max:
            return (dist - r_max) / r_max
        # 在壳层内损失为0
        return 0.0

    shell_losses = np.array(
        [shell_loss(d, surround_distance_min, surround_distance_max)
         for d in dists_to_prey]
    )
    shell_potential = np.mean(shell_losses)
    shell_reward = -shell_potential
    components["shell_field"] = shell_reward

    # =========================
    # 2. 极坐标分区协同：鼓励多扇区合围，抑制扎堆
    # =========================
    # 对猎物为原点，建立极坐标系
    rel_vecs = adv_positions - prey_position
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])  # [-pi, pi]
    # 将角度映射到 [-pi, pi) 上的扇区
    n_sectors = 6
    sector_width = 2.0 * np.pi / n_sectors
    sectors = np.floor((angles + np.pi) / sector_width).astype(int)
    sectors = np.clip(sectors, 0, n_sectors - 1)

    # 2.1 扇区覆盖度：覆盖扇区数越多越好，最多3个就足够
    unique_sectors = np.unique(sectors)
    n_covered = len(unique_sectors)

    # 将覆盖度转为 [0, 1]，2~3扇区为最佳（避免过度扩散）
    if n_covered <= 1:
        coverage_score = 0.0
    elif n_covered == 2:
        coverage_score = 0.7
    elif n_covered == 3:
        coverage_score = 1.0
    else:
        # 覆盖过多方向也稍微降低
        coverage_score = max(0.0, 1.0 - 0.1 * (n_covered - 3))

    components["sector_coverage"] = coverage_score

    # 2.2 扇区均衡：防止扎堆同一扇区
    counts_per_sector = np.zeros(n_sectors)
    for s in sectors:
        counts_per_sector[s] += 1

    # 偏离均衡的平方损失（仅对有追捕者的扇区）
    nonzero_counts = counts_per_sector[counts_per_sector > 0]
    if len(nonzero_counts) > 0:
        mean_count = np.mean(nonzero_counts)
        imbalance = np.mean((nonzero_counts - mean_count) ** 2)
        # 归一化，最大不平衡假定为全部扎堆：var ~ (N-1)^2
        n_adv = len(adversaries)
        max_imbalance = max(1.0, (n_adv - 1.0) ** 2)
        imbalance_norm = imbalance / max_imbalance
        balance_reward = -imbalance_norm
    else:
        balance_reward = 0.0

    components["sector_balance"] = balance_reward

    # =========================
    # 3. 局部安全势场：追捕者之间的排斥，避免碰撞与扎堆
    # =========================
    repulsion_sum = 0.0
    n_adv = len(adv_global_indices)
    for i_idx in range(n_adv):
        gi = adv_global_indices[i_idx]
        for j_idx in range(i_idx + 1, n_adv):
            gj = adv_global_indices[j_idx]
            d = inter_agent_distances[gi, gj] + 1e-8
            if d < safety_distance:
                # 在安全距离内使用 1/d 势场排斥
                repulsion_sum += (safety_distance / d) - 1.0

    repulsion_penalty = -0.2 * repulsion_sum
    components["repulsion_field"] = repulsion_penalty

    # 硬碰撞惩罚：重罚实际重叠
    collision_penalty = 0.0
    for i_idx in range(n_adv):
        gi = adv_global_indices[i_idx]
        for j_idx in range(i_idx + 1, n_adv):
            gj = adv_global_indices[j_idx]
            if inter_agent_distances[gi, gj] < 2.0 * adv_size:
                collision_penalty -= 2.0

    components["hard_collision_penalty"] = collision_penalty

    # =========================
    # 4. 环形包围扇区对齐：每个扇区选一个代表点，鼓励分布到互补扇区
    # =========================
    # 定义3个目标扇区中心：0, 120deg, 240deg
    target_sector_centers = np.array(
        [-np.pi + sector_width / 2.0,
         -np.pi + sector_width / 2.0 + 2.0 * np.pi / 3.0,
         -np.pi + sector_width / 2.0 + 4.0 * np.pi / 3.0]
    )

    # 对每个目标中心，计算最近追捕者角度误差
    angle_errors = []
    for center in target_sector_centers:
        diffs = np.abs(np.arctan2(
            np.sin(angles - center),
            np.cos(angles - center)
        ))
        angle_errors.append(np.min(diffs))

    angle_errors = np.array(angle_errors)
    # 将误差限制在 [0, pi] 并归一为 [0,1]
    angle_errors_norm = angle_errors / np.pi
    angle_alignment_reward = -np.mean(angle_errors_norm)
    components["angle_alignment"] = angle_alignment_reward

    # =========================
    # 5. 局部个体进展：当前追捕者的势场增量
    # =========================
    self_pos = agent_positions[agent_index_global]
    self_vec_to_prey = prey_position - self_pos
    self_dist_to_prey = float(np.linalg.norm(self_vec_to_prey) + 1e-8)

    # 基于“壳层圆环”定义个体势能：越接近壳层中心半径越好
    shell_center = 0.5 * (surround_distance_min + surround_distance_max)
    self_shell_loss = shell_loss(self_dist_to_prey,
                                 surround_distance_min,
                                 surround_distance_max)
    # 将壳层中心视作势能最小点
    self_shell_center_loss = abs(self_dist_to_prey - shell_center) / shell_center
    self_shell_potential = 0.5 * self_shell_loss + 0.5 * self_shell_center_loss
    self_shell_reward = -self_shell_potential
    components["self_shell_potential"] = self_shell_reward

    # =========================
    # 6. 捕获与强奖励：距离足够近
    # =========================
    # 捕获定义为任一追捕者与猎物距离小于 capture_distance
    capture_reward = 0.0
    min_dist_to_prey = np.min(dists_to_prey)
    if min_dist_to_prey < capture_distance:
        # 使用平滑近距离势场并给一个较大额外奖励
        close_potential = min_dist_to_prey / capture_distance
        capture_reward = 8.0 * (1.0 - close_potential)

    components["capture_bonus"] = capture_reward

    # =========================
    # 7. 时间步惩罚（轻微）
    # =========================
    components["time_penalty"] = time_penalty

    total_reward = sum(components.values())
    return total_reward, components
```
```
