# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker1_F2
- **Timestamp**: 20260409_135237

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
```
