# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker3_L1
- **Timestamp**: 20260414_135749

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

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
capture_penalty 与 escape_reward 主导，强惩罚失败但成功奖励极弱（capture_bonus 极低），time_penalty 恒定、无区分度。formation_angle_reward、formation_radius_reward、distance_shaping_* 与 boundary_penalty 多为负值，说明“围捕/站位/边界”设计在当前策略下起反作用，压制探索与收缩。协同上无碰撞问题，但 encirclement_angle_std 高、avg_distance_to_prey 偏大，说明包围不成形、个体间距尚可但整体过松，存在“各自为战”的协同缺陷。

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
    # 非追捕者奖励恒为 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ---------------------------
    # 物理与环境常量（硬编码）
    # ---------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 势场与权重参数（可调试）
    w_global_orbit = 1.0        # 全局环形势场权重
    w_local_alignment = 0.6     # 局部队友对齐权重
    w_radial_flow = 0.8         # 径向收缩权重
    w_squeeze = 0.7             # 压缩猎物通道权重
    w_safe_distance = 0.6       # 追捕者最小间距权重
    w_boundary = 0.3            # 边界势场权重
    w_capture = 10.0            # 成功围捕奖励权重
    w_step_cost = 0.01          # 步长惩罚权重（小）

    # ---------------------------
    # 状态解析
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

    adversary_indices = list(range(n_adv))
    if agent_index not in adversary_indices:
        return 0.0, components

    local_adv_idx = adversary_indices.index(agent_index)

    agent_pos = agent_positions[agent_index]
    agent_vel = agent_velocities[agent_index]

    # ---------------------------
    # 极坐标系构造（相对猎物）
    # ---------------------------
    rel_vec = agent_pos - prey_pos
    rel_dist = np.linalg.norm(rel_vec) + 1e-8
    rel_angle = np.arctan2(rel_vec[1], rel_vec[0])

    # 动态目标半径：从外圈收缩到 capture_threshold 附近
    base_capture_radius = capture_threshold * 0.9
    max_orbit_radius = world_size * 0.8
    # 根据猎物速度决定当前理想包围半径（跑得快时略放大）
    prey_speed = np.linalg.norm(prey_vel)
    speed_factor = np.clip(prey_speed / 1.3, 0.0, 1.0)
    target_radius = base_capture_radius + (max_orbit_radius - base_capture_radius) * (0.3 * (1.0 - speed_factor))

    # ---------------------------
    # 1. 全局环形势场：鼓励形成近似圆环
    # ---------------------------
    # 半径势场：鼓励 rel_dist 接近 target_radius，而不是一味贴近猎物
    radius_error = (rel_dist - target_radius) / max(target_radius, 1e-6)
    radial_potential = -radius_error ** 2

    # 角度势场：鼓励全局角度均匀分布（但只作为 shaping，不做强约束）
    angle_potential = 0.0
    if n_adv > 1:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        rel_angles_sorted = np.sort(rel_angles)
        deltas = np.diff(np.concatenate([rel_angles_sorted, rel_angles_sorted[:1] + 2.0 * np.pi]))
        target_delta = 2.0 * np.pi / float(n_adv)
        delta_errors = deltas - target_delta
        # 当前 agent 角度在排序序列中的位置，用其所属边的误差作为局部角度势场
        order = np.argsort(rel_angles)
        rank = int(np.where(order == local_adv_idx)[0][0])
        local_delta_error = delta_errors[rank]
        angle_potential = -local_delta_error ** 2

    components["global_orbit_radius_potential"] = w_global_orbit * radial_potential
    components["global_orbit_angle_potential"] = w_global_orbit * 0.5 * angle_potential

    # ---------------------------
    # 2. 局部速度对齐势场（基于邻居）
    # ---------------------------
    local_alignment_potential = 0.0
    neighbor_count = 0
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j]
        # 将 0.3 视为邻居感知半径（小于此认为是局部邻居）
        if d_aa < 0.3:
            neighbor_vel = agent_velocities[j]
            if np.linalg.norm(neighbor_vel) > 1e-8:
                # 只对对猎物的径向/切向方向进行对齐
                # 径向单位向量
                radial_dir = rel_vec / rel_dist
                tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

                # 当前与邻居在径向、切向方向上的速度投影差
                proj_self_r = np.dot(agent_vel, radial_dir)
                proj_self_t = np.dot(agent_vel, tangential_dir)
                proj_nb_r = np.dot(neighbor_vel, radial_dir)
                proj_nb_t = np.dot(neighbor_vel, tangential_dir)

                diff_r = proj_self_r - proj_nb_r
                diff_t = proj_self_t - proj_nb_t

                local_alignment_potential -= (diff_r ** 2 + 0.5 * diff_t ** 2)
                neighbor_count += 1

    if neighbor_count > 0:
        local_alignment_potential /= float(neighbor_count)

    components["local_alignment_potential"] = w_local_alignment * local_alignment_potential

    # ---------------------------
    # 3. 径向收缩势场：整体向内合围
    # ---------------------------
    radial_flow_potential = 0.0
    if rel_dist > target_radius:
        # 当在目标半径之外时，鼓励径向速度向内（负的径向距离）
        radial_dir = rel_vec / rel_dist
        radial_speed = np.dot(agent_vel, -radial_dir)
        # 大于 0 表示向内移动
        radial_flow_potential = radial_speed
    else:
        # 在目标环内时，鼓励保持较小的径向速度，避免穿过猎物
        radial_dir = rel_vec / rel_dist
        radial_speed = np.dot(agent_vel, radial_dir)
        radial_flow_potential = -abs(radial_speed)

    components["radial_flow_potential"] = w_radial_flow * radial_flow_potential

    # ---------------------------
    # 4. 压缩猎物通道（夹击）势场
    # ---------------------------
    squeeze_potential = 0.0
    if n_adv >= 2:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
        unit_vecs = rel_vecs / rel_dists[:, None]

        # 寻找与当前追捕者在猎物两侧大致对称的另一个追捕者
        best_sym_score = -1.0
        best_pair_idx = None
        for j_idx, j in enumerate(adversary_indices):
            if j == agent_index:
                continue
            v_self = rel_vec / rel_dist
            v_j = unit_vecs[j_idx]
            # 对称性度量：cos(π - θ) = -cos θ，越接近 -1 越对称
            cos_theta = np.clip(np.dot(v_self, v_j), -1.0, 1.0)
            sym_score = -cos_theta  # -1 -> 1 (高好)
            if sym_score > best_sym_score:
                best_sym_score = sym_score
                best_pair_idx = j

        if best_pair_idx is not None:
            # 通道宽度 = 两追捕者连线在猎物处的投影
            other_pos = agent_positions[best_pair_idx]
            mid_point = 0.5 * (agent_pos + other_pos)
            # 猎物到中点距离越小，通道越窄
            channel_half_width = np.linalg.norm(mid_point - prey_pos)
            # 期望的通道宽度（稍大于猎物尺寸）
            desired_width = prey_size * 3.0
            width_error = channel_half_width - desired_width
            # 对通道过宽施加惩罚，适度鼓励压缩
            squeeze_potential = -width_error

            # 对称性补偿：如果对称性很差，不鼓励压缩（避免一边挤压）
            squeeze_potential *= max(best_sym_score, 0.0)

    components["squeeze_potential"] = w_squeeze * squeeze_potential

    # ---------------------------
    # 5. 追捕者安全间距势场（防止拥挤与碰撞）
    # ---------------------------
    safe_distance = 2.0 * adv_size + 0.02
    safe_distance_potential = 0.0
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j]
        if d_aa < safe_distance:
            # 类似软势场惩罚，越近越大惩罚
            safe_distance_potential -= (safe_distance - d_aa) ** 2

    components["safe_distance_potential"] = w_safe_distance * safe_distance_potential

    # ---------------------------
    # 6. 边界势场：避免远离猎物追到边缘
    # ---------------------------
    boundary_potential = 0.0
    # 使用软墙势场：越靠近边界惩罚越大
    margin = world_size * 0.1
    for k in range(2):
        dist_to_pos_wall = (world_size - margin) - agent_pos[k]
        dist_to_neg_wall = (world_size - margin) + agent_pos[k]
        if dist_to_pos_wall < 0.0:
            boundary_potential -= dist_to_pos_wall ** 2
        if dist_to_neg_wall < 0.0:
            boundary_potential -= dist_to_neg_wall ** 2

    components["boundary_potential"] = w_boundary * boundary_potential

    # ---------------------------
    # 7. 围捕成功势场：全局合围完成奖励
    # ---------------------------
    capture_reward = 0.0
    if n_adv > 0:
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            # 使用一份全局结构性指标作为质量评分：
            adv_positions = agent_positions[adversary_indices]
            rel_vecs = adv_positions - prey_pos
            rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-8
            rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

            # 半径统一性
            radius_spread = np.max(rel_dists) - np.min(rel_dists)
            radius_score = np.exp(-radius_spread / (capture_threshold + 1e-6))

            # 角度均匀性
            rel_angles_sorted = np.sort(rel_angles)
            deltas = np.diff(np.concatenate([rel_angles_sorted, rel_angles_sorted[:1] + 2.0 * np.pi]))
            target_delta = 2.0 * np.pi / float(n_adv)
            angle_error_mean = np.mean(np.abs(deltas - target_delta))
            angle_score = np.exp(-angle_error_mean / np.pi)

            quality_score = 0.5 * (radius_score + angle_score)
            capture_reward = w_capture * (0.5 + 0.5 * quality_score)

    components["capture_reward"] = capture_reward

    # ---------------------------
    # 8. 轻微时间成本
    # ---------------------------
    step_cost = -w_step_cost
    components["step_cost"] = step_cost

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
