# LLM Interaction Log

- **Generation**: 4
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker1_F2
- **Timestamp**: 20260414_164934

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
    adversary_indices = list(range(n_adv))

    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    d_ap = np.linalg.norm(agent.state.p_pos - prey_pos)

    # ---------------------------
    # 1. 距离引导奖励（重构：从固定目标半径改为分段 / 平滑靠近奖励）
    # ---------------------------
    # 诊断显示：包围半径偏大，平均距猎物较远
    # 重构逻辑：
    #   - 当距离 > capture_threshold 时：主要奖励向猎物靠近（相对 world_size 归一化）
    #   - 当距离 <= capture_threshold 时：弱化惩罚，给予轻微的“保持在圈内”奖励
    #   - 全局平均距离使用 smooth-min 型函数，避免极端负值主导
    far_scale = world_size
    # 归一化距离
    norm_dist = d_ap / (far_scale + 1e-6)
    # 远距离段：单调增加的负奖励，但上限截断
    if d_ap > capture_threshold:
        # 使用平滑函数 1 - exp(-k * norm_dist)，再取负，限制最远惩罚
        k_far = 3.0
        distance_shaping_self = - (1.0 - np.exp(-k_far * norm_dist))
    else:
        # 已在捕获半径内：轻微鼓励保持（平缓正奖励，防止强负）
        margin = max(capture_threshold - d_ap, 0.0) / (capture_threshold + 1e-6)
        distance_shaping_self = 0.2 * margin

    # 全局平均距离 - 使用 softplus 技巧平滑并压缩尺度
    if n_adv > 0:
        mean_dist_to_prey = float(np.mean(distances_to_prey))
    else:
        mean_dist_to_prey = d_ap
    norm_mean_dist = mean_dist_to_prey / (far_scale + 1e-6)
    k_global = 2.0
    # 负的平滑函数，避免极端大负值
    global_distance_reward = -np.log1p(np.exp(k_global * norm_mean_dist))

    components["distance_shaping_self"] = 0.6 * distance_shaping_self
    components["distance_shaping_global"] = 0.4 * global_distance_reward

    # ---------------------------
    # 2. 防碰撞奖励（追捕者-追捕者）
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
    # 3. 队形奖励：均匀包围圈（重构，使其不再强烈主导负向）
    # ---------------------------
    formation_angle_reward = 0.0
    formation_radius_reward = 0.0
    containment_bonus = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-6
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        # 角度均匀性：使用相对误差+饱和的 smooth penalty
        target_delta = 2.0 * np.pi / float(n_adv)
        angles_sorted = np.sort(rel_angles)
        deltas = []
        for i in range(n_adv - 1):
            deltas.append(angles_sorted[i + 1] - angles_sorted[i])
        deltas.append((angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        deltas = np.array(deltas)

        # 归一化角度误差
        rel_angle_error = np.abs(deltas - target_delta) / (target_delta + 1e-6)
        # 使用 1 - 1 / (1 + x) 映射到 (0, 1)，再取负避免大负值
        angle_error_smooth = np.mean(1.0 - 1.0 / (1.0 + rel_angle_error))
        formation_angle_reward = -angle_error_smooth

        # 半径均匀性：相对 spread, 再做平滑
        radius_spread = np.max(rel_dists) - np.min(rel_dists)
        mean_radius = float(np.mean(rel_dists))
        if mean_radius > 1e-6:
            rel_radius_spread = radius_spread / mean_radius
        else:
            rel_radius_spread = 0.0
        radius_error_smooth = 1.0 - 1.0 / (1.0 + rel_radius_spread)
        formation_radius_reward = -radius_error_smooth

        # 包含性保持不变
        if n_adv == 3:
            p = prey_pos
            a, b, c = adv_positions

            def _sign(p1, p2, p3):
                return (
                    (p1[0] - p3[0]) * (p2[1] - p3[1])
                    - (p2[0] - p3[0]) * (p1[1] - p3[1])
                )

            b1 = _sign(p, a, b) < 0.0
            b2 = _sign(p, b, c) < 0.0
            b3 = _sign(p, c, a) < 0.0
            is_inside = (b1 == b2) and (b2 == b3)
            if is_inside:
                containment_bonus = 2.0

    # 下调权重，避免主导强负
    components["formation_angle_reward"] = 0.3 * formation_angle_reward
    components["formation_radius_reward"] = 0.3 * formation_radius_reward
    components["containment_bonus"] = containment_bonus

    # ---------------------------
    # 4. 时间惩罚（重构：从固定步惩罚改为基于距离的软惩罚）
    # ---------------------------
    # 诊断：依赖“耗时间+偶然合围”，固定时间惩罚可能过强
    # 新逻辑：离得越近，时间惩罚越弱；远距离时适度惩罚拖延
    norm_mean_dist_for_time = mean_dist_to_prey / (far_scale + 1e-6)
    base_time_penalty = -0.01
    # 使用 sigmoid 缩放：越近惩罚越接近 0
    k_time = 5.0
    time_scale = 1.0 / (1.0 + np.exp(-k_time * (norm_mean_dist_for_time - 0.3)))
    time_penalty = base_time_penalty * time_scale
    components["time_penalty"] = time_penalty

    # ---------------------------
    # 5. 围捕完成奖励（重构：从固定+质量线性耦合改为平滑质量加成）
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            # 将 formation 指标转换为 [0,1] 的质量得分，再进行平滑加成
            # 角度与半径部分已为负数（越接近0越好）
            angle_quality = np.exp(formation_angle_reward)  # in (0,1]
            radius_quality = np.exp(formation_radius_reward)
            quality_score = 0.5 * angle_quality + 0.5 * radius_quality
            # 使用 log1p 防止极端放大
            capture_bonus = 8.0 + 4.0 * np.log1p(quality_score)
    components["capture_bonus"] = capture_bonus

    # ---------------------------
    # 6. 协同收缩奖励（重构：避免与 distance_shaping 叠加过强）
    # ---------------------------
    cooperative_shrink_reward = 0.0
    if n_adv > 1:
        team_mean_radius = float(np.mean(distances_to_prey))
        norm_team_radius = team_mean_radius / (far_scale + 1e-6)
        # 使用软负奖励，并在进入捕获半径后快速饱和
        k_team = 4.0
        cooperative_shrink_reward = -0.05 * (
            1.0 - np.exp(-k_team * max(norm_team_radius - capture_threshold / far_scale, 0.0))
        )
    components["cooperative_shrink_reward"] = cooperative_shrink_reward

    # ---------------------------
    # 7. 协同对称收缩奖励（重构：从线性绝对误差改为平滑+归一化）
    # ---------------------------
    symmetric_shrink_reward = 0.0
    if n_adv > 1:
        team_mean_radius = float(np.mean(distances_to_prey))
        local_distance = float(distances_to_prey[local_adv_idx])
        if team_mean_radius > 1e-6:
            rel_diff = abs(local_distance - team_mean_radius) / team_mean_radius
        else:
            rel_diff = 0.0
        # 使用 1 / (1 + rel_diff) ∈ (0,1]，鼓励对称
        symmetric_shrink_reward = 0.05 * (1.0 / (1.0 + rel_diff) - 0.5)
    components["symmetric_shrink_reward"] = symmetric_shrink_reward

    # ---------------------------
    # 8. 协同进攻方向一致性奖励（保持逻辑，仅稍作平滑）
    # ---------------------------
    cooperative_direction_reward = 0.0
    if n_adv > 1:
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

        to_prey_align = np.einsum("ij,ij->i", adv_vels, adv_dir_to_prey)
        local_to_prey_align = float(to_prey_align[local_adv_idx])

        mean_vel_dir = np.mean(adv_vels, axis=0)
        mean_vel_dir_norm = mean_vel_dir / (np.linalg.norm(mean_vel_dir) + 1e-8)
        team_align = float(np.dot(adv_vels[local_adv_idx], mean_vel_dir_norm))

        # 将对齐度从 [-1,1] 映射到 [-0.5,0.5] 再缩放
        align_self_scaled = 0.5 * local_to_prey_align
        align_team_scaled = 0.5 * team_align
        cooperative_direction_reward = 0.08 * align_self_scaled + \
            0.08 * align_team_scaled

    components["cooperative_direction_reward"] = cooperative_direction_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
